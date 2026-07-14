from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from engine_emclip import _finish_optimizer_step, train_one_epoch
from main_emclip import DistributedEvalSampler
from models import EMCLIP, EMCLIPConfig


class _SyntheticCompressedDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "i_frames": torch.randn(4, 3, 64, 64),
            "motion_vectors": torch.randn(4, 2, 64, 64),
            "residuals": torch.randn(4, 3, 64, 64),
            "valid_mask": torch.ones(4, dtype=torch.bool),
            "label": torch.tensor(index, dtype=torch.long),
        }


def test_distributed_eval_sampler_never_pads_duplicate_samples():
    dataset = list(range(5))
    rank_zero = list(DistributedEvalSampler(dataset, num_replicas=2, rank=0))
    rank_one = list(DistributedEvalSampler(dataset, num_replicas=2, rank=1))

    assert rank_zero == [0, 2, 4]
    assert rank_one == [1, 3]
    assert sorted(rank_zero + rank_one) == list(range(5))


def _build_engine_components(lambda_mg=1.0):
    config = EMCLIPConfig(
        num_classes=2,
        class_names=["class 0", "class 1"],
        candidate_frames=4,
        selected_frames=2,
        input_size=64,
        patch_size=16,
        width=32,
        layers=1,
        heads=4,
        embed_dim=16,
        text_width=32,
        text_heads=4,
        text_layers=1,
        lambda_mg=lambda_mg,
    )
    model = EMCLIP(config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = SimpleNamespace(
        micro_batch_size=1,
        amp=False,
        grad_clip_norm=1.0,
        print_freq=100,
        debug_unused_parameters=False,
    )
    return model, optimizer, scheduler, scaler, args


def test_engine_rejects_contrastive_micro_batch_split():
    model, optimizer, scheduler, scaler, args = _build_engine_components(lambda_mg=1.0)

    with pytest.raises(ValueError, match="global positives and negatives"):
        train_one_epoch(
            model,
            DataLoader(_SyntheticCompressedDataset(), batch_size=2),
            optimizer,
            scheduler,
            scaler,
            torch.device("cpu"),
            epoch=0,
            args=args,
        )


def test_engine_accumulates_micro_batches_when_motion_kl_is_disabled():
    model, optimizer, scheduler, scaler, args = _build_engine_components(lambda_mg=0.0)

    stats = train_one_epoch(
        model,
        DataLoader(_SyntheticCompressedDataset(), batch_size=2),
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        epoch=0,
        args=args,
    )

    assert torch.isfinite(torch.tensor(stats["loss"]))
    assert torch.isfinite(torch.tensor(stats["acc1"]))
    assert "loss_mg_mv2text" in stats
    assert "loss_mg_text2mv" in stats


class _OverflowScaler:
    def __init__(self, scale=1024.0):
        self.scale = float(scale)
        self.step_called = False

    def unscale_(self, optimizer):
        return None

    def get_scale(self):
        return self.scale

    def is_enabled(self):
        return True

    def step(self, optimizer):
        # A real GradScaler skips optimizer.step after unscale_ found inf.
        self.step_called = True

    def update(self):
        self.scale *= 0.5


def test_amp_overflow_uses_grad_scaler_backoff_instead_of_immediate_failure():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    model.bias.grad = torch.zeros_like(model.bias)
    scaler = _OverflowScaler(scale=1024.0)

    result = _finish_optimizer_step(model, optimizer, scaler, grad_clip_norm=1.0)

    assert not result["stepped"]
    assert result["scale_before"] == 1024.0
    assert result["scale_after"] == 512.0
    assert result["bad_gradients"] == ["weight"]
    assert scaler.step_called


def test_non_amp_nonfinite_gradient_still_raises():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    model.bias.grad = torch.zeros_like(model.bias)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    with pytest.raises(FloatingPointError, match="without AMP"):
        _finish_optimizer_step(model, optimizer, scaler, grad_clip_norm=1.0)
