"""Compare real two-rank training gradients with a single global batch."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from models import EMCLIP, EMCLIPConfig


def _global_batch_training_worker(rank, rendezvous, variant):
    torch.set_num_threads(1)
    torch.manual_seed(49)
    config = EMCLIPConfig(
        num_classes=3, class_names=["jumping", "walking", "running"],
        emclip_variant=variant, candidate_frames=4, selected_frames=2,
        input_size=32, width=32, layers=2, heads=4, embed_dim=16,
        text_width=32, text_layers=2, text_heads=4,
    )
    reference = EMCLIP(config)
    initial = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    frames = torch.randn(4, 4, 3, 32, 32)
    motion = torch.randn(4, 4, 2, 32, 32)
    residuals = torch.randn_like(frames)
    labels = torch.tensor([0, 1, 0, 2])  # Same-class positives span both ranks.
    valid = torch.ones(4, 4, dtype=torch.bool)
    optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad],
                                  lr=8e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    # Compute the reference before initializing distributed collectives.
    expected = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = reference(frames, motion, residuals, labels=labels, valid_mask=valid)
        out["loss"].backward()
        gradients = {name: p.grad.detach().clone() for name, p in reference.named_parameters()
                     if p.requires_grad}
        torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        expected.append((float(out["loss"].detach()), gradients,
                         {name: p.detach().clone() for name, p in reference.named_parameters()
                          if p.requires_grad}))
    del reference, optimizer

    store = dist.FileStore(rendezvous, 2)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        model = EMCLIP(config)
        model.load_state_dict(initial)
        ddp = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad],
                                      lr=8e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        selected = slice(rank * 2, (rank + 1) * 2)
        for expected_loss, expected_gradients, expected_parameters in expected:
            optimizer.zero_grad(set_to_none=True)
            out = ddp(frames[selected], motion[selected], residuals[selected],
                      labels=labels[selected], valid_mask=valid[selected])
            out["loss"].backward()
            loss = out["loss"].detach().clone()
            dist.all_reduce(loss)
            torch.testing.assert_close(loss / 2, torch.tensor(expected_loss), rtol=3e-5, atol=3e-5)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, name
                    assert torch.isfinite(parameter.grad).all(), name
                    # FP32 LayerNorm/temperature=0.01 can produce large text
                    # gradients. Different batch/reduction groupings round
                    # differently, especially near zero. Compare relative
                    # vector error, which detects world-size scaling errors
                    # without requiring bitwise per-element equivalence.
                    error = (parameter.grad - expected_gradients[name]).norm()
                    tolerance = 2e-5 * expected_gradients[name].norm() + 3e-5
                    assert error <= tolerance, "%s: gradient error=%g tolerance=%g" % (name, error, tolerance)
            torch.nn.utils.clip_grad_norm_([p for p in ddp.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, expected_parameters[name],
                                               rtol=3e-5, atol=3e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", ["emclip", "diamond"])
def test_two_rank_gradients_and_optimizer_updates_match_global_batch(tmp_path, variant):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    mp.spawn(_global_batch_training_worker,
             args=(str(tmp_path / "gloo_store"), variant), nprocs=2, join=True)
