from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import dataset_coviar as cvd
from engine_emclip import load_checkpoint, load_model_checkpoint, save_checkpoint
from main_emclip import resolve_implementation
from models import EMCLIP, EMCLIPConfig
from models.emclip_mgse import select_topk_indices


def _config(**overrides):
    values = dict(
        num_classes=3, class_names=["pulling left", "pulling right", "opening"],
        candidate_frames=4, selected_frames=2, input_size=32,
        width=32, layers=2, heads=4, embed_dim=16,
        text_width=32, text_heads=4, text_layers=2,
    )
    values.update(overrides)
    return EMCLIPConfig(**values)


def _inputs():
    return (torch.randn(2, 4, 3, 32, 32), torch.randn(2, 4, 2, 32, 32),
            torch.randn(2, 4, 3, 32, 32))


@pytest.mark.parametrize("layers", [2, 12])
def test_every_layer_shares_eq16_residual_input_and_generates_two_prompts(layers):
    model = EMCLIP(_config(layers=layers)).eval()
    captured = {layer: {} for layer in range(layers)}
    hooks = []
    for layer in range(layers):
        def gs_hook(module, inputs, layer=layer):
            captured[layer]["gs_kv"] = inputs[1]
        def lm_hook(module, inputs, layer=layer):
            captured[layer]["lm_kv"] = inputs[1]
        def sag_hook(module, inputs, layer=layer):
            captured[layer]["sag_shape"] = inputs[0].shape
        hooks.extend([
            model.melsc.gs_attn[layer].register_forward_pre_hook(gs_hook),
            model.melsc.lm_attn[layer].register_forward_pre_hook(lm_hook),
            model.melsc.i_encoder.blocks[layer].register_forward_pre_hook(sag_hook),
        ])
    try:
        with torch.no_grad():
            i_frames, _, residuals = _inputs()
            model.melsc(i_frames[:, :2], residuals[:, :2])
    finally:
        for hook in hooks:
            hook.remove()
    for layer in range(layers):
        assert captured[layer]["gs_kv"] is captured[layer]["lm_kv"]
        assert captured[layer]["gs_kv"].shape == (2, 2, 32)
        # Four 16x16 patches + CLS + two fresh prompts at every layer.
        assert captured[layer]["sag_shape"] == (4, 7, 32)


def test_acg_standardization_equals_population_variance_formula_without_affine_parameters():
    model = EMCLIP(_config())
    features = torch.randn(2, 4, 16) * 3 + 5
    centered = features - features.mean(dim=-1, keepdim=True)
    standardized = centered / torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = F.normalize(standardized, dim=-1)
    assert list(model.mgse.feature_ln.parameters()) == []
    assert torch.allclose(model.mgse._prepare_features(features), expected, atol=2e-7)
    assert torch.allclose(model.mgse._normalize_token_features(features), expected, atol=2e-7)


def test_fixed_classification_tau_ignores_checkpoint_logit_scale():
    torch.manual_seed(9)
    model = EMCLIP(_config()).eval()
    inputs = _inputs()
    with torch.no_grad():
        first = model(*inputs, training_mode=False)
        model.logit_scale.fill_(-3)
        second = model(*inputs, training_mode=False)
    expected = (first["video_features"] @ first["class_text_features"].t()) / 0.01
    assert first["classification_logit_scale"].item() == 100.0
    assert torch.allclose(first["logits"], expected, atol=1e-5)
    assert torch.equal(first["logits"], second["logits"])
    assert not model.logit_scale.requires_grad


def test_category_conditioned_training_and_label_independent_eval_can_run_in_one_model():
    torch.manual_seed(3)
    model = EMCLIP(_config())
    inputs = _inputs()
    assert model.mgse.train_text_mode == "ground_truth"
    assert model.mgse.text_mode == "class_bank"
    train_output = model(*inputs, labels=torch.tensor([0, 2]), training_mode=True)
    train_output["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        first = model(*inputs, labels=torch.tensor([0, 1]), training_mode=False)
        changed_labels = model(*inputs, labels=torch.tensor([2, 0]), training_mode=False)
        no_labels = model(*inputs, training_mode=False)
    assert torch.equal(first["selected_indices"], changed_labels["selected_indices"])
    assert torch.equal(first["selected_indices"], no_labels["selected_indices"])
    assert torch.equal(first["logits"], no_labels["logits"])


@pytest.mark.parametrize("gops,k", [(8, 8), (3, 8), (1, 8)])
def test_short_video_selects_all_distinct_gops_before_repeating(gops, k):
    candidates, mask, _ = cvd.sample_gop_indices(gops * 12, 16, 12, False)
    ids = torch.tensor(candidates)
    saliency = (ids.float() + 1)[None, :]
    selected = select_topk_indices(saliency, mask[None, :], k)[0]
    selected_gops = ids[selected]
    assert mask.sum().item() == gops
    assert selected_gops.unique().numel() == min(gops, k)
    assert mask[selected].all()
    assert selected.tolist() == sorted(selected.tolist())


def test_ssv2_flip_is_disabled_and_rgb_residual_matches_rgb_stem(tmp_path, monkeypatch):
    video = tmp_path / "1.mp4"
    video.touch()
    listing = tmp_path / "list.txt"
    listing.write_text("1.mp4 12 86\n", encoding="utf-8")
    monkeypatch.setattr(cvd, "coviar_load", lambda *args: None)
    dataset = cvd.CoviarDataSet("ssv2_mpeg4", str(listing), str(tmp_path), 174,
                              candidate_frames=1, input_size=4)
    image = torch.arange(48).reshape(1, 3, 4, 4).float()
    motion = torch.ones(1, 2, 4, 4)
    monkeypatch.setattr(cvd.random, "random", lambda: 0.0)
    result_i, result_mv, _ = dataset._augment(image, motion, image)
    assert torch.equal(result_i, image)
    assert torch.equal(result_mv, motion)
    bgr = np.array([[[10, 20, 30]]], dtype=np.int32)
    assert torch.equal(dataset._prepare_residual(bgr, (1, 1)), dataset._prepare_iframe(bgr))
    with pytest.raises(ValueError, match="label permutation"):
        cvd.CoviarDataSet("ssv2_mpeg4", str(listing), str(tmp_path), 174, horizontal_flip=True)


def test_order_encoding_is_an_explicit_ablation_and_changes_reversed_video():
    torch.manual_seed(5)
    model = EMCLIP(_config(temporal_position_encoding="sinusoidal")).eval()
    i_frames, _, residuals = _inputs()
    with torch.no_grad():
        forward = model.melsc(i_frames, residuals)["video_features"]
        reversed_frames = model.melsc(i_frames.flip(1), residuals.flip(1))["video_features"]
    assert _config().temporal_position_encoding == "none"
    assert (forward - reversed_frames).abs().max().item() > 1e-5


def _args(path=None, implementation="auto", resume=True):
    return SimpleNamespace(
        resume=str(path) if path and resume else None,
        init_checkpoint=str(path) if path and not resume else None,
        emclip_implementation=implementation, eval=False,
        residual_channel_order=None, duplicate_gop_policy=None,
    )


def test_old_checkpoint_is_identified_without_silent_migration(tmp_path):
    source = EMCLIP(_config(implementation="legacy"))
    path = tmp_path / "old.pth"
    torch.save({"model": source.state_dict()}, path)
    args = _args(path)
    resolve_implementation(args, explicit_options=set())
    assert args.emclip_implementation == "legacy"
    assert args.residual_channel_order == "bgr"
    assert args.duplicate_gop_policy == "keep"
    with pytest.raises(ValueError, match="No silent parameter migration"):
        resolve_implementation(_args(path, "paper"), explicit_options=set())
    restored = EMCLIP(_config(implementation="legacy"))
    load_checkpoint(path, restored)
    inputs = _inputs()
    source.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(source(*inputs)["logits"], restored(*inputs)["logits"])


def test_new_checkpoint_restores_model_input_settings_and_optimizer_states(tmp_path):
    source = EMCLIP(_config(temporal_position_encoding="sinusoidal"))
    optimizer = torch.optim.AdamW([p for p in source.parameters() if p.requires_grad], lr=8e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    source(*_inputs(), labels=torch.tensor([0, 1]))["loss"].backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "latest.pth"
    save_checkpoint(str(path), source, optimizer, scheduler, None, 4, 12.0,
                    run_config={"residual_channel_order": "rgb", "duplicate_gop_policy": "mask", "mv_clamp": 15.0})
    args = _args(path)
    resolve_implementation(args, explicit_options=set())
    assert args.emclip_implementation == "paper"
    assert args.temporal_position_encoding == "sinusoidal"
    assert args.mgse_train_text_mode == "ground_truth"
    assert args.mv_clamp == 15.0
    target = EMCLIP(EMCLIPConfig(**asdict(source.config)))
    target_optimizer = torch.optim.AdamW([p for p in target.parameters() if p.requires_grad], lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda _: 1.0)
    epoch, best = load_checkpoint(path, target, target_optimizer, target_scheduler)
    assert (epoch, best) == (5, 12.0)
    assert target_optimizer.param_groups[0]["lr"] == 8e-6
    assert target_scheduler.last_epoch == scheduler.last_epoch
    assert len(target_optimizer.state) == len(optimizer.state)
    for original, restored in zip(optimizer.state.values(), target_optimizer.state.values()):
        assert torch.equal(original["exp_avg"], restored["exp_avg"])
    # Dataset text banks may differ for model-only initialization.
    transfer = EMCLIP(_config(class_names=["running", "sitting", "jumping"],
                             temporal_position_encoding="sinusoidal"))
    load_model_checkpoint(path, transfer)
    with pytest.raises(ValueError, match="class text/label order"):
        load_checkpoint(path, transfer)


def test_resume_detects_changed_trainable_parameter_order(tmp_path):
    model = EMCLIP(_config())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-6)
    path = tmp_path / "latest.pth"
    save_checkpoint(str(path), model, optimizer, None, None, 0, 0)
    model.apply_train_mode("freeze_text")
    changed_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-6)
    with pytest.raises(RuntimeError, match="parameter order/train mode"):
        load_checkpoint(path, model, changed_optimizer)


def test_tau_001_correlations_and_losses_stay_float32_under_cpu_autocast():
    if not hasattr(torch, "autocast"):
        pytest.skip("CPU autocast is unavailable in this PyTorch version")
    model = EMCLIP(_config())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(*_inputs(), labels=torch.tensor([0, 1]))
    assert output["logits"].dtype == torch.float32
    assert output["saliency"].dtype == torch.float32
    assert output["loss"].dtype == torch.float32
    output["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
