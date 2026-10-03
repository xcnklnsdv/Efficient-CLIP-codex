import pytest
import torch

from models.emclip import EMCLIP, EMCLIPConfig


def test_emclip_synthetic_forward_shapes_and_losses_are_finite():
    torch.manual_seed(0)
    config = EMCLIPConfig(
        num_classes=5,
        class_names=[f"class {i}" for i in range(5)],
        candidate_frames=4,
        selected_frames=2,
        input_size=64,
        patch_size=16,
        width=32,
        layers=2,
        heads=4,
        embed_dim=16,
        text_width=32,
        text_heads=4,
        temporal_aggregator_layers=1,
        mgse_temperature=0.01,
        emclip_variant="emclip",
    )
    model = EMCLIP(config)
    i_frames = torch.randn(2, 4, 3, 64, 64)
    motion_vectors = torch.randn(2, 4, 2, 64, 64)
    residuals = torch.randn(2, 4, 3, 64, 64)
    labels = torch.tensor([0, 3])

    out = model(
        i_frames=i_frames,
        motion_vectors=motion_vectors,
        residuals=residuals,
        labels=labels,
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        training_mode=True,
    )

    assert out["motion_frame_features"].shape == (2, 4, 16)
    assert out["saliency"].shape == (2, 4)
    assert out["selected_indices"].shape == (2, 2)
    assert out["selected_i_frames"].shape == (2, 2, 3, 64, 64)
    assert out["selected_residuals"].shape == (2, 2, 3, 64, 64)
    assert out["melsc_debug"]["gspl"].shape == (2, 2, 32)
    assert out["melsc_debug"]["lmpl"].shape == (2, 2, 32)
    assert out["melsc_debug"]["residual_frame_cls"].shape == (2, 2, 32)
    assert out["video_features"].shape == (2, 16)
    assert out["logits"].shape == (2, 5)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_mg"])
    assert torch.isfinite(out["loss_me"])


def test_emclip_diamond_uses_i_and_residual_without_mgse_loss():
    config = EMCLIPConfig(
        num_classes=4,
        class_names=[f"class {i}" for i in range(4)],
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
        temporal_aggregator_layers=1,
        emclip_variant="diamond",
    )
    model = EMCLIP(config)
    assert not model.melsc.r_encoder.proj.requires_grad
    assert not any(param.requires_grad for param in model.melsc.r_encoder.ln_post.parameters())
    assert not any(param.requires_grad for param in model.melsc.r_encoder.blocks[-1].parameters())

    residual_block_calls = []
    hook = model.melsc.r_encoder.blocks[-1].register_forward_hook(
        lambda *_: residual_block_calls.append(True)
    )
    out = model(
        i_frames=torch.randn(2, 4, 3, 64, 64),
        motion_vectors=None,
        residuals=torch.randn(2, 4, 3, 64, 64),
        labels=torch.tensor([0, 1]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        training_mode=True,
    )
    hook.remove()

    assert out["selected_indices"].shape == (2, 2)
    assert out["loss_mg"].item() == 0.0
    assert torch.isfinite(out["loss"])
    assert residual_block_calls == [True]


@pytest.mark.parametrize("implementation", ["paper", "legacy"])
def test_emclip_full_protocol_backward_has_no_unused_trainable_parameters(implementation):
    torch.manual_seed(0)
    config = EMCLIPConfig(
        num_classes=5,
        class_names=[f"class {i}" for i in range(5)],
        candidate_frames=4,
        selected_frames=2,
        input_size=64,
        patch_size=16,
        width=32,
        layers=2,
        heads=4,
        embed_dim=16,
        text_width=32,
        text_heads=4,
        text_layers=2,
        temporal_aggregator_layers=1,
        mgse_temperature=0.01,
        emclip_variant="emclip",
        emclip_train_mode="full",
        mgse_text_mode="class_bank",
        motion_pooling="saliency",
        lambda_mg=1.0,
        lambda_me=1.0,
        implementation=implementation,
    )
    model = EMCLIP(config)

    out = model(
        i_frames=torch.randn(2, 4, 3, 64, 64),
        motion_vectors=torch.randn(2, 4, 2, 64, 64),
        residuals=torch.randn(2, 4, 3, 64, 64),
        labels=torch.tensor([0, 3]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        training_mode=True,
    )

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_mg"])
    assert torch.isfinite(out["loss_me"])
    out["loss"].backward()

    unused = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert unused == []
    nonfinite = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    assert nonfinite == []
    params = dict(model.named_parameters())
    expected_grad_names = [
        "text_encoder.token_embedding.weight",
        "mgse.motion_encoder.conv1.weight",
        "melsc.i_encoder.conv1.weight",
        "melsc.r_encoder.conv1.weight",
        "melsc.r_encoder.blocks.0.attn.in_proj_weight",
        "melsc.temporal_blocks.0.attn.in_proj_weight",
    ]
    if implementation == "legacy":
        expected_grad_names.extend(["logit_scale", "mgse.feature_ln.weight"])
    else:
        assert not model.logit_scale.requires_grad
        assert "mgse.feature_ln.weight" not in params
    for name in expected_grad_names:
        assert params[name].requires_grad
        assert params[name].grad is not None, name


def test_emclip_mean_motion_pooling_freezes_saliency_only_layernorm():
    config = EMCLIPConfig(
        num_classes=5,
        class_names=[f"class {i}" for i in range(5)],
        candidate_frames=4,
        selected_frames=2,
        input_size=64,
        patch_size=16,
        width=32,
        layers=2,
        heads=4,
        embed_dim=16,
        text_width=32,
        text_heads=4,
        text_layers=2,
        temporal_aggregator_layers=1,
        emclip_variant="emclip",
        emclip_train_mode="full",
        mgse_text_mode="class_bank",
        motion_pooling="mean",
    )

    model = EMCLIP(config)

    assert not any(param.requires_grad for param in model.mgse.feature_ln.parameters())


def test_ground_truth_diagnostic_uses_separate_selection_labels_during_eval():
    config = EMCLIPConfig(
        num_classes=3,
        class_names=["running", "jumping", "sitting"],
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
        mgse_text_mode="ground_truth",
        allow_mgse_label_leakage_for_diagnostic=True,
    )
    model = EMCLIP(config).eval()

    output = model(
        i_frames=torch.randn(2, 4, 3, 64, 64),
        motion_vectors=torch.randn(2, 4, 2, 64, 64),
        residuals=torch.randn(2, 4, 3, 64, 64),
        labels=None,
        mgse_labels=torch.tensor([0, 2]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        training_mode=False,
    )

    assert output["loss"] is None
    assert output["selected_indices"].shape == (2, 2)


def test_freeze_clip_only_trains_new_prompt_and_aggregation_parameters():
    config = EMCLIPConfig(
        num_classes=2,
        class_names=["running", "jumping"],
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
        emclip_train_mode="freeze_clip",
    )
    model = EMCLIP(config)

    assert not model.logit_scale.requires_grad
    assert not model.melsc.i_encoder.proj.requires_grad
    assert not model.text_encoder.token_embedding.weight.requires_grad
    assert model.mgse.motion_encoder.proj.requires_grad
    assert model.melsc.gs_i_proj[0].weight.requires_grad
    assert model.melsc.temporal_blocks[0].attn.in_proj_weight.requires_grad
