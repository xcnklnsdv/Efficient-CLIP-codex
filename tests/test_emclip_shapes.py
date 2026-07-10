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
        temporal_aggregator_layers=1,
        emclip_variant="diamond",
    )
    model = EMCLIP(config)

    out = model(
        i_frames=torch.randn(2, 4, 3, 64, 64),
        motion_vectors=None,
        residuals=torch.randn(2, 4, 3, 64, 64),
        labels=torch.tensor([0, 1]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        training_mode=True,
    )

    assert out["selected_indices"].shape == (2, 2)
    assert out["loss_mg"].item() == 0.0
    assert torch.isfinite(out["loss"])
