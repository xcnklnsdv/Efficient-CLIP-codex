import pytest
import torch

from models.emclip_mgse import (
    MotionGuidedSaliencyExtraction,
    build_mv_patch_embed_weight,
    gather_temporal,
    select_topk_indices,
)


def test_mv_patch_embed_weight_uses_rgb_channel_mean_with_scale():
    rgb = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).view(4, 3, 2, 2)

    mv = build_mv_patch_embed_weight(rgb, scale=True)

    expected = rgb.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1) * 1.5
    assert mv.shape == (4, 2, 2, 2)
    assert torch.equal(mv, expected)


def test_topk_respects_mask_sorts_time_and_gathers_per_batch():
    saliency = torch.tensor([
        [0.1, 0.9, 0.8, 0.7],
        [0.8, 0.1, 0.7, 0.9],
    ])
    valid_mask = torch.tensor([
        [True, True, True, False],
        [True, False, True, True],
    ])
    frames = torch.arange(2 * 4 * 1 * 1 * 1).view(2, 4, 1, 1, 1)

    indices = select_topk_indices(saliency, valid_mask, k=2)
    selected = gather_temporal(frames, indices)

    assert indices.tolist() == [[1, 2], [0, 3]]
    assert selected[:, :, 0, 0, 0].tolist() == [[1, 2], [4, 7]]


def test_topk_repeats_valid_indices_when_valid_count_is_smaller_than_k():
    saliency = torch.tensor([[0.2, 0.9, 0.3]])
    valid_mask = torch.tensor([[False, True, False]])

    indices = select_topk_indices(saliency, valid_mask, k=2)

    assert indices.tolist() == [[1, 1]]


def test_topk_rejects_k_larger_than_candidate_count():
    with pytest.raises(ValueError, match="selected_frames"):
        select_topk_indices(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), k=3)


def test_ground_truth_mode_is_blocked_during_eval_without_diagnostic_flag():
    mgse = MotionGuidedSaliencyExtraction(
        width=32,
        layers=1,
        heads=4,
        output_dim=16,
        patch_size=16,
        selected_frames=2,
        temperature=0.01,
        text_mode="ground_truth",
    )
    mv = torch.randn(2, 4, 2, 64, 64)
    text = torch.randn(5, 16)
    token_features = torch.randn(5, 3, 16)
    token_mask = torch.ones(5, 3, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="label leakage"):
        mgse(
            mv,
            class_text_features=text,
            class_token_features=token_features,
            class_token_mask=token_mask,
            labels=torch.tensor([0, 1]),
            valid_mask=torch.ones(2, 4, dtype=torch.bool),
            training_mode=False,
        )


def test_class_bank_mode_does_not_use_labels_for_selection():
    torch.manual_seed(0)
    mgse = MotionGuidedSaliencyExtraction(
        width=32,
        layers=1,
        heads=4,
        output_dim=16,
        patch_size=16,
        selected_frames=2,
        temperature=0.01,
        text_mode="class_bank",
    )
    mv = torch.randn(2, 4, 2, 64, 64)
    text = torch.randn(5, 16)
    token_features = torch.randn(5, 3, 16)
    token_mask = torch.ones(5, 3, dtype=torch.bool)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)

    out_a = mgse(
        mv,
        class_text_features=text,
        class_token_features=token_features,
        class_token_mask=token_mask,
        labels=torch.tensor([0, 1]),
        valid_mask=valid_mask,
        training_mode=False,
    )
    out_b = mgse(
        mv,
        class_text_features=text,
        class_token_features=token_features,
        class_token_mask=token_mask,
        labels=torch.tensor([4, 3]),
        valid_mask=valid_mask,
        training_mode=False,
    )

    assert torch.equal(out_a["selected_indices"], out_b["selected_indices"])
