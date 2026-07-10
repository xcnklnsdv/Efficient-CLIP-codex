import torch

from losses.emclip_loss import build_multi_positive_target, motion_text_kl_loss


def test_multi_positive_target_groups_same_labels():
    labels = torch.tensor([1, 2, 1])

    target = build_multi_positive_target(labels, labels)

    expected = torch.tensor([
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5],
    ])
    assert torch.equal(target, expected)


def test_motion_text_kl_loss_is_finite_for_batch_size_one():
    motion = torch.randn(1, 4, 16)
    saliency = torch.ones(1, 4) / 4
    text = torch.randn(3, 16)
    labels = torch.tensor([2])

    out = motion_text_kl_loss(motion, saliency, text, labels, temperature=0.01)

    assert torch.isfinite(out["loss_mg"])
    assert out["loss_mg"].ndim == 0


def test_motion_text_kl_loss_handles_amp_sensitive_temperature_in_float32():
    motion = torch.randn(2, 4, 16).half()
    saliency = torch.softmax(torch.randn(2, 4), dim=1).half()
    text = torch.randn(3, 16).half()
    labels = torch.tensor([1, 1])

    out = motion_text_kl_loss(motion, saliency, text, labels, temperature=0.01)

    assert torch.isfinite(out["loss_mg"])
    assert torch.isfinite(out["loss_mg_mv2text"])
    assert torch.isfinite(out["loss_mg_text2mv"])
