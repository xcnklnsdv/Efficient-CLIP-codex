from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from engine_emclip import load_checkpoint, save_checkpoint
from main_emclip import resolve_implementation
from models import EMCLIP, EMCLIPConfig
from models.emclip_mgse import MotionGuidedSaliencyExtraction


def _config(**overrides):
    values = dict(num_classes=2, class_names=["jumping", "walking"],
                  candidate_frames=4, selected_frames=2, input_size=32,
                  width=32, layers=2, heads=4, embed_dim=16,
                  text_width=32, text_layers=2, text_heads=4)
    values.update(overrides)
    return EMCLIPConfig(**values)


def test_eq23_temporal_aggregation_consumes_raw_last_layer_cls():
    model = EMCLIP(_config()).eval()
    captured = {}

    def capture_cls(module, inputs, output):
        captured["cls"] = output[:, 0].reshape(2, 2, 32)

    def capture_temporal(module, inputs):
        captured["temporal"] = inputs[0]

    hooks = [model.melsc.i_encoder.blocks[-1].register_forward_hook(capture_cls),
             model.melsc.temporal_blocks[0].register_forward_pre_hook(capture_temporal)]
    try:
        with torch.no_grad():
            model.melsc(torch.randn(2, 2, 3, 32, 32), torch.randn(2, 2, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()
    assert torch.equal(captured["temporal"], captured["cls"])
    assert not any(p.requires_grad for p in model.melsc.i_encoder.ln_post.parameters())


@pytest.mark.parametrize("historical_implementation", ["paper", "legacy"])
def test_historical_checkpoint_restores_previous_norm_order(tmp_path, historical_implementation):
    source = EMCLIP(_config(implementation=historical_implementation,
                           melsc_norm_order="pre_and_post"))
    stored = asdict(source.config)
    stored.pop("melsc_norm_order")  # Checkpoints written before this repair.
    path = tmp_path / "historical.pth"
    torch.save({"model": source.state_dict(), "model_config": stored}, path)
    args = SimpleNamespace(resume=str(path), init_checkpoint=None, eval=True,
                           emclip_implementation="auto", melsc_norm_order=None,
                           residual_channel_order=None, duplicate_gop_policy=None)
    resolve_implementation(args, explicit_options=set())
    assert args.melsc_norm_order == "pre_and_post"
    target = EMCLIP(_config(implementation=args.emclip_implementation,
                           melsc_norm_order=args.melsc_norm_order))
    load_checkpoint(path, target)
    frames = torch.randn(2, 2, 3, 32, 32)
    residuals = torch.randn_like(frames)
    with torch.no_grad():
        expected = source.melsc(frames, residuals)["video_features"]
        actual = target.melsc(frames, residuals)["video_features"]
    assert torch.equal(actual, expected)


def test_loading_weights_invalidates_the_evaluation_text_bank(tmp_path):
    target = EMCLIP(_config()).eval()
    donor = EMCLIP(_config())
    with torch.no_grad():
        donor.text_encoder.text_projection.add_(0.5)
        old = target.text_encoder.encode_class_prompts(False)[0].clone()
    path = tmp_path / "changed.pth"
    save_checkpoint(str(path), donor, None, None, None, 0, 0.0)
    load_checkpoint(path, target)
    with torch.no_grad():
        loaded = target.text_encoder.encode_class_prompts(False)[0]
        expected = donor.text_encoder.encode_class_prompts(False)[0]
    assert torch.allclose(loaded, expected)
    assert not torch.allclose(loaded, old)


def test_evaluation_text_bank_does_not_reuse_an_old_autograd_graph():
    model = EMCLIP(_config()).eval()
    with torch.no_grad():
        model.text_encoder.encode_class_prompts(False)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        features = model.text_encoder.encode_class_prompts(False)[0]
        features[:, 0].sum().backward()
        assert model.text_encoder.text_projection.grad is not None
        assert torch.isfinite(model.text_encoder.text_projection.grad).all()


def test_predicted_class_uses_the_projection_space_trained_by_l_mg():
    mgse = MotionGuidedSaliencyExtraction(width=16, layers=1, heads=2, output_dim=3,
                                         input_resolution=32, selected_frames=1,
                                         text_mode="predicted_class")
    # The projected motion is aligned with class 0. Standardizing it gives
    # class 1, so a second normalization space must not drive the prediction.
    raw_features = torch.tensor([[[4.0, 3.0, 3.0], [8.0, 6.0, 7.0]]])
    mgse._encode_motion = lambda _: raw_features
    classes = F.normalize(torch.stack([raw_features.mean(1)[0],
                                       mgse._prepare_features(raw_features).mean(1)[0]]), dim=-1)
    tokens = raw_features[0, :, None, :]
    mask = torch.ones(2, 1, dtype=torch.bool)
    valid = torch.ones(1, 2, dtype=torch.bool)
    expected = mgse._ground_truth_saliency(mgse._prepare_features(raw_features),
                                          mgse._normalize_token_features(tokens),
                                          mask, torch.tensor([0]), valid)
    actual = mgse(torch.zeros(1, 2, 2, 32, 32), classes, tokens, mask,
                  valid_mask=valid, training_mode=False)
    assert torch.allclose(actual["saliency"], expected)
