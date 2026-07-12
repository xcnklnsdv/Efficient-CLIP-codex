from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

import main_emclip
from engine_emclip import load_checkpoint
from models.clip_checkpoint import load_clip_source_checkpoint
from models.emclip import EMCLIP, EMCLIPConfig
from models.emclip_layers import interpolate_positional_embedding
from models.emclip_mgse import build_mv_patch_embed_weight


def _tiny_config(checkpoint=None):
    return EMCLIPConfig(
        num_classes=2,
        class_names=["jump", "sit"],
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
        text_context_length=8,
        vocab_size=64,
        temporal_aggregator_layers=1,
        clip_checkpoint=str(checkpoint) if checkpoint is not None else None,
    )


def _openai_clip_state(reference, source_grid=None):
    state = OrderedDict()
    for key, value in reference.melsc.i_encoder.state_dict().items():
        source_key = key.replace("blocks.", "transformer.resblocks.")
        tensor = value.detach().clone()
        if key == "positional_embedding" and source_grid is not None:
            tensor = interpolate_positional_embedding(tensor, (source_grid, source_grid))
        state["visual." + source_key] = tensor
    for key, value in reference.text_encoder.state_dict().items():
        source_key = key.replace("blocks.", "transformer.resblocks.")
        state[source_key] = value.detach().clone()
    state["logit_scale"] = reference.logit_scale.detach().clone()
    return state


def test_torchscript_checkpoint_extracts_state_dict(tmp_path):
    path = tmp_path / "scripted.pt"
    scripted = torch.jit.script(nn.Linear(3, 2))
    torch.jit.save(scripted, str(path))

    state, checkpoint_type, checkpoint_path = load_clip_source_checkpoint(path)

    assert checkpoint_type == "torchscript"
    assert checkpoint_path == str(path.resolve())
    assert set(state) == {"weight", "bias"}
    assert all(torch.is_tensor(value) for value in state.values())
    # Regression: a RecursiveScriptModule is no longer handed to
    # load_state_dict; the extracted mapping can be loaded normally.
    nn.Linear(3, 2).load_state_dict(state)


@pytest.mark.parametrize("container_key", [None, "state_dict", "model"])
def test_regular_state_dict_containers_are_supported(tmp_path, container_key):
    path = tmp_path / ("%s.pth" % (container_key or "direct"))
    expected = {"module.weight": torch.randn(2, 3), "module.bias": torch.randn(2)}
    payload = expected if container_key is None else {container_key: expected}
    torch.save(payload, path)

    state, checkpoint_type, _ = load_clip_source_checkpoint(path)

    assert checkpoint_type == "state_dict"
    assert set(state) == {"weight", "bias"}
    assert torch.equal(state["weight"], expected["module.weight"])


def test_serialized_nn_module_is_supported(tmp_path):
    path = tmp_path / "module.pth"
    torch.save(nn.Linear(3, 2), path)
    state, checkpoint_type, _ = load_clip_source_checkpoint(path)
    assert checkpoint_type == "state_dict"
    assert set(state) == {"weight", "bias"}


def test_clip_weights_are_mapped_to_independent_branches(tmp_path):
    torch.manual_seed(7)
    reference = EMCLIP(_tiny_config())
    source = _openai_clip_state(reference, source_grid=3)
    path = tmp_path / "clip_state.pth"
    torch.save({"model": source}, path)

    loaded = EMCLIP(_tiny_config(path))

    assert loaded.pretrained_audit["checkpoint_type"] == "state_dict"
    assert loaded.pretrained_audit["position_grids"]["I_encoder"] == (3, 4)
    for branch in loaded.pretrained_audit["branches"].values():
        assert branch["coverage"] >= 0.99
        assert branch["shape_mismatch_keys"] == []

    i_conv = loaded.melsc.i_encoder.conv1.weight
    r_conv = loaded.melsc.r_encoder.conv1.weight
    assert torch.equal(i_conv, source["visual.conv1.weight"])
    assert torch.equal(r_conv, source["visual.conv1.weight"])
    assert i_conv is not r_conv
    assert i_conv.data_ptr() != r_conv.data_ptr()
    assert torch.equal(
        loaded.melsc.i_encoder.blocks[0].attn.in_proj_weight,
        source["visual.transformer.resblocks.0.attn.in_proj_weight"],
    )
    expected_mv = build_mv_patch_embed_weight(source["visual.conv1.weight"], scale=True)
    assert loaded.mgse.motion_encoder.conv1.in_channels == 2
    assert torch.equal(loaded.mgse.motion_encoder.conv1.weight, expected_mv)
    assert torch.equal(
        loaded.mgse.motion_encoder.blocks[1].mlp.c_proj.weight,
        source["visual.transformer.resblocks.1.mlp.c_proj.weight"],
    )
    assert torch.equal(
        loaded.text_encoder.token_embedding.weight,
        source["token_embedding.weight"],
    )
    assert torch.equal(
        loaded.text_encoder.blocks[0].attn.out_proj.weight,
        source["transformer.resblocks.0.attn.out_proj.weight"],
    )
    assert torch.equal(loaded.text_encoder.text_projection, source["text_projection"])
    assert torch.equal(loaded.logit_scale, source["logit_scale"])


def test_low_coverage_checkpoint_fails_clearly(tmp_path):
    reference = EMCLIP(_tiny_config())
    complete = _openai_clip_state(reference)
    sparse = OrderedDict(
        (key, value) for key, value in complete.items()
        if not key.startswith("visual.") or key == "visual.conv1.weight"
    )
    path = tmp_path / "low_coverage.pth"
    torch.save(sparse, path)

    with pytest.raises(RuntimeError, match="coverage=.*required"):
        EMCLIP(_tiny_config(path))


def test_missing_checkpoint_path_fails_clearly(tmp_path):
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_clip_source_checkpoint(missing)


def test_non_tensor_state_value_is_rejected(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"weight": torch.randn(2, 2), "metadata": "not-a-tensor"}, path)
    with pytest.raises(TypeError, match="non-Tensor"):
        load_clip_source_checkpoint(path)


def test_resume_loader_rejects_an_original_clip_state_dict(tmp_path):
    path = tmp_path / "not-an-emclip-resume.pth"
    torch.save({"visual.conv1.weight": torch.randn(4, 3, 2, 2)}, path)
    with pytest.raises(TypeError, match="--resume expects"):
        load_checkpoint(path, nn.Linear(2, 2))


def test_main_preserves_error_and_destroys_process_group(monkeypatch):
    destroyed = []

    def fail_worker():
        raise RuntimeError("original failure")

    monkeypatch.setattr(main_emclip, "main_worker", fail_worker)
    monkeypatch.setattr(main_emclip.dist, "is_available", lambda: True)
    monkeypatch.setattr(main_emclip.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        main_emclip.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )
    with pytest.raises(RuntimeError, match="original failure"):
        main_emclip.main()
    assert destroyed == [True]
