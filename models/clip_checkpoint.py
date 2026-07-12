"""Utilities for loading original OpenAI CLIP source checkpoints.

The ``--clip-checkpoint`` input is deliberately kept separate from an
EM-CLIP training checkpoint.  It may be an OpenAI TorchScript archive or a
regular PyTorch state-dict container, but it is never loaded into the whole
EMCLIP module directly.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .emclip_layers import interpolate_positional_embedding


@dataclass
class BranchLoadAudit:
    branch_name: str
    source_tensor_count: int
    loaded_tensor_count: int
    loaded_parameter_numel: int
    total_parameter_numel: int
    coverage: float
    missing_keys: list
    unexpected_keys: list
    shape_mismatch_keys: list
    excluded_keys: list

    def to_dict(self):
        return asdict(self)


def _torch_load_compat(path):
    """Use the explicit legacy-unpickling behavior where PyTorch supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # ``weights_only`` was introduced after the oldest supported PyTorch.
        return torch.load(path, map_location="cpu")


def _strip_module_prefix(state_dict):
    stripped = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        if clean_key in stripped:
            raise ValueError(
                "checkpoint contains duplicate key %r after removing the 'module.' prefix"
                % clean_key
            )
        stripped[clean_key] = value
    return stripped


def _extract_state_dict(obj, path):
    if isinstance(obj, nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, Mapping) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, Mapping) and "model" in obj:
        state_dict = obj["model"]
    elif isinstance(obj, Mapping):
        state_dict = obj
    else:
        raise TypeError(
            "CLIP checkpoint %s must contain a state_dict, {'state_dict': ...}, "
            "{'model': ...}, or nn.Module; got %s" % (path, type(obj))
        )
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "resolved CLIP state_dict in %s is not dict-like; got %s"
            % (path, type(state_dict))
        )
    non_tensor = [(key, type(value).__name__) for key, value in state_dict.items()
                  if not torch.is_tensor(value)]
    if non_tensor:
        preview = non_tensor[:10]
        raise TypeError(
            "CLIP state_dict %s contains non-Tensor values (first 10): %s"
            % (path, preview)
        )
    return _strip_module_prefix(state_dict)


def load_clip_source_checkpoint(path):
    """Return ``(state_dict, checkpoint_type, absolute_path)`` for CLIP input.

    TorchScript is attempted first because calling ``torch.load`` first on an
    OpenAI CLIP JIT archive can return a ``RecursiveScriptModule`` rather than
    the mapping expected by ``load_state_dict``.
    """
    if not path:
        raise ValueError("a non-empty --clip-checkpoint path is required")
    checkpoint_path = str(Path(path).expanduser().resolve())
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError("CLIP checkpoint does not exist: %s" % checkpoint_path)

    try:
        scripted = torch.jit.load(checkpoint_path, map_location="cpu").eval()
    except Exception as jit_error:
        try:
            obj = _torch_load_compat(checkpoint_path)
        except Exception as load_error:
            raise RuntimeError(
                "failed to load CLIP checkpoint %s as TorchScript (%s: %s) or "
                "as a PyTorch checkpoint (%s: %s)"
                % (
                    checkpoint_path,
                    type(jit_error).__name__,
                    jit_error,
                    type(load_error).__name__,
                    load_error,
                )
            ) from load_error
        state_dict = _extract_state_dict(obj, checkpoint_path)
        checkpoint_type = "state_dict"
    else:
        state_dict = _extract_state_dict(scripted, checkpoint_path)
        checkpoint_type = "torchscript"
    return state_dict, checkpoint_type, checkpoint_path


def extract_visual_state(source_state):
    """Map OpenAI ``visual.*`` names to :class:`PatchTokenEncoder` names."""
    mapped = OrderedDict()
    for key, value in source_state.items():
        if not key.startswith("visual."):
            continue
        target_key = key[len("visual."):]
        target_key = target_key.replace("transformer.resblocks.", "blocks.")
        mapped[target_key] = value
    if not mapped:
        raise KeyError("CLIP checkpoint has no 'visual.*' tensors")
    return mapped


def extract_text_state(source_state):
    """Map OpenAI text encoder names to :class:`PromptTextEncoder` names."""
    mapped = OrderedDict()
    prefixes = (
        "token_embedding.",
        "transformer.resblocks.",
        "ln_final.",
    )
    exact = {"positional_embedding", "text_projection"}
    for key, value in source_state.items():
        if key in exact or key.startswith(prefixes):
            target_key = key.replace("transformer.resblocks.", "blocks.")
            mapped[target_key] = value
    if not mapped:
        raise KeyError("CLIP checkpoint has no recognizable text encoder tensors")
    return mapped


def adapt_visual_positional_embedding(module, source_state):
    """Interpolate a source visual position embedding to the branch grid."""
    adapted = OrderedDict(source_state)
    key = "positional_embedding"
    if key not in adapted:
        return adapted, None
    source = adapted[key]
    target = module.state_dict()[key]
    if source.shape == target.shape:
        source_grid = int((source.size(0) - 1) ** 0.5)
        return adapted, (source_grid, source_grid)
    if source.ndim != 2 or target.ndim != 2 or source.size(1) != target.size(1):
        return adapted, None
    source_grid = int((source.size(0) - 1) ** 0.5)
    target_grid = int((target.size(0) - 1) ** 0.5)
    if source_grid * source_grid != source.size(0) - 1:
        return adapted, None
    if target_grid * target_grid != target.size(0) - 1:
        return adapted, None
    adapted[key] = interpolate_positional_embedding(source, (target_grid, target_grid))
    return adapted, (source_grid, target_grid)


def load_submodule_with_audit(
    module,
    source_state,
    branch_name,
    allowed_missing=None,
    excluded_from_coverage=None,
    minimum_coverage=0.99,
    required_keys=None,
):
    """Load shape-compatible tensors and enforce an audited coverage floor."""
    allowed_missing = set(allowed_missing or ())
    excluded = set(excluded_from_coverage or ())
    required_keys = set(required_keys or ())
    target_state = module.state_dict()
    parameter_numel = {name: parameter.numel() for name, parameter in module.named_parameters()}
    loadable = OrderedDict()
    shape_mismatches = []
    for key, value in source_state.items():
        if key not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(
                "%s: source=%s target=%s"
                % (key, tuple(value.shape), tuple(target_state[key].shape))
            )
            continue
        loadable[key] = value

    missing = sorted(key for key in target_state if key not in loadable and key not in allowed_missing)
    unexpected = sorted(key for key in source_state if key not in target_state)
    loaded_keys_for_coverage = [key for key in loadable if key not in excluded]
    total_keys_for_coverage = [key for key in parameter_numel if key not in excluded]
    loaded_numel = sum(parameter_numel.get(key, 0) for key in loaded_keys_for_coverage)
    total_numel = sum(parameter_numel[key] for key in total_keys_for_coverage)
    coverage = float(loaded_numel) / float(max(1, total_numel))
    audit = BranchLoadAudit(
        branch_name=branch_name,
        source_tensor_count=len([key for key in source_state if key not in excluded]),
        loaded_tensor_count=len(loaded_keys_for_coverage),
        loaded_parameter_numel=loaded_numel,
        total_parameter_numel=total_numel,
        coverage=coverage,
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatch_keys=sorted(shape_mismatches),
        excluded_keys=sorted(excluded),
    )

    missing_required = sorted(key for key in required_keys if key not in loadable)
    if shape_mismatches or missing_required or coverage < float(minimum_coverage):
        raise RuntimeError(
            "%s CLIP initialization audit failed: coverage=%.4f (required >= %.4f), "
            "missing_required=%s, shape_mismatches=%s, missing=%s, unexpected=%s"
            % (
                branch_name,
                coverage,
                minimum_coverage,
                missing_required,
                shape_mismatches,
                missing[:20],
                unexpected[:20],
            )
        )
    module.load_state_dict(loadable, strict=False)
    return audit
