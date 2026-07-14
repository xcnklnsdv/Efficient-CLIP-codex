"""Small AMP compatibility helpers shared by EM-CLIP model and losses."""

from contextlib import nullcontext

import torch


def autocast_disabled(device):
    """Disable CUDA autocast for numerically sensitive float32 operations."""
    device = torch.device(device)
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=False)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", enabled=False)
    return torch.cuda.amp.autocast(enabled=False)
