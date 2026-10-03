"""Small AMP compatibility helpers shared by EM-CLIP model and losses."""

from contextlib import nullcontext

import torch


def autocast_disabled(device):
    """Disable device autocast for numerically sensitive float32 operations."""
    device = torch.device(device)
    if device.type not in ("cuda", "cpu"):
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device.type, enabled=False)
        except TypeError:
            return torch.amp.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cpu", enabled=False)
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "amp"):
        return torch.cpu.amp.autocast(enabled=False)
    return nullcontext()
