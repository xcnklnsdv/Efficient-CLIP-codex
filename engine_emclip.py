import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F


def dist_ready():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not dist_ready()) or dist.get_rank() == 0


def reduce_sum(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def move_batch_to_device(batch, device):
    return {
        "i_frames": batch["i_frames"].to(device, non_blocking=True),
        "motion_vectors": batch["motion_vectors"].to(device, non_blocking=True),
        "residuals": batch["residuals"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def _autocast(device, enabled):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def _flatten_views(batch):
    i_frames = batch["i_frames"]
    motion_vectors = batch["motion_vectors"]
    residuals = batch["residuals"]
    valid_mask = batch["valid_mask"]
    if i_frames.ndim == 6:
        B, V = i_frames.shape[:2]
        flat = {
            "i_frames": i_frames.flatten(0, 1),
            "motion_vectors": motion_vectors.flatten(0, 1),
            "residuals": residuals.flatten(0, 1),
            "valid_mask": valid_mask.flatten(0, 1),
            "views": V,
            "batch": B,
        }
    else:
        flat = {
            "i_frames": i_frames,
            "motion_vectors": motion_vectors,
            "residuals": residuals,
            "valid_mask": valid_mask,
            "views": 1,
            "batch": i_frames.size(0),
        }
    return flat


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, args):
    model.train()
    start = time.time()
    totals = {
        "loss": 0.0,
        "loss_mg": 0.0,
        "loss_me": 0.0,
        "acc1": 0.0,
        "acc5": 0.0,
        "count": 0,
    }
    for step, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, args.amp):
            out = model(
                i_frames=batch["i_frames"],
                motion_vectors=batch["motion_vectors"],
                residuals=batch["residuals"],
                labels=labels,
                valid_mask=batch["valid_mask"],
                training_mode=True,
            )
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        logits = out["logits"].detach()
        max_k = min(5, logits.size(1))
        pred = logits.topk(max_k, dim=1).indices
        acc1 = (pred[:, :1] == labels[:, None]).any(dim=1).float().sum().item()
        acc5 = (pred[:, :max_k] == labels[:, None]).any(dim=1).float().sum().item()
        bs = labels.numel()
        totals["loss"] += loss.detach().item() * bs
        totals["loss_mg"] += out["loss_mg"].detach().item() * bs
        totals["loss_me"] += out["loss_me"].detach().item() * bs
        totals["acc1"] += acc1
        totals["acc5"] += acc5
        totals["count"] += bs

        if is_main_process() and step % args.print_freq == 0:
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0
            print(
                "epoch=%d step=%d/%d loss=%.4f loss_mg=%.4f loss_me=%.4f "
                "acc1=%.2f acc5=%.2f lr=%.8f grad_norm=%.4f entropy=%.4f "
                "selected_mean=%.2f max_mem=%.0fMB"
                % (
                    epoch,
                    step,
                    len(loader),
                    loss.detach().item(),
                    out["loss_mg"].detach().item(),
                    out["loss_me"].detach().item(),
                    100.0 * acc1 / bs,
                    100.0 * acc5 / bs,
                    optimizer.param_groups[0]["lr"],
                    float(grad_norm),
                    float(out["mgse_saliency_entropy"].detach()),
                    float(out["selected_indices"].float().mean().detach()),
                    mem,
                ),
                flush=True,
            )
    device_for_reduce = device
    count = reduce_sum(totals["count"], device_for_reduce)
    result = {
        "loss": reduce_sum(totals["loss"], device_for_reduce) / max(1.0, count),
        "loss_mg": reduce_sum(totals["loss_mg"], device_for_reduce) / max(1.0, count),
        "loss_me": reduce_sum(totals["loss_me"], device_for_reduce) / max(1.0, count),
        "acc1": 100.0 * reduce_sum(totals["acc1"], device_for_reduce) / max(1.0, count),
        "acc5": 100.0 * reduce_sum(totals["acc5"], device_for_reduce) / max(1.0, count),
        "seconds": time.time() - start,
    }
    return result


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    totals = {"acc1": 0.0, "acc5": 0.0, "count": 0, "loss_me": 0.0}
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        labels = batch["label"]
        flat = _flatten_views(batch)
        with _autocast(device, args.amp):
            out = model(
                i_frames=flat["i_frames"],
                motion_vectors=flat["motion_vectors"],
                residuals=flat["residuals"],
                labels=None,
                valid_mask=flat["valid_mask"],
                training_mode=False,
            )
        logits = out["logits"].view(flat["batch"], flat["views"], -1).mean(dim=1)
        loss_me = F.cross_entropy(logits.float(), labels)
        max_k = min(5, logits.size(1))
        pred = logits.topk(max_k, dim=1).indices
        totals["acc1"] += (pred[:, :1] == labels[:, None]).any(dim=1).float().sum().item()
        totals["acc5"] += (pred[:, :max_k] == labels[:, None]).any(dim=1).float().sum().item()
        totals["loss_me"] += loss_me.item() * labels.numel()
        totals["count"] += labels.numel()
    count = reduce_sum(totals["count"], device)
    return {
        "val_acc1": 100.0 * reduce_sum(totals["acc1"], device) / max(1.0, count),
        "val_acc5": 100.0 * reduce_sum(totals["acc5"], device) / max(1.0, count),
        "val_loss_me": reduce_sum(totals["loss_me"], device) / max(1.0, count),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_acc1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_acc1": best_acc1,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_acc1", -math.inf))
