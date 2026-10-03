import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass

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


def reduce_max(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def move_batch_to_device(batch, device):
    moved = {}
    for key in ("i_frames", "motion_vectors", "residuals", "valid_mask", "label", "candidate_gop_indices"):
        value = batch.get(key)
        if value is not None:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _slice_batch(batch, start, end):
    """Slice only the sample dimension while keeping metadata usable on CPU."""
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:end]
        elif isinstance(value, dict):
            sliced[key] = {
                sub_key: sub_value[start:end] if isinstance(sub_value, (list, tuple)) else sub_value
                for sub_key, sub_value in value.items()
            }
        elif isinstance(value, (list, tuple)):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _micro_batch_ranges(batch_size, requested_size):
    requested_size = int(requested_size)
    micro_size = batch_size if requested_size <= 0 else max(1, min(requested_size, batch_size))
    return [(start, min(batch_size, start + micro_size)) for start in range(0, batch_size, micro_size)]


def _no_sync_context(model, enabled):
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _autocast(device, enabled):
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast("cuda", enabled=enabled)
            except TypeError:
                return torch.amp.autocast(device_type="cuda", enabled=enabled)
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


def _debug_unused_parameters(model, step, args):
    if not getattr(args, "debug_unused_parameters", False) or step >= 3:
        return
    if not is_main_process():
        return
    raw_model = model.module if hasattr(model, "module") else model
    unused = [
        (name, tuple(param.shape), param.numel())
        for name, param in raw_model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    print("[DDP unused params] step=%d count=%d" % (step, len(unused)), flush=True)
    for name, shape, numel in unused:
        print("  %s shape=%s numel=%d" % (name, shape, numel), flush=True)


def _finish_optimizer_step(model, optimizer, scaler, grad_clip_norm):
    """Unscale, validate/clip gradients and let GradScaler recover fp16 overflow."""
    scaler.unscale_(optimizer)
    raw_model = model.module if hasattr(model, "module") else model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    scale_before = float(scaler.get_scale())
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    if not torch.isfinite(grad_norm):
        # Keep the expensive per-parameter scan on the exceptional path; doing
        # hundreds of GPU synchronizations every normal step is prohibitively slow.
        bad_gradients = [
            name
            for name, parameter in raw_model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if not bad_gradients:
            raise FloatingPointError(
                "gradient norm overflowed although every individual gradient is finite"
            )
        if not scaler.is_enabled():
            raise FloatingPointError(
                "non-finite unscaled gradients without AMP; bad parameters=%s"
                % bad_gradients[:20]
            )
        # unscale_ recorded found_inf. GradScaler now skips the parameter
        # update and lowers its scale, which is the standard AMP recovery path.
        scaler.step(optimizer)
        scaler.update()
        return {
            "stepped": False,
            "grad_norm": float("nan"),
            "scale_before": scale_before,
            "scale_after": float(scaler.get_scale()),
            "bad_gradients": bad_gradients,
        }

    scaler.step(optimizer)
    scaler.update()
    return {
        "stepped": True,
        "grad_norm": float(grad_norm),
        "scale_before": scale_before,
        "scale_after": float(scaler.get_scale()),
        "bad_gradients": [],
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, args):
    model.train()
    start_time = time.time()
    raw_model = model.module if hasattr(model, "module") else model
    rank = dist.get_rank() if dist_ready() else 0
    consecutive_amp_overflows = 0
    amp_skipped_steps = 0
    print("[emclip][rank=%d] epoch=%d entering DataLoader" % (rank, epoch), flush=True)
    totals = {
        "loss": 0.0,
        "loss_mg": 0.0,
        "loss_mg_mv2text": 0.0,
        "loss_mg_text2mv": 0.0,
        "loss_me": 0.0,
        "acc1": 0.0,
        "acc5": 0.0,
        "entropy": 0.0,
        "selected_unique_gops": 0.0,
        "valid_candidates": 0.0,
        "count": 0,
    }
    for step, raw_batch in enumerate(loader):
        batch_size = int(raw_batch["label"].size(0))
        if step == 0:
            print(
                "[emclip][rank=%d] first batch loaded CPU shapes I=%s MV=%s R=%s"
                % (
                    rank,
                    tuple(raw_batch["i_frames"].shape),
                    tuple(raw_batch["motion_vectors"].shape),
                    tuple(raw_batch["residuals"].shape),
                ),
                flush=True,
            )
        micro_ranges = _micro_batch_ranges(
            batch_size,
            getattr(args, "micro_batch_size", 0),
        )
        if (
            len(micro_ranges) > 1
            and getattr(raw_model, "use_mgse", False)
            and float(raw_model.config.lambda_mg) != 0.0
        ):
            raise ValueError(
                "--micro-batch-size cannot split a batch while L_MG is enabled: "
                "the split would change DDP global positives and negatives. Use 0/full batch, "
                "or set --lambda-mg 0 for a classification-only ablation."
            )
        optimizer.zero_grad(set_to_none=True)
        out = None
        batch_loss_total = 0.0
        batch_loss_mg_total = 0.0
        batch_loss_mg_mv2text_total = 0.0
        batch_loss_mg_text2mv_total = 0.0
        batch_loss_me_total = 0.0
        for micro_idx, (micro_start, micro_end) in enumerate(micro_ranges):
            batch = move_batch_to_device(
                _slice_batch(raw_batch, micro_start, micro_end), device
            )
            labels = batch["label"]
            if step == 0 and micro_idx == 0:
                print(
                    "[emclip][rank=%d] first micro-batch moved to %s shape I=%s"
                    % (rank, device, tuple(batch["i_frames"].shape)),
                    flush=True,
                )
            with _no_sync_context(model, enabled=micro_idx + 1 < len(micro_ranges)):
                with _autocast(device, args.amp):
                    micro_out = model(
                        i_frames=batch["i_frames"],
                        motion_vectors=batch.get("motion_vectors"),
                        residuals=batch["residuals"],
                        labels=labels,
                        valid_mask=batch["valid_mask"],
                        training_mode=True,
                    )
                    micro_loss = micro_out["loss"]
                if micro_loss is None or not torch.is_tensor(micro_loss) or micro_loss.ndim != 0:
                    raise RuntimeError("EM-CLIP training loss must be a scalar Tensor.")
                # Weight by sample count so a short final micro-batch does not
                # change the effective loss, while bounding GPU activation.
                micro_weight = float(micro_end - micro_start) / float(batch_size)
                scaler.scale(micro_loss * micro_weight).backward()
            out = micro_out
            micro_bs = micro_end - micro_start
            batch_loss_total += micro_loss.detach().item() * micro_bs
            batch_loss_mg_total += micro_out["loss_mg"].detach().item() * micro_bs
            batch_loss_mg_mv2text_total += micro_out["loss_mg_mv2text"].detach().item() * micro_bs
            batch_loss_mg_text2mv_total += micro_out["loss_mg_text2mv"].detach().item() * micro_bs
            batch_loss_me_total += micro_out["loss_me"].detach().item() * micro_bs
            totals["loss"] += micro_loss.detach().item() * micro_bs
            totals["loss_mg"] += micro_out["loss_mg"].detach().item() * micro_bs
            totals["loss_mg_mv2text"] += micro_out["loss_mg_mv2text"].detach().item() * micro_bs
            totals["loss_mg_text2mv"] += micro_out["loss_mg_text2mv"].detach().item() * micro_bs
            totals["loss_me"] += micro_out["loss_me"].detach().item() * micro_bs

            logits = micro_out["logits"].detach()
            max_k = min(5, logits.size(1))
            pred = logits.topk(max_k, dim=1).indices
            totals["acc1"] += (pred[:, :1] == labels[:, None]).any(dim=1).float().sum().item()
            totals["acc5"] += (pred[:, :max_k] == labels[:, None]).any(dim=1).float().sum().item()
            totals["entropy"] += micro_out["mgse_saliency_entropy"].detach().item() * micro_bs
            gop_ids = batch.get("candidate_gop_indices")
            selected = micro_out["selected_indices"]
            if gop_ids is None:
                gop_ids = torch.arange(batch["valid_mask"].size(1), device=device).expand(micro_bs, -1)
            selected_gops = torch.gather(gop_ids, 1, selected).sort(dim=1).values
            unique_counts = 1 + (selected_gops[:, 1:] != selected_gops[:, :-1]).sum(dim=1)
            totals["selected_unique_gops"] += unique_counts.sum().item()
            totals["valid_candidates"] += batch["valid_mask"].sum().item()
            totals["count"] += micro_bs

        if out is None:
            raise RuntimeError("EM-CLIP received an empty DataLoader batch.")
        _debug_unused_parameters(model, step, args)
        step_result = _finish_optimizer_step(
            model, optimizer, scaler, args.grad_clip_norm
        )
        if not step_result["stepped"]:
            amp_skipped_steps += 1
            consecutive_amp_overflows += 1
            max_overflows = int(getattr(args, "max_consecutive_amp_overflows", 8))
            if step_result["scale_after"] >= step_result["scale_before"]:
                raise FloatingPointError(
                    "GradScaler found non-finite gradients but did not reduce its scale: "
                    "scale=%.1f bad parameters=%s"
                    % (
                        step_result["scale_before"],
                        step_result["bad_gradients"][:20],
                    )
                )
            if consecutive_amp_overflows > max_overflows:
                raise FloatingPointError(
                    "AMP gradients remained non-finite for %d consecutive batches; "
                    "last scale %.1f->%.1f bad parameters=%s"
                    % (
                        consecutive_amp_overflows,
                        step_result["scale_before"],
                        step_result["scale_after"],
                        step_result["bad_gradients"][:20],
                    )
                )
            if is_main_process():
                print(
                    "[emclip][amp] skipped optimizer step epoch=%d step=%d "
                    "scale=%.1f->%.1f consecutive=%d bad=%s"
                    % (
                        epoch,
                        step,
                        step_result["scale_before"],
                        step_result["scale_after"],
                        consecutive_amp_overflows,
                        step_result["bad_gradients"][:5],
                    ),
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            continue
        consecutive_amp_overflows = 0
        grad_norm = step_result["grad_norm"]
        if not math.isfinite(grad_norm):
            raise FloatingPointError(
                "non-finite gradient norm after a successful optimizer step at epoch=%d step=%d"
                % (epoch, step)
            )
        if scheduler is not None:
            scheduler.step()

        loss = batch_loss_total / float(batch_size)
        loss_mg = batch_loss_mg_total / float(batch_size)
        loss_mg_mv2text = batch_loss_mg_mv2text_total / float(batch_size)
        loss_mg_text2mv = batch_loss_mg_text2mv_total / float(batch_size)
        loss_me = batch_loss_me_total / float(batch_size)

        if is_main_process() and step % args.print_freq == 0:
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0
            print(
                "epoch=%d step=%d/%d loss=%.4f loss_mg=%.4f "
                "loss_mg_mv2text=%.4f loss_mg_text2mv=%.4f loss_me=%.4f "
                "acc1=%.2f acc5=%.2f lr=%.8f grad_norm=%.4f entropy=%.4f "
                "selected_mean=%.2f unique_gops=%.2f valid_candidates=%.2f "
                "classification_scale=%.4f max_mem=%.0fMB amp_scale=%.1f amp_skips=%d"
                % (
                    epoch,
                    step,
                    len(loader),
                    loss,
                    loss_mg,
                    loss_mg_mv2text,
                    loss_mg_text2mv,
                    loss_me,
                    100.0 * totals["acc1"] / max(1, totals["count"]),
                    100.0 * totals["acc5"] / max(1, totals["count"]),
                    optimizer.param_groups[0]["lr"],
                    float(grad_norm),
                    float(totals["entropy"] / max(1, totals["count"])),
                    float(out["selected_indices"].float().mean().detach()),
                    totals["selected_unique_gops"] / max(1, totals["count"]),
                    totals["valid_candidates"] / max(1, totals["count"]),
                    float(out["classification_logit_scale"]),
                    mem,
                    step_result["scale_after"],
                    amp_skipped_steps,
                ),
                flush=True,
            )
    device_for_reduce = device
    count = reduce_sum(totals["count"], device_for_reduce)
    elapsed_seconds = reduce_max(time.time() - start_time, device_for_reduce)
    candidate_frames = int(raw_model.config.candidate_frames)
    result = {
        "loss": reduce_sum(totals["loss"], device_for_reduce) / max(1.0, count),
        "loss_mg": reduce_sum(totals["loss_mg"], device_for_reduce) / max(1.0, count),
        "loss_mg_mv2text": reduce_sum(totals["loss_mg_mv2text"], device_for_reduce) / max(1.0, count),
        "loss_mg_text2mv": reduce_sum(totals["loss_mg_text2mv"], device_for_reduce) / max(1.0, count),
        "loss_me": reduce_sum(totals["loss_me"], device_for_reduce) / max(1.0, count),
        "acc1": 100.0 * reduce_sum(totals["acc1"], device_for_reduce) / max(1.0, count),
        "acc5": 100.0 * reduce_sum(totals["acc5"], device_for_reduce) / max(1.0, count),
        "mgse_saliency_entropy": reduce_sum(totals["entropy"], device_for_reduce) / max(1.0, count),
        "selected_unique_gops": reduce_sum(totals["selected_unique_gops"], device_for_reduce) / max(1.0, count),
        "valid_candidates": reduce_sum(totals["valid_candidates"], device_for_reduce) / max(1.0, count),
        "amp_skipped_steps": amp_skipped_steps,
        "seconds": elapsed_seconds,
        "global_videos_per_second": count / max(elapsed_seconds, 1e-12),
        "global_candidate_gops_per_second": (
            count * candidate_frames / max(elapsed_seconds, 1e-12)
        ),
    }
    return result


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    start_time = time.time()
    totals = {"acc1": 0.0, "acc5": 0.0, "count": 0, "loss_me": 0.0}
    raw_model = model.module if hasattr(model, "module") else model
    diagnostic_ground_truth = (
        getattr(raw_model, "use_mgse", False)
        and raw_model.config.mgse_text_mode == "ground_truth"
        and raw_model.config.allow_mgse_label_leakage_for_diagnostic
    )
    for raw_batch in loader:
        labels = raw_batch["label"].to(device, non_blocking=True)
        flat = _flatten_views(raw_batch)
        flat_logits = []
        flat_selection_labels = None
        if diagnostic_ground_truth:
            flat_selection_labels = labels[:, None].expand(-1, flat["views"]).reshape(-1)
        for start, end in _micro_batch_ranges(
            flat["i_frames"].size(0),
            getattr(args, "micro_batch_size", 0),
        ):
            micro = {
                "i_frames": flat["i_frames"][start:end],
                "motion_vectors": flat["motion_vectors"][start:end],
                "residuals": flat["residuals"][start:end],
                "valid_mask": flat["valid_mask"][start:end],
            }
            batch = move_batch_to_device(micro, device)
            with _autocast(device, args.amp):
                out = model(
                    i_frames=batch["i_frames"],
                    motion_vectors=batch.get("motion_vectors"),
                    residuals=batch["residuals"],
                    labels=None,
                    mgse_labels=(
                        flat_selection_labels[start:end]
                        if flat_selection_labels is not None
                        else None
                    ),
                    valid_mask=batch["valid_mask"],
                    training_mode=False,
                )
            flat_logits.append(out["logits"])
        logits = torch.cat(flat_logits, dim=0).view(flat["batch"], flat["views"], -1).mean(dim=1)
        loss_me = F.cross_entropy(logits.float(), labels)
        max_k = min(5, logits.size(1))
        pred = logits.topk(max_k, dim=1).indices
        totals["acc1"] += (pred[:, :1] == labels[:, None]).any(dim=1).float().sum().item()
        totals["acc5"] += (pred[:, :max_k] == labels[:, None]).any(dim=1).float().sum().item()
        totals["loss_me"] += loss_me.item() * labels.numel()
        totals["count"] += labels.numel()
    count = reduce_sum(totals["count"], device)
    elapsed_seconds = reduce_max(time.time() - start_time, device)
    return {
        "val_acc1": 100.0 * reduce_sum(totals["acc1"], device) / max(1.0, count),
        "val_acc5": 100.0 * reduce_sum(totals["acc5"], device) / max(1.0, count),
        "val_loss_me": reduce_sum(totals["loss_me"], device) / max(1.0, count),
        "val_seconds": elapsed_seconds,
        "val_global_videos_per_second": count / max(elapsed_seconds, 1e-12),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_acc1, run_config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    config = getattr(module, "config", None)
    torch.save(
        {
            "model": module.state_dict(),
            "format_version": 2,
            "model_config": asdict(config) if is_dataclass(config) else None,
            "run_config": run_config,
            "optimizer_parameter_names": [
                name for name, parameter in module.named_parameters() if parameter.requires_grad
            ],
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_acc1": best_acc1,
        },
        path,
    )


def _load_checkpoint_file(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_checkpoint(path, model, map_location="cpu"):
    """Load model weights for transfer without restoring training progress."""
    ckpt = _load_checkpoint_file(path, map_location)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise TypeError(
            "--init-checkpoint expects an EM-CLIP checkpoint containing a 'model' "
            "state_dict; got %s" % type(ckpt)
        )
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"], strict=True)
    return {
        "source_epoch": int(ckpt.get("epoch", -1)),
        "source_best_acc1": float(ckpt.get("best_acc1", -math.inf)),
    }


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = _load_checkpoint_file(path, map_location)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise TypeError(
            "--resume expects an EM-CLIP training checkpoint containing a 'model' state_dict; got %s"
            % type(ckpt)
        )
    module = model.module if hasattr(model, "module") else model
    saved_config = ckpt.get("model_config") or {}
    if hasattr(module, "config") and saved_config.get("class_names") is not None:
        if saved_config["class_names"] != module.config.class_names:
            raise ValueError("Resume class text/label order differs from the checkpoint; use --init-checkpoint for cross-dataset transfer")
    module.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        expected_names = ckpt.get("optimizer_parameter_names")
        current_names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
        if expected_names is not None and expected_names != current_names:
            raise RuntimeError("Resume optimizer parameter order/train mode differs from the checkpoint; use --init-checkpoint for a new run")
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_acc1", -math.inf))
