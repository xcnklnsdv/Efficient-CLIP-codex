import torch
import torch.distributed as dist
import torch.nn.functional as F

from amp_compat import autocast_disabled


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


class _AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        world_size = dist.get_world_size()
        local_size = torch.tensor([value.size(0)], dtype=torch.long, device=value.device)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        sizes = [int(size.item()) for size in sizes]
        if len(set(sizes)) != 1:
            raise RuntimeError(
                "gradient-preserving DDP all_gather requires equal local batch sizes; got %s"
                % sizes
            )
        gathered = [torch.zeros_like(value) for _ in range(world_size)]
        dist.all_gather(gathered, value)
        ctx.local_size = value.size(0)
        ctx.rank = dist.get_rank()
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.local_size
        return grad_output.narrow(0, start, ctx.local_size).contiguous()


def all_gather_with_grad(x):
    if not _dist_ready():
        return x
    try:
        from torch.distributed.nn.functional import all_gather
    except (ImportError, AttributeError):
        return _AllGatherWithGrad.apply(x)
    return torch.cat(tuple(all_gather(x)), dim=0)


def all_gather_no_grad(x):
    if not _dist_ready():
        return x
    gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def build_multi_positive_target(local_labels, all_labels):
    local_labels = local_labels.view(-1, 1)
    all_labels = all_labels.view(1, -1)
    target = (local_labels == all_labels).float()
    return target / target.sum(dim=1, keepdim=True).clamp_min(1.0)


def _pool_motion_features(motion_frame_features, saliency, valid_mask=None, pooling="saliency"):
    assert motion_frame_features.ndim == 3, "motion_frame_features must be [B, T, D]."
    B, T, _ = motion_frame_features.shape
    if valid_mask is None:
        valid_mask = torch.ones(B, T, dtype=torch.bool, device=motion_frame_features.device)
    valid_mask = valid_mask.to(device=motion_frame_features.device, dtype=torch.bool)
    if pooling == "saliency":
        assert saliency is not None and saliency.shape == (B, T), "saliency must be [B, T]."
        weights = saliency.float().masked_fill(~valid_mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    elif pooling == "mean":
        weights = valid_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    else:
        raise ValueError("Unsupported motion pooling: %s" % pooling)
    pooled = (motion_frame_features.float() * weights[:, :, None]).sum(dim=1)
    return F.normalize(pooled, dim=-1)


def motion_text_kl_loss(
    motion_frame_features,
    saliency,
    class_text_features,
    labels,
    valid_mask=None,
    pooling="saliency",
    temperature=0.01,
):
    assert labels.ndim == 1, "labels must be [B]."
    with autocast_disabled(motion_frame_features.device):
        motion_video = _pool_motion_features(
            motion_frame_features.float(), saliency.float(), valid_mask, pooling=pooling
        )
        text_local = class_text_features.to(device=motion_video.device)[labels].float()
        text_local = F.normalize(text_local, dim=-1)
        motion_all = all_gather_with_grad(motion_video)
        text_all = all_gather_with_grad(text_local)
        labels_all = all_gather_no_grad(labels.to(device=motion_video.device))
        target = build_multi_positive_target(labels, labels_all).to(device=motion_video.device)

        logits_mv2text = motion_video @ text_all.t() / float(temperature)
        logits_text2mv = text_local @ motion_all.t() / float(temperature)
        loss_mv2text = F.kl_div(
            F.log_softmax(logits_mv2text, dim=-1),
            target.float(),
            reduction="batchmean",
        )
        loss_text2mv = F.kl_div(
            F.log_softmax(logits_text2mv, dim=-1),
            target.float(),
            reduction="batchmean",
        )
        loss = 0.5 * (loss_mv2text + loss_text2mv)
    if not torch.isfinite(loss):
        raise FloatingPointError("L_MG produced a non-finite loss.")
    return {
        "loss_mg": loss,
        "loss_mg_mv2text": loss_mv2text,
        "loss_mg_text2mv": loss_text2mv,
        "motion_video_features": motion_video,
        "target": target,
    }
