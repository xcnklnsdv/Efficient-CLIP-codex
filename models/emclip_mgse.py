import torch
import torch.nn as nn
import torch.nn.functional as F

from amp_compat import autocast_disabled
from .emclip_layers import LayerNorm, PatchTokenEncoder


def build_mv_patch_embed_weight(rgb_weight, scale=True):
    """Initialize a 2-channel MV patch embedding from CLIP RGB conv weights."""
    assert rgb_weight.ndim == 4 and rgb_weight.size(1) == 3, (
        "rgb_weight must be [out_channels, 3, patch_h, patch_w]."
    )
    mv_weight = rgb_weight.float().mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
    if scale:
        mv_weight = mv_weight * 1.5
    return mv_weight.to(dtype=rgb_weight.dtype)


def gather_temporal(x, indices):
    """Batch-aware gather over temporal dimension.

    x: [B, T, ...], indices: [B, K], return: [B, K, ...]
    """
    assert x.ndim >= 2, "x must be [B, T, ...]."
    assert indices.ndim == 2, "indices must be [B, K]."
    assert x.size(0) == indices.size(0), "batch size mismatch in temporal gather."
    view_shape = [indices.size(0), indices.size(1)] + [1] * (x.ndim - 2)
    gather_index = indices.view(*view_shape).expand(-1, -1, *x.shape[2:])
    return torch.gather(x, dim=1, index=gather_index)


def select_topk_indices(saliency, valid_mask, k):
    """Select top-k valid candidates, then sort selected indices by time."""
    assert saliency.ndim == 2, "saliency must be [B, T]."
    B, T = saliency.shape
    if k > T:
        raise ValueError("selected_frames K=%d cannot exceed candidate count T=%d." % (k, T))
    if valid_mask is None:
        valid_mask = torch.ones(B, T, dtype=torch.bool, device=saliency.device)
    valid_mask = valid_mask.to(device=saliency.device, dtype=torch.bool)
    assert valid_mask.shape == saliency.shape, "valid_mask must be [B, T]."
    valid_counts = valid_mask.sum(dim=1)
    if (valid_counts == 0).any():
        bad = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError("every sample needs at least one valid candidate; invalid rows=%s" % bad)

    masked = saliency.float().masked_fill(~valid_mask, float("-inf"))
    indices = torch.topk(masked, k=k, dim=1).indices
    for b in range(B):
        count = int(valid_counts[b].item())
        if count < k:
            best = torch.argmax(masked[b])
            indices[b, count:] = best
    indices = torch.sort(indices, dim=1).values
    assert indices.shape == (B, k)
    return indices


class MotionGuidedSaliencyExtraction(nn.Module):
    """Motion-Guided Saliency Extraction and Action Correlation Generator."""

    def __init__(
        self,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        input_resolution=256,
        patch_size=16,
        selected_frames=8,
        temperature=0.01,
        text_mode="class_bank",
        class_aggregation="mean",
        motion_pooling="saliency",
        dropout=0.0,
        allow_label_leakage_for_diagnostic=False,
        train_text_mode=None,
        implementation="paper",
    ):
        super().__init__()
        self.selected_frames = selected_frames
        self.temperature = temperature
        self.text_mode = text_mode
        self.train_text_mode = train_text_mode or text_mode
        self.implementation = implementation
        if temperature <= 0:
            raise ValueError("MGSE temperature must be positive")
        for mode in (self.text_mode, self.train_text_mode):
            if mode not in ("ground_truth", "class_bank", "predicted_class"):
                raise ValueError("Unsupported MGSE text mode: %s" % mode)
        self.class_aggregation = class_aggregation
        self.motion_pooling = motion_pooling
        self.allow_label_leakage_for_diagnostic = allow_label_leakage_for_diagnostic
        self.motion_encoder = PatchTokenEncoder(
            in_channels=2,
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
            dropout=dropout,
        )
        # Eq. (5) has no learned scale/bias. Keep the old affine LN only when
        # loading the historical implementation and its checkpoints.
        self.feature_ln = LayerNorm(output_dim, elementwise_affine=(implementation == "legacy"))

    @property
    def output_dim(self):
        return self.motion_encoder.output_dim

    def initialize_mv_patch_from_rgb(self, rgb_conv_weight, scale=True):
        weight = build_mv_patch_embed_weight(rgb_conv_weight, scale=scale)
        if weight.shape != self.motion_encoder.conv1.weight.shape:
            raise ValueError(
                "converted MV patch weight shape %s does not match encoder conv1 %s"
                % (tuple(weight.shape), tuple(self.motion_encoder.conv1.weight.shape))
            )
        with torch.no_grad():
            self.motion_encoder.conv1.weight.copy_(weight)

    def _encode_motion(self, motion_vectors):
        assert motion_vectors.ndim == 5, "motion_vectors must be [B, T, 2, H, W]."
        B, T, C, H, W = motion_vectors.shape
        assert C == 2, "motion vector channel dimension must be 2."
        mv = motion_vectors.reshape(B * T, C, H, W)
        frame_features, _ = self.motion_encoder(mv)
        frame_features = frame_features.reshape(B, T, -1)
        assert frame_features.shape[:2] == (B, T)
        return frame_features

    def _prepare_features(self, features):
        features = self.feature_ln(features.float())
        return F.normalize(features, dim=-1)

    def _normalize_token_features(self, token_features):
        token_features = F.layer_norm(token_features.float(), (token_features.size(-1),))
        return F.normalize(token_features, dim=-1)

    def _temporal_softmax(self, similarity, valid_mask):
        # similarity shape can be [B, T, S] or [B, T, C, S].
        mask_shape = [valid_mask.size(0), valid_mask.size(1)] + [1] * (similarity.ndim - 2)
        similarity = similarity.masked_fill(~valid_mask.view(*mask_shape), float("-inf"))
        attn = F.softmax(similarity.float(), dim=1)
        if not torch.isfinite(attn).all():
            raise FloatingPointError("MGSE temporal softmax produced non-finite values.")
        return attn

    def _ground_truth_saliency(self, motion, token_features, token_mask, labels, valid_mask):
        if labels is None:
            raise RuntimeError("mgse_text_mode='ground_truth' requires labels.")
        tokens = token_features[labels]
        mask = token_mask[labels]
        sim = torch.einsum("btd,bsd->bts", motion, tokens) / float(self.temperature)
        attn = self._temporal_softmax(sim, valid_mask)
        attn = attn * mask[:, None, :].float()
        denom = mask.sum(dim=1).clamp_min(1).float()[:, None]
        saliency = attn.sum(dim=-1) / denom
        return saliency

    def _class_bank_saliency(self, motion, token_features, token_mask, valid_mask):
        sim = torch.einsum("btd,csd->btcs", motion, token_features) / float(self.temperature)
        attn = self._temporal_softmax(sim, valid_mask)
        attn = attn * token_mask[None, None, :, :].float()
        denom = token_mask.sum(dim=1).clamp_min(1).float()[None, None, :]
        per_class = attn.sum(dim=-1) / denom
        if self.class_aggregation == "mean":
            saliency = per_class.mean(dim=-1)
        elif self.class_aggregation == "max":
            saliency = per_class.max(dim=-1).values
        elif self.class_aggregation == "logsumexp":
            saliency = torch.logsumexp(per_class.float(), dim=-1)
        else:
            raise ValueError("Unsupported mgse_class_aggregation: %s" % self.class_aggregation)
        return saliency

    def _predicted_class_saliency(self, motion, class_text_features, token_features, token_mask, valid_mask,
                                 motion_frame_features):
        weights = valid_mask.float()
        # L_MG aligns projected motion features (Eq. 4) with EOT text. The
        # non-affine standardization in Eq. 5 is only for word correlation;
        # applying it to class prediction changes the trained cosine space.
        video_motion = (motion_frame_features.float() * weights[:, :, None]).sum(dim=1)
        video_motion = video_motion / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pred = (F.normalize(video_motion, dim=-1) @ F.normalize(class_text_features.float(), dim=-1).t()).argmax(dim=1)
        return self._ground_truth_saliency(motion, token_features, token_mask, pred, valid_mask)

    def _normalize_saliency(self, saliency, valid_mask):
        saliency = saliency.float().masked_fill(~valid_mask, 0.0)
        denom = saliency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        saliency = saliency / denom
        if not torch.isfinite(saliency).all():
            raise FloatingPointError("MGSE saliency contains non-finite values.")
        return saliency

    def _pool_motion_video(self, motion_features, saliency, valid_mask):
        if self.motion_pooling == "saliency":
            weights = saliency.float().masked_fill(~valid_mask, 0.0)
        elif self.motion_pooling == "mean":
            weights = valid_mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            raise ValueError("Unsupported motion_pooling: %s" % self.motion_pooling)
        video = (motion_features.float() * weights[:, :, None]).sum(dim=1)
        return F.normalize(video, dim=-1)

    def forward(
        self,
        motion_vectors,
        class_text_features,
        class_token_features=None,
        class_token_mask=None,
        labels=None,
        valid_mask=None,
        training_mode=True,
    ):
        assert class_text_features.ndim == 2, "class_text_features must be [C, D]."
        if class_token_features is None:
            class_token_features = class_text_features[:, None, :]
            class_token_mask = torch.ones(
                class_text_features.size(0),
                1,
                dtype=torch.bool,
                device=class_text_features.device,
            )
        if class_token_mask is None:
            class_token_mask = torch.ones(
                class_token_features.shape[:2],
                dtype=torch.bool,
                device=class_token_features.device,
            )
        assert class_token_features.ndim == 3, "class_token_features must be [C, S, D]."
        assert class_token_mask.shape == class_token_features.shape[:2], "class_token_mask must be [C, S]."

        B, T = motion_vectors.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=motion_vectors.device)
        valid_mask = valid_mask.to(device=motion_vectors.device, dtype=torch.bool)
        class_text_features = class_text_features.to(device=motion_vectors.device)
        class_token_features = class_token_features.to(device=motion_vectors.device)
        class_token_mask = class_token_mask.to(device=motion_vectors.device)
        if labels is not None:
            labels = labels.to(device=motion_vectors.device, dtype=torch.long)

        motion_frame_features = self._encode_motion(motion_vectors)
        # ``.float()`` alone is insufficient inside an outer CUDA autocast
        # region: einsum/matmul can still be selected for fp16.  The paper's
        # tau=0.01 makes this complete correlation path explicitly float32.
        with autocast_disabled(motion_frame_features.device):
            motion_norm = self._prepare_features(motion_frame_features.float())
            token_norm = self._normalize_token_features(class_token_features.float())
            class_text_norm = F.normalize(class_text_features.float(), dim=-1)

            text_mode = self.train_text_mode if training_mode else self.text_mode
            if text_mode == "ground_truth":
                if (not training_mode) and (not self.allow_label_leakage_for_diagnostic):
                    raise RuntimeError(
                        "mgse_text_mode='ground_truth' would cause validation label leakage; "
                        "use class_bank or pass allow_label_leakage_for_diagnostic=True."
                    )
                saliency = self._ground_truth_saliency(
                    motion_norm, token_norm, class_token_mask, labels, valid_mask
                )
            elif text_mode == "class_bank":
                saliency = self._class_bank_saliency(
                    motion_norm, token_norm, class_token_mask, valid_mask
                )
            elif text_mode == "predicted_class":
                saliency = self._predicted_class_saliency(
                    motion_norm,
                    class_text_norm,
                    token_norm,
                    class_token_mask,
                    valid_mask,
                    motion_frame_features,
                )
            else:
                raise ValueError("Unsupported mgse_text_mode: %s" % text_mode)

            saliency = self._normalize_saliency(saliency, valid_mask)
            selected_indices = select_topk_indices(saliency, valid_mask, self.selected_frames)
            motion_video_features = self._pool_motion_video(
                motion_frame_features.float(), saliency, valid_mask
            )
            mg_logits_mv2text = (
                motion_video_features @ class_text_norm.t() / float(self.temperature)
            )
            mg_logits_text2mv = mg_logits_mv2text.t()
            if not torch.isfinite(mg_logits_mv2text).all():
                raise FloatingPointError("MGSE logits contain non-finite values.")

        return {
            "selected_indices": selected_indices,
            "saliency": saliency,
            "motion_frame_features": motion_frame_features,
            "motion_video_features": motion_video_features,
            "mg_logits_mv2text": mg_logits_mv2text,
            "mg_logits_text2mv": mg_logits_text2mv,
        }
