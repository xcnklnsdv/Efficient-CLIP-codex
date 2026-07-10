import torch
import torch.nn as nn
import torch.nn.functional as F

from .emclip_layers import LayerNorm, PatchTokenEncoder, TransformerBlock, init_linear_or_attention


class MotionEmbeddedLongTermSpatiotemporalCorrelation(nn.Module):
    """MELSC with IFE, GSPL, LMPL, SAG, and temporal aggregation."""

    def __init__(
        self,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        input_resolution=256,
        patch_size=16,
        temporal_aggregator_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim
        self.i_encoder = PatchTokenEncoder(
            in_channels=3,
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.r_encoder = PatchTokenEncoder(
            in_channels=3,
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.gs_i_proj = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.gs_r_proj = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.lm_r_proj = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.gs_i_ln = nn.ModuleList([LayerNorm(width) for _ in range(layers)])
        self.gs_r_ln = nn.ModuleList([LayerNorm(width) for _ in range(layers)])
        self.lm_ln = nn.ModuleList([LayerNorm(width) for _ in range(layers)])
        self.gs_attn = nn.ModuleList([
            nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.lm_attn = nn.ModuleList([
            nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(width, heads, dropout=dropout)
            for _ in range(temporal_aggregator_layers)
        ])
        self.final_ln = LayerNorm(width)
        self.apply(init_linear_or_attention)

    def _initial_tokens(self, branch, frames, channels):
        assert frames.ndim == 5, "frames must be [B, K, C, H, W]."
        B, K, C, H, W = frames.shape
        assert C == channels, "expected %d channels, got %d" % (channels, C)
        tokens = branch.embed_patches(frames.reshape(B * K, C, H, W))
        N, D = tokens.size(1), tokens.size(2)
        tokens = tokens.reshape(B, K, N, D)
        assert tokens.shape == (B, K, N, self.width)
        return tokens

    def _layer_prompts(self, layer_idx, z_i, z_r):
        cls_i = z_i[:, :, 0, :]
        cls_r = z_r[:, :, 0, :]
        assert cls_i.shape == cls_r.shape and cls_i.ndim == 3

        a_i = self.gs_i_proj[layer_idx](cls_i)
        a_r = self.gs_r_proj[layer_idx](cls_r)
        q_i = self.gs_i_ln[layer_idx](a_i)
        kv_r = self.gs_r_ln[layer_idx](a_r)
        gs_hat = self.gs_attn[layer_idx](q_i, kv_r, kv_r, need_weights=False)[0]
        gs = a_i + self.dropout(gs_hat)

        a_lm = self.lm_r_proj[layer_idx](cls_r)
        q_lm = self.lm_ln[layer_idx](a_lm)
        lm_hat = self.lm_attn[layer_idx](q_lm, q_lm, q_lm, need_weights=False)[0]
        lm = a_lm + self.dropout(lm_hat)
        assert gs.shape == lm.shape == cls_i.shape
        return gs, lm

    def forward(self, i_selected, r_selected):
        assert i_selected.ndim == 5, "I_selected must be [B, K, 3, H, W]."
        assert r_selected.ndim == 5, "R_selected must be [B, K, 3, H, W]."
        assert i_selected.shape[:2] == r_selected.shape[:2], "I/R must share [B, K]."
        B, K = i_selected.shape[:2]
        z_i = self._initial_tokens(self.i_encoder, i_selected, channels=3)
        z_r = self._initial_tokens(self.r_encoder, r_selected, channels=3)
        N = z_i.size(2)
        last_gs = None
        last_lm = None

        for layer_idx in range(self.layers):
            gs, lm = self._layer_prompts(layer_idx, z_i, z_r)
            last_gs, last_lm = gs, lm
            gs_prompt = gs.unsqueeze(2)
            lm_prompt = lm.unsqueeze(2)
            i_with_prompt = torch.cat([z_i, gs_prompt, lm_prompt], dim=2)
            assert i_with_prompt.shape == (B, K, N + 2, self.width)

            i_flat = i_with_prompt.reshape(B * K, N + 2, self.width)
            i_flat = self.i_encoder.blocks[layer_idx](i_flat)
            z_i = i_flat[:, :N, :].reshape(B, K, N, self.width)
            assert z_i.shape == (B, K, N, self.width)

            r_flat = z_r.reshape(B * K, N, self.width)
            r_flat = self.r_encoder.blocks[layer_idx](r_flat)
            z_r = r_flat.reshape(B, K, N, self.width)
            assert z_r.shape == (B, K, N, self.width)

        z_i = self.i_encoder.ln_post(z_i.reshape(B * K, N, self.width)).reshape(B, K, N, self.width)
        frame_cls = z_i[:, :, 0, :]
        assert frame_cls.shape == (B, K, self.width)
        temporal = frame_cls
        for block in self.temporal_blocks:
            temporal = block(temporal)
        pooled = temporal.mean(dim=1)
        pooled = self.final_ln(pooled)
        video_features = pooled @ self.i_encoder.proj.to(dtype=pooled.dtype, device=pooled.device)
        video_features = F.normalize(video_features.float(), dim=-1)
        assert video_features.shape == (B, self.output_dim)
        return {
            "video_features": video_features,
            "debug": {
                "gspl": last_gs,
                "lmpl": last_lm,
                "frame_cls": frame_cls,
            },
        }
