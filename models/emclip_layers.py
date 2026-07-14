import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """LayerNorm that keeps CLIP-style fp16 inputs numerically stable."""

    def forward(self, x):
        dtype = x.dtype
        return super().forward(x.float()).to(dtype=dtype)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with full self-attention."""

    def __init__(self, width, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ln_2 = LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, hidden)),
            ("gelu", QuickGELU()),
            ("drop_1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(hidden, width)),
            ("drop_2", nn.Dropout(dropout)),
        ]))
        self._init_new_weights()

    def _init_new_weights(self):
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: [B, N, D]
        assert x.ndim == 3, "TransformerBlock expects [B, N, D]."
        normalized = self.ln_1(x)
        attn_out = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


def interpolate_positional_embedding(positional_embedding, target_grid):
    """Bicubic interpolation for absolute ViT position embeddings.

    positional_embedding: [1 + old_h * old_w, D]
    target_grid: (new_h, new_w)
    """
    assert positional_embedding.ndim == 2, "position embedding must be [N, D]."
    cls_pos = positional_embedding[:1]
    patch_pos = positional_embedding[1:]
    old_grid = int(math.sqrt(patch_pos.size(0)))
    assert old_grid * old_grid == patch_pos.size(0), "patch positions must form a square grid."
    new_h, new_w = target_grid
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_h * new_w, -1)
    return torch.cat([cls_pos.float(), patch_pos], dim=0).to(dtype=positional_embedding.dtype)


class PatchTokenEncoder(nn.Module):
    """CLIP-style ViT token encoder used by I, residual, and motion branches."""

    def __init__(
        self,
        in_channels,
        input_resolution,
        patch_size,
        width,
        layers,
        heads,
        output_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        grid = max(1, input_resolution // patch_size)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid * grid + 1, width))
        self.ln_pre = LayerNorm(width)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) if output_dim is not None else None
        self._init_patch_weights()

    def _init_patch_weights(self):
        nn.init.normal_(self.conv1.weight, std=self.width ** -0.5)

    def embed_patches(self, x):
        # x: [B, C, H, W]
        assert x.ndim == 4, "PatchTokenEncoder input must be [B, C, H, W]."
        assert x.size(1) == self.in_channels, (
            "expected %d channels, got %d" % (self.in_channels, x.size(1))
        )
        assert x.size(-2) >= self.patch_size and x.size(-1) >= self.patch_size, (
            "input spatial size must be at least the patch size."
        )
        x = self.conv1(x)
        grid_h, grid_w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        cls = self.class_embedding.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        x = torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)
        pos = interpolate_positional_embedding(
            self.positional_embedding.to(dtype=x.dtype, device=x.device),
            (grid_h, grid_w),
        )
        x = x + pos.unsqueeze(0)
        x = self.ln_pre(x)
        assert x.shape == (x.size(0), grid_h * grid_w + 1, self.width)
        return x

    def forward_tokens(self, x):
        x = self.embed_patches(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x

    def forward(self, x):
        tokens = self.forward_tokens(x)
        cls = tokens[:, 0]
        if self.proj is not None:
            cls = cls @ self.proj.to(dtype=cls.dtype, device=cls.device)
        return cls, tokens


def init_linear_or_attention(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight)
        nn.init.zeros_(module.out_proj.bias)
