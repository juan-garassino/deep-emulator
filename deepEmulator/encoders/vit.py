"""ViT-tiny implementation for 96×96 grayscale emulator frames.

Design:
- Conv2d patch-embed (kernel=stride=patch_size)
- Frozen 2D sincos positional embedding (no learned posenc to avoid collapse on small data)
- Pre-LN transformer blocks; each block stashes its last attention map for rollout
- CLS token → projection head → latent

Reference: `lucidrains/vit-pytorch:vit.py` (clean baseline), `facebookresearch/mae:models_mae.py:get_2d_sincos_pos_embed`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 2D sincos positional embedding (vendored from MAE) ---------------------
def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True) -> np.ndarray:
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0)  # (2, H, W)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (N, embed_dim)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return pos_embed.astype(np.float32)


# --- building blocks --------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class Attention(nn.Module):
    """Multi-head self-attention that stashes the attention map for rollout."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- the model --------------------------------------------------------------
@dataclass
class ViTConfig:
    image_size: int = 96
    in_chans: int = 1
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    out_dim: int = 256


class ViTTiny(nn.Module):
    """ViT-tiny for 96×96 grayscale emulator frames.

    Output:
        cls_latent: (B, out_dim) — what the agent / DINO head consumes
    Forward returns latent; call `forward_tokens()` for (cls, patch_tokens)
    and `get_attention_maps()` after a forward pass for rollout.
    """

    def __init__(self, cfg: ViTConfig | None = None):
        super().__init__()
        self.cfg = cfg or ViTConfig()
        c = self.cfg
        assert c.image_size % c.patch_size == 0, "image_size must divide patch_size"
        self.grid_size = c.image_size // c.patch_size
        self.num_patches = self.grid_size**2

        self.patch_embed = PatchEmbed(c.in_chans, c.embed_dim, c.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Frozen sincos posenc, registered as buffer
        pe = get_2d_sincos_pos_embed(c.embed_dim, self.grid_size, cls_token=True)
        self.register_buffer("pos_embed", torch.from_numpy(pe).unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [TransformerBlock(c.embed_dim, c.num_heads, c.mlp_ratio) for _ in range(c.depth)]
        )
        self.norm = nn.LayerNorm(c.embed_dim)
        self.head = nn.Linear(c.embed_dim, c.out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk; return (cls_token, patch_tokens) after final LN."""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        x = x + self.pos_embed[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.forward_tokens(x)
        return self.head(cls)

    def get_attention_maps(self) -> list[torch.Tensor]:
        """List of (B, heads, N+1, N+1) tensors, one per block. Call after a forward."""
        return [blk.attn.last_attn for blk in self.blocks if blk.attn.last_attn is not None]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
