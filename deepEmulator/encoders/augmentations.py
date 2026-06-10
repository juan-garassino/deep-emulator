"""Multi-crop augmentations for DINO on emulator grayscale frames.

Grayscale-appropriate ops only: RandomResizedCrop, brightness/contrast jitter,
Gaussian blur. No color jitter, no solarization (natural-image tricks). All
crops are resized back to 96×96 so a single ViT consumes them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _gaussian_kernel(radius: float, size: int = 7) -> torch.Tensor:
    sigma = max(radius, 1e-6)
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)  # (size, size)


def _gaussian_blur(x: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Apply 2D gaussian blur to (B, 1, H, W). Pure-torch, no cv2/PIL."""
    k = _gaussian_kernel(radius).to(x.device, x.dtype)
    k = k.view(1, 1, *k.shape)
    pad = k.shape[-1] // 2
    return F.conv2d(x, k, padding=pad)


def _brightness_contrast(x: torch.Tensor, brightness: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, H, W) in [0, 1]. brightness/contrast: (B, 1, 1, 1) per-sample draws."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x + brightness
    return x.clamp(0.0, 1.0)


def _random_resized_crop(x: torch.Tensor, out_size: int, scale: tuple[float, float]) -> torch.Tensor:
    """PER-SAMPLE random crop boxes via affine_grid/grid_sample (batched, no
    python loop). Batch-shared crops gave every sample in a batch geometrically
    identical view pairs — much weaker invariances than reference DINO.

    x: (B, 1, H, W) → (B, 1, out_size, out_size)
    """
    B, _, H, W = x.shape
    short = float(min(H, W))
    s = torch.empty(B, device=x.device).uniform_(*scale)
    side = (s.sqrt() * short).clamp(min=8.0, max=short)  # crop side in pixels
    y0 = torch.rand(B, device=x.device) * (H - side)
    x0 = torch.rand(B, device=x.device) * (W - side)
    # normalized [-1, 1] center + scale for affine_grid (x = horizontal first)
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=torch.float32)
    theta[:, 0, 0] = side / W
    theta[:, 0, 2] = (2 * x0 + side) / W - 1
    theta[:, 1, 1] = side / H
    theta[:, 1, 2] = (2 * y0 + side) / H - 1
    grid = F.affine_grid(theta, (B, 1, out_size, out_size), align_corners=False)
    out = F.grid_sample(x.float(), grid, mode="bilinear", align_corners=False)
    return out.clamp(0.0, 1.0)


@dataclass
class MultiCropConfig:
    n_global: int = 2
    n_local: int = 6
    out_size: int = 96
    global_scale: tuple[float, float] = (0.4, 1.0)
    local_scale: tuple[float, float] = (0.05, 0.4)
    blur_prob: float = 0.5
    blur_radius_range: tuple[float, float] = (0.1, 2.0)
    brightness_range: tuple[float, float] = (-0.2, 0.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)


class MultiCropAugment:
    """Produce `n_global + n_local` augmented views of each input batch.

    Input: x of shape (B, 1, H, W) with H, W >= out_size, values in [0, 255] uint8 or [0, 1] float.
    Output: list of tensors, each (B, 1, out_size, out_size), float in [0, 1].

    The first `n_global` entries are global views; the rest are local. This
    ordering matters for the DINO loss (teacher only sees globals).
    """

    def __init__(self, cfg: MultiCropConfig | None = None):
        self.cfg = cfg or MultiCropConfig()

    def _augment_one(self, x: torch.Tensor, scale: tuple[float, float]) -> torch.Tensor:
        B = x.shape[0]
        x = _random_resized_crop(x, self.cfg.out_size, scale)
        # Brightness / contrast — per-sample draws
        b = torch.empty(B, 1, 1, 1, device=x.device).uniform_(*self.cfg.brightness_range)
        c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(*self.cfg.contrast_range)
        x = _brightness_contrast(x, b, c)
        # Gaussian blur with prob — batch-shared radius per view (the kernel
        # conv is the expensive part; per-sample radii buy little here)
        if random.random() < self.cfg.blur_prob:
            r = random.uniform(*self.cfg.blur_radius_range)
            # kernel normalization is only float32-exact, so blur can overshoot
            # the [0, 1] contract by ~1e-7 on saturated regions — clamp it back
            x = _gaussian_blur(x, r).clamp(0.0, 1.0)
        return x

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()
        crops: list[torch.Tensor] = []
        for _ in range(self.cfg.n_global):
            crops.append(self._augment_one(x, self.cfg.global_scale))
        for _ in range(self.cfg.n_local):
            crops.append(self._augment_one(x, self.cfg.local_scale))
        return crops

    @property
    def n_total(self) -> int:
        return self.cfg.n_global + self.cfg.n_local
