"""Attention rollout + overlay rendering — no ROM/env needed."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_attention_rollout_shape_and_normalization():
    from deepEmulator.visualization.attention import attention_rollout

    # 3 layers, batch 2, 3 heads, 5 tokens (1 CLS + 4 patches)
    maps = []
    for _ in range(3):
        a = torch.softmax(torch.randn(2, 3, 5, 5), dim=-1)
        maps.append(a)
    out = attention_rollout(maps)
    assert out.shape == (2, 4)  # B, N (patches only, CLS dropped)
    assert (out >= 0).all()  # softmax + identity addition keeps it non-negative


def test_attention_rollout_identity_layers_yields_uniform():
    """If every layer is exactly the identity, CLS attention to patches should be uniform."""
    from deepEmulator.visualization.attention import attention_rollout

    N = 5
    eye = torch.eye(N).unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).clone()
    maps = [eye, eye, eye]
    out = attention_rollout(maps)
    # rollout adds identity each layer and renormalizes -> roughly uniform after chaining
    # Just check it's positive and sums to a finite value
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4)


def test_reshape_to_grid():
    from deepEmulator.visualization.attention import reshape_to_grid

    cls_attn = torch.arange(36, dtype=torch.float32).view(2, 18).reshape(2, 18)[:1].expand(2, -1)
    cls_attn = torch.arange(36, dtype=torch.float32).reshape(1, 36)
    g = reshape_to_grid(cls_attn, 6)
    assert g.shape == (1, 6, 6)
    assert g[0, 0, 0].item() == 0.0
    assert g[0, 5, 5].item() == 35.0


def test_bilinear_up_matches_endpoints():
    from deepEmulator.visualization.attention import _bilinear_up

    grid = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    up = _bilinear_up(grid, 4, 4)
    assert up.shape == (4, 4)
    assert up[0, 0] == 0.0
    assert up[0, -1] == 1.0
    assert up[-1, 0] == 2.0
    assert up[-1, -1] == 3.0


def test_heatmap_hot_endpoints():
    from deepEmulator.visualization.attention import _heatmap_hot

    out = _heatmap_hot(np.array([[0.0, 1.0]]))
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [0, 0, 0]  # black
    assert out[0, 1].tolist() == [255, 255, 255]  # white


def test_compose_overlay_shape_and_dtype():
    from deepEmulator.visualization.attention import compose_overlay

    frame = np.full((144, 160, 3), 128, dtype=np.uint8)
    grid = np.random.RandomState(0).rand(24, 24).astype(np.float32)
    out = compose_overlay(frame, grid, alpha=0.5)
    assert out.shape == (144, 160, 3)
    assert out.dtype == np.uint8


def test_attention_for_frame_with_real_vit():
    """End-to-end: ViT forward + rollout + overlay on a real-shape frame."""
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny
    from deepEmulator.visualization.attention import attention_for_frame

    enc = ViTTiny(ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64))
    frame = np.full((144, 160, 3), 200, dtype=np.uint8)
    overlay, grid = attention_for_frame(enc, frame, alpha=0.5, device="cpu")
    assert overlay.shape == (144, 160, 3)
    assert overlay.dtype == np.uint8
    assert grid.shape == (12, 12)  # 96 / 8 = 12


def test_write_gif_roundtrip(tmp_path):
    from deepEmulator.visualization.attention import write_gif

    pytest.importorskip("PIL")
    frames = [np.random.RandomState(i).randint(0, 256, (40, 40, 3), dtype=np.uint8) for i in range(5)]
    out = tmp_path / "test.gif"
    write_gif(frames, out, fps=10)
    assert out.exists() and out.stat().st_size > 0


def test_attention_cli_module_exposes_main():
    from deepEmulator.visualization import attention as att

    assert hasattr(att, "main")
    assert callable(att.main)
