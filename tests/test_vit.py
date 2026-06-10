"""ViT-tiny unit tests — shape contract + attention extractability."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_pos_embed_shape():
    from deepEmulator.encoders.vit import get_2d_sincos_pos_embed

    pe = get_2d_sincos_pos_embed(192, 24, cls_token=True)
    assert pe.shape == (24 * 24 + 1, 192)
    assert pe.dtype == np.float32
    # CLS row is zeros
    assert np.allclose(pe[0], 0.0)


def test_vit_forward_shapes():
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    model = ViTTiny(ViTConfig(image_size=96, patch_size=4, embed_dim=192, depth=4, num_heads=3))
    x = torch.zeros(2, 1, 96, 96)
    z = model(x)
    assert z.shape == (2, 256)


def test_vit_forward_tokens_returns_cls_and_patches():
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    model = ViTTiny(ViTConfig(image_size=96, patch_size=4, embed_dim=192, depth=4, num_heads=3))
    x = torch.zeros(3, 1, 96, 96)
    cls, patches = model.forward_tokens(x)
    assert cls.shape == (3, 192)
    assert patches.shape == (3, 576, 192)


def test_vit_attention_maps_extractable():
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    model = ViTTiny(ViTConfig(image_size=96, patch_size=4, embed_dim=192, depth=4, num_heads=3))
    x = torch.zeros(2, 1, 96, 96)

    # training mode must NOT stash attention — at DINO batch sizes the stash
    # held ~6 GB of dead GPU memory across student + teacher
    model.train()
    _ = model(x)
    assert all(blk.attn.last_attn is None for blk in model.blocks)

    # eval mode (all rollout/visualization paths) stashes as before
    model.eval()
    _ = model(x)
    maps = model.get_attention_maps()
    assert len(maps) == 4
    for m in maps:
        # (B, heads, N+1, N+1) with N = 576
        assert m.shape == (2, 3, 577, 577)


def test_vit_tiny_param_count_is_sane():
    """Full ViT-tiny (depth 12, dim 192) should be ~5M params."""
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    model = ViTTiny(ViTConfig())  # default = full ViT-tiny
    n = model.num_parameters()
    assert 4_500_000 < n < 6_500_000, f"unexpected param count: {n:,}"


def test_vit_backward_works():
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    model = ViTTiny(ViTConfig(image_size=96, patch_size=4, embed_dim=96, depth=2, num_heads=3))
    x = torch.randn(2, 1, 96, 96)
    z = model(x)
    loss = z.sum()
    loss.backward()
    # CLS token + patch_embed.proj.weight should have grads
    assert model.cls_token.grad is not None
    assert model.patch_embed.proj.weight.grad is not None
