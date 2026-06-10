"""Hot-path perf primitives — correctness vs the old reference implementations."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _reference_grab(frame: np.ndarray) -> np.ndarray:
    """The old two-float64-pass implementation."""
    full = frame[:, :, :3].mean(axis=-1).astype(np.uint8)
    h2, w2 = frame.shape[0] // 2, frame.shape[1] // 2
    return full.reshape(h2, 2, w2, 2).mean(axis=(1, 3)).astype(np.uint8)


def test_downscale_luma_2x_matches_reference_within_1():
    from deepEmulator.platforms.gameboy import downscale_luma_2x

    rng = np.random.default_rng(0)
    for channels in (3, 4):
        frame = rng.integers(0, 256, (144, 160, channels), dtype=np.uint8)
        fast = downscale_luma_2x(frame)
        ref = _reference_grab(frame)
        assert fast.shape == (72, 80)
        assert fast.dtype == np.uint8
        assert int(np.abs(fast.astype(np.int16) - ref.astype(np.int16)).max()) <= 1


def test_inplace_shift_matches_np_roll():
    stack = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    new = np.full((4, 5), 99, dtype=np.uint8)

    expected = np.roll(stack.copy(), 1, axis=0)
    expected[0] = new

    got = stack.copy()
    for i in range(got.shape[0] - 1, 0, -1):
        got[i] = got[i - 1]
    got[0] = new

    assert np.array_equal(got, expected)


class _StackingInnerEnv:
    """Fake inner env with REAL frame-stack semantics: newest at index 0,
    older frames shift down — the contract the latent cache relies on."""

    def __init__(self, frame_stack=3, h=72, w=80, n_actions=7):
        from deepEmulator.core.spaces import Box, Discrete

        class _Cart:
            cartridge_title = "FAKE"
            platform = "gameboy"

        self.cartridge = _Cart()
        self.observation_space = Box(0, 255, (frame_stack, h, w))
        self.action_space = Discrete(n_actions)
        self._rng = np.random.default_rng(0)
        self._stack = np.zeros((frame_stack, h, w), dtype=np.uint8)

    def _new_frame(self):
        return self._rng.integers(0, 256, self._stack.shape[1:], dtype=np.uint8)

    def reset(self, *, seed=None):
        first = self._new_frame()
        self._stack[:] = first
        return self._stack.copy(), {}

    def step(self, action):
        self._stack[1:] = self._stack[:-1]
        self._stack[0] = self._new_frame()
        return self._stack.copy(), 0.1, False, False, {}

    def close(self):
        pass


def test_frozen_encoder_latent_cache_matches_full_reencode():
    """Cached (encode newest only) vs uncached (re-encode all): same values
    up to batched-matmul float noise."""
    from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    enc = ViTTiny(cfg)

    cached_env = FrozenEncoderEnv(_StackingInnerEnv(), enc, device="cpu")
    uncached_env = FrozenEncoderEnv(_StackingInnerEnv(), enc, device="cpu")
    uncached_env._cache_ok = False  # force the full re-encode path

    o1, _ = cached_env.reset()
    o2, _ = uncached_env.reset()
    assert np.allclose(o1, o2, atol=1e-5)
    for step in range(6):
        a1 = cached_env.step(0)[0]
        a2 = uncached_env.step(0)[0]
        assert np.allclose(a1, a2, atol=1e-5), f"divergence at step {step}"
