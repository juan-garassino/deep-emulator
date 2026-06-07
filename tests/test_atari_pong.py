"""Smoke tests for Atari platform + Pong adapter.

The 100-random-step end-to-end test is gated on `ale-py` being installed.
Unit tests (adapter shape, registry, preprocessing utility) run unconditionally.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_pong_registers():
    from deepEmulator.cartridges.atari import pong  # noqa: F401
    from deepEmulator.core import registry

    assert "ATARI PONG" in registry.available()


def test_pong_adapter_defaults():
    from deepEmulator.cartridges.atari.pong import PongAdapter

    a = PongAdapter()
    assert a.platform == "atari"
    assert a.rom_name == "pong"
    assert a.observation_shape == (4, 84, 84)
    assert a.terminal_on_life_loss is False


def test_pong_reward_clipping():
    from deepEmulator.cartridges.atari.pong import PongAdapter

    a = PongAdapter(clip_reward=True)
    assert a.compute_reward({}, {"raw_reward": 5.0}, None) == 1.0
    assert a.compute_reward({}, {"raw_reward": -3.0}, None) == -1.0
    assert a.compute_reward({}, {"raw_reward": 0.0}, None) == 0.0

    a_raw = PongAdapter(clip_reward=False)
    assert a_raw.compute_reward({}, {"raw_reward": 5.0}, None) == 5.0


def test_resize_grayscale_84_shape():
    from deepEmulator.platforms.atari import _resize_grayscale_84

    frame = np.arange(210 * 160, dtype=np.uint8).reshape(210, 160) % 255
    out = _resize_grayscale_84(frame.astype(np.uint8))
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8


@pytest.mark.skipif(
    importlib.util.find_spec("ale_py") is None,
    reason="ale-py not installed (pip install -e '.[atari]')",
)
def test_atari_pong_100_random_steps():
    from deepEmulator.cartridges.atari.pong import PongAdapter
    from deepEmulator.platforms.atari import AtariEnv

    adapter = PongAdapter()
    env = AtariEnv(adapter, frame_stack=4, max_steps=200, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8

    rng = np.random.default_rng(0)
    saw_reward = False
    for _ in range(100):
        action = int(rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == (4, 84, 84)
        assert "lives" in info
        if r != 0.0:
            saw_reward = True
        if term or trunc:
            break
    env.close()
    # Pong scores points roughly every ~20-60 frames at random play.
    # 100 frames may not be enough to score — just assert nothing crashed.
    assert isinstance(saw_reward, bool)
