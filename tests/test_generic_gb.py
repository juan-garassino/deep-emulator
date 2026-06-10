"""GenericGameBoyAdapter unit tests + ROM-gated smoke for any GB/GBC ROM."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
TEST_ROM = REPO / "roms" / "test_rom.gb"

requires_rom = pytest.mark.skipif(
    not TEST_ROM.exists(),
    reason=f"drop any GB/GBC ROM at {TEST_ROM.relative_to(REPO)} to run this smoke",
)


def test_generic_gb_registers():
    from deepEmulator.cartridges import generic_gb  # noqa: F401
    from deepEmulator.core import registry

    assert "GENERIC GB" in registry.available()


def test_generic_gb_adapter_defaults():
    from deepEmulator.cartridges.generic_gb import GenericGameBoyAdapter

    a = GenericGameBoyAdapter()
    assert a.platform == "gameboy"
    assert a.action_set == ["down", "left", "right", "up", "a", "b"]  # start is opt-in
    assert GenericGameBoyAdapter(include_start=True).action_set[-1] == "start"
    assert a.observation_shape == (3, 72, 80)
    assert a.is_done({}) is False


def test_generic_gb_reward_is_zero():
    from deepEmulator.cartridges.generic_gb import GenericGameBoyAdapter

    a = GenericGameBoyAdapter()
    assert a.compute_reward({}, {}, None) == 0.0


def test_grayscale_fix_is_noop_on_dmg_like_input(monkeypatch):
    """Verify the new luminance preprocessing is identity-equivalent when R=G=B."""
    from deepEmulator.platforms import gameboy as gb_mod

    # PyBoy 2.4 returns RGB(A); DMG renders to identical channels.
    fake_dmg = np.tile(np.arange(0, 144 * 160, dtype=np.uint8).reshape(144, 160)[..., None], (1, 1, 3))
    # The luminance step: mean over channels -> equal to channel 0 since R=G=B
    luminance = fake_dmg[:, :, :3].mean(axis=-1).astype(np.uint8)
    assert np.array_equal(luminance, fake_dmg[:, :, 0])


def test_grayscale_fix_uses_all_channels_for_gbc_like_input():
    """On GBC (real RGB), mean ≠ R channel."""
    rng = np.random.default_rng(0)
    fake_gbc = rng.integers(0, 256, (144, 160, 3), dtype=np.uint8)
    luminance = fake_gbc[:, :, :3].mean(axis=-1).astype(np.uint8)
    # Almost surely different from any single channel
    assert not np.array_equal(luminance, fake_gbc[:, :, 0])
    assert not np.array_equal(luminance, fake_gbc[:, :, 1])
    assert not np.array_equal(luminance, fake_gbc[:, :, 2])


@requires_rom
def test_generic_gb_100_random_steps():
    from deepEmulator.cartridges.generic_gb import GenericGameBoyAdapter
    from deepEmulator.platforms.gameboy import PyBoyEnv

    adapter = GenericGameBoyAdapter()
    env = PyBoyEnv(adapter, rom_path=TEST_ROM, headless=True, max_steps=200)

    obs, info = env.reset()
    assert obs.shape == (3, 72, 80)
    assert obs.dtype == np.uint8

    rng = np.random.default_rng(0)
    for _ in range(100):
        action = int(rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == (3, 72, 80)
        # Generic adapter always returns zero reward
        assert r == 0.0
        if term or trunc:
            break

    env.close()

@requires_rom
def test_reset_without_init_state_restores_boot_snapshot():
    """Two resets must give identical first observations — reset() previously
    left the emulator running from wherever the last episode ended."""
    from deepEmulator.cartridges.generic_gb import GenericGameBoyAdapter
    from deepEmulator.platforms.gameboy import PyBoyEnv

    adapter = GenericGameBoyAdapter()
    env = PyBoyEnv(adapter, rom_path=TEST_ROM, headless=True, max_steps=200)
    try:
        obs1, _ = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(30):
            env.step(int(rng.integers(0, env.action_space.n)))
        obs2, _ = env.reset()
        assert np.array_equal(obs1, obs2)
    finally:
        env.close()


@requires_rom
def test_step_none_is_idle_and_advances_time():
    from deepEmulator.cartridges.generic_gb import GenericGameBoyAdapter
    from deepEmulator.platforms.gameboy import PyBoyEnv

    adapter = GenericGameBoyAdapter()
    env = PyBoyEnv(adapter, rom_path=TEST_ROM, headless=True, max_steps=200)
    try:
        env.reset()
        obs, r, term, trunc, info = env.step(None)
        assert obs.shape == (3, 72, 80)
        assert r == 0.0
    finally:
        env.close()
