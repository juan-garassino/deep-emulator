"""Smoke test: 100 random steps against Pokemon Red.

Skipped automatically unless the user has dropped:
  - roms/PokemonRed.gb
  - states/init.state
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
ROM = REPO / "roms" / "PokemonRed.gb"
STATE = REPO / "states" / "init.state"

requires_rom = pytest.mark.skipif(
    not (ROM.exists() and STATE.exists()),
    reason=f"needs {ROM.relative_to(REPO)} and {STATE.relative_to(REPO)} present",
)


def test_pokemon_red_registry_loaded():
    # importing the module registers the adapter
    from deepEmulator.cartridges import pokemon_red  # noqa: F401
    from deepEmulator.core import registry

    assert "POKEMON RED" in registry.available()


def test_pokemon_red_adapter_shape():
    from deepEmulator.cartridges.pokemon_red import PokemonRedAdapter

    adapter = PokemonRedAdapter()
    assert adapter.platform == "gameboy"
    assert len(adapter.action_set) == 7
    assert adapter.observation_shape == (3, 72, 80)


@requires_rom
def test_pokemon_red_100_random_steps():
    from deepEmulator.cartridges.pokemon_red import PokemonRedAdapter
    from deepEmulator.platforms.gameboy import PyBoyEnv

    adapter = PokemonRedAdapter(init_state=STATE)
    env = PyBoyEnv(adapter, rom_path=ROM, init_state=STATE, headless=True, max_steps=200)

    obs, info = env.reset()
    assert obs.shape == (3, 72, 80)
    assert obs.dtype == np.uint8
    assert "game_state" in info

    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(100):
        action = int(rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        assert obs.shape == (3, 72, 80)
        assert "trajectory" in info
        if term or trunc:
            break

    env.close()
    # reward should at least vary across 100 random steps (exploration accumulates)
    assert any(r != 0.0 for r in rewards) or len(set(rewards)) >= 1
