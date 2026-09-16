"""Reward shaping reads a mock RAM dict correctly — no ROM/PyBoy needed.

lixado/PyBoy-RL's AISettings interface computes reward from raw RAM addresses.
Our CartridgeAdapter equivalent (PokemonRedAdapter) reads `pyboy.memory[addr]`;
here a stub emulator exposes `.memory` as an address->byte dict so we can assert
the RAM-driven reads and reward deltas without an emulator.
"""
from __future__ import annotations

import pytest


class _MockRAM:
    """address -> byte, defaulting to 0 for unmapped addresses."""

    def __init__(self, values: dict[int, int] | None = None):
        self._m: dict[int, int] = dict(values or {})

    def __getitem__(self, addr: int) -> int:
        return self._m.get(addr, 0)

    def __setitem__(self, addr: int, val: int) -> None:
        self._m[addr] = val


class _MockEmulator:
    def __init__(self, values: dict[int, int] | None = None):
        self.memory = _MockRAM(values)


def test_read_game_state_reads_ram_addresses():
    from deepEmulator.cartridges.pokemon_red import (
        BADGE_COUNT,
        MAP_N,
        PARTY_SIZE,
        X_POS,
        Y_POS,
        PokemonRedAdapter,
    )

    emu = _MockEmulator(
        {
            X_POS: 12,
            Y_POS: 7,
            MAP_N: 3,
            PARTY_SIZE: 2,
            BADGE_COUNT: 0b0000_0111,  # 3 badge bits set
        }
    )
    adapter = PokemonRedAdapter()
    state = adapter.read_game_state(emu)
    assert state["x"] == 12
    assert state["y"] == 7
    assert state["map_id"] == 3
    assert state["party_size"] == 2
    assert state["badges"] == 3  # popcount of 0b0000_0111


def test_hp_fraction_from_mock_ram():
    from deepEmulator.cartridges.pokemon_red import HP, MAX_HP, PokemonRedAdapter

    # first party slot: hp=50/100 (big-endian byte pairs), rest zero
    emu = _MockEmulator(
        {
            HP[0]: 0, HP[0] + 1: 50,
            MAX_HP[0]: 0, MAX_HP[0] + 1: 100,
        }
    )
    adapter = PokemonRedAdapter()
    assert adapter._hp_fraction(emu) == pytest.approx(0.5)


def test_explore_reward_increases_as_new_tiles_are_seen():
    """The explore component is RAM-driven: it counts unique (x,y,map) tiles.
    Moving to a new tile must raise the reward delta above zero."""
    from deepEmulator.cartridges.pokemon_red import (
        IN_BATTLE,
        MAP_N,
        X_POS,
        Y_POS,
        PokemonRedAdapter,
    )

    emu = _MockEmulator({X_POS: 1, Y_POS: 1, MAP_N: 0, IN_BATTLE: 0})
    adapter = PokemonRedAdapter()
    adapter.reset_episode(emu)  # establishes the RAM baseline

    r1 = adapter.compute_reward({}, {}, emu)  # first tile seen
    emu.memory[X_POS] = 2  # step to a brand-new tile
    r2 = adapter.compute_reward({}, {}, emu)
    assert r2 > 0.0  # exploring a new tile yields positive shaped reward
    assert len(adapter.seen_coords) == 2  # two distinct tiles recorded from RAM


def test_phased_reward_boot_helper_reads_mock_state():
    """PhasedReward.constant_per_step: reads a mock game-state dict, adds a
    bonus the step party_size transitions 0 -> 1 (picked a starter)."""
    from deepEmulator.core.reward import PhasedReward, RewardPhase, constant_per_step

    boot = constant_per_step(value=0.01, first_acquire_bonus=1.0, acquire_key="party_size")
    phased = PhasedReward(strict=True).add(
        RewardPhase(name="boot", is_active=lambda s, e: True, compute=boot)
    )
    # no acquisition -> just the per-step trickle
    assert phased.compute({"party_size": 0}, {"party_size": 0}, None) == pytest.approx(0.01)
    # 0 -> 1 party transition -> trickle + bonus
    assert phased.compute({"party_size": 0}, {"party_size": 1}, None) == pytest.approx(1.01)
    assert phased.last_phase == "boot"
