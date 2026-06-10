"""Tests for the central cartridge loader (deepEmulator/cartridges/__init__.py).

Motivated by the audit finding that 4 of 7 CLIs shipped private import lists
covering only pokemon_red + pong, making `make play/eval/attention/collect_frames`
fail for the default CART="POKEMON CORAL".
"""
from __future__ import annotations

import importlib.util

from deepEmulator.cartridges import load_all

_HAS_ALE = importlib.util.find_spec("ale_py") is not None

_GB_TITLES = {"POKEMON RED", "POKEMON CRYSTAL", "POKEMON CORAL", "GENERIC GB"}


def test_load_all_registers_every_gameboy_cartridge():
    available = set(load_all())
    assert _GB_TITLES <= available, available


def test_load_all_registers_pong_when_ale_present():
    available = set(load_all())
    if _HAS_ALE:
        assert "ATARI PONG" in available


def test_load_all_is_idempotent():
    first = set(load_all())
    second = set(load_all())
    assert first == second


def test_coral_adapter_constructible_via_registry():
    from deepEmulator.core import registry

    load_all()
    AdapterCls = registry.get("POKEMON CORAL")
    adapter = AdapterCls()
    assert adapter.platform == "gameboy"


def test_arrows_resolver_routes_coral_to_crystal_map():
    from deepEmulator.cartridges.pokemon_crystal import local_to_global
    from deepEmulator.visualization.arrows import _resolve_local_to_global

    assert _resolve_local_to_global("POKEMON CORAL") is local_to_global
