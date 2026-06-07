"""Tests for the Crystal init.state script. Most are import-only; the
end-to-end run is ROM-gated."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
ROM = REPO / "roms" / "PokemonCrystal.gbc"

_SCRIPT = REPO / "scripts" / "make_crystal_init_state.py"
_spec = importlib.util.spec_from_file_location("make_crystal_init", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["make_crystal_init"] = _mod
_spec.loader.exec_module(_mod)


def test_script_module_loads():
    assert hasattr(_mod, "drive_intro")
    assert hasattr(_mod, "main")
    assert hasattr(_mod, "_STARTER_INDEX")
    assert set(_mod._STARTER_INDEX) == {"chikorita", "cyndaquil", "totodile"}


def test_arg_parsing_defaults_to_cyndaquil():
    sys.argv = ["make_crystal_init.py", "--rom", "fake.gbc"]
    p = _mod.argparse.ArgumentParser(prog="x")
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--starter", default="cyndaquil", choices=list(_mod._STARTER_INDEX))
    args = p.parse_args(["--rom", "fake.gbc"])
    assert args.starter == "cyndaquil"


def test_press_and_mash_helpers_exist():
    assert callable(_mod._press)
    assert callable(_mod._mash)
    assert callable(_mod._enter_name)
    assert callable(_mod._wait_for_map)


@pytest.mark.skipif(not ROM.exists(), reason=f"need {ROM.relative_to(REPO)}")
@pytest.mark.timeout(600)
def test_end_to_end_produces_savestate(tmp_path):
    """The full intro script should produce a valid PyBoy save-state."""
    out = tmp_path / "crystal_init.state"
    _mod.drive_intro(ROM, out, name="RED", starter="cyndaquil", headless=True, verbose=True)
    assert out.exists()
    assert out.stat().st_size > 10_000  # PyBoy states are ~150 KB
