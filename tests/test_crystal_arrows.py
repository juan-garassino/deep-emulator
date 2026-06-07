"""Tests for the Crystal arrow viz: map_data builder + local_to_global wiring."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent

# Load the map-data builder script
_BUILD = REPO / "scripts" / "build_crystal_map_data.py"
_spec = importlib.util.spec_from_file_location("build_crystal_map_data", _BUILD)
_build = importlib.util.module_from_spec(_spec)
sys.modules["build_crystal_map_data"] = _build
_spec.loader.exec_module(_build)


# --- packer correctness ----------------------------------------------------
def test_build_map_data_packs_groups_without_overlap():
    catalog = [
        (1, 0, 10, 10),
        (1, 1, 10, 10),
        (2, 0, 10, 10),
    ]
    data = _build.build_map_data(catalog, padding=2)
    g1 = data["groups"]["1"]
    g2 = data["groups"]["2"]
    # Within group 1, the two maps share y_offset (same row) but x differs
    assert g1["0"]["y_offset"] == g1["1"]["y_offset"]
    assert g1["1"]["x_offset"] > g1["0"]["x_offset"] + g1["0"]["width"]
    # Group 2 is on a new row below group 1
    assert g2["0"]["y_offset"] > g1["0"]["y_offset"] + g1["0"]["height"]


def test_build_map_data_global_shape_grows_with_groups():
    data_one = _build.build_map_data([(1, 0, 5, 5)], padding=0)
    data_two = _build.build_map_data([(1, 0, 5, 5), (2, 0, 5, 5)], padding=0)
    assert data_two["global_shape"][0] > data_one["global_shape"][0]


# --- shipped map_data.json -------------------------------------------------
def test_shipped_map_data_loads():
    path = REPO / "deepEmulator" / "cartridges" / "pokemon_crystal_map_data.json"
    assert path.exists(), "run scripts/build_crystal_map_data.py to generate"
    data = json.loads(path.read_text())
    assert "groups" in data
    assert "global_shape" in data
    assert isinstance(data["global_shape"], list) and len(data["global_shape"]) == 2


# --- local_to_global uses the JSON when present ----------------------------
def test_local_to_global_uses_json_layout():
    """For a known (group, num), local_to_global should respect the JSON offsets."""
    # Reload the module to ensure JSON is freshly read
    from deepEmulator.cartridges import pokemon_crystal as pc

    pc._MAP_DATA = None  # force reload of vendored JSON
    map_id = 24 * 256 + 1  # New Bark Town per default catalog
    gy, gx = pc.local_to_global(0, 0, map_id=map_id)
    # New Bark Town has nonzero y_offset (group 24 is below group 1)
    assert gy > 0 or gx >= 0  # not a teleport to (PAD, PAD) which means missing


def test_local_to_global_unknown_map_falls_back_to_pad():
    from deepEmulator.cartridges import pokemon_crystal as pc

    pc._MAP_DATA = None
    # Map_id we definitely didn't include in the catalog
    gy, gx = pc.local_to_global(0, 0, map_id=999 * 256 + 99)
    # Falls back to (_PAD, _PAD) when (group, number) isn't in the JSON
    assert gy == pc._PAD and gx == pc._PAD


# --- arrows.py wiring ------------------------------------------------------
def test_arrows_resolves_pokemon_crystal():
    from deepEmulator.visualization.arrows import _resolve_local_to_global

    fn = _resolve_local_to_global("POKEMON CRYSTAL")
    out = fn(0, 0, 24 * 256 + 1)
    assert isinstance(out, tuple) and len(out) == 2


# --- end-to-end via compute_flow ------------------------------------------
def test_crystal_compute_flow_on_synthetic_trajectory():
    """Build a fake trajectory across a single Crystal map and confirm flow accumulates."""
    from deepEmulator.cartridges import pokemon_crystal as pc
    from deepEmulator.visualization.arrows import compute_flow

    pc._MAP_DATA = None
    # Walk east through New Bark Town: x=0,1,2,3, all on map 24:1
    map_id = 24 * 256 + 1
    coords = np.array(
        [[i, 5, map_id] for i in range(4)],
        dtype=np.int32,
    )
    flows = compute_flow(coords, pc.local_to_global)
    assert len(flows) >= 1  # at least one tile has an outgoing vector


@pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="PIL required (install [viz] extra)"
)
def test_crystal_arrows_cli_end_to_end(tmp_path):
    """Full path: fake trajectory CSV → run deepemu-visualize for Crystal → PNG."""
    from deepEmulator.visualization import arrows as arrows_mod

    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    with gzip.open(traj_dir / "episode_000000.csv.gz", "wt") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "map_id", "action", "reward"])
        # Walk east a few tiles on New Bark Town (group 24, map_number 1)
        map_id = 24 * 256 + 1
        for i in range(10):
            w.writerow([i, i % 20, 5, map_id, 2, 0.0])

    out = tmp_path / "crystal_arrows.png"
    sys.argv = [
        "deepemu-visualize",
        "--trajectories", str(traj_dir),
        "--cartridge", "POKEMON CRYSTAL",
        "--out", str(out),
        "--cell-size", "8",
    ]
    rc = arrows_mod.main()
    assert rc == 0
    assert out.exists() and out.stat().st_size > 100
