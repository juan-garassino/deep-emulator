"""Arrow trajectory viz tests — no ROM needed."""
from __future__ import annotations

import csv
import gzip
import importlib
import sys

import numpy as np
import pytest


def _fake_local_to_global(r: int, c: int, map_n: int) -> tuple[int, int]:
    """Trivial identity-ish mapping for tests."""
    return r + 100 * map_n, c + 100 * map_n


def test_compute_flow_accumulates_per_tile():
    from deepEmulator.visualization.arrows import compute_flow

    # Walk in y: (x=0,y=0)→(x=0,y=1)→(x=0,y=2), all on map 0. Two south-moves.
    # compute_flow calls local_to_global(cy, cx, cm) so the diff lives in row (y).
    coords = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.int32)
    flows = compute_flow(coords, _fake_local_to_global)
    assert len(flows) == 2
    for vec in flows.values():
        assert vec.tolist() == [1, 0]  # dgy=+1 (south), dgx=0


def test_compute_flow_skips_cross_map_jumps():
    from deepEmulator.visualization.arrows import compute_flow

    # map change between steps → must be skipped
    coords = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.int32)
    flows = compute_flow(coords, _fake_local_to_global)
    # Only the (0,0,1)→(0,1,1) transition counts (same map)
    assert len(flows) == 1


def test_compute_flow_skips_large_jumps():
    """A 'diff' with manhattan distance > 2 is treated as an artifact."""
    from deepEmulator.visualization.arrows import compute_flow

    def teleport_lookup(r, c, m):
        # Same input always maps to wildly different outputs for testing
        return (r * 100, c * 100)

    coords = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int32)
    flows = compute_flow(coords, teleport_lookup)
    assert flows == {}


def test_resolve_local_to_global_pokemon_red():
    from deepEmulator.visualization.arrows import _resolve_local_to_global

    fn = _resolve_local_to_global("POKEMON RED")
    out = fn(5, 7, 40)
    assert isinstance(out, tuple) and len(out) == 2


def test_resolve_local_to_global_rejects_unknown():
    from deepEmulator.visualization.arrows import _resolve_local_to_global

    with pytest.raises(KeyError):
        _resolve_local_to_global("UNKNOWN GAME")


def test_load_trajectory_dir(tmp_path):
    from deepEmulator.visualization.arrows import load_trajectory_dir

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    for ep in range(2):
        with gzip.open(traj_dir / f"episode_{ep:06d}.csv.gz", "wt") as f:
            w = csv.writer(f)
            w.writerow(["step", "x", "y", "map_id", "action", "reward"])
            for s in range(5):
                w.writerow([s, s, s * 2, 40, s % 4, 0.0])

    coords = load_trajectory_dir(traj_dir)
    assert coords.shape == (10, 3)
    assert coords[0].tolist() == [0, 0, 40]


def test_load_trajectory_dir_raises_when_empty(tmp_path):
    from deepEmulator.visualization.arrows import load_trajectory_dir

    (tmp_path / "trajectories").mkdir()
    with pytest.raises(FileNotFoundError):
        load_trajectory_dir(tmp_path / "trajectories")


@pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="PIL required (install [viz] extra)"
)
def test_render_arrows_produces_png(tmp_path):
    from deepEmulator.visualization.arrows import render_arrows

    flows = {
        (0, 0): np.array([0, 1], dtype=np.int32),  # → east
        (0, 1): np.array([1, 0], dtype=np.int32),  # → south
        (1, 1): np.array([0, -1], dtype=np.int32),  # → west
        (1, 0): np.array([-1, 0], dtype=np.int32),  # → north
    }
    out = render_arrows(flows, tmp_path / "arrows.png", cell_size=16)
    assert out.exists() and out.stat().st_size > 100

    from PIL import Image  # type: ignore

    img = Image.open(out)
    assert img.size == (2 * 16, 2 * 16)


@pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="PIL required (install [viz] extra)"
)
def test_render_arrows_rejects_empty():
    from deepEmulator.visualization.arrows import render_arrows

    with pytest.raises(ValueError):
        render_arrows({}, "/tmp/_unused.png")


@pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="PIL required (install [viz] extra)"
)
def test_cli_end_to_end_with_real_pokemon_red_mapping(tmp_path):
    """Full path: write fake trajectories → run main → assert PNG."""
    from deepEmulator.visualization import arrows as arrows_mod

    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    with gzip.open(traj_dir / "episode_000000.csv.gz", "wt") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "map_id", "action", "reward"])
        # Use real Pokemon Red map_id 40 with adjacent positions so flow accumulates
        for i in range(20):
            w.writerow([i, 5 + (i % 4), 3, 40, 0, 0.0])

    out = tmp_path / "arrows.png"
    sys.argv = [
        "deepemu-visualize",
        "--trajectories", str(traj_dir),
        "--cartridge", "POKEMON RED",
        "--out", str(out),
        "--cell-size", "8",
    ]
    rc = arrows_mod.main()
    assert rc == 0
    assert out.exists() and out.stat().st_size > 100
