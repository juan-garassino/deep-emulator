"""Arrow trajectory flow visualization.

Port of `RL-pokemon-red-experiments/visualization/BetterMapVis_script_version_FLOW.py:90-198`.

Pipeline:
- Load trajectory CSV(s) (step, x, y, map_id, action, reward) produced by
  `deepEmulator.utils.checkpoints.TrajectoryWriter`.
- `compute_flow(coords)` — accumulate movement vectors per global tile,
  skipping cross-map jumps.
- `render_arrows(flows, ...)` — for each tile, rotate a small arrow sprite by
  the resultant angle and color it via an HSL hue ramp (no matplotlib dep —
  pure HSV→RGB conversion).
- `main()` CLI — `deepemu-visualize --trajectories <bundle>/trajectories
  --cartridge "POKEMON RED" --out arrows.png`.

PIL is in `[viz]` extra; the CLI raises a clear error if missing.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# --- per-cartridge global-coord lookup --------------------------------------
def _resolve_local_to_global(cartridge: str) -> Callable[[int, int, int], tuple[int, int]]:
    """Return the cartridge's `local_to_global(r, c, map_n)` function."""
    cart = cartridge.upper()
    if cart == "POKEMON RED":
        from deepEmulator.cartridges.pokemon_red import local_to_global

        return local_to_global
    if cart == "POKEMON CRYSTAL":
        from deepEmulator.cartridges.pokemon_crystal import local_to_global

        return local_to_global
    raise KeyError(
        f"no local_to_global mapping registered for cartridge {cartridge!r}. "
        "Add a `local_to_global(r, c, map_n)` function to that cartridge module."
    )


# --- trajectory loading -----------------------------------------------------
def load_trajectory_dir(traj_dir: Path | str) -> np.ndarray:
    """Concatenate all `episode_*.csv.gz` in a dir into (N, 3) of (x, y, map_id)."""
    traj_dir = Path(traj_dir)
    files = sorted(traj_dir.glob("episode_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no episode_*.csv.gz in {traj_dir}")
    coords: list[tuple[int, int, int]] = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                coords.append((int(row["x"]), int(row["y"]), int(row["map_id"])))
    return np.array(coords, dtype=np.int32)


def load_trajectory_csv(csv_path: Path | str) -> np.ndarray:
    csv_path = Path(csv_path)
    opener = gzip.open if csv_path.suffix == ".gz" else open
    coords: list[tuple[int, int, int]] = []
    with opener(csv_path, "rt") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            coords.append((int(row["x"]), int(row["y"]), int(row["map_id"])))
    return np.array(coords, dtype=np.int32)


# --- flow accumulation ------------------------------------------------------
def compute_flow(
    coords: np.ndarray,
    local_to_global: Callable[[int, int, int], tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """Accumulate movement vectors per global tile, skipping cross-map jumps.

    Args:
        coords: (N, 3) int array of (x, y, map_id) per step
        local_to_global: callable(r, c, map_n) -> (gy, gx)
    Returns:
        dict mapping (gy, gx) -> resultant (dgy, dgx) movement vector
    """
    flows: dict[tuple[int, int], np.ndarray] = {}
    for i in range(1, len(coords)):
        cx, cy, cm = (int(v) for v in coords[i])
        px, py, pm = (int(v) for v in coords[i - 1])
        if cm != pm:
            continue  # skip cross-map transitions
        gy_c, gx_c = local_to_global(cy, cx, cm)
        gy_p, gx_p = local_to_global(py, px, pm)
        diff = np.array([gy_c - gy_p, gx_c - gx_p], dtype=np.int32)
        if abs(int(diff[0])) + abs(int(diff[1])) > 2:
            continue  # jump artifact
        key = (gy_p, gx_p)
        if key in flows:
            flows[key] = flows[key] + diff
        else:
            flows[key] = diff.copy()
    return flows


# --- rendering helpers ------------------------------------------------------
def _hsv_to_rgb(h: float, s: float = 0.85, v: float = 0.95) -> tuple[int, int, int]:
    """h in [0, 1] -> (r, g, b) in [0, 255] uint8."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


def _arrow_sprite(size: int, color: tuple[int, int, int]):
    """Build a flat-color RGBA arrow pointing right, size×size."""
    from PIL import Image, ImageDraw  # type: ignore

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    margin = max(1, size // 8)
    cx, cy = size // 2, size // 2
    shaft_h = max(2, size // 4)
    # Shaft + arrowhead, pointing right (will be rotated by render_arrows)
    d.rectangle(
        [margin, cy - shaft_h // 2, size - size // 3, cy + shaft_h // 2],
        fill=(*color, 255),
    )
    d.polygon(
        [
            (size - size // 3, cy - size // 3),
            (size - margin, cy),
            (size - size // 3, cy + size // 3),
        ],
        fill=(*color, 255),
    )
    return img


def render_arrows(
    flows: dict[tuple[int, int], np.ndarray],
    out_path: Path | str,
    *,
    cell_size: int = 16,
    background: tuple[int, int, int] = (30, 30, 30),
) -> Path:
    """Render an arrow grid PNG.

    Each tile gets a sprite rotated by atan2(-dgy, dgx) and colored by hue
    proportional to that angle. Background defaults to dark gray; pass a
    different RGB triple or set to white for printable output.
    """
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("PIL not installed — run `pip install -e '.[viz]'`")
    from PIL import Image  # type: ignore

    if not flows:
        raise ValueError("no flows to render (trajectory may be empty or all cross-map)")

    keys = list(flows.keys())
    min_y = min(k[0] for k in keys)
    max_y = max(k[0] for k in keys)
    min_x = min(k[1] for k in keys)
    max_x = max(k[1] for k in keys)
    h_cells = max_y - min_y + 1
    w_cells = max_x - min_x + 1

    canvas = Image.new("RGBA", (w_cells * cell_size, h_cells * cell_size), (*background, 255))
    for (gy, gx), vec in flows.items():
        dy, dx = float(vec[0]), float(vec[1])
        angle = math.atan2(-dy, dx)  # negate dy: image y grows downward
        hue = 0.5 * angle / math.pi + 0.5  # [0, 1]
        color = _hsv_to_rgb(hue)
        sprite = _arrow_sprite(cell_size, color)
        rotated = sprite.rotate(180.0 * angle / math.pi, resample=Image.Resampling.BICUBIC)
        canvas.paste(rotated, ((gx - min_x) * cell_size, (gy - min_y) * cell_size), rotated)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# --- CLI --------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-visualize")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--trajectories", type=Path, help="bundle/trajectories dir of episode_*.csv.gz")
    g.add_argument("--csv", type=Path, help="single trajectory CSV (gz or plain)")
    p.add_argument("--cartridge", default="POKEMON RED", help="for global-coord lookup")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell-size", type=int, default=16)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    local_to_global = _resolve_local_to_global(args.cartridge)
    if args.trajectories is not None:
        coords = load_trajectory_dir(args.trajectories)
        source = args.trajectories
    else:
        coords = load_trajectory_csv(args.csv)
        source = args.csv
    flows = compute_flow(coords, local_to_global)
    out = render_arrows(flows, args.out, cell_size=args.cell_size)
    print(
        f"[deepemu-visualize] read {len(coords)} steps from {source}; "
        f"rendered {len(flows)} tiles → {out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
