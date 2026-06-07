#!/usr/bin/env python3
"""Generate `cartridges/pokemon_crystal_map_data.json` — synthetic global
canvas layout for arrow viz.

We don't have a real Johto world map vendored; instead, every Crystal map
gets a non-overlapping rectangular region on a large canvas, grouped by
region (= map_group). Output is functionally identical to PWhiddy's Red
`pokemon_red_map_data.json` for our `local_to_global` purposes.

Map dimensions ideally come from pret/pokecrystal `data/maps/attributes.asm`
(or similar). If `--source` isn't given, fall back to a synthetic catalog
that lists every (group, number) combination we expect to encounter in early
Johto with a default 20×20 tile cell.

Run:
    python scripts/build_crystal_map_data.py \\
        --out deepEmulator/cartridges/pokemon_crystal_map_data.json \\
        [--source path/to/pokecrystal/data/maps/attributes.asm] \\
        [--padding 2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# When run as a script, allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Default synthetic catalog. Each entry: (group, number, height, width).
# Sizes are approximate; the exact value doesn't matter for arrow viz since
# we just need non-overlapping rectangles. Covers early-Johto maps the agent
# is likely to visit; unknown maps fall through to a default cell at the origin.
_DEFAULT_CATALOG: list[tuple[int, int, int, int]] = [
    # (group, number, h, w)
    (1, 1, 9, 10),   # Olivine City  -- placeholder dim
    (24, 1, 18, 20), # New Bark Town -- placeholder dim
    (24, 2, 18, 30), # Route 29
    (24, 3, 18, 30), # Route 30
    (24, 4, 18, 30), # Route 31
    (24, 5, 18, 20), # Violet City
    (24, 6, 9, 10),  # Sprout Tower
    (26, 1, 18, 20), # Cherrygrove City
    (26, 2, 9, 10),  # Cherrygrove Mart
    (26, 3, 9, 10),  # Cherrygrove Pokecenter
    # Add more here as needed; unknown maps render as size 0 cells at canvas origin
]


# Pattern matches lines like:
#   map_attributes NEW_BARK_TOWN, NEW_BARK_TOWN, $05, EAST | WEST
# in pokecrystal/data/maps/attributes.asm. We don't try to extract dimensions
# from this file (they're elsewhere); just extract names.
_ATTR_LINE = re.compile(r"map_attributes\s+(\w+),\s*(\w+)\b")


def _parse_pret_attributes(path: Path) -> list[tuple[str, str]]:
    """Extract (map_const, group_const) pairs from pret/pokecrystal attributes.asm."""
    pairs = []
    for line in path.read_text().splitlines():
        m = _ATTR_LINE.match(line.strip())
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def build_map_data(
    catalog: list[tuple[int, int, int, int]],
    *,
    padding: int = 2,
    canvas_cols: int = 8,
) -> dict:
    """Pack each (group, number) into a non-overlapping cell on a global canvas.

    Layout strategy: by group, in rows; within a group, maps lay out left→right.
    Each cell gets a (height + padding) × (width + padding) box.
    """
    # Bucket by group
    by_group: dict[int, list[tuple[int, int, int]]] = {}
    for group, number, h, w in catalog:
        by_group.setdefault(group, []).append((number, h, w))

    groups: dict[str, dict[str, dict]] = {}
    cursor_y = 0
    max_x = 0
    for group, entries in sorted(by_group.items()):
        entries.sort(key=lambda e: e[0])  # by map_number
        row_x = 0
        row_max_h = 0
        groups[str(group)] = {}
        for number, h, w in entries:
            groups[str(group)][str(number)] = {
                "y_offset": cursor_y,
                "x_offset": row_x,
                "height": h,
                "width": w,
            }
            row_x += w + padding
            row_max_h = max(row_max_h, h)
        max_x = max(max_x, row_x)
        cursor_y += row_max_h + padding

    return {
        "groups": groups,
        "global_shape": [cursor_y, max_x],
        "padding": padding,
    }


def main() -> int:
    p = argparse.ArgumentParser(prog="build-crystal-map-data")
    p.add_argument("--out", type=Path,
                   default=Path("deepEmulator/cartridges/pokemon_crystal_map_data.json"))
    p.add_argument("--source", type=Path, default=None,
                   help="Optional path to pret/pokecrystal data/maps/attributes.asm")
    p.add_argument("--padding", type=int, default=2)
    args = p.parse_args()

    catalog = list(_DEFAULT_CATALOG)
    if args.source is not None:
        pairs = _parse_pret_attributes(args.source)
        print(f"[build-crystal-map-data] parsed {len(pairs)} maps from {args.source}")
        # We don't have dimensions from this file; use 20x20 default for parsed maps
        # Real-future improvement: parse data/maps/blockdata.asm for actual dims
        # For each parsed name, add a placeholder entry. Use index as (synthetic_group, synthetic_num)
        # to avoid collisions with the curated _DEFAULT_CATALOG.
        for i, (map_name, group_name) in enumerate(pairs):
            # Synthesize a unique (group, number) — not the real Crystal IDs, but
            # enough for layout when real IDs aren't known.
            synth_group = 100 + (i // 16)
            synth_num = i % 16
            catalog.append((synth_group, synth_num, 16, 16))

    data = build_map_data(catalog, padding=args.padding)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2))
    print(
        f"[build-crystal-map-data] wrote {args.out} "
        f"({len(data['groups'])} groups, canvas {data['global_shape']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
