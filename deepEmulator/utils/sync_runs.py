"""Sync the runs root to GCS — the container's artifact lifeline.

    python -m deepEmulator.utils.sync_runs /runs gs://bucket/prefix

Called by entrypoint.sh from the periodic background loop AND the EXIT trap.
Exits non-zero if ANY directory fails to sync — the entrypoint surfaces that
loudly instead of shrugging a lost run off as "non-fatal".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepEmulator.utils import gcs


def sync(runs_root: Path | str, prefix_uri: str) -> list[str]:
    """Push every directory under `runs_root` to `<prefix_uri>/<name>`.

    Returns the list of failed directory names (empty = full success).
    """
    root = Path(runs_root)
    prefix_uri = prefix_uri.rstrip("/")
    if not root.exists():
        print(f"[sync_runs] {root} does not exist — nothing to sync")
        return []
    failed: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        target = f"{prefix_uri}/{child.name}"
        try:
            gcs.upload_dir(child, target)
            print(f"[sync_runs] pushed {child} -> {target}")
        except Exception as e:
            print(f"[sync_runs] FAILED {child} -> {target}: {e}", file=sys.stderr)
            failed.append(child.name)
    return failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sync_runs")
    p.add_argument("runs_root", type=Path)
    p.add_argument("prefix_uri")
    args = p.parse_args(argv)
    failed = sync(args.runs_root, args.prefix_uri)
    if failed:
        print(f"[sync_runs] {len(failed)} dir(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
