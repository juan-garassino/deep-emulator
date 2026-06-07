"""Wrap deepemu-eval with SHA256-signed result rows.

Used by the Phase 0.5 self-improve loop so Claude Code cannot silently
edit improvement_log.tsv to claim wins it did not earn. Each row carries
sha256(bundle || baseline || score || timestamp). A reviewer recomputes
the digest from the bundle directories on disk and rejects mismatches.

Audit, not sandbox: Claude has bash and could in principle bypass this.
But doing so leaves a missing/invalid signature trail for the reviewer.

Usage:
    python scripts/eval_signed.py \\
        --baseline checkpoints/coral/baseline_run \\
        --treatment checkpoints/coral/candidate_run \\
        --cartridge "POKEMON CORAL" \\
        --rom roms/PokemonCoral.gbc \\
        --episodes 10 \\
        --log improvement_log.tsv
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from deepEmulator.utils import gcs


def _digest(baseline: Path, treatment: Path, score: float, ts: str) -> str:
    h = hashlib.sha256()
    h.update(gcs.sha256_dir(baseline).encode())
    h.update(gcs.sha256_dir(treatment).encode())
    h.update(f"{score:.6f}".encode())
    h.update(ts.encode())
    return h.hexdigest()


def _parse_score_from_report(html_path: Path) -> float:
    """Extract the treatment mean-reward from deepemu-eval's HTML report."""
    text = html_path.read_text()
    marker = 'data-mean-reward="'
    i = text.find(marker)
    if i < 0:
        return float("nan")
    j = text.find('"', i + len(marker))
    return float(text[i + len(marker): j])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--treatment", required=True, type=Path)
    p.add_argument("--cartridge", required=True)
    p.add_argument("--rom", required=True)
    p.add_argument("--init-state", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--log", type=Path, default=Path("improvement_log.tsv"))
    args = p.parse_args(argv)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        report = Path(f.name)

    cmd = [
        sys.executable, "-m", "deepEmulator.training.eval",
        "--baseline", str(args.baseline),
        "--treatment", str(args.treatment),
        "--cartridge", args.cartridge,
        "--rom", args.rom,
        "--episodes", str(args.episodes),
        "--out", str(report),
    ]
    if args.init_state:
        cmd += ["--init-state", args.init_state]

    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[eval_signed] deepemu-eval exited rc={rc}", file=sys.stderr)
        return rc

    score = _parse_score_from_report(report)
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    sig = _digest(args.baseline, args.treatment, score, ts)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    if not args.log.exists():
        args.log.write_text("ts\tbaseline\ttreatment\tscore\tsig\n")
    with args.log.open("a") as out:
        out.write(f"{ts}\t{args.baseline}\t{args.treatment}\t{score:.6f}\t{sig}\n")
    print(json.dumps({"ts": ts, "score": score, "sig": sig}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
