"""Wrap deepemu-eval with SHA256-signed result rows.

Used by the Phase 0.5 self-improve loop so silent edits to
improvement_log.tsv are DETECTABLE. Each row carries
sha256(bundle_tree || baseline_tree || episodes_tsv_sha || score || ts);
a reviewer recomputes the digest from the artifacts on disk and rejects
mismatches.

HONEST FRAMING — tamper-EVIDENCE, not tamper-proof: the digest is unkeyed
and this script ships in the repo, so an agent with bash could rewrite a
row AND its signature. What it cannot do without detection is claim a
score that disagrees with the eval_episodes.tsv files committed alongside
(their sha is folded into the digest), and any log surgery shows up in
the branch diff the reviewer reads anyway.

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


def _digest(baseline: Path, treatment: Path, episodes_sha: str, score: float, ts: str) -> str:
    h = hashlib.sha256()
    h.update(gcs.sha256_dir(baseline).encode())
    h.update(gcs.sha256_dir(treatment).encode())
    h.update(episodes_sha.encode())
    h.update(f"{score:.6f}".encode())
    h.update(ts.encode())
    return h.hexdigest()


def _parse_report_json(json_path: Path) -> tuple[float, str]:
    """Return (treatment mean reward, combined eval_episodes sha)."""
    payload = json.loads(json_path.read_text())
    summaries = payload["summaries"]
    score = float(summaries[-1]["reward_mean"])  # treatment = last bundle
    shas = payload.get("eval_episodes_sha256", {})
    combined = hashlib.sha256(
        "".join(shas[k] or "" for k in sorted(shas)).encode()
    ).hexdigest()
    return score, combined


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

    json_path = report.with_suffix(".json")
    if not json_path.exists():
        print(f"[eval_signed] {json_path} missing — eval did not produce JSON", file=sys.stderr)
        return 3
    score, episodes_sha = _parse_report_json(json_path)
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    sig = _digest(args.baseline, args.treatment, episodes_sha, score, ts)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    if not args.log.exists():
        args.log.write_text("ts\tbaseline\ttreatment\tscore\tepisodes_sha\tsig\n")
    with args.log.open("a") as out:
        out.write(
            f"{ts}\t{args.baseline}\t{args.treatment}\t{score:.6f}\t{episodes_sha}\t{sig}\n"
        )
    print(json.dumps({"ts": ts, "score": score, "episodes_sha": episodes_sha, "sig": sig}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
