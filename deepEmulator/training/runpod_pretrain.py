"""RunPod entrypoint — DINO / V-JEPA pretraining wrapper.

Mirrors `deepEmulator/training/colab_pretrain.py` (Drive seam) but for
the RunPod + GCS seam. Artifact upload is entrypoint.sh's job (periodic
sync + EXIT trap), so no bucket params here.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from deepEmulator.utils import gcs


def run_in_runpod(
    *,
    algo: str = "dino",
    corpus_uri: str,
    steps: int = 10_000,
    batch_size: int = 32,
    save_every: int = 250,
    data_dir: str | Path = "/data",
    runs_root: str | Path = "/runs",
    run_id: str | None = None,
) -> int:
    """Stage the frame corpus from GCS and run the appropriate pretrain CLI."""
    # validate the algo BEFORE staging gigabytes of corpus
    if algo == "dino":
        from deepEmulator.training.pretrain_dino import main
    elif algo == "vjepa":
        from deepEmulator.training.pretrain_vjepa import IMPLEMENTED, main  # type: ignore

        if not IMPLEMENTED:
            print("[runpod_pretrain] V-JEPA is Phase 1 — not implemented; refusing before staging")
            return 2
    else:
        raise ValueError(f"unknown pretrain algo {algo!r}")

    data_dir = Path(data_dir)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    corpus_local = data_dir / "corpus"
    if not corpus_local.exists():
        print(f"[runpod_pretrain] staging corpus {corpus_uri} -> {corpus_local}")
        gcs.download_dir(corpus_uri, corpus_local)

    # run dir gets a RUN_ID component so latest.txt lands at
    # <runs_root>/<algo>/latest.txt (the documented marker location) instead
    # of colliding with the train marker at the runs root
    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    argv = [
        "--corpus", str(corpus_local),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--save-every", str(save_every),
        "--run-dir", str(runs_root / algo / run_id),
    ]
    print(f"[runpod_pretrain] algo={algo} argv={argv}")
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(
        run_in_runpod(
            algo=os.environ.get("ALGO", "dino"),
            corpus_uri=os.environ["CORPUS_GCS_URI"],
            steps=int(os.environ.get("STEPS", "10000")),
            batch_size=int(os.environ.get("BATCH_SIZE", "32")),
            save_every=int(os.environ.get("SAVE_EVERY", "250")),
            run_id=os.environ.get("RUN_ID"),
        )
    )
