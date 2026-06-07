"""RunPod entrypoint — DINO / V-JEPA pretraining wrapper.

Mirrors `deepEmulator/training/colab_pretrain.py` (Drive seam) but for
the RunPod + GCS seam.
"""
from __future__ import annotations

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
    gcs_bucket: str = "gs://garassino-ml-artifacts",
    gcs_prefix: str = "deepemulator/encoders/default",
    data_dir: str | Path = "/data",
    runs_root: str | Path = "/runs",
) -> int:
    """Stage the frame corpus from GCS and run the appropriate pretrain CLI."""
    data_dir = Path(data_dir)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    corpus_local = data_dir / "corpus"
    if not corpus_local.exists():
        print(f"[runpod_pretrain] staging corpus {corpus_uri} -> {corpus_local}")
        gcs.download_dir(corpus_uri, corpus_local)

    if algo == "dino":
        from deepEmulator.training.pretrain_dino import main
    elif algo == "vjepa":
        from deepEmulator.training.pretrain_vjepa import main  # type: ignore
    else:
        raise ValueError(f"unknown pretrain algo {algo!r}")

    argv = [
        "--corpus", str(corpus_local),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--save-every", str(save_every),
        "--run-dir", str(runs_root / algo),
    ]
    print(f"[runpod_pretrain] algo={algo} argv={argv}")
    return main(argv)


if __name__ == "__main__":
    run_in_runpod(
        algo=os.environ.get("ALGO", "dino"),
        corpus_uri=os.environ["CORPUS_GCS_URI"],
        steps=int(os.environ.get("STEPS", "10000")),
        gcs_bucket=os.environ.get("GCS_BUCKET", "gs://garassino-ml-artifacts"),
        gcs_prefix=os.environ.get("GCS_PREFIX", "deepemulator/encoders/default"),
    )
