"""Colab wrapper around `pretrain_dino.main` — Drive mount + path resolution.

Mirrors `training/colab_train.py`. Relative paths resolve to MyDrive.
"""
from __future__ import annotations

from pathlib import Path

from deepEmulator.training.colab_train import _in_colab, _mount_drive, _print_gpu, _resolve


def run_in_colab(
    *,
    corpus: str | Path,
    steps: int = 50_000,
    batch_size: int = 64,
    save_every: int = 2_000,
    out_dim: int = 4096,
    n_local_crops: int = 6,
    run_dir: str | Path | None = None,
    drive_prefix: str = "deepEmulator",
    resume: bool = True,
) -> int:
    from deepEmulator.training.pretrain_dino import main

    drive_root: Path | None = None
    if _in_colab():
        drive_root = _mount_drive()
        print(f"[colab] drive mounted at {drive_root}")
        _print_gpu()

    corpus_p = _resolve(corpus, drive_root)
    runs_root = (drive_root / drive_prefix / "encoders") if drive_root else Path("encoders")

    argv = [
        "--corpus", str(corpus_p),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--save-every", str(save_every),
        "--out-dim", str(out_dim),
        "--n-local-crops", str(n_local_crops),
        "--runs-root", str(runs_root),
    ]
    if run_dir is not None:
        argv += ["--run-dir", str(_resolve(run_dir, drive_root))]
    if resume:
        argv += ["--resume"]
    return main(argv)
