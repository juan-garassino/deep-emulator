"""DINO pretraining CLI.

    deepemu-pretrain-dino --corpus data/frames/multi --steps 50000 \\
                          --batch-size 64 --runs-root encoders

Writes a portable encoder bundle to `{runs-root}/dino/<timestamp>/`:
    encoder.pt             # trainer state_dict (student + teacher + opt + center)
    encoder_only.pt        # just the student trunk weights (for RL inference)
    metadata.json          # vit_cfg, dino_cfg, crop_cfg, hyperparams, versions
    metrics.tsv            # iteration, loss, time
    latest.txt             # marker in {runs-root}/dino/ pointing at this run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from deepEmulator.data.frame_corpus import FrameStorage
from deepEmulator.encoders.augmentations import MultiCropConfig
from deepEmulator.encoders.dino import DINOConfig, DINOTrainer
from deepEmulator.encoders.vit import ViTConfig
from deepEmulator.utils.checkpoints import _versions, find_latest_run


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="deepemu-pretrain-dino")
    p.add_argument("--corpus", type=Path, required=True, help="FrameStorage dir (with index.json)")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-every", type=int, default=2_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=Path("encoders"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dim", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--n-local-crops", type=int, default=6)
    return p.parse_args(argv)


def _write_bundle(run_dir: Path, trainer: DINOTrainer, hyper: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), run_dir / "encoder.pt")
    torch.save(trainer.encoder_only_state_dict(), run_dir / "encoder_only.pt")
    metadata = {
        "algo": "dino",
        "vit_config": trainer.vit_cfg.__dict__,
        "dino_config": trainer.dino_cfg.__dict__,
        "crop_config": trainer.crop_cfg.__dict__,
        "hyper": hyper,
        "step_count": trainer.step_count,
        "versions": _versions(),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    # latest.txt sibling of the run dir — stores the run NAME (relative),
    # so the marker stays meaningful after a GCS round-trip
    (run_dir.parent / "latest.txt").write_text(run_dir.name + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    dino_root = args.runs_root / "dino"

    resume_from: Path | None = None
    if args.resume:
        resume_from = find_latest_run(dino_root)
        if resume_from is None:
            print(f"[pretrain-dino] --resume requested but no prior run under {dino_root}")

    if args.run_dir is None:
        if resume_from is not None:
            args.run_dir = resume_from
        else:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_dir = dino_root / stamp
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Build trainer
    trainer = DINOTrainer(
        vit_cfg=ViTConfig(),
        dino_cfg=DINOConfig(out_dim=args.out_dim, learning_rate=args.learning_rate),
        crop_cfg=MultiCropConfig(n_local=args.n_local_crops),
    )
    print(f"[pretrain-dino] device={trainer.device} run_dir={args.run_dir}")
    print(f"[pretrain-dino] ViT params: {trainer.student.num_parameters():,}")

    if resume_from is not None:
        state = torch.load(resume_from / "encoder.pt", map_location=trainer.device, weights_only=False)
        trainer.load_state_dict(state)
        print(f"[pretrain-dino] resumed at step {trainer.step_count}")

    # Open corpus
    storage = FrameStorage(args.corpus)
    print(f"[pretrain-dino] corpus: {storage.total_frames:,} frames at {args.corpus}")

    # Metrics file
    metrics_path = args.run_dir / "metrics.tsv"
    if not metrics_path.exists():
        metrics_path.write_text("Step\tLoss\tTime\n")

    hyper = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "n_local_crops": args.n_local_crops,
    }

    from deepEmulator.utils.wandb_logger import WandbLogger

    wb = WandbLogger(config={"algo": "dino", **hyper})

    t0 = time.time()
    losses_window: list[float] = []
    try:
        while trainer.step_count < args.steps:
            # Cycle the corpus
            for batch in storage.iter_batches(args.batch_size, shuffle=True, seed=trainer.step_count):
                loss = trainer.step(batch)
                losses_window.append(loss)

                if trainer.step_count % args.log_every == 0:
                    mean = sum(losses_window) / len(losses_window)
                    losses_window = []
                    with open(metrics_path, "a") as f:
                        f.write(f"{trainer.step_count}\t{mean:.4f}\t{time.time() - t0:.1f}\n")
                    print(f"  step {trainer.step_count:6d} | loss {mean:.4f}")
                    wb.log({"loss": mean, "time": time.time() - t0}, step=trainer.step_count)

                if trainer.step_count > 0 and trainer.step_count % args.save_every == 0:
                    _write_bundle(args.run_dir, trainer, hyper)

                if trainer.step_count >= args.steps:
                    break
    finally:
        _write_bundle(args.run_dir, trainer, hyper)
        wb.finish()
    print(f"[pretrain-dino] done. bundle at {args.run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
