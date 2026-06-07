"""Colab entrypoint — Drive-mount + path resolution wrapper around `train.main`.

Usage from a Colab cell:

    !pip install -q git+https://github.com/<you>/deepEmulator.git
    from deepEmulator.training.colab_train import run_in_colab
    run_in_colab(
        cartridge="POKEMON RED",
        rom="deepEmulator/roms/PokemonRed.gb",      # path under MyDrive
        init_state="deepEmulator/states/init.state",
        steps=100_000,
    )

Resolves paths relative to /content/drive/MyDrive/ when in Colab; falls back
to literal paths when run outside Colab (so the same call works locally).
"""
from __future__ import annotations

import os
from pathlib import Path


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except ImportError:
        return False


def _mount_drive(mountpoint: str = "/content/drive") -> Path:
    from google.colab import drive  # type: ignore

    if not os.path.ismount(mountpoint):
        drive.mount(mountpoint)
    return Path(mountpoint) / "MyDrive"


def _resolve(path: str | Path, drive_root: Path | None) -> Path:
    p = Path(path)
    if p.is_absolute() or drive_root is None:
        return p
    return drive_root / p


def run_in_colab(
    *,
    cartridge: str,
    rom: str | Path,
    init_state: str | Path | None = None,
    steps: int = 100_000,
    max_episode_steps: int = 2048,
    save_every: int = 10_000,
    run_dir: str | Path | None = None,
    drive_prefix: str = "deepEmulator",
    resume: bool = True,
    encoder: str | Path | None = None,
) -> int:
    """Mount Drive (if on Colab), resolve paths, and run `train.main`.

    Run output goes to `MyDrive/{drive_prefix}/checkpoints/<slug>/<timestamp>/`
    unless `run_dir` is explicitly set. When `resume=True` (default), continues
    from the latest run under `MyDrive/{drive_prefix}/checkpoints/<slug>/`
    (resolved via `latest.txt`), which is the intended Colab-session pattern:
    re-running the same cell after a disconnect picks up where you left off.
    """
    from deepEmulator.training.train import main

    drive_root: Path | None = None
    if _in_colab():
        drive_root = _mount_drive()
        print(f"[colab] drive mounted at {drive_root}")
        _print_gpu()

    rom_p = _resolve(rom, drive_root)
    init_p = _resolve(init_state, drive_root) if init_state else None

    runs_root = (drive_root / drive_prefix / "checkpoints") if drive_root else Path("checkpoints")

    argv = [
        "--cartridge", cartridge,
        "--rom", str(rom_p),
        "--steps", str(steps),
        "--max-episode-steps", str(max_episode_steps),
        "--save-every", str(save_every),
        "--runs-root", str(runs_root),
        "--headless",
    ]
    if init_p is not None:
        argv += ["--init-state", str(init_p)]
    if run_dir is not None:
        argv += ["--run-dir", str(_resolve(run_dir, drive_root))]
    if resume:
        argv += ["--resume"]
    if encoder is not None:
        argv += ["--encoder", str(_resolve(encoder, drive_root))]

    return main(argv)


def _print_gpu() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            print(f"[colab] cuda OK — {torch.cuda.get_device_name(0)}")
        else:
            print("[colab] no CUDA — running on CPU (training will be slow)")
    except Exception:
        pass


def main():  # CLI shim, mirrors train.main but with Colab path resolution
    import argparse

    p = argparse.ArgumentParser(prog="deepemu-colab-train")
    p.add_argument("--cartridge", required=True)
    p.add_argument("--rom", required=True)
    p.add_argument("--init-state", default=None)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--max-episode-steps", type=int, default=2048)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--drive-prefix", default="deepEmulator")
    args = p.parse_args()
    return run_in_colab(**vars(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
