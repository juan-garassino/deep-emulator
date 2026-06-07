"""RunPod entrypoint — GCS-stage + path resolution wrapper around `train.main`.

Mirrors `deepEmulator/training/colab_train.py` (Drive seam) but for the
RunPod + GCS seam. Usage from a notebook or a local docker-run:

    from deepEmulator.training.runpod_train import run_in_runpod
    run_in_runpod(
        cartridge="POKEMON CORAL",
        rom_uri="gs://garassino-ml-artifacts/roms/PokemonCoral.gbc",
        init_state_uri="gs://garassino-ml-artifacts/states/coral_init.state",
        steps=100_000,
        gcs_bucket="gs://garassino-ml-artifacts",
        gcs_prefix="coral/run-001",
    )

Staging is idempotent — re-running pulls only what's missing.
"""
from __future__ import annotations

import os
from pathlib import Path

from deepEmulator.utils import gcs


def _stage(uri: str | None, dest_dir: Path) -> str | None:
    if uri is None:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(uri).name
    if dest.exists():
        return str(dest)
    gcs.download(uri, dest)
    return str(dest)


def run_in_runpod(
    *,
    cartridge: str,
    rom_uri: str | None = None,
    init_state_uri: str | None = None,
    encoder_uri: str | None = None,
    steps: int = 100_000,
    max_episode_steps: int = 2048,
    save_every: int = 10_000,
    gcs_bucket: str = "gs://garassino-ml-artifacts",
    gcs_prefix: str = "deepemulator/default",
    data_dir: str | Path = "/data",
    runs_root: str | Path = "/runs",
    resume: bool = True,
) -> int:
    """Stage GCS objects locally and invoke `train.main`.

    On exit, the calling layer (entrypoint.sh) is expected to rsync
    `runs_root/` back to `${gcs_bucket}/${gcs_prefix}/`.
    """
    from deepEmulator.training.train import main

    data_dir = Path(data_dir)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    rom_local = _stage(rom_uri, data_dir)
    init_local = _stage(init_state_uri, data_dir)
    enc_local = _stage(encoder_uri, data_dir / "encoders") if encoder_uri else None

    prefix_uri = f"{gcs_bucket.rstrip('/')}/{gcs_prefix.strip('/')}"
    resume_uri = gcs.latest_run_uri(prefix_uri) if resume else None
    if resume_uri:
        print(f"[runpod_train] resume marker -> {resume_uri}")
    else:
        print(f"[runpod_train] no resume marker at {prefix_uri}/latest.txt — fresh run")

    argv = [
        "--cartridge", cartridge,
        "--steps", str(steps),
        "--max-episode-steps", str(max_episode_steps),
        "--save-every", str(save_every),
        "--runs-root", str(runs_root),
        "--headless",
    ]
    if rom_local:
        argv += ["--rom", rom_local]
    if init_local:
        argv += ["--init-state", init_local]
    if enc_local:
        argv += ["--encoder", enc_local]
    if resume:
        argv += ["--resume"]

    print(f"[runpod_train] argv: {argv}")
    return main(argv)


if __name__ == "__main__":
    # Allow `python -m deepEmulator.training.runpod_train` for ad-hoc invocation.
    run_in_runpod(
        cartridge=os.environ.get("CARTRIDGE", "POKEMON CORAL"),
        rom_uri=os.environ.get("ROM_GCS_URI"),
        init_state_uri=os.environ.get("INIT_STATE_GCS_URI"),
        encoder_uri=os.environ.get("ENCODER_GCS_URI"),
        steps=int(os.environ.get("STEPS", "5000")),
        gcs_bucket=os.environ.get("GCS_BUCKET", "gs://garassino-ml-artifacts"),
        gcs_prefix=os.environ.get("GCS_PREFIX", "deepemulator/default"),
    )
