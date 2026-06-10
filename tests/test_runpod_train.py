"""runpod_train resume seam — the marker must lead to a DOWNLOADED bundle.

file:// end-to-end with a stubbed train.main, per the project stub pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _seed_remote_bundle(bucket_root: Path, prefix: str) -> None:
    run = bucket_root / prefix / "train" / "run-007"
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"weights")
    (run / "metadata.json").write_text(json.dumps({"cartridge_title": "GENERIC GB"}))
    (bucket_root / prefix / "train" / "latest.txt").write_text("run-007\n")


def test_resume_downloads_bundle_and_passes_run_dir(tmp_path, monkeypatch):
    from deepEmulator.training import runpod_train
    from deepEmulator.training import train as train_mod

    bucket_root = tmp_path / "bucket"
    _seed_remote_bundle(bucket_root, "deepemulator/x")

    received: dict = {}

    def _fake_main(argv):
        received["argv"] = argv
        return 0

    monkeypatch.setattr(train_mod, "main", _fake_main)

    rc = runpod_train.run_in_runpod(
        cartridge="GENERIC GB",
        gcs_bucket=f"file://{bucket_root}",
        gcs_prefix="deepemulator/x",
        data_dir=tmp_path / "data",
        runs_root=tmp_path / "runs",
        steps=10,
    )
    assert rc == 0
    argv = received["argv"]
    assert "--resume" in argv
    assert "--run-dir" in argv
    run_dir = Path(argv[argv.index("--run-dir") + 1])
    # the bundle was actually downloaded, not just name-dropped
    assert (run_dir / "model.pt").read_bytes() == b"weights"
    assert (run_dir / "metadata.json").exists()


def test_fresh_run_without_marker(tmp_path, monkeypatch):
    from deepEmulator.training import runpod_train
    from deepEmulator.training import train as train_mod

    received: dict = {}
    monkeypatch.setattr(train_mod, "main", lambda argv: received.update(argv=argv) or 0)

    rc = runpod_train.run_in_runpod(
        cartridge="GENERIC GB",
        gcs_bucket=f"file://{tmp_path}/bucket",
        gcs_prefix="deepemulator/fresh",
        data_dir=tmp_path / "data",
        runs_root=tmp_path / "runs",
        steps=10,
    )
    assert rc == 0
    assert "--run-dir" not in received["argv"]
    assert "--resume" in received["argv"]
