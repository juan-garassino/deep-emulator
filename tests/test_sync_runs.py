"""utils/sync_runs — the container's artifact lifeline (file:// exercised)."""
from __future__ import annotations

from pathlib import Path

from deepEmulator.utils import sync_runs
from deepEmulator.utils import gcs


def _make_runs(root: Path) -> None:
    (root / "train" / "run-1").mkdir(parents=True)
    (root / "train" / "run-1" / "model.pt").write_bytes(b"weights")
    (root / "train" / "latest.txt").write_text("run-1\n")
    (root / "collect_frames" / "run-2").mkdir(parents=True)
    (root / "collect_frames" / "run-2" / "chunk_0.npz").write_bytes(b"frames")


def test_sync_pushes_every_mode_dir(tmp_path):
    runs = tmp_path / "runs"
    _make_runs(runs)
    bucket = f"file://{tmp_path}/bucket/prefix"

    failed = sync_runs.sync(runs, bucket)
    assert failed == []
    assert (tmp_path / "bucket/prefix/train/run-1/model.pt").read_bytes() == b"weights"
    assert (tmp_path / "bucket/prefix/train/latest.txt").read_text().strip() == "run-1"
    assert (tmp_path / "bucket/prefix/collect_frames/run-2/chunk_0.npz").exists()


def test_sync_missing_root_is_noop(tmp_path):
    assert sync_runs.sync(tmp_path / "nope", f"file://{tmp_path}/bucket") == []


def test_main_exit_code_nonzero_on_failure(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _make_runs(runs)

    def _boom(local, uri):
        raise ConnectionError("network down")

    monkeypatch.setattr(gcs, "upload_dir", _boom)
    rc = sync_runs.main([str(runs), f"file://{tmp_path}/bucket"])
    assert rc == 1


def test_main_exit_code_zero_on_success(tmp_path):
    runs = tmp_path / "runs"
    _make_runs(runs)
    assert sync_runs.main([str(runs), f"file://{tmp_path}/bucket"]) == 0
