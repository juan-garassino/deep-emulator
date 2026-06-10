"""Round-trip tests against a local file:// URI — no GCS network required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepEmulator.utils import gcs


def _make_bundle(root: Path) -> Path:
    run = root / "run-001"
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"\x00\x01\x02fake-weights")
    (run / "metadata.json").write_text(json.dumps({"algo": "ddqn", "steps": 500}))
    (run / "trajectories").mkdir()
    (run / "trajectories" / "episode_0.csv.gz").write_bytes(b"gz-bytes-here")
    return run


def test_parse_uri_file_and_gs():
    assert gcs.parse_uri("file:///tmp/x/y") == ("file", "", "tmp/x/y")
    assert gcs.parse_uri("gs://bucket/path/to/key") == ("gs", "bucket", "path/to/key")


def test_parse_uri_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        gcs.parse_uri("s3://nope/key")


def test_upload_dir_then_download_dir_roundtrip(tmp_path):
    local_src = tmp_path / "src"
    run = _make_bundle(local_src)

    remote_uri = f"file://{tmp_path}/remote/runs/coral/run-001"
    gcs.upload_dir(run, remote_uri)

    pulled = tmp_path / "pulled"
    gcs.download_dir(remote_uri, pulled)

    assert (pulled / "model.pt").read_bytes() == (run / "model.pt").read_bytes()
    assert json.loads((pulled / "metadata.json").read_text())["steps"] == 500
    assert (pulled / "trajectories" / "episode_0.csv.gz").exists()


def test_latest_run_name_returns_none_when_missing(tmp_path):
    prefix = f"file://{tmp_path}/never-created"
    assert gcs.latest_run_name(prefix) is None


def test_latest_run_name_reads_marker(tmp_path):
    prefix_dir = tmp_path / "runs" / "coral"
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "latest.txt").write_text("run-007\n")
    prefix_uri = f"file://{prefix_dir}"
    assert gcs.latest_run_name(prefix_uri) == "run-007"


def test_latest_run_name_degrades_old_absolute_markers(tmp_path):
    """Pre-fix markers stored absolute container paths — degrade to basename."""
    prefix_dir = tmp_path / "runs" / "coral"
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "latest.txt").write_text("/runs/train/20260607-101500\n")
    assert gcs.latest_run_name(f"file://{prefix_dir}") == "20260607-101500"


def test_upload_single_file(tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")
    uri = f"file://{tmp_path}/dest/x/f.bin"
    gcs.upload(src, uri)
    assert (tmp_path / "dest" / "x" / "f.bin").read_bytes() == b"hello"


def test_sha256_dir_is_stable(tmp_path):
    a = _make_bundle(tmp_path / "a")
    b = _make_bundle(tmp_path / "b")
    assert gcs.sha256_dir(a) == gcs.sha256_dir(b)
    (b / "model.pt").write_bytes(b"\x00\x01\x02fake-weights-modified")
    assert gcs.sha256_dir(a) != gcs.sha256_dir(b)
