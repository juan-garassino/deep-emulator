"""gs:// branch of utils/gcs.py against a stubbed GCS client (no network).

Stub-class pattern (no MagicMock): StubBlob/StubBucket/StubClient share one
dict-of-bytes "object store"; monkeypatch gcs._gcs_client to return it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deepEmulator.utils import gcs


class _StubBlob:
    def __init__(self, store: dict, bucket: str, name: str):
        self.store = store
        self.bucket = bucket
        self.name = name
        self.fail_remaining = 0  # for retry tests

    def _key(self) -> tuple[str, str]:
        return (self.bucket, self.name)

    def _maybe_fail(self):
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            raise ConnectionError("transient")

    def upload_from_filename(self, path: str) -> None:
        self._maybe_fail()
        self.store[self._key()] = Path(path).read_bytes()

    def download_to_filename(self, path: str) -> None:
        self._maybe_fail()
        Path(path).write_bytes(self.store[self._key()])

    def download_as_text(self) -> str:
        return self.store[self._key()].decode()

    def exists(self) -> bool:
        return self._key() in self.store


class _StubBucket:
    def __init__(self, store: dict, name: str):
        self.store = store
        self.name = name
        self.blobs: dict[str, _StubBlob] = {}

    def blob(self, name: str) -> _StubBlob:
        # reuse instances so tests can pre-arm fail_remaining
        if name not in self.blobs:
            self.blobs[name] = _StubBlob(self.store, self.name, name)
        return self.blobs[name]


class _StubClient:
    def __init__(self, store: dict):
        self.store = store
        self.buckets: dict[str, _StubBucket] = {}

    def bucket(self, name: str) -> _StubBucket:
        if name not in self.buckets:
            self.buckets[name] = _StubBucket(self.store, name)
        return self.buckets[name]

    def list_blobs(self, bucket_name: str, prefix: str = ""):
        b = self.bucket(bucket_name)
        return [
            b.blob(name)
            for (bkt, name) in sorted(self.store)
            if bkt == bucket_name and name.startswith(prefix)
        ]


@pytest.fixture()
def stub(monkeypatch):
    store: dict[tuple[str, str], bytes] = {}
    client = _StubClient(store)
    monkeypatch.setattr(gcs, "_gcs_client", lambda: client)
    monkeypatch.setattr(gcs.time, "sleep", lambda _s: None)  # fast retries
    return client


def test_gs_upload_download_roundtrip(stub, tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"payload")
    gcs.upload(src, "gs://bkt/deepemulator/run/f.bin")
    assert gcs.exists("gs://bkt/deepemulator/run/f.bin")

    out = tmp_path / "out.bin"
    gcs.download("gs://bkt/deepemulator/run/f.bin", out)
    assert out.read_bytes() == b"payload"


def test_gs_upload_dir_and_download_dir_with_prefix_trim(stub, tmp_path):
    src = tmp_path / "bundle"
    (src / "trajectories").mkdir(parents=True)
    (src / "model.pt").write_bytes(b"weights")
    (src / "trajectories" / "ep0.csv.gz").write_bytes(b"rows")

    gcs.upload_dir(src, "gs://bkt/deepemulator/train/run-1")
    dest = tmp_path / "pulled"
    gcs.download_dir("gs://bkt/deepemulator/train/run-1", dest)
    assert (dest / "model.pt").read_bytes() == b"weights"
    assert (dest / "trajectories" / "ep0.csv.gz").read_bytes() == b"rows"


def test_gs_download_dir_merges_instead_of_wiping(stub, tmp_path):
    src = tmp_path / "bundle"
    src.mkdir()
    (src / "a.txt").write_bytes(b"a")
    gcs.upload_dir(src, "gs://bkt/p/run")

    dest = tmp_path / "pulled"
    dest.mkdir()
    (dest / "keep.txt").write_bytes(b"local")
    gcs.download_dir("gs://bkt/p/run", dest)
    assert (dest / "keep.txt").exists()  # additive merge — parity with gs://
    assert (dest / "a.txt").exists()


def test_gs_latest_run_name(stub, tmp_path):
    marker = tmp_path / "latest.txt"
    marker.write_text("run-042\n")
    gcs.upload(marker, "gs://bkt/deepemulator/train/latest.txt")
    assert gcs.latest_run_name("gs://bkt/deepemulator/train") == "run-042"
    assert gcs.latest_run_name("gs://bkt/deepemulator/never") is None


def test_gs_retry_recovers_from_transient_failures(stub, tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"x")
    blob = stub.bucket("bkt").blob("k/f.bin")
    blob.fail_remaining = 2  # fails twice, succeeds on attempt 3
    gcs.upload(src, "gs://bkt/k/f.bin")
    assert gcs.exists("gs://bkt/k/f.bin")


def test_gs_retry_gives_up_after_attempts(stub, tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"x")
    blob = stub.bucket("bkt").blob("k/g.bin")
    blob.fail_remaining = 99
    with pytest.raises(ConnectionError):
        gcs.upload(src, "gs://bkt/k/g.bin")


def test_file_upload_dir_merges_instead_of_replacing(tmp_path):
    """file:// must match gs:// additive-merge semantics — the old
    rmtree+copytree hid stale-object bugs from local smokes."""
    src1 = tmp_path / "v1"
    src1.mkdir()
    (src1 / "old.txt").write_bytes(b"old")
    uri = f"file://{tmp_path}/bucket/run"
    gcs.upload_dir(src1, uri)

    src2 = tmp_path / "v2"
    src2.mkdir()
    (src2 / "new.txt").write_bytes(b"new")
    gcs.upload_dir(src2, uri)

    assert (tmp_path / "bucket/run/old.txt").exists()  # merge, not replace
    assert (tmp_path / "bucket/run/new.txt").exists()
