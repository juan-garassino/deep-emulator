"""Storage backend for RunPod training bundles.

Supports two URI schemes:

    gs://bucket/key       — real Google Cloud Storage (requires google-cloud-storage).
    file:///abs/path      — local filesystem (used by tests + local smoke runs).

The same functions work for both. Callers pass URIs; this module does the
right thing under the hood. The `gs` import is deferred so a `file://`-only
test run does not need google-cloud-storage installed.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse


def parse_uri(uri: str) -> tuple[str, str, str]:
    """Return (scheme, bucket-or-host, key/path)."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("gs", "file"):
        raise ValueError(f"unsupported scheme {parsed.scheme!r} in {uri!r}")
    return parsed.scheme, parsed.netloc, parsed.path.lstrip("/")


def _file_path(bucket: str, key: str) -> Path:
    # file:///tmp/x → bucket='', key='tmp/x' → /tmp/x
    if bucket:
        return Path("/") / bucket / key
    return Path("/") / key


def _gcs_client():
    from google.cloud import storage  # type: ignore

    return storage.Client()


def download(uri: str, dest: Path | str) -> Path:
    """Download a single object to `dest`. Returns the local path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        src = _file_path(bucket, key)
        shutil.copy2(src, dest)
        return dest
    client = _gcs_client()
    blob = client.bucket(bucket).blob(key)
    blob.download_to_filename(str(dest))
    return dest


def upload(local: Path | str, uri: str) -> str:
    """Upload one local file to `uri`. Returns the URI."""
    local = Path(local)
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        dst = _file_path(bucket, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)
        return uri
    client = _gcs_client()
    client.bucket(bucket).blob(key).upload_from_filename(str(local))
    return uri


def upload_dir(local: Path | str, uri: str) -> str:
    """Recursively upload a directory tree under `uri`. Returns the URI."""
    local = Path(local)
    if not local.is_dir():
        raise NotADirectoryError(local)
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        dst = _file_path(bucket, key)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(local, dst)
        return uri
    client = _gcs_client()
    b = client.bucket(bucket)
    for path in local.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local)
            b.blob(f"{key.rstrip('/')}/{rel.as_posix()}").upload_from_filename(str(path))
    return uri


def download_dir(uri: str, dest: Path | str) -> Path:
    """Recursively download a directory tree from `uri` to `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        src = _file_path(bucket, key)
        if not src.is_dir():
            raise NotADirectoryError(src)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest
    client = _gcs_client()
    b = client.bucket(bucket)
    prefix = key.rstrip("/") + "/"
    for blob in client.list_blobs(b.name, prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(out))
    return dest


def exists(uri: str) -> bool:
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        return _file_path(bucket, key).exists()
    client = _gcs_client()
    return client.bucket(bucket).blob(key).exists()


def latest_run_uri(prefix_uri: str) -> str | None:
    """Read `<prefix_uri>/latest.txt` and return its contents trimmed, or None.

    Missing markers must NOT raise — fresh runs are a normal case.
    """
    marker_uri = prefix_uri.rstrip("/") + "/latest.txt"
    if not exists(marker_uri):
        return None
    scheme, bucket, key = parse_uri(marker_uri)
    if scheme == "file":
        return _file_path(bucket, key).read_text().strip() or None
    client = _gcs_client()
    return client.bucket(bucket).blob(key).download_as_text().strip() or None


def sha256_dir(local: Path | str) -> str:
    """Stable digest over a directory tree — used by `eval_signed.py`."""
    local = Path(local)
    h = hashlib.sha256()
    for path in sorted(local.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(local).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()
