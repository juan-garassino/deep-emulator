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
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _retry(fn: Callable[[], _T], *, attempts: int = 3, backoff: float = 2.0) -> _T:
    """Retry a flaky network call with exponential backoff. Re-raises the
    final failure — losing a run's artifacts to one transient error is worse
    than a few seconds of waiting."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception:
            if attempt == attempts:
                raise
            wait = backoff ** (attempt - 1)
            logger.warning("gcs op failed (attempt %d/%d), retrying in %.1fs",
                           attempt, attempts, wait, exc_info=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


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
    _retry(lambda: blob.download_to_filename(str(dest)))
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
    blob = client.bucket(bucket).blob(key)
    _retry(lambda: blob.upload_from_filename(str(local)))
    return uri


def upload_dir(local: Path | str, uri: str) -> str:
    """Recursively upload a directory tree under `uri` (additive merge —
    matches the gs:// per-object semantics so local smokes exercise the
    same behavior pods see). Returns the URI."""
    local = Path(local)
    if not local.is_dir():
        raise NotADirectoryError(local)
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        dst = _file_path(bucket, key)
        for path in local.rglob("*"):
            if path.is_file():
                out = dst / path.relative_to(local)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
        return uri
    client = _gcs_client()
    b = client.bucket(bucket)
    for path in local.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local)
            blob = b.blob(f"{key.rstrip('/')}/{rel.as_posix()}")
            _retry(lambda _b=blob, _p=path: _b.upload_from_filename(str(_p)))
    return uri


def download_dir(uri: str, dest: Path | str) -> Path:
    """Recursively download a directory tree from `uri` to `dest` (additive
    merge — never wipes existing local files)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        src = _file_path(bucket, key)
        if not src.is_dir():
            raise NotADirectoryError(src)
        for path in src.rglob("*"):
            if path.is_file():
                out = dest / path.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
        return dest
    client = _gcs_client()
    b = client.bucket(bucket)
    prefix = key.rstrip("/") + "/"
    for blob in _retry(lambda: list(client.list_blobs(b.name, prefix=prefix))):
        rel = blob.name[len(prefix):]
        if not rel:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        _retry(lambda _b=blob, _o=out: _b.download_to_filename(str(_o)))
    return dest


def exists(uri: str) -> bool:
    scheme, bucket, key = parse_uri(uri)
    if scheme == "file":
        return _file_path(bucket, key).exists()
    client = _gcs_client()
    return client.bucket(bucket).blob(key).exists()


def latest_run_name(prefix_uri: str) -> str | None:
    """Read `<prefix_uri>/latest.txt` and return the run NAME it contains, or None.

    The marker stores a bare run-dir name (e.g. `20260611-031500`) — callers
    compose `<prefix_uri>/<name>` themselves. Markers written before the
    relative-name fix contained an absolute container path; we degrade those
    to their basename so old prefixes stay resumable.

    Missing markers must NOT raise — fresh runs are a normal case.
    """
    marker_uri = prefix_uri.rstrip("/") + "/latest.txt"
    if not exists(marker_uri):
        return None
    scheme, bucket, key = parse_uri(marker_uri)
    if scheme == "file":
        content = _file_path(bucket, key).read_text().strip()
    else:
        client = _gcs_client()
        content = _retry(lambda: client.bucket(bucket).blob(key).download_as_text()).strip()
    if not content:
        return None
    return content.rsplit("/", 1)[-1]  # absolute-path markers degrade to basename


def sha256_dir(local: Path | str) -> str:
    """Stable digest over a directory tree — used by `eval_signed.py`."""
    local = Path(local)
    h = hashlib.sha256()
    for path in sorted(local.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(local).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()
