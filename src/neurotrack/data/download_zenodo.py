"""Zenodo dataset acquisition: pure functions.

The Zenodo record at https://zenodo.org/records/14386134 ships the TrackML +
RedVid tarballs.  This module exposes a small set of side-effect-isolated
helpers consumed by ``scripts/download_zenodo.py``:

* :func:`download_zenodo_file` -- resumable HTTP Range download with atomic
  rename and tenacity retries (5 attempts, exp backoff 2 s -> 60 s, retries
  limited to connection errors / timeouts / 5xx).
* :func:`compute_md5`, :func:`compute_sha256` -- streaming hashes.
* :func:`verify_md5` -- equality check (case-insensitive on hex).
* :func:`check_disk_space` -- pre-flight free-space guard returning a bool.

No CLI lives here -- the CLI is in ``scripts/download_zenodo.py``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import requests
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

log = structlog.get_logger(__name__)

CHUNK_SIZE: Final[int] = 1 << 20  # 1 MiB
DEFAULT_TIMEOUT: Final[float] = 60.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ZenodoDownloadError(RuntimeError):
    """Base class for any failure inside this module."""


class ChecksumMismatchError(ZenodoDownloadError):
    """Raised when an on-disk file's hash disagrees with the manifest."""


class DiskSpaceError(ZenodoDownloadError):
    """Raised / signalled when free disk space is below the required headroom."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def compute_md5(path: Path, *, chunk: int = CHUNK_SIZE) -> str:
    """Stream-hash ``path`` with MD5; return the lower-case hex digest."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def compute_sha256(path: Path, *, chunk: int = CHUNK_SIZE) -> str:
    """Stream-hash ``path`` with SHA-256; return the lower-case hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def verify_md5(path: Path, expected: str) -> bool:
    """Return True iff ``path``'s MD5 equals ``expected`` (case-insensitive).

    An empty ``expected`` returns False -- treat as "unknown, populate first".
    """
    if not expected:
        return False
    return compute_md5(path).lower() == expected.strip().lower()


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------
def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Return True iff ``path`` (or its nearest existing ancestor) has at
    least ``required_bytes`` bytes free.
    """
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    free = shutil.disk_usage(p).free
    return free >= required_bytes


# ---------------------------------------------------------------------------
# HTTP retry policy
# ---------------------------------------------------------------------------
def _is_retriable(exc: BaseException) -> bool:
    """Retry on connection errors, timeouts, and 5xx -- never on 4xx."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        if resp is not None and 500 <= resp.status_code < 600:
            return True
    return False


_retry: Callable[[Callable[..., Any]], Callable[..., Any]] = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retriable),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Resumable download
# ---------------------------------------------------------------------------
@_retry
def _stream_to_partial(
    url: str,
    partial: Path,
    *,
    timeout: float,
    progress: bool,
) -> int:
    """One attempt: stream ``url`` into ``partial`` using HTTP Range to resume.

    Returns the on-disk size at the end of the call.  Idempotent across
    retries: a partially downloaded file is appended to.
    """
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    resp = requests.get(url, stream=True, headers=headers, timeout=timeout)
    try:
        # 416 = "range not satisfiable" -> the file on disk is already complete.
        if resp.status_code == 416:
            log.info("zenodo.range_satisfied", url=url, existing=existing)
            return existing
        # Promote 5xx into HTTPError so tenacity retries.
        if 500 <= resp.status_code < 600:
            resp.raise_for_status()
        # 4xx: terminal; raise without retry (handled by _is_retriable returning False).
        if resp.status_code >= 400:
            resp.raise_for_status()
        if resp.status_code not in (200, 206):
            resp.raise_for_status()

        # Server may ignore Range and reply 200 (full body) -- restart from zero.
        if resp.status_code == 200 and existing > 0:
            log.warning("zenodo.range_ignored_restart", url=url, existing=existing)
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing > 0 else "wb"

        total_remaining = int(resp.headers.get("Content-Length", 0) or 0)
        total = (existing + total_remaining) if total_remaining else None

        bar = tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=partial.stem,
            disable=not progress,
        )
        try:
            with partial.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()
    finally:
        resp.close()

    return partial.stat().st_size


def download_zenodo_file(
    filename: str,
    out_dir: Path,
    base_url: str,
    *,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    progress: bool = True,
) -> Path:
    """Download ``base_url + filename`` into ``out_dir``.

    Resumable via HTTP Range and atomic via ``<filename>.partial`` ->
    ``<filename>`` rename.  Re-running with the destination already present
    is a no-op unless ``force=True``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / filename
    partial = out_dir / f"{filename}.partial"

    if final.exists() and not force:
        log.info("zenodo.skip_existing", path=str(final))
        return final
    if force and partial.exists():
        partial.unlink()
    if force and final.exists():
        final.unlink()

    url = base_url.rstrip("/") + "/" + filename
    log.info("zenodo.download_start", url=url, dest=str(final))
    _stream_to_partial(url, partial, timeout=timeout, progress=progress)
    os.replace(partial, final)
    log.info(
        "zenodo.download_complete",
        url=url,
        dest=str(final),
        size=final.stat().st_size,
    )
    return final


__all__ = [
    "ChecksumMismatchError",
    "DiskSpaceError",
    "ZenodoDownloadError",
    "check_disk_space",
    "compute_md5",
    "compute_sha256",
    "download_zenodo_file",
    "verify_md5",
]
