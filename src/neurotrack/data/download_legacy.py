# LEGACY. Kaggle+CERN path. kept for ref. new code in download_zenodo.py.
"""Dataset acquisition: pure functions for fetching, verifying, and unpacking
the TrackML dataset from Kaggle, CERN Open Data, or a manual URL / local copy.

The public surface is intentionally small and side-effect-isolated so the
orchestrator script (``scripts/download_data.py``) can compose it freely:

* :func:`download_kaggle` -- Kaggle competition file pull.
* :func:`download_cern`   -- CERN Open Data record pull.
* :func:`download_url`    -- single-file resumable HTTP download.
* :func:`copy_local`      -- copy from an existing local archive.
* :func:`verify_checksum` -- SHA-256 integrity check.
* :func:`compute_sha256`  -- streaming hash of a file.
* :func:`safe_extract_zip`-- zip-slip-safe archive extraction.
* :func:`check_disk_space`-- pre-flight free-space guard.

All HTTP activity goes through ``_http_get_stream`` which is wrapped in a
tenacity retry policy: 5 attempts, exponential backoff 2 s -> 60 s, retries
limited to connection errors, timeouts, and 5xx responses (4xx is treated
as terminal).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import requests
import structlog
from requests import Response
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
DEFAULT_DISK_HEADROOM: Final[float] = 2.0  # require 2x archive size free

CERN_RECORD_API: Final[str] = "https://opendata.cern.ch/api/records/{record_id}"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class DownloadError(RuntimeError):
    """Base class for any failure inside this module."""


class ChecksumMismatchError(DownloadError):
    """Raised when an on-disk file's SHA-256 disagrees with the manifest."""


class DiskSpaceError(DownloadError):
    """Raised when free disk space is below the required headroom."""


class ZipSlipError(DownloadError):
    """Raised when an archive entry would land outside the extraction root."""


class KaggleAuthError(DownloadError):
    """Raised when the Kaggle CLI cannot authenticate."""


# ---------------------------------------------------------------------------
# Disk space + checksums
# ---------------------------------------------------------------------------
def check_disk_space(
    path: Path,
    required_bytes: int,
    *,
    headroom: float = DEFAULT_DISK_HEADROOM,
) -> None:
    """Raise :class:`DiskSpaceError` if ``path`` has less than headroom-x free.

    ``path`` need not exist; the nearest existing ancestor is queried.
    """
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    free = shutil.disk_usage(p).free
    needed = int(required_bytes * headroom)
    if free < needed:
        raise DiskSpaceError(
            f"Need {needed:,} bytes free at {p} (got {free:,}); "
            f"{required_bytes:,} bytes required + {headroom}x headroom.",
        )


def compute_sha256(path: Path, *, chunk: int = CHUNK_SIZE) -> str:
    """Stream-hash ``path`` with SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    """Return True iff ``path``'s SHA-256 equals ``expected_sha256``.

    Comparison is case-insensitive on the hex digest.  An empty
    ``expected_sha256`` returns False (treat as "unknown -> populate first").
    """
    if not expected_sha256:
        return False
    return compute_sha256(path).lower() == expected_sha256.strip().lower()


# ---------------------------------------------------------------------------
# Zip extraction (zip-slip safe)
# ---------------------------------------------------------------------------
def safe_extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract ``zip_path`` into ``out_dir`` with zip-slip protection.

    Each member's resolved destination must lie inside ``out_dir`` -- entries
    with absolute paths or ``..`` traversal raise :class:`ZipSlipError` *before*
    any extraction happens.

    Returns the list of extracted file paths (excluding directories).
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        # Pre-flight every member; nothing is written until all are validated.
        for info in zf.infolist():
            name = info.filename
            # Reject absolute or drive-letter paths up front.
            if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
                raise ZipSlipError(f"absolute path in archive: {name!r}")
            target = (out_dir / name).resolve()
            try:
                target.relative_to(out_dir)
            except ValueError as e:
                raise ZipSlipError(
                    f"archive entry escapes extraction root: {name!r} -> {target}",
                ) from e

        # Now actually extract.
        for info in zf.infolist():
            zf.extract(info, out_dir)
            target = out_dir / info.filename
            if not info.is_dir():
                extracted.append(target)

    return extracted


# ---------------------------------------------------------------------------
# HTTP retry policy + streaming download
# ---------------------------------------------------------------------------
def _is_retriable(exc: BaseException) -> bool:
    """Retry on connection errors, timeouts, and 5xx HTTPErrors only."""
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


@contextmanager
def _http_get_stream(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[Response]:
    """Streaming GET that turns 5xx into HTTPError so tenacity can retry."""
    resp = requests.get(url, stream=True, headers=headers or {}, timeout=timeout)
    try:
        if 500 <= resp.status_code < 600:
            resp.raise_for_status()
        yield resp
    finally:
        resp.close()


@_retry
def _download_to_part(
    url: str,
    part_path: Path,
    *,
    timeout: float,
    progress: bool,
) -> int:
    """Download ``url`` into ``part_path`` with HTTP Range resume.

    Returns the final on-disk size in bytes.  Idempotent: if ``part_path``
    already has bytes and the server honours ``Range``, the download appends.
    """
    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with _http_get_stream(url, headers=headers, timeout=timeout) as resp:
        # 416 = range not satisfiable -> the file is already complete on disk.
        if resp.status_code == 416:
            log.info("download.range_satisfied", url=url, size=existing)
            return existing
        if resp.status_code not in (200, 206):
            resp.raise_for_status()

        # If the server didn't honour the range, restart from zero.
        if resp.status_code == 200 and existing > 0:
            log.warning(
                "download.range_ignored_restart",
                url=url,
                existing=existing,
            )
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing > 0 else "wb"

        total_remaining = int(resp.headers.get("Content-Length", 0))
        total = existing + total_remaining if total_remaining else None

        bar = tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=part_path.name,
            disable=not progress,
        )
        try:
            with part_path.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()

    return part_path.stat().st_size


# ---------------------------------------------------------------------------
# Public download functions
# ---------------------------------------------------------------------------
def download_url(
    url: str,
    out_dir: Path,
    *,
    dest_name: str | None = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    progress: bool = True,
    expected_size: int | None = None,
) -> Path:
    """Download ``url`` to ``out_dir`` with resumable HTTP Range + atomic rename.

    On success the file is moved from ``<dest>.part`` to ``<dest>`` via
    :func:`os.replace`, which is atomic on the same filesystem on both POSIX
    and Windows.  Re-running with the same ``url`` and ``out_dir`` returns
    immediately if the destination already exists (unless ``force=True``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = dest_name or url.rsplit("/", 1)[-1].split("?", 1)[0] or "download.bin"
    final = out_dir / name
    part = out_dir / f"{name}.part"

    if final.exists() and not force:
        log.info("download.skip_existing", path=str(final))
        return final
    if force and part.exists():
        part.unlink()

    if expected_size is not None:
        check_disk_space(out_dir, expected_size)

    log.info("download.start", url=url, dest=str(final))
    _download_to_part(url, part, timeout=timeout, progress=progress)
    os.replace(part, final)
    log.info("download.complete", url=url, dest=str(final), size=final.stat().st_size)
    return final


def download_kaggle(
    competition: str,
    files: list[str],
    out_dir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Pull each file in ``files`` from a Kaggle competition.

    Authentication is via ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``
    / ``KAGGLE_KEY`` environment variables.  Failure to authenticate raises
    :class:`KaggleAuthError`; any per-file failure is wrapped in
    :class:`DownloadError` so the caller can fall back to another source.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:  # pragma: no cover -- exercised only if kaggle missing
        raise KaggleAuthError(f"kaggle package not installed: {e}") from e

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # kaggle wraps everything; surface a clean error
        raise KaggleAuthError(f"kaggle authenticate() failed: {e}") from e

    paths: list[Path] = []
    for f in files:
        target = out_dir / f
        zip_alias = out_dir / f"{f}.zip"
        if target.exists() and not force:
            log.info("kaggle.skip_existing", file=f)
            paths.append(target)
            continue
        log.info("kaggle.fetch", competition=competition, file=f)
        try:
            api.competition_download_file(
                competition=competition,
                file_name=f,
                path=str(out_dir),
                force=force,
                quiet=False,
            )
        except Exception as e:
            raise DownloadError(f"kaggle pull failed for {f}: {e}") from e

        if target.exists():
            paths.append(target)
        elif zip_alias.exists():  # Kaggle sometimes appends .zip.
            paths.append(zip_alias)
        else:
            raise DownloadError(
                f"kaggle reported success but {target} (or {zip_alias}) is missing",
            )
    return paths


def download_cern(
    record_id: int,
    out_dir: Path,
    *,
    file_filter: list[str] | None = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[Path]:
    """Pull files from a CERN Open Data record (TrackML uses record 14).

    If ``file_filter`` is given, only files whose ``key`` matches one of the
    listed names are downloaded; otherwise every file in the record is pulled.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = CERN_RECORD_API.format(record_id=record_id)
    log.info("cern.fetch_metadata", record_id=record_id, url=api_url)
    meta_resp = requests.get(api_url, timeout=timeout)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    # The shape is sometimes {files: [...]} and sometimes nested under metadata.
    raw_files = meta.get("files") or meta.get("metadata", {}).get("files") or []
    if not raw_files:
        raise DownloadError(f"CERN record {record_id} returned no files; check the API.")

    paths: list[Path] = []
    for f in raw_files:
        key = f.get("key") or f.get("filename")
        if not key:
            continue
        if file_filter is not None and key not in file_filter:
            continue
        url = (
            f.get("uri")
            or f.get("links", {}).get("self")
            or f.get("links", {}).get("download")
        )
        if not url:
            log.warning("cern.skip_file_no_url", key=key)
            continue
        size = f.get("size")
        path = download_url(
            url,
            out_dir,
            dest_name=key,
            force=force,
            timeout=timeout,
            expected_size=size,
        )
        paths.append(path)
    return paths


def copy_local(src: Path, out_dir: Path, *, force: bool = False) -> list[Path]:
    """Copy a local file or directory into ``out_dir`` (for ``--from-local``).

    A single file is copied as-is; a directory is mirrored shallowly (one
    level of files copied, no recursion into subdirectories).  Returns the
    list of resulting paths under ``out_dir``.
    """
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise DownloadError(f"--from-local path does not exist: {src}")

    out: list[Path] = []
    if src.is_file():
        target = out_dir / src.name
        if target.exists() and not force:
            return [target]
        shutil.copy2(src, target)
        return [target]

    for item in src.iterdir():
        if not item.is_file():
            continue
        target = out_dir / item.name
        if target.exists() and not force:
            out.append(target)
            continue
        shutil.copy2(item, target)
        out.append(target)
    return out


__all__ = [
    "CERN_RECORD_API",
    "ChecksumMismatchError",
    "DiskSpaceError",
    "DownloadError",
    "KaggleAuthError",
    "ZipSlipError",
    "check_disk_space",
    "compute_sha256",
    "copy_local",
    "download_cern",
    "download_kaggle",
    "download_url",
    "safe_extract_zip",
    "verify_checksum",
]
