"""Unit tests for the dataset acquisition layer.

These tests are deliberately offline -- network calls to Kaggle and CERN
are exercised under integration tests (``@pytest.mark.requires_data``)
not included in the default suite.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
import requests

from neurotrack.data.download_legacy import (
    ChecksumMismatchError,
    DiskSpaceError,
    DownloadError,
    ZipSlipError,
    _is_retriable,
    check_disk_space,
    compute_sha256,
    copy_local,
    download_url,
    safe_extract_zip,
    verify_checksum,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
class TestSha256:
    def test_compute_matches_hashlib(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        payload = b"hello world\n" * 100
        f.write_bytes(payload)
        assert compute_sha256(f) == hashlib.sha256(payload).hexdigest()

    def test_verify_checksum_accepts_match(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        assert verify_checksum(f, hashlib.sha256(b"abc").hexdigest())

    def test_verify_checksum_rejects_mismatch(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        assert not verify_checksum(f, "0" * 64)

    def test_verify_checksum_empty_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        assert not verify_checksum(f, "")


# ---------------------------------------------------------------------------
# Disk-space guard
# ---------------------------------------------------------------------------
class TestCheckDiskSpace:
    def test_passes_for_small_request(self, tmp_path: Path) -> None:
        # 1 KiB needed -- always available.
        check_disk_space(tmp_path, 1024)

    def test_raises_for_implausible_request(self, tmp_path: Path) -> None:
        with pytest.raises(DiskSpaceError):
            check_disk_space(tmp_path, 1 << 60)  # 1 EiB

    def test_walks_up_to_existing_ancestor(self, tmp_path: Path) -> None:
        # Path that doesn't exist yet -- function should walk up to tmp_path.
        check_disk_space(tmp_path / "nope" / "still_nope", 1024)


# ---------------------------------------------------------------------------
# Zip-slip protection
# ---------------------------------------------------------------------------
class TestSafeExtractZip:
    def test_extracts_clean_archive(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "ok.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.txt", b"hi")
            zf.writestr("sub/b.txt", b"there")
        out = tmp_path / "out"
        members = safe_extract_zip(zip_path, out)
        assert {p.name for p in members} == {"a.txt", "b.txt"}
        assert (out / "a.txt").read_bytes() == b"hi"
        assert (out / "sub" / "b.txt").read_bytes() == b"there"

    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../escape.txt", b"pwn")
        out = tmp_path / "out"
        with pytest.raises(ZipSlipError):
            safe_extract_zip(zip_path, out)
        # No partial extraction outside the root.
        assert not (tmp_path / "escape.txt").exists()

    def test_rejects_absolute_path(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "evil_abs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/etc/passwd", b"pwn")
        out = tmp_path / "out"
        with pytest.raises(ZipSlipError):
            safe_extract_zip(zip_path, out)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
class TestRetryPolicy:
    def test_retries_connection_error(self) -> None:
        assert _is_retriable(requests.ConnectionError("boom"))

    def test_retries_timeout(self) -> None:
        assert _is_retriable(requests.Timeout("slow"))

    def test_retries_5xx(self) -> None:
        resp = requests.Response()
        resp.status_code = 503
        err = requests.HTTPError(response=resp)
        assert _is_retriable(err)

    def test_does_not_retry_4xx(self) -> None:
        resp = requests.Response()
        resp.status_code = 404
        err = requests.HTTPError(response=resp)
        assert not _is_retriable(err)

    def test_does_not_retry_unrelated(self) -> None:
        assert not _is_retriable(ValueError("nope"))


# ---------------------------------------------------------------------------
# Local copy (used by --source local)
# ---------------------------------------------------------------------------
class TestCopyLocal:
    def test_copies_single_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        dst = tmp_path / "out"
        result = copy_local(src, dst)
        assert result == [dst / "src.bin"]
        assert (dst / "src.bin").read_bytes() == b"payload"

    def test_copies_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.csv").write_bytes(b"a")
        (src / "b.csv").write_bytes(b"b")
        dst = tmp_path / "out"
        result = copy_local(src, dst)
        assert {p.name for p in result} == {"a.csv", "b.csv"}

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        src = tmp_path / "x.bin"
        src.write_bytes(b"original")
        dst = tmp_path / "out"
        copy_local(src, dst)

        # Mutate the source; without --force the destination should not change.
        src.write_bytes(b"updated")
        copy_local(src, dst)
        assert (dst / "x.bin").read_bytes() == b"original"

        # With force=True it does update.
        copy_local(src, dst, force=True)
        assert (dst / "x.bin").read_bytes() == b"updated"

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DownloadError):
            copy_local(tmp_path / "missing", tmp_path / "out")


# ---------------------------------------------------------------------------
# download_url -- HTTP path mocked via monkeypatch
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
        self._body = body
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(body))}

    def iter_content(self, chunk_size: int):  # type: ignore[no-untyped-def]
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)  # type: ignore[arg-type]

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc) -> None:  # type: ignore[no-untyped-def]
        return None


class TestDownloadUrl:
    def test_downloads_and_atomically_renames(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = b"x" * 4096

        def fake_get(url, stream, headers, timeout):  # type: ignore[no-untyped-def]
            return _FakeResponse(payload)

        monkeypatch.setattr(requests, "get", fake_get)
        out = download_url(
            "https://example.com/data.bin",
            tmp_path,
            progress=False,
        )
        assert out.read_bytes() == payload
        # No leftover .part file.
        assert not (tmp_path / "data.bin.part").exists()

    def test_skip_existing_without_force(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        existing = tmp_path / "data.bin"
        existing.write_bytes(b"already here")

        # If anyone calls requests.get, fail loudly -- skip should short-circuit.
        def boom(*a, **kw):  # type: ignore[no-untyped-def]
            raise AssertionError("should not have called HTTP")

        monkeypatch.setattr(requests, "get", boom)
        out = download_url("https://example.com/data.bin", tmp_path, progress=False)
        assert out.read_bytes() == b"already here"


# ---------------------------------------------------------------------------
# Manifest helpers (loaded from the script)
# ---------------------------------------------------------------------------
class TestManifest:
    def test_populate_then_verify(self, tmp_path: Path) -> None:
        # Import lazily; the helpers live in the script module.
        import importlib.util
        import sys

        script = Path(__file__).resolve().parents[2] / "scripts" / "download_data_legacy.py"
        spec = importlib.util.spec_from_file_location("download_script", script)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["download_script"] = mod
        spec.loader.exec_module(mod)

        f = tmp_path / "data.zip"
        f.write_bytes(b"hello")
        manifest: dict = {"files": {}}

        # First call populates and returns False (= "newly populated").
        assert mod.populate_or_verify(manifest, f) is False
        assert manifest["files"]["data.zip"]["sha256"] == compute_sha256(f)
        assert manifest["files"]["data.zip"]["size_bytes"] == 5

        # Second call verifies and returns True.
        assert mod.populate_or_verify(manifest, f) is True

        # Mutate the file -> mismatch raises.
        f.write_bytes(b"corrupted")
        with pytest.raises(ChecksumMismatchError):
            mod.populate_or_verify(manifest, f)
