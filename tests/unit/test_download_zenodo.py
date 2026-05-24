"""Unit tests for the Zenodo dataset acquisition layer.

All HTTP is mocked; no test in this module hits the real Zenodo host.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from neurotrack.data.download_zenodo import (
    ZenodoDownloadError,
    _is_retriable,
    check_disk_space,
    compute_md5,
    compute_sha256,
    download_zenodo_file,
    verify_md5,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
class TestHashing:
    def test_compute_md5_matches_hashlib(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        payload = b"hello zenodo\n" * 50
        f.write_bytes(payload)
        assert compute_md5(f) == hashlib.md5(payload).hexdigest()

    def test_compute_sha256_matches_hashlib(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        payload = b"hello zenodo\n" * 50
        f.write_bytes(payload)
        assert compute_sha256(f) == hashlib.sha256(payload).hexdigest()

    def test_verify_md5_match(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        assert verify_md5(f, hashlib.md5(b"abc").hexdigest())
        # Case insensitive on the hex digest.
        assert verify_md5(f, hashlib.md5(b"abc").hexdigest().upper())

    def test_verify_md5_mismatch_raises_when_caller_asserts(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        # Direct API: returns False (does not raise).  The orchestrator turns
        # this into ChecksumMismatchError; we assert the boolean contract here.
        assert not verify_md5(f, "0" * 32)

    def test_verify_md5_empty_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        assert not verify_md5(f, "")


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------
class TestCheckDiskSpace:
    def test_passes_for_small_request(self, tmp_path: Path) -> None:
        assert check_disk_space(tmp_path, 1024) is True

    def test_refuses_implausible_request(self, tmp_path: Path) -> None:
        # 1 EiB -- larger than any disk.
        assert check_disk_space(tmp_path, 1 << 60) is False

    def test_walks_to_existing_ancestor(self, tmp_path: Path) -> None:
        # Path that doesn't exist yet -- function should walk up.
        assert check_disk_space(tmp_path / "nope" / "nope2", 1024) is True

    def test_refuses_when_free_below_required(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from neurotrack.data import download_zenodo as mod

        class _Fake:
            free = 100  # 100 bytes

        monkeypatch.setattr(mod.shutil, "disk_usage", lambda *a, **kw: _Fake())
        assert check_disk_space(tmp_path, 1024) is False


# ---------------------------------------------------------------------------
# Retry classifier
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
# Resumable download (mocked HTTP)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(
        self,
        body: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
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


class TestDownloadZenodoFile:
    def test_downloads_full_then_renames_atomically(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = b"x" * 4096

        def fake_get(url, stream, headers, timeout):  # type: ignore[no-untyped-def]
            return _FakeResp(payload)

        monkeypatch.setattr(requests, "get", fake_get)
        out = download_zenodo_file(
            "data.bin",
            tmp_path,
            "https://zenodo.invalid/records/0/files/",
            progress=False,
        )
        assert out.read_bytes() == payload
        assert not (tmp_path / "data.bin.partial").exists()

    def test_skip_existing_without_force(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        existing = tmp_path / "data.bin"
        existing.write_bytes(b"already here")

        def boom(*a, **kw):  # type: ignore[no-untyped-def]
            raise AssertionError("must not call HTTP when file present")

        monkeypatch.setattr(requests, "get", boom)
        out = download_zenodo_file(
            "data.bin",
            tmp_path,
            "https://zenodo.invalid/records/0/files/",
            progress=False,
        )
        assert out.read_bytes() == b"already here"

    def test_resume_appends_via_range(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two-step resume: first 1 KiB, then a Range request returning the
        remaining bytes; final file must equal the full payload.
        """
        full = bytes(range(256)) * 16  # 4096 bytes
        first_half = full[:1024]
        second_half = full[1024:]

        partial = tmp_path / "big.bin.partial"
        # Pre-seed a partially written file as if a previous attempt died.
        partial.write_bytes(first_half)

        captured_headers: dict[str, str] = {}

        def fake_get(url, stream, headers, timeout):  # type: ignore[no-untyped-def]
            captured_headers.update(headers or {})
            return _FakeResp(
                second_half,
                status=206,
                headers={
                    "Content-Length": str(len(second_half)),
                    "Content-Range": f"bytes 1024-{len(full) - 1}/{len(full)}",
                },
            )

        monkeypatch.setattr(requests, "get", fake_get)
        out = download_zenodo_file(
            "big.bin",
            tmp_path,
            "https://zenodo.invalid/records/0/files/",
            progress=False,
        )
        assert captured_headers.get("Range") == "bytes=1024-"
        assert out.read_bytes() == full
        assert not partial.exists()


# ---------------------------------------------------------------------------
# Manifest round-trip (script-level helpers)
# ---------------------------------------------------------------------------
def _load_script() -> Any:
    """Import scripts/download_zenodo.py as a module."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "download_zenodo.py"
    spec = importlib.util.spec_from_file_location("download_zenodo_script", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_zenodo_script"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestManifestRoundTrip:
    def test_load_and_writeback(self, tmp_path: Path) -> None:
        mod = _load_script()
        path = tmp_path / "m.yaml"
        original = {
            "base_url": "https://zenodo.invalid/records/0/files/",
            "files": {
                "a.tar.gz": {"md5": "0" * 32, "size_mb": 1.0, "required": True},
            },
            "sha256": {},
        }
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(original, fh)
        loaded = mod.load_manifest(path)
        assert loaded["base_url"] == original["base_url"]
        assert "a.tar.gz" in loaded["files"]
        loaded["sha256"]["a.tar.gz"] = "f" * 64
        mod.save_manifest(path, loaded)
        again = mod.load_manifest(path)
        assert again["sha256"]["a.tar.gz"] == "f" * 64

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        mod = _load_script()
        with pytest.raises(ZenodoDownloadError):
            mod.load_manifest(tmp_path / "missing.yaml")

    def test_resolve_local_path_handles_renamed_readme(self, tmp_path: Path) -> None:
        mod = _load_script()
        # When trackml_README.html exists, that's the canonical local path.
        (tmp_path / "trackml_README.html").write_text("hi")
        p = mod.resolve_local_path(tmp_path, "README.html")
        assert p.name == "trackml_README.html"

    def test_resolve_local_path_uses_zenodo_name_pre_rename(
        self,
        tmp_path: Path,
    ) -> None:
        mod = _load_script()
        # No rename yet -> resolves to README.html under out_dir.
        p = mod.resolve_local_path(tmp_path, "README.html")
        assert p.name == "README.html"
