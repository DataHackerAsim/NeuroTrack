"""Zenodo dataset acquisition CLI for the TrackML + RedVid tarballs.

Reads :file:`configs/data_checksums_zenodo.yaml` for the file list, base URL,
and authoritative MD5s; downloads each missing file resumably from Zenodo;
verifies MD5 against the manifest; populates SHA-256 on first run.

Usage::

    python scripts/download_zenodo.py
    python scripts/download_zenodo.py --verify-only
    python scripts/download_zenodo.py --only trackml_40k-events-10-to-50-tracks.tar.gz
    python scripts/download_zenodo.py --force
    python scripts/download_zenodo.py --no-rename-readme
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from neurotrack.data.download_zenodo import (
    ZenodoDownloadError,
    check_disk_space,
    compute_md5,
    compute_sha256,
    download_zenodo_file,
    verify_md5,
)

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_MANIFEST = Path("configs/data_checksums_zenodo.yaml")
README_ZENODO_NAME = "README.html"
README_LOCAL_NAME = "trackml_README.html"


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, Any]:
    import yaml  # ships with hydra-core / pyyaml

    if not path.exists():
        raise ZenodoDownloadError(f"manifest missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "base_url" not in data or "files" not in data:
        raise ZenodoDownloadError(
            f"manifest {path} missing 'base_url' or 'files' section",
        )
    data.setdefault("sha256", {})
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(json: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    procs: list[Any] = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    procs.append(
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(),
    )
    structlog.configure(
        processors=procs,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Path resolution: handle the README post-rename to trackml_README.html
# ---------------------------------------------------------------------------
def resolve_local_path(out_dir: Path, manifest_name: str) -> Path:
    """Return where the file should live locally.

    The Zenodo record names the readme ``README.html``; we rename the local
    copy to ``trackml_README.html`` (avoids clashing with the project README).
    Any other filename is used verbatim.
    """
    if manifest_name == README_ZENODO_NAME:
        primary = out_dir / README_LOCAL_NAME
        if primary.exists():
            return primary
        # If --rename-readme has not yet run, the original Zenodo name is on disk.
        return out_dir / manifest_name
    return out_dir / manifest_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="FILENAME",
        help="restrict download/verify to a subset of manifest filenames",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="re-hash existing files against the manifest; do not download",
    )
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument(
        "--rename-readme",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after download, rename README.html -> trackml_README.html",
    )
    p.add_argument("--json-logs", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def selected_files(manifest: dict[str, Any], only: list[str] | None) -> dict[str, Any]:
    files: dict[str, Any] = manifest["files"]
    if only is None:
        return files
    sel = {k: files[k] for k in only if k in files}
    missing = [k for k in only if k not in files]
    if missing:
        raise ZenodoDownloadError(f"--only includes unknown files: {missing}")
    return sel


def preflight_disk(
    out_dir: Path,
    files: dict[str, Any],
    log: structlog.stdlib.BoundLogger,
) -> bool:
    """Refuse to start if free space < 2x sum of expected sizes."""
    expected_mb = sum(float(meta.get("size_mb") or 0) for meta in files.values())
    expected_bytes = int(expected_mb * (1 << 20))
    required = expected_bytes * 2
    ok = check_disk_space(out_dir, required)
    log.info(
        "zenodo.preflight_disk",
        out_dir=str(out_dir),
        expected_bytes=expected_bytes,
        required_bytes=required,
        ok=ok,
    )
    return ok


def report_table(
    rows: list[tuple[str, int, str, str]],
    log: structlog.stdlib.BoundLogger,
) -> None:
    log.info("zenodo.summary", n=len(rows))
    print()
    print(f"{'file':<60s}  {'bytes':>14s}  {'status':<8s}  md5")
    print("-" * 110)
    for name, size, status, md5 in rows:
        print(f"{name:<60s}  {size:>14,}  {status:<8s}  {md5}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.json_logs)
    log = structlog.get_logger("download_zenodo")

    try:
        manifest = load_manifest(args.manifest)
    except ZenodoDownloadError as e:
        log.error("zenodo.manifest_error", error=str(e))
        return 1

    base_url = manifest["base_url"]
    try:
        files = selected_files(manifest, args.only)
    except ZenodoDownloadError as e:
        log.error("zenodo.bad_only", error=str(e))
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight (skipped on --verify-only since no writes happen).
    if not args.verify_only and not preflight_disk(args.output_dir, files, log):
        log.error("zenodo.preflight_failed", reason="insufficient disk space")
        return 1

    # Track the table rows for the final report.
    rows: list[tuple[str, int, str, str]] = []
    sha_section: dict[str, str] = manifest.setdefault("sha256", {}) or {}
    any_failure = False

    for fname, meta in files.items():
        local = resolve_local_path(args.output_dir, fname)
        expected_md5 = (meta.get("md5") or "").strip()

        # ----- VERIFY-ONLY path
        if args.verify_only:
            if not local.exists():
                log.error("zenodo.verify_missing", file=fname, looked_at=str(local))
                rows.append((local.name, 0, "MISSING", "-"))
                any_failure = True
                continue
            got = compute_md5(local)
            ok = got.lower() == expected_md5.lower()
            log.info(
                "zenodo.verify",
                file=fname,
                local=str(local),
                ok=ok,
                expected=expected_md5,
                got=got,
            )
            rows.append(
                (
                    local.name,
                    local.stat().st_size,
                    "OK" if ok else "BAD",
                    got,
                ),
            )
            if not ok:
                any_failure = True
            else:
                sha_section.setdefault(fname, compute_sha256(local))
            continue

        # ----- DOWNLOAD path
        if local.exists() and not args.force:
            got = compute_md5(local)
            if not verify_md5(local, expected_md5):
                log.error(
                    "zenodo.existing_md5_mismatch",
                    file=fname,
                    local=str(local),
                    expected=expected_md5,
                    got=got,
                )
                rows.append(
                    (local.name, local.stat().st_size, "BAD", got),
                )
                any_failure = True
                continue
            log.info("zenodo.skip_verified", file=fname, md5=got)
            sha_section.setdefault(fname, compute_sha256(local))
            rows.append((local.name, local.stat().st_size, "OK", got))
            continue

        # File not present (or --force) -> download.
        try:
            local = download_zenodo_file(
                fname, args.output_dir, base_url, force=args.force,
            )
        except ZenodoDownloadError as e:
            log.error("zenodo.download_failed", file=fname, error=str(e))
            rows.append((fname, 0, "FAIL", "-"))
            any_failure = True
            continue
        except Exception as e:
            log.error("zenodo.download_failed", file=fname, error=str(e))
            rows.append((fname, 0, "FAIL", "-"))
            any_failure = True
            continue

        got = compute_md5(local)
        if got.lower() != expected_md5.lower():
            # Hard fail; do NOT delete the file (per spec).
            log.error(
                "zenodo.checksum_mismatch",
                file=fname,
                local=str(local),
                expected=expected_md5,
                got=got,
            )
            rows.append((local.name, local.stat().st_size, "BAD", got))
            any_failure = True
            continue
        sha_section[fname] = compute_sha256(local)
        rows.append((local.name, local.stat().st_size, "OK", got))
        log.info("zenodo.download_verified", file=fname, md5=got)

    # README rename: only after a successful download / verify pass.
    if not args.verify_only and args.rename_readme:
        old = args.output_dir / README_ZENODO_NAME
        new = args.output_dir / README_LOCAL_NAME
        if old.exists() and not new.exists():
            old.rename(new)
            log.info("zenodo.readme_renamed", old=str(old), new=str(new))

    # Persist the SHA-256 section back into the manifest.
    manifest["sha256"] = sha_section
    save_manifest(args.manifest, manifest)

    report_table(rows, log)

    if any_failure:
        log.error("zenodo.done_with_failures")
        return 1
    log.info("zenodo.done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
