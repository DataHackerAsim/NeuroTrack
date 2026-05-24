# LEGACY. Kaggle+CERN path. kept for ref. new code in download_zenodo.py.
"""TrackML dataset acquisition CLI.

Source priority (``--source auto``)::

    1. Kaggle  -- requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY
    2. CERN Open Data record 14  -- automatic fallback on any Kaggle failure
    3. --url / --from-local       -- explicit user-supplied paths

Outputs land under ``data/raw/`` so the existing ``scripts/ingest.py`` finds
the layout it expects:

    data/raw/train_sample/event000001000-hits.csv
    data/raw/detectors.csv

Re-runs are idempotent: an existing file is not re-downloaded unless
``--force`` is set.  Checksums are verified against
``configs/data_checksums.yaml`` -- empty manifest entries are populated on
first success; subsequent mismatches abort.

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --source cern
    python scripts/download_data.py --full
    python scripts/download_data.py --verify-only
    python scripts/download_data.py --from-local /mnt/usb/trackml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from neurotrack.data.download_legacy import (
    ChecksumMismatchError,
    DownloadError,
    KaggleAuthError,
    ZipSlipError,
    compute_sha256,
    copy_local,
    download_cern,
    download_kaggle,
    download_url,
    safe_extract_zip,
    verify_checksum,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAGGLE_COMPETITION = "trackml-particle-identification"
CERN_RECORD_ID = 14

TRAIN_SAMPLE_FILES: list[str] = ["train_sample.zip"]
TRAIN_FULL_FILES: list[str] = [
    "train_1.zip",
    "train_2.zip",
    "train_3.zip",
    "train_4.zip",
    "train_5.zip",
]
DETECTORS_FILE = "detectors.csv"

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_MANIFEST = Path("configs/data_checksums.yaml")


# ---------------------------------------------------------------------------
# Manifest (configs/data_checksums.yaml)
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, Any]:
    import yaml  # bundled with hydra-core / omegaconf

    if not path.exists():
        return {"files": {}}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "files" not in data:
        data["files"] = {}
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    tmp.replace(path)


def populate_or_verify(manifest: dict[str, Any], file_path: Path) -> bool:
    """Populate the manifest entry on first run, otherwise verify the hash.

    Returns True iff the file's hash matches an existing manifest entry.
    Raises :class:`ChecksumMismatchError` on a populated-but-mismatched entry.
    """
    key = file_path.name
    files = manifest.setdefault("files", {})
    entry = files.setdefault(key, {"sha256": "", "size_bytes": None, "source": ""})
    expected = (entry.get("sha256") or "").strip()

    if expected:
        if not verify_checksum(file_path, expected):
            raise ChecksumMismatchError(
                f"{file_path.name}: SHA-256 mismatch; expected {expected}, "
                f"got {compute_sha256(file_path)}.  Delete the file and re-download, "
                f"or fix the manifest entry in configs/data_checksums.yaml.",
            )
        return True

    entry["sha256"] = compute_sha256(file_path)
    entry["size_bytes"] = file_path.stat().st_size
    return False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def configure_logging(json: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    procs: list[Any] = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    if json:
        procs.append(structlog.processors.JSONRenderer())
    else:
        procs.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=procs,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=("auto", "kaggle", "cern", "url", "local"),
        default="auto",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="fetch the full training set (~80 GB) instead of train_sample",
    )
    p.add_argument(
        "--accept-rules",
        action="store_true",
        help="acknowledgement that you accepted Kaggle's competition rules",
    )
    p.add_argument("--url", help="explicit URL when --source=url")
    p.add_argument(
        "--from-local",
        type=Path,
        help="copy from an existing local file/dir when --source=local",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="re-hash existing files against the manifest; do not download",
    )
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="path to checksum manifest yaml",
    )
    p.add_argument("--json-logs", action="store_true", help="emit structlog JSON")
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="skip zip extraction after download",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def expected_files(args: argparse.Namespace) -> list[str]:
    return (TRAIN_FULL_FILES if args.full else TRAIN_SAMPLE_FILES) + [DETECTORS_FILE]


def verify_only(args: argparse.Namespace, log: structlog.stdlib.BoundLogger) -> int:
    manifest = load_manifest(args.manifest)
    files_meta = manifest.get("files") or {}
    if not files_meta or not any((e.get("sha256") or "") for e in files_meta.values()):
        log.error(
            "verify.manifest_empty",
            manifest=str(args.manifest),
            hint="run `python scripts/download_data.py` first to fetch the dataset",
        )
        return 1

    missing: list[str] = []
    bad: list[str] = []
    for fname in expected_files(args):
        entry = files_meta.get(fname, {})
        expected = (entry.get("sha256") or "").strip()
        path = args.output_dir / fname
        if not path.exists():
            missing.append(fname)
            continue
        if not expected:
            log.warning("verify.entry_unpopulated", file=fname)
            continue
        if not verify_checksum(path, expected):
            bad.append(fname)

    if missing or bad:
        log.error("verify.failed", missing=missing, mismatched=bad)
        return 1
    log.info("verify.ok", checked=expected_files(args))
    return 0


def fetch_files(
    args: argparse.Namespace,
    log: structlog.stdlib.BoundLogger,
    sources: list[str],
) -> list[Path]:
    files = expected_files(args)

    last_err: Exception | None = None
    for src in sources:
        try:
            if src == "kaggle":
                if not args.accept_rules:
                    log.warning(
                        "kaggle.rules_unconfirmed",
                        hint="pass --accept-rules once you've accepted the competition rules online",
                    )
                paths = download_kaggle(
                    KAGGLE_COMPETITION, files, args.output_dir, force=args.force,
                )
            elif src == "cern":
                paths = download_cern(
                    CERN_RECORD_ID,
                    args.output_dir,
                    file_filter=files,
                    force=args.force,
                )
            elif src == "url":
                if not args.url:
                    raise DownloadError("--source=url requires --url <URL>")
                paths = [download_url(args.url, args.output_dir, force=args.force)]
            elif src == "local":
                if not args.from_local:
                    raise DownloadError("--source=local requires --from-local <PATH>")
                paths = copy_local(args.from_local, args.output_dir, force=args.force)
            else:
                raise DownloadError(f"unknown source: {src}")
            log.info("download.source_ok", source=src, files=[p.name for p in paths])
            return paths
        except (ChecksumMismatchError, ZipSlipError):
            raise  # never fall through on integrity failures
        except (KaggleAuthError, DownloadError, Exception) as e:
            last_err = e
            log.warning("download.source_failed", source=src, error=str(e))

    raise DownloadError(
        f"all sources exhausted; last error: {last_err}",
    )


def maybe_extract(paths: list[Path], log: structlog.stdlib.BoundLogger) -> None:
    """Extract any *.zip in ``paths`` to a sibling directory named after its stem.

    train_sample.zip -> data/raw/train_sample/...
    """
    for p in paths:
        if p.suffix.lower() != ".zip":
            continue
        target = p.with_suffix("")  # e.g. data/raw/train_sample
        log.info("extract.start", zip=str(p), dest=str(target))
        try:
            members = safe_extract_zip(p, target)
        except ZipSlipError:
            raise
        log.info("extract.done", zip=str(p), n_members=len(members))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.json_logs)
    log = structlog.get_logger("download_data")

    if args.verify_only:
        return verify_only(args, log)

    sources = {
        "auto": ["kaggle", "cern"],
        "kaggle": ["kaggle"],
        "cern": ["cern"],
        "url": ["url"],
        "local": ["local"],
    }[args.source]

    try:
        paths = fetch_files(args, log, sources)
    except DownloadError as e:
        log.error("download.failed", error=str(e))
        return 1

    # Verify or populate manifest for every file we have.
    manifest = load_manifest(args.manifest)
    for p in paths:
        try:
            verified = populate_or_verify(manifest, p)
            log.info(
                "manifest.entry",
                file=p.name,
                action="verified" if verified else "populated",
            )
        except ChecksumMismatchError as e:
            log.error("manifest.mismatch", file=p.name, error=str(e))
            return 1
    save_manifest(args.manifest, manifest)

    if not args.no_extract:
        try:
            maybe_extract(paths, log)
        except ZipSlipError as e:
            log.error("extract.zip_slip", error=str(e))
            return 1

    log.info("download.done", out=str(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
