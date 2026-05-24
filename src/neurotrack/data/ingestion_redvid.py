"""Streaming ingestion for REDVID tarballs.

Each REDVID tarball contains a single big CSV (semicolon-separated, quoted
strings) with all events stacked by ``event_id``.  We never extract the CSV
to disk -- we stream rows out of the tar via :class:`tarfile.TarFile` and
flush per-event buffers to Parquet as we cross ``event_id`` boundaries.

Output layout per shard::

    <out_dir>/<shard>/events/<event_id>.parquet
    <out_dir>/<shard>/_metadata.json

Idempotent: an event whose Parquet already exists is skipped unless
``force=True``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Final

import polars as pl

from .schemas_zenodo import RedvidRow
from .unified_schema import Source, build_unified_features

CSV_DELIMITER: Final[str] = ";"
EXPECTED_COLUMNS: Final[tuple[str, ...]] = RedvidRow.csv_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_data_member(tf: tarfile.TarFile) -> tarfile.TarInfo:
    """Return the actual data CSV member, skipping macOS resource forks."""
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = Path(m.name).name
        if name.startswith("._"):  # AppleDouble metadata
            continue
        if name.lower().endswith(".csv"):
            return m
    raise FileNotFoundError("no data CSV member found in tarball")


def _open_csv_stream(
    tf: tarfile.TarFile, member: tarfile.TarInfo,
) -> "csv.DictReader[str]":
    fobj = tf.extractfile(member)
    if fobj is None:
        raise RuntimeError(f"cannot extract {member.name}")
    text = io.TextIOWrapper(fobj, encoding="utf-8", newline="", errors="strict")
    reader = csv.DictReader(text, delimiter=CSV_DELIMITER)
    if reader.fieldnames is None:
        raise ValueError("empty CSV header")
    cols = tuple(reader.fieldnames)
    if cols != EXPECTED_COLUMNS:
        raise ValueError(
            f"REDVID column mismatch.\n"
            f"  expected: {EXPECTED_COLUMNS}\n"
            f"  got     : {cols}",
        )
    return reader


def _flush_event(
    event_id: int,
    rows: list[dict[str, str]],
    out_events_dir: Path,
    source: Source,
    force: bool,
    validate_sample: int,
) -> tuple[bool, int]:
    """Write one event's Parquet file. Returns ``(wrote_new, n_rows)``."""
    out_path = out_events_dir / f"{event_id}.parquet"
    if out_path.exists() and not force:
        return False, len(rows)

    # Build a polars DataFrame with explicit dtypes (csv stage was all str).
    df = pl.DataFrame(
        {
            "event_id": [int(r["event_id"]) for r in rows],
            "sub_detector_id": [int(r["sub_detector_id"]) for r in rows],
            "sub_detector_type": [r["sub_detector_type"] for r in rows],
            "track_id": [int(r["track_id"]) for r in rows],
            "track_type": [r["track_type"] for r in rows],
            "radial_const": [float(r["radial_const"]) for r in rows],
            "azimuthal_const": [float(r["azimuthal_const"]) for r in rows],
            "pitch_const": [float(r["pitch_const"]) for r in rows],
            "radial_coeff": [float(r["radial_coeff"]) for r in rows],
            "azimuthal_coeff": [float(r["azimuthal_coeff"]) for r in rows],
            "pitch_coeff": [float(r["pitch_coeff"]) for r in rows],
            "hit_id": [int(r["hit_id"]) for r in rows],
            "hit_r": [float(r["hit_r"]) for r in rows],
            "hit_theta": [float(r["hit_theta"]) for r in rows],
            "hit_z": [float(r["hit_z"]) for r in rows],
        },
    )

    # Spot-validate a sample of rows through Pydantic.
    if validate_sample != 0:
        sample = df.head(df.height if validate_sample < 0 else validate_sample)
        for row in sample.iter_rows(named=True):
            RedvidRow.model_validate(row)

    # Build the unified feature columns (24 cols) and stitch on bookkeeping.
    features = build_unified_features(df, source)
    out_df = features.with_columns(
        df["event_id"].alias("event_id"),
        df["hit_id"].alias("hit_id"),
        df["track_id"].alias("particle_id"),  # REDVID's track_id is the particle proxy
        pl.lit(1.0).alias("weight"),          # no per-hit weight in REDVID
        pl.lit(source.value).alias("source"),
    )

    out_events_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_events_dir / f".{event_id}.parquet.tmp.{os.getpid()}"
    try:
        out_df.write_parquet(tmp_path, compression="zstd", statistics=True)
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return True, len(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ingest_redvid_tarball(
    tar_path: Path,
    out_dir: Path,
    source: Source,
    *,
    num_workers: int = 1,  # noqa: ARG001  -- accepted for interface symmetry
    force: bool = False,
    validate_sample: int = 200,
) -> dict[str, object]:
    """Stream a REDVID tarball into per-event Parquet files.

    Parameters mirror :func:`ingest_trackml_tarball` for orchestration symmetry.
    Returns a stats dict written verbatim to ``<shard>/_metadata.json``.
    """
    if not source.is_redvid:
        raise ValueError(f"source {source!r} is not a REDVID source")
    tar_path = Path(tar_path)
    out_dir = Path(out_dir)
    shard_dir = out_dir / source.value
    events_dir = shard_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_events = 0
    n_events_written = 0
    n_events_skipped = 0
    n_rows = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        member = _find_data_member(tf)
        reader = _open_csv_stream(tf, member)

        current_eid: int | None = None
        buffer: list[dict[str, str]] = []
        for raw in reader:
            try:
                eid = int(raw["event_id"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"REDVID row has bad event_id: {raw.get('event_id')!r}",
                ) from e
            if current_eid is None:
                current_eid = eid
            if eid != current_eid:
                wrote, n = _flush_event(
                    current_eid, buffer, events_dir, source, force, validate_sample,
                )
                n_events += 1
                n_events_written += int(wrote)
                n_events_skipped += int(not wrote)
                n_rows += n
                buffer = []
                current_eid = eid
            buffer.append(raw)
        # Flush the last event.
        if buffer and current_eid is not None:
            wrote, n = _flush_event(
                current_eid, buffer, events_dir, source, force, validate_sample,
            )
            n_events += 1
            n_events_written += int(wrote)
            n_events_skipped += int(not wrote)
            n_rows += n

    stats: dict[str, object] = {
        "source": source.value,
        "tar_path": str(tar_path),
        "shard_dir": str(shard_dir),
        "n_events": n_events,
        "n_events_written": n_events_written,
        "n_events_skipped": n_events_skipped,
        "n_rows": n_rows,
        "wall_time_s": round(time.time() - t0, 2),
        "schema_version": "zenodo-redvid-1",
    }
    (shard_dir / "_metadata.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8",
    )
    return stats


__all__ = ["EXPECTED_COLUMNS", "ingest_redvid_tarball"]
