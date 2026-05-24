"""Streaming ingestion for the TrackML-reduced tarballs.

Each TrackML-reduced tarball ships one big CSV (comma-separated, no quoting)
with all events stacked by ``event_id``.  We stream rows out of the tar,
buffer per-event, then write two Parquet files per event:

* ``<shard>/events/<event_id>.parquet``     -- 24 unified features + bookkeeping
* ``<shard>/particles/<event_id>.parquet``  -- per-(event,particle) kinematics

The particles table is deduped from the per-row denormalised kinematics
(``vx, vy, vz, px, py, pz, q``) -- these are constant per particle, so we
take the first row's values for each ``particle_id`` group.
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

from .schemas_zenodo import TrackMLReducedRow
from .unified_schema import Source, build_unified_features

CSV_DELIMITER: Final[str] = ","
EXPECTED_COLUMNS: Final[tuple[str, ...]] = TrackMLReducedRow.csv_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_data_member(tf: tarfile.TarFile) -> tarfile.TarInfo:
    """Return the actual data CSV member, skipping macOS resource forks."""
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = Path(m.name).name
        if name.startswith("._"):
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
            f"TrackML column mismatch.\n"
            f"  expected: {EXPECTED_COLUMNS}\n"
            f"  got     : {cols}",
        )
    return reader


def _build_event_df(rows: list[dict[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x":           [float(r["x"]) for r in rows],
            "y":           [float(r["y"]) for r in rows],
            "z":           [float(r["z"]) for r in rows],
            "volume_id":   [int(r["volume_id"]) for r in rows],
            "vx":          [float(r["vx"]) for r in rows],
            "vy":          [float(r["vy"]) for r in rows],
            "vz":          [float(r["vz"]) for r in rows],
            "px":          [float(r["px"]) for r in rows],
            "py":          [float(r["py"]) for r in rows],
            "pz":          [float(r["pz"]) for r in rows],
            "q":           [int(r["q"]) for r in rows],
            "particle_id": [int(r["particle_id"]) for r in rows],
            "weight":      [float(r["weight"]) for r in rows],
            "event_id":    [int(r["event_id"]) for r in rows],
        },
    )


def _atomic_write(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".{out_path.name}.tmp.{os.getpid()}"
    try:
        df.write_parquet(tmp, compression="zstd", statistics=True)
        os.replace(tmp, out_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _flush_event(
    event_id: int,
    rows: list[dict[str, str]],
    events_dir: Path,
    particles_dir: Path,
    source: Source,
    force: bool,
    validate_sample: int,
) -> tuple[bool, int, int]:
    """Write one event's hits + particles Parquets.

    Returns ``(wrote_new, n_hits, n_particles)``.  ``wrote_new`` is False when
    both files already exist and ``force`` is False.
    """
    hits_path = events_dir / f"{event_id}.parquet"
    parts_path = particles_dir / f"{event_id}.parquet"
    if hits_path.exists() and parts_path.exists() and not force:
        return False, len(rows), 0

    df = _build_event_df(rows)

    # Spot-validate a sample.
    if validate_sample != 0:
        sample = df.head(df.height if validate_sample < 0 else validate_sample)
        for row in sample.iter_rows(named=True):
            TrackMLReducedRow.model_validate(row)

    # Hits parquet (24 features + bookkeeping).
    features = build_unified_features(df, source)
    hits = features.with_columns(
        df["event_id"].alias("event_id"),
        pl.arange(0, df.height, dtype=pl.Int64).alias("hit_id"),  # synthesised
        df["particle_id"].alias("particle_id"),
        df["weight"].alias("weight"),
        pl.lit(source.value).alias("source"),
    )
    _atomic_write(hits, hits_path)

    # Particles parquet (deduped per particle; n_hits per group).
    particles = (
        df.group_by("particle_id")
        .agg(
            pl.col("event_id").first().alias("event_id"),
            pl.col("vx").first().alias("vx"),
            pl.col("vy").first().alias("vy"),
            pl.col("vz").first().alias("vz"),
            pl.col("px").first().alias("px"),
            pl.col("py").first().alias("py"),
            pl.col("pz").first().alias("pz"),
            pl.col("q").first().alias("q"),
            pl.len().alias("n_hits"),
        )
        .select(
            ["event_id", "particle_id", "vx", "vy", "vz",
             "px", "py", "pz", "q", "n_hits"],
        )
        .sort("particle_id")
    )
    _atomic_write(particles, parts_path)

    return True, hits.height, particles.height


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ingest_trackml_tarball(
    tar_path: Path,
    out_dir: Path,
    source: Source,
    *,
    num_workers: int = 1,  # noqa: ARG001  -- interface symmetry
    force: bool = False,
    validate_sample: int = 200,
) -> dict[str, object]:
    """Stream a TrackML-reduced tarball into per-event Parquet files."""
    if not source.is_trackml:
        raise ValueError(f"source {source!r} is not a TrackML source")
    tar_path = Path(tar_path)
    out_dir = Path(out_dir)
    shard_dir = out_dir / source.value
    events_dir = shard_dir / "events"
    particles_dir = shard_dir / "particles"
    events_dir.mkdir(parents=True, exist_ok=True)
    particles_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_events = 0
    n_events_written = 0
    n_events_skipped = 0
    n_rows = 0
    n_particles = 0

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
                    f"TrackML row has bad event_id: {raw.get('event_id')!r}",
                ) from e
            if current_eid is None:
                current_eid = eid
            if eid != current_eid:
                wrote, nh, np_ = _flush_event(
                    current_eid, buffer, events_dir, particles_dir,
                    source, force, validate_sample,
                )
                n_events += 1
                n_events_written += int(wrote)
                n_events_skipped += int(not wrote)
                n_rows += nh
                n_particles += np_
                buffer = []
                current_eid = eid
            buffer.append(raw)
        if buffer and current_eid is not None:
            wrote, nh, np_ = _flush_event(
                current_eid, buffer, events_dir, particles_dir,
                source, force, validate_sample,
            )
            n_events += 1
            n_events_written += int(wrote)
            n_events_skipped += int(not wrote)
            n_rows += nh
            n_particles += np_

    stats: dict[str, object] = {
        "source": source.value,
        "tar_path": str(tar_path),
        "shard_dir": str(shard_dir),
        "n_events": n_events,
        "n_events_written": n_events_written,
        "n_events_skipped": n_events_skipped,
        "n_rows": n_rows,
        "n_particles": n_particles,
        "wall_time_s": round(time.time() - t0, 2),
        "schema_version": "zenodo-trackml-1",
    }
    (shard_dir / "_metadata.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8",
    )
    return stats


__all__ = ["EXPECTED_COLUMNS", "ingest_trackml_tarball"]
