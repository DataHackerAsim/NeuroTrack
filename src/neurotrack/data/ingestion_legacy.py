# LEGACY. 4-CSV Kaggle layout. New code in ingestion_redvid.py / ingestion_trackml_reduced.py.
"""L1 -- raw TrackML CSV -> validated, denormalised Parquet.

Two input layouts are supported:

1. **Per-event directory** (used by ``tests/fixtures/`` and our own
   re-exports)::

       event_dir/
         hits.csv
         particles.csv
         truth.csv
         cells.csv      # optional

   The event_id is parsed from the directory name (``event_000001000`` -> 1000).

2. **Flat TrackML layout** (the format Kaggle ships)::

       event_dir/
         event000001000-hits.csv
         event000001000-particles.csv
         event000001000-truth.csv
         event000001000-cells.csv

   Pass ``event_id=`` explicitly when the dir contains many events.

The output is a single Parquet file per event named
``event_{event_id:09d}.parquet`` in ``out_dir``, written via temp-file +
``os.replace`` for atomicity.  Re-running on an existing output is a no-op
unless ``force=True``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

import polars as pl

from .schemas import (
    Cell,
    EventData,
    Hit,
    Particle,
    SchemaValidationError,
    Truth,
)

SCHEMA_VERSION: Final[str] = "1"

_EVENT_DIR_RE = re.compile(r"event_?(\d+)$")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _resolve_event_id(event_dir: Path, event_id: int | None) -> int:
    if event_id is not None:
        return event_id
    m = _EVENT_DIR_RE.search(event_dir.name)
    if not m:
        raise ValueError(
            f"Cannot derive event_id from {event_dir.name!r}; pass event_id= explicitly.",
        )
    return int(m.group(1))


def _candidate_csv_paths(event_dir: Path, event_id: int, kind: str) -> list[Path]:
    return [
        event_dir / f"{kind}.csv",                       # per-event-dir layout
        event_dir / f"event{event_id:09d}-{kind}.csv",   # flat TrackML layout
    ]


def _required_csv_path(event_dir: Path, event_id: int, kind: str) -> Path:
    """Resolve a mandatory CSV; raise if missing."""
    for p in _candidate_csv_paths(event_dir, event_id, kind):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No {kind} CSV for event {event_id} under {event_dir} "
        f"(looked for {kind}.csv and event{event_id:09d}-{kind}.csv)",
    )


def _optional_csv_path(event_dir: Path, event_id: int, kind: str) -> Path | None:
    """Resolve an optional CSV; return ``None`` if missing."""
    for p in _candidate_csv_paths(event_dir, event_id, kind):
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CSV readers -- enforce the schema's polars dtypes at read time.
# ---------------------------------------------------------------------------
def _read_csv(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    df = pl.read_csv(path, schema_overrides=schema)
    extra = set(df.columns) - set(schema.keys())
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        raise SchemaValidationError(f"{path.name}: missing columns {sorted(missing)}")
    if extra:
        # Drop unexpected columns rather than failing -- TrackML dumps sometimes
        # carry extra debug fields.
        df = df.drop(list(extra))
    return df.select(list(schema.keys()))


# ---------------------------------------------------------------------------
# Cell aggregation -- one row per hit_id.
# ---------------------------------------------------------------------------
def _aggregate_cells(cells: pl.DataFrame) -> pl.DataFrame:
    """Compute per-hit cell summaries: count, total charge, ch0/ch1 span."""
    return cells.group_by("hit_id").agg(
        pl.len().alias("n_cells"),
        pl.col("value").sum().alias("cell_value_sum"),
        (pl.col("ch0").max() - pl.col("ch0").min() + 1).alias("cell_ch0_span"),
        (pl.col("ch1").max() - pl.col("ch1").min() + 1).alias("cell_ch1_span"),
    )


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def ingest_event(
    event_dir: Path,
    out_dir: Path,
    *,
    event_id: int | None = None,
    force: bool = False,
    validate_sample: int = 100,
) -> EventData:
    """Ingest one event's CSVs into a single denormalised Parquet file.

    Parameters
    ----------
    event_dir
        Directory containing this event's raw CSVs (per-event layout) or a
        flat directory containing many events' CSVs (with ``event_id`` set).
    out_dir
        Directory to write the output Parquet into; created if missing.
    event_id
        Optional override; otherwise derived from ``event_dir.name``.
    force
        Overwrite an existing output file.
    validate_sample
        Number of rows from each CSV to validate row-by-row through Pydantic.
        Set to 0 to skip, -1 for full validation (slow on real events).

    Returns
    -------
    EventData
        Manifest describing the written file.

    Raises
    ------
    SchemaValidationError
        If any CSV's columns or sample rows fail validation.
    FileNotFoundError
        If a required (non-cells) CSV is missing.
    """
    event_dir = Path(event_dir)
    out_dir = Path(out_dir)
    eid = _resolve_event_id(event_dir, event_id)

    out_path = out_dir / f"event_{eid:09d}.parquet"
    if out_path.exists() and not force:
        # Idempotent: rebuild manifest from the existing file.
        existing = pl.read_parquet(out_path)
        return EventData(
            event_id=eid,
            parquet_path=out_path,
            n_hits=existing.height,
            n_particles=int(existing["particle_id"].n_unique() - int((existing["particle_id"] == 0).any())),
            n_noise_hits=int((existing["particle_id"] == 0).sum()),
            has_cells="n_cells" in existing.columns,
            schema_version=SCHEMA_VERSION,
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ read
    hits = _read_csv(_required_csv_path(event_dir, eid, "hits"), Hit.polars_schema())
    parts = _read_csv(_required_csv_path(event_dir, eid, "particles"), Particle.polars_schema())
    truth = _read_csv(_required_csv_path(event_dir, eid, "truth"), Truth.polars_schema())
    cells_path = _optional_csv_path(event_dir, eid, "cells")
    cells: pl.DataFrame | None = (
        _read_csv(cells_path, Cell.polars_schema()) if cells_path is not None else None
    )

    # -------------------------------------------------------------- validate
    Hit.validate_dataframe(hits, sample_size=validate_sample)
    Particle.validate_dataframe(parts, sample_size=validate_sample)
    Truth.validate_dataframe(truth, sample_size=validate_sample)
    if cells is not None:
        Cell.validate_dataframe(cells, sample_size=validate_sample)

    # ------------------------------------------------------------------ join
    # rename truth.weight -> hit_weight to avoid future clashes.
    truth_renamed = truth.rename({"weight": "hit_weight"})
    parts_renamed = parts.rename(
        {
            "vx": "particle_vx",
            "vy": "particle_vy",
            "vz": "particle_vz",
            "px": "particle_px",
            "py": "particle_py",
            "pz": "particle_pz",
            "q":  "particle_q",
            "nhits": "particle_nhits",
        },
    )

    joined = hits.join(truth_renamed, on="hit_id", how="left")
    joined = joined.join(parts_renamed, on="particle_id", how="left")
    if cells is not None:
        joined = joined.join(_aggregate_cells(cells), on="hit_id", how="left")
        joined = joined.with_columns(
            pl.col("n_cells").fill_null(0),
            pl.col("cell_value_sum").fill_null(0.0),
            pl.col("cell_ch0_span").fill_null(0),
            pl.col("cell_ch1_span").fill_null(0),
        )

    # particle_id is null only if a hit has no truth row (shouldn't happen in
    # TrackML, but be defensive).  Treat null as noise (==0).
    joined = joined.with_columns(pl.col("particle_id").fill_null(0))

    # Sort by hit_id for determinism.
    joined = joined.sort("hit_id")

    # ----------------------------------------------------------------- write
    n_hits = joined.height
    n_noise = int((joined["particle_id"] == 0).sum())
    n_particles = parts.height

    tmp_path = out_dir / f".event_{eid:09d}.parquet.tmp.{os.getpid()}"
    try:
        joined.write_parquet(
            tmp_path,
            compression="zstd",
            statistics=True,
        )
        os.replace(tmp_path, out_path)  # atomic on same filesystem (Windows + POSIX)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return EventData(
        event_id=eid,
        parquet_path=out_path,
        n_hits=n_hits,
        n_particles=n_particles,
        n_noise_hits=n_noise,
        has_cells=cells is not None,
        schema_version=SCHEMA_VERSION,
    )


__all__ = ["SCHEMA_VERSION", "ingest_event"]
