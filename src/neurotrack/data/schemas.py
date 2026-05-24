"""Pydantic v2 schemas for raw TrackML records and the joined per-event payload.

These models define the source-of-truth for column names, dtypes, and value
ranges.  They are used to:

* validate sample rows from the four raw CSVs at ingestion time;
* declare the expected column set so downstream code can rely on it;
* enforce dtype coercion when reading with polars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, get_type_hints

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Helper: map a Pydantic model -> {column: polars dtype}.
# Kept here so the schemas are the single source of truth.
# ---------------------------------------------------------------------------
_PY_TO_POLARS: dict[type, pl.DataType] = {
    int: pl.Int64(),
    float: pl.Float64(),
    str: pl.Utf8(),
    bool: pl.Boolean(),
}


class _CsvRecord(BaseModel):
    """Common base for raw-CSV row models."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    csv_columns: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def polars_schema(cls) -> dict[str, pl.DataType]:
        """Return the polars dtype mapping for this record's CSV columns."""
        # Use get_type_hints so PEP 563 stringified annotations resolve to
        # real types; raw __annotations__ would be strings.
        hints = get_type_hints(cls)
        out: dict[str, pl.DataType] = {}
        for col in cls.csv_columns:
            py_t = hints[col]
            origin = getattr(py_t, "__origin__", None)
            if origin is not None:  # Annotated / Optional / Union
                args = [a for a in py_t.__args__ if a is not type(None)]
                py_t = args[0]
            out[col] = _PY_TO_POLARS.get(py_t, pl.Object())
        return out

    @classmethod
    def validate_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        sample_size: int = 100,
    ) -> None:
        """Spot-check ``df`` against this model.

        Verifies (1) the column set matches ``csv_columns``, and (2) up to
        ``sample_size`` rows pass full Pydantic validation. Row-level checks
        are sampled rather than exhaustive because production events have
        ~100k rows; full validation is opt-in via ``sample_size=-1``.
        """
        missing = set(cls.csv_columns) - set(df.columns)
        if missing:
            raise SchemaValidationError(
                f"{cls.__name__}: missing columns {sorted(missing)}",
            )
        if sample_size == 0:
            return
        head = df.head(len(df) if sample_size < 0 else sample_size)
        for row in head.iter_rows(named=True):
            cls.model_validate(row)


# ---------------------------------------------------------------------------
# Per-record Pydantic models -- mirror the four raw TrackML CSVs.
# ---------------------------------------------------------------------------
class Hit(_CsvRecord):
    """A single detector hit (one row of ``*-hits.csv``)."""

    csv_columns: ClassVar[tuple[str, ...]] = (
        "hit_id",
        "x",
        "y",
        "z",
        "volume_id",
        "layer_id",
        "module_id",
    )

    hit_id: int = Field(..., ge=1, description="1-indexed hit id, unique within event")
    x: float = Field(..., description="global x [mm]")
    y: float = Field(..., description="global y [mm]")
    z: float = Field(..., description="global z [mm]")
    volume_id: int = Field(..., ge=0)
    layer_id: int = Field(..., ge=0)
    module_id: int = Field(..., ge=0)


class Particle(_CsvRecord):
    """A single Monte-Carlo particle (one row of ``*-particles.csv``)."""

    csv_columns: ClassVar[tuple[str, ...]] = (
        "particle_id",
        "vx",
        "vy",
        "vz",
        "px",
        "py",
        "pz",
        "q",
        "nhits",
    )

    particle_id: int = Field(..., ge=1, description="MC particle id (>=1)")
    vx: float
    vy: float
    vz: float
    px: float
    py: float
    pz: float
    q: int = Field(..., description="electric charge in elementary units")
    nhits: int = Field(..., ge=0, description="number of generator-level hits")

    @field_validator("q")
    @classmethod
    def _q_in_range(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError(f"unexpected charge {v}; expected -1, 0, or +1")
        return v


class Truth(_CsvRecord):
    """A single hit-to-particle truth association (one row of ``*-truth.csv``).

    The ``particle_id`` is 0 for noise hits.  ``tx, ty, tz, tpx, tpy, tpz`` are
    the truth-level position and momentum of the particle at the hit; we keep
    them for QC plots and downstream physics studies.
    """

    csv_columns: ClassVar[tuple[str, ...]] = (
        "hit_id",
        "particle_id",
        "tx",
        "ty",
        "tz",
        "tpx",
        "tpy",
        "tpz",
        "weight",
    )

    hit_id: int = Field(..., ge=1)
    particle_id: int = Field(..., ge=0, description="0 = noise / unassociated")
    tx: float
    ty: float
    tz: float
    tpx: float
    tpy: float
    tpz: float
    weight: float = Field(..., ge=0.0, description="hit weight in the TrackML score")


class Cell(_CsvRecord):
    """A single readout-cell entry (one row of ``*-cells.csv``).

    Cells are pixel-level activations associated with a hit.  We aggregate
    them per ``hit_id`` during ingestion to obtain compact features.
    """

    csv_columns: ClassVar[tuple[str, ...]] = ("hit_id", "ch0", "ch1", "value")

    hit_id: int = Field(..., ge=1)
    ch0: int = Field(..., ge=0)
    ch1: int = Field(..., ge=0)
    value: float = Field(..., ge=0.0, description="charge / energy in arbitrary units")


# ---------------------------------------------------------------------------
# EventData -- the joined per-event payload written to Parquet.
# ---------------------------------------------------------------------------
class EventData(BaseModel):
    """Lightweight per-event manifest.

    The actual hit/particle tables live in Parquet; this model captures the
    metadata + paths and is what ``ingest_event`` returns to its caller.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    event_id: int = Field(..., ge=0)
    parquet_path: Path
    n_hits: int = Field(..., ge=0)
    n_particles: int = Field(..., ge=0)
    n_noise_hits: int = Field(..., ge=0)
    has_cells: bool = True
    schema_version: str = "1"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class SchemaValidationError(ValueError):
    """Raised when a raw CSV does not match the declared schema."""


__all__ = [
    "Cell",
    "EventData",
    "Hit",
    "Particle",
    "SchemaValidationError",
    "Truth",
]
