"""Pydantic v2 schemas for the Zenodo TrackML + RedVid raw rows.

These are the source-of-truth for the raw CSV columns inside each tarball.
They are used to spot-check rows during ingestion (the per-event buffers are
small enough -- typically a few hundred to a few thousand rows -- that
full validation is cheap, but the call sites still default to a 200-row
sample for safety).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Raw row models
# ---------------------------------------------------------------------------
class RedvidRow(BaseModel):
    """One row of the REDVID per-hit-per-track-per-event CSV.

    The CSV uses ';' as separator and quotes string fields.  ``track_type`` and
    ``sub_detector_type`` are free-form strings (e.g. ``"helical_expanding"``,
    ``"short_strip"``); we accept any non-empty string.
    """

    model_config = ConfigDict(extra="forbid", strict=False, frozen=True)

    csv_columns: ClassVar[tuple[str, ...]] = (
        "event_id", "sub_detector_id", "sub_detector_type",
        "track_id", "track_type",
        "radial_const", "azimuthal_const", "pitch_const",
        "radial_coeff", "azimuthal_coeff", "pitch_coeff",
        "hit_id", "hit_r", "hit_theta", "hit_z",
    )

    event_id: int = Field(..., ge=0)
    sub_detector_id: int = Field(..., ge=0)
    sub_detector_type: str = Field(..., min_length=1)
    track_id: int = Field(..., ge=0)
    track_type: str = Field(..., min_length=1)
    radial_const: float
    azimuthal_const: float
    pitch_const: float
    radial_coeff: float
    azimuthal_coeff: float
    pitch_coeff: float
    hit_id: int = Field(..., ge=0)
    hit_r: float = Field(..., ge=0.0)
    hit_theta: float
    hit_z: float


class TrackMLReducedRow(BaseModel):
    """One row of the TrackML-reduced per-hit-denormalised CSV."""

    model_config = ConfigDict(extra="forbid", strict=False, frozen=True)

    csv_columns: ClassVar[tuple[str, ...]] = (
        "x", "y", "z", "volume_id",
        "vx", "vy", "vz", "px", "py", "pz", "q",
        "particle_id", "weight", "event_id",
    )

    x: float
    y: float
    z: float
    volume_id: int = Field(..., ge=0)
    vx: float
    vy: float
    vz: float
    px: float
    py: float
    pz: float
    q: int = Field(..., description="charge in elementary units; expected -1, 0, +1")
    particle_id: int = Field(..., ge=0, description="0 = noise / unassociated")
    weight: float = Field(..., ge=0.0)
    event_id: int = Field(..., ge=0)

    @field_validator("q")
    @classmethod
    def _q_in_set(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError(f"unexpected charge {v}; expected -1, 0, or +1")
        return v


# ---------------------------------------------------------------------------
# Output / per-event models
# ---------------------------------------------------------------------------
class UnifiedHit(BaseModel):
    """One row of the unified per-hit Parquet (post-ingestion).

    The 24 unified geometry + volume-onehot features live in their own
    columns (so we don't repeat them here); this model just captures the
    bookkeeping columns the ingestion pipeline emits alongside the features.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(..., ge=0)
    hit_id: int = Field(..., ge=0)
    particle_id: int = Field(..., ge=0)
    weight: float = Field(..., ge=0.0)
    source: str = Field(..., min_length=1)


class ParticleKinematics(BaseModel):
    """One row of the per-event particle kinematics Parquet (TrackML only).

    REDVID rows carry track parameters per-row but no separate particle
    kinematics; this model is therefore TrackML-specific.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(..., ge=0)
    particle_id: int = Field(..., ge=0)
    vx: float
    vy: float
    vz: float
    px: float
    py: float
    pz: float
    q: int
    n_hits: int = Field(..., ge=0)


__all__ = [
    "ParticleKinematics",
    "RedvidRow",
    "TrackMLReducedRow",
    "UnifiedHit",
]
