"""Unified per-hit feature schema across the four Zenodo shards.

Both REDVID (cylindrical) and TrackML-reduced (cartesian) inputs are
projected into the same feature space so downstream code (embedding model,
GNN) can be source-agnostic.

NOTE on the feature count
-------------------------
Prompt R-B describes the feature list as "22 names" but enumerates
``[x, y, z, r, phi, eta, conformal_u, conformal_v, volume_onehot_0 ...
volume_onehot_15]`` -- that's 8 + 16 = 24 entries.  We honour the
literal list (24 features); two extra columns vs. "22" is the smaller
deviation than dropping conformal coordinates, which the upstream system
plan calls out as load-bearing for the metric-learning embedding (the
conformal map turns helices into approximate straight lines).
"""

from __future__ import annotations

import enum
import math
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

NDArrayF = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Feature names -- order is part of the contract
# ---------------------------------------------------------------------------
GEOMETRIC_FEATURES: Final[tuple[str, ...]] = (
    "x", "y", "z",
    "r", "phi", "eta",
    "conformal_u", "conformal_v",
)

NUM_VOLUMES: Final[int] = 16

VOLUME_ONEHOT_FEATURES: Final[tuple[str, ...]] = tuple(
    f"volume_onehot_{i}" for i in range(NUM_VOLUMES)
)

UNIFIED_FEATURES: Final[tuple[str, ...]] = (
    GEOMETRIC_FEATURES + VOLUME_ONEHOT_FEATURES
)

# Numerical guard for conformal mapping.
_CONFORMAL_EPS: Final[float] = 1.0e-12


# ---------------------------------------------------------------------------
# Source enum -- one entry per Zenodo tarball / shard
# ---------------------------------------------------------------------------
class Source(str, enum.Enum):
    """Identifier for the four shards.  Stored as a string column in Parquet
    so downstream code can filter without an enum import.
    """

    REDVID_HELICAL_5050 = "redvid_helical_5050"
    REDVID_HELICAL_100 = "redvid_helical_100"
    TRACKML_SMALL = "trackml_small"
    TRACKML_LARGE = "trackml_large"

    @property
    def is_redvid(self) -> bool:
        return self in (Source.REDVID_HELICAL_5050, Source.REDVID_HELICAL_100)

    @property
    def is_trackml(self) -> bool:
        return self in (Source.TRACKML_SMALL, Source.TRACKML_LARGE)


# ---------------------------------------------------------------------------
# Coordinate transforms (pure functions over numpy arrays)
# ---------------------------------------------------------------------------
def cartesian_from_cylindrical(
    r: NDArrayF, theta: NDArrayF, z: NDArrayF,
) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
    """(r, theta, z) -> (x, y, z).  Theta is in radians."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def cylindrical_from_cartesian(
    x: NDArrayF, y: NDArrayF, z: NDArrayF,
) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
    """(x, y, z) -> (r, phi, z).  phi in (-pi, pi]."""
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return r, phi, z


def compute_eta(r: NDArrayF, z: NDArrayF) -> NDArrayF:
    """Pseudorapidity eta = -ln(tan(theta/2)) from (r, z).

    theta is the polar angle from the +z axis.  We compute
        tan(theta/2) = r / (sqrt(r**2 + z**2) + z)
    which is numerically stable for both forward (r small, z>0) and
    backward (r small, z<0) hits.
    """
    rho = np.sqrt(r * r + z * z)
    denom = rho + z
    # Guard: when r == 0 and z < 0, denom -> 0 -- clamp to avoid div by zero.
    denom = np.where(np.abs(denom) < _CONFORMAL_EPS, _CONFORMAL_EPS, denom)
    arg = r / denom
    arg = np.clip(arg, _CONFORMAL_EPS, None)
    out: NDArrayF = -np.log(arg)
    return out


def compute_conformal(
    x: NDArrayF, y: NDArrayF,
) -> tuple[NDArrayF, NDArrayF]:
    """Conformal mapping (x, y) -> (u, v) = (x/(x^2+y^2), y/(x^2+y^2)).

    Helices that pass through the origin map to straight lines in (u, v).
    A small epsilon clamp on the denominator keeps the transform well-defined
    at r -> 0.
    """
    r2 = x * x + y * y
    r2 = np.where(r2 < _CONFORMAL_EPS, _CONFORMAL_EPS, r2)
    return x / r2, y / r2


def volume_onehot(
    volume_id: npt.ArrayLike, max_volumes: int = NUM_VOLUMES,
) -> npt.NDArray[np.float32]:
    """One-hot encode an integer ``volume_id`` array into shape ``(N, max_volumes)``.

    Values < 0 are clipped to bucket 0; values >= ``max_volumes`` are clipped
    to the last bucket.  TrackML's volume_id reaches 18, so the >=16 entries
    saturate; REDVID's sub_detector_id is 0..2 and fits exactly.
    """
    v = np.asarray(volume_id, dtype=np.int64)
    v = np.clip(v, 0, max_volumes - 1)
    out = np.zeros((v.shape[0], max_volumes), dtype=np.float32)
    out[np.arange(v.shape[0]), v] = 1.0
    return out


# ---------------------------------------------------------------------------
# Build the unified DataFrame
# ---------------------------------------------------------------------------
def build_unified_features(df: pl.DataFrame, source: Source) -> pl.DataFrame:
    """Project ``df`` (raw per-row hit table) into the unified feature schema.

    For REDVID inputs the source columns are cylindrical (``hit_r, hit_theta,
    hit_z``) plus ``sub_detector_id``; we convert to cartesian here.

    For TrackML inputs the columns are cartesian (``x, y, z``) plus
    ``volume_id``; we convert to cylindrical and compute eta from (r, z).

    Returns a NEW DataFrame with exactly :data:`UNIFIED_FEATURES` plus the
    pass-through columns the caller needs to emit downstream (the caller
    decides which raw columns to keep alongside the features).
    """
    n = df.height
    if source.is_redvid:
        r = df["hit_r"].to_numpy().astype(np.float64)
        theta = df["hit_theta"].to_numpy().astype(np.float64)
        z = df["hit_z"].to_numpy().astype(np.float64)
        x, y, _ = cartesian_from_cylindrical(r, theta, z)
        phi = np.where(theta > math.pi, theta - 2 * math.pi, theta)
        vid_src = df["sub_detector_id"].to_numpy().astype(np.int64)
    elif source.is_trackml:
        x = df["x"].to_numpy().astype(np.float64)
        y = df["y"].to_numpy().astype(np.float64)
        z = df["z"].to_numpy().astype(np.float64)
        r, phi, _ = cylindrical_from_cartesian(x, y, z)
        vid_src = df["volume_id"].to_numpy().astype(np.int64)
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown source: {source}")

    eta = compute_eta(r, z)
    cu, cv = compute_conformal(x, y)
    onehot = volume_onehot(vid_src)  # (N, NUM_VOLUMES)

    cols: dict[str, pl.Series] = {
        "x": pl.Series("x", x.astype(np.float32)),
        "y": pl.Series("y", y.astype(np.float32)),
        "z": pl.Series("z", z.astype(np.float32)),
        "r": pl.Series("r", r.astype(np.float32)),
        "phi": pl.Series("phi", phi.astype(np.float32)),
        "eta": pl.Series("eta", eta.astype(np.float32)),
        "conformal_u": pl.Series("conformal_u", cu.astype(np.float32)),
        "conformal_v": pl.Series("conformal_v", cv.astype(np.float32)),
    }
    for i in range(NUM_VOLUMES):
        cols[f"volume_onehot_{i}"] = pl.Series(
            f"volume_onehot_{i}", onehot[:, i].astype(np.float32),
        )
    out = pl.DataFrame(cols)
    assert out.height == n
    assert tuple(out.columns) == UNIFIED_FEATURES
    return out


__all__ = [
    "GEOMETRIC_FEATURES",
    "NUM_VOLUMES",
    "Source",
    "UNIFIED_FEATURES",
    "VOLUME_ONEHOT_FEATURES",
    "build_unified_features",
    "cartesian_from_cylindrical",
    "compute_conformal",
    "compute_eta",
    "cylindrical_from_cartesian",
    "volume_onehot",
]
