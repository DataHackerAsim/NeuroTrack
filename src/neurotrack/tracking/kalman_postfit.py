"""Geometric helix post-fit + chi2 arbitration of predicted tracks.

A "full" Kalman filter for tracking takes a measurement model, a process
noise model, and iterates one hit at a time.  For v1 we use a much
lighter geometric fit that captures the same single number we actually
need downstream: the per-track ``chi2_per_dof`` against a perfect-helix
hypothesis.

The fit decomposes into:

* a 2-D circle fit in (x, y) using the algebraic Kasa method
  (linear least-squares on (A, B, C) with x^2 + y^2 + Ax + By + C = 0),
  yielding (xc, yc, R);
* a 1-D linear fit of z vs arc length s, yielding (z0, theta) and
  therefore eta and the helix pitch.

Then we compute residuals at each hit:
* radial residual:  |hit_xy - center| - R
* longitudinal residual: z - (z0 + s_i * cot(theta))

and sum them into chi2 with a uniform measurement sigma of 0.3 mm
(see R-H spec, refine later).

No torch dependency; runs CPU-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

DEFAULT_SIGMA_MM: Final[float] = 0.3
DEFAULT_N_PARAMS: Final[int] = 5  # xc, yc, R, theta, z0


@dataclass(frozen=True)
class HelixParams:
    xc: float       # circle centre x in (xy) plane
    yc: float       # circle centre y
    R: float        # radius (signed = curvature direction is implicit)
    z0: float       # z at s = 0
    theta: float    # polar angle (cot(theta) = dz/ds)

    @property
    def eta(self) -> float:
        # eta = -ln(tan(theta/2));  guard the corner cases.
        if not math.isfinite(self.theta) or self.theta <= 0 or self.theta >= math.pi:
            return float("nan")
        t = math.tan(self.theta / 2.0)
        if t <= 0.0:
            return float("nan")
        return -math.log(t)

    @property
    def pT_proxy(self) -> float:
        # pT proxy proportional to R (true pT requires B-field strength).
        return abs(self.R)


# ---------------------------------------------------------------------------
def _circle_fit_kasa(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Kasa method: minimize sum((x - xc)^2 + (y - yc)^2 - R^2)^2.

    Returns (xc, yc, R).  Falls back to (mean_x, mean_y, mean_radius)
    when the linear system is singular (collinear points).
    """
    n = x.size
    if n < 3:
        return float(x.mean()) if n else 0.0, float(y.mean()) if n else 0.0, 0.0
    # x^2 + y^2 + Ax + By + C = 0
    A = np.column_stack([x, y, np.ones(n)])
    b = -(x * x + y * y)
    try:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float(x.mean()), float(y.mean()), float(np.hypot(x - x.mean(), y - y.mean()).mean())
    A_, B_, C_ = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    xc = -A_ / 2.0
    yc = -B_ / 2.0
    R2 = xc * xc + yc * yc - C_
    R = math.sqrt(R2) if R2 > 0 else 0.0
    return xc, yc, R


def _arc_length(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    xc: float, yc: float, R: float,
) -> npt.NDArray[np.float64]:
    """Return per-hit arc length along the circle, measured CCW from the
    first hit.  Returns straight-line distance if R == 0.
    """
    if R <= 0.0:
        # Degenerate (line): arc length = chord distance from first point.
        chord: npt.NDArray[np.float64] = np.hypot(x - x[0], y - y[0])
        return chord
    # Unwrap angles so s is monotone with respect to the hit order.
    phi = np.arctan2(y - yc, x - xc)
    dphi = np.diff(phi)
    dphi = np.where(dphi > math.pi, dphi - 2 * math.pi, dphi)
    dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
    phi_unwrapped = np.concatenate([[0.0], np.cumsum(dphi)])
    arc: npt.NDArray[np.float64] = R * np.abs(phi_unwrapped - phi_unwrapped[0])
    return arc


def fit_helix_chi2(
    hits_xyz: npt.NDArray[np.float64],
    *,
    sigma_mm: float = DEFAULT_SIGMA_MM,
) -> tuple[HelixParams, float, npt.NDArray[np.float64]]:
    """Geometric helix fit + chi2.

    Returns ``(helix, chi2_per_dof, residuals_mm)``.  ``residuals_mm`` is
    the per-hit total residual length (sqrt of 2D radial^2 + longitudinal^2),
    aligned to the caller's input hit order.
    """
    n = hits_xyz.shape[0]
    if n < 3:
        return (
            HelixParams(xc=0.0, yc=0.0, R=0.0, z0=0.0, theta=math.pi / 2),
            float("inf"),
            np.zeros(n, dtype=np.float64),
        )
    xyz = np.asarray(hits_xyz, dtype=np.float64)
    x_all = xyz[:, 0]
    y_all = xyz[:, 1]
    z_all = xyz[:, 2]

    # 1) Circle fit -- order-invariant.
    xc, yc, R = _circle_fit_kasa(x_all, y_all)

    # 2) Arc-length parameterisation -- order the hits along the helix.
    # Sort by z (monotone along any helix with non-zero pz), with a phi
    # tie-break for purely transverse tracks (pz ~= 0).
    phi_all = np.arctan2(y_all - yc, x_all - xc)
    if z_all.max() - z_all.min() > 1.0e-9:
        order = np.argsort(z_all, kind="stable")
    else:
        order = np.argsort(phi_all, kind="stable")
    phi_sorted = phi_all[order]
    z_sorted = z_all[order]

    dphi = np.diff(phi_sorted)
    dphi = np.where(dphi > math.pi, dphi - 2 * math.pi, dphi)
    dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
    phi_unwrapped = np.concatenate([[0.0], np.cumsum(dphi)])
    s_sorted = R * np.abs(phi_unwrapped - phi_unwrapped[0]) if R > 0 \
        else np.linspace(0.0, 1.0, n)

    # 3) Linear z vs s.
    if s_sorted[-1] - s_sorted[0] > 1.0e-9:
        A = np.column_stack([np.ones_like(s_sorted), s_sorted])
        try:
            zcoef, *_ = np.linalg.lstsq(A, z_sorted, rcond=None)
            z0 = float(zcoef[0])
            cot_theta = float(zcoef[1])
        except np.linalg.LinAlgError:
            z0 = float(z_sorted.mean())
            cot_theta = 0.0
    else:
        z0 = float(z_sorted.mean())
        cot_theta = 0.0
    theta = math.atan2(1.0, cot_theta) if cot_theta else math.pi / 2.0

    # 4) Per-hit residuals, in caller's original order.
    s_all = np.empty(n, dtype=np.float64)
    s_all[order] = s_sorted
    r_xy_all = np.hypot(x_all - xc, y_all - yc)
    res_r = r_xy_all - R
    res_z = z_all - (z0 + cot_theta * s_all)
    res_total: npt.NDArray[np.float64] = np.sqrt(res_r * res_r + res_z * res_z)

    chi2 = float(np.sum((res_total / sigma_mm) ** 2))
    dof = max(1, n - DEFAULT_N_PARAMS)
    chi2_per_dof = chi2 / dof
    return (
        HelixParams(xc=xc, yc=yc, R=R, z0=z0, theta=theta),
        chi2_per_dof,
        res_total,
    )


# ---------------------------------------------------------------------------
__all__ = [
    "DEFAULT_SIGMA_MM",
    "HelixParams",
    "fit_helix_chi2",
]
