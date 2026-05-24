"""Kalman-driven track arbitration: split outlier-laden tracks.

Given a list of candidate tracks (e.g. from
:func:`build_tracks_uncertainty`) and the per-event hit positions, fit
each track with :func:`fit_helix_chi2` and:

* keep tracks with ``chi2_per_dof <= chi2_threshold``
* try to split tracks above the threshold at their largest residual; if
  both halves now pass the threshold and have at least
  ``min_hits_after_split`` hits, keep both; if only one passes, keep
  that one; otherwise keep the original (splitting made it worse).

Returns the arbitrated track list plus a small stats dict (counts of
kept / split / dropped) that the orchestrator can log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .builder import Track
from .kalman_postfit import fit_helix_chi2


@dataclass
class ArbitrateStats:
    n_kept: int = 0
    n_split_attempted: int = 0
    n_split_succeeded: int = 0
    n_dropped: int = 0
    n_one_half_kept: int = 0
    sum_chi2: float = 0.0
    n_fit: int = 0

    @property
    def mean_chi2(self) -> float:
        return self.sum_chi2 / max(1, self.n_fit)


def _fit(hits_xyz: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
    _, chi2, res = fit_helix_chi2(hits_xyz)
    return chi2, res


def arbitrate_tracks(
    tracks: list[Track],
    hit_xyz: npt.NDArray[np.float64],
    *,
    chi2_threshold: float = 3.0,
    min_hits_after_split: int = 3,
) -> tuple[list[Track], ArbitrateStats]:
    """Run Kalman post-fit on each track; split or drop outliers.

    Parameters
    ----------
    tracks
        Candidate tracks (each contains ``hit_indices`` into ``hit_xyz``).
    hit_xyz
        ``(N, 3)`` array of hit positions in the event.
    chi2_threshold
        Tracks with ``chi2_per_dof <= chi2_threshold`` are kept verbatim.
    min_hits_after_split
        Each surviving sub-track must have at least this many hits.
    """
    out: list[Track] = []
    stats = ArbitrateStats()

    for t in tracks:
        idx = t.hit_indices
        if idx.size < min_hits_after_split:
            stats.n_dropped += 1
            continue
        xyz = hit_xyz[idx]
        chi2, res = _fit(xyz)
        stats.sum_chi2 += chi2
        stats.n_fit += 1
        if chi2 <= chi2_threshold or not np.isfinite(chi2):
            out.append(_with_chi2(t, chi2))
            stats.n_kept += 1
            continue

        # Above threshold: try a single split at the largest-residual hit.
        stats.n_split_attempted += 1
        # Sort by radial distance so the split has physical meaning.
        r2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        radial_order = np.argsort(r2)
        sorted_idx = idx[radial_order]
        sorted_res = res[radial_order]
        worst = int(sorted_res.argmax())
        # Two halves: [0 .. worst-1] and [worst+1 .. end] (drop the bad hit).
        left = sorted_idx[:worst]
        right = sorted_idx[worst + 1 :]

        keep_left = left.size >= min_hits_after_split
        keep_right = right.size >= min_hits_after_split
        if not keep_left and not keep_right:
            # Splitting cannot produce two viable halves: keep original.
            out.append(_with_chi2(t, chi2))
            stats.n_kept += 1
            continue

        c_left = c_right = float("inf")
        if keep_left:
            c_left, _ = _fit(hit_xyz[left])
        if keep_right:
            c_right, _ = _fit(hit_xyz[right])

        left_ok = keep_left and c_left <= chi2_threshold
        right_ok = keep_right and c_right <= chi2_threshold

        if left_ok and right_ok:
            out.append(_make_track(left, t, c_left))
            out.append(_make_track(right, t, c_right))
            stats.n_split_succeeded += 1
        elif left_ok:
            out.append(_make_track(left, t, c_left))
            stats.n_one_half_kept += 1
        elif right_ok:
            out.append(_make_track(right, t, c_right))
            stats.n_one_half_kept += 1
        else:
            # Splitting made nothing pass; keep the original (it was the GNN's call).
            out.append(_with_chi2(t, chi2))
            stats.n_kept += 1

    return out, stats


def _with_chi2(t: Track, chi2: float) -> Track:
    extras = dict(t.extras)
    extras["chi2_per_dof"] = float(chi2)
    return Track(hit_indices=t.hit_indices, score=t.score, extras=extras)


def _make_track(idx: npt.NDArray[np.int64], parent: Track, chi2: float) -> Track:
    extras = dict(parent.extras)
    extras["chi2_per_dof"] = float(chi2)
    extras["arbitrated"] = True
    return Track(hit_indices=np.asarray(sorted(idx.tolist()), dtype=np.int64),
                 score=parent.score, extras=extras)


__all__ = ["ArbitrateStats", "arbitrate_tracks"]
