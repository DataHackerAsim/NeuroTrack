"""Tests for tracking/kalman_postfit.py and tracking/arbitrate.py."""

from __future__ import annotations

import math

import numpy as np

from neurotrack.tracking.arbitrate import arbitrate_tracks
from neurotrack.tracking.builder import Track
from neurotrack.tracking.kalman_postfit import fit_helix_chi2


def _perfect_helix(n: int, R: float = 50.0, pitch: float = 5.0) -> np.ndarray:
    """Build a perfect 3-D helix of ``n`` points with center at the origin."""
    ts = np.linspace(0.0, math.pi * 1.5, n)
    x = R * np.cos(ts)
    y = R * np.sin(ts)
    z = pitch * ts
    return np.column_stack([x, y, z])


class TestKalmanFit:
    def test_perfect_helix_low_chi2(self) -> None:
        xyz = _perfect_helix(n=12, R=50.0, pitch=5.0)
        helix, chi2, res = fit_helix_chi2(xyz, sigma_mm=0.3)
        assert chi2 < 0.1, f"chi2_per_dof={chi2}"
        assert abs(helix.R - 50.0) < 0.5
        assert np.all(res < 0.1)

    def test_helix_with_outlier(self) -> None:
        xyz = _perfect_helix(n=10)
        # Push one hit far off the helix.
        xyz[5] += np.array([15.0, -10.0, 8.0])
        _, chi2, res = fit_helix_chi2(xyz)
        # chi2 should be much larger than the no-outlier case.
        assert chi2 > 10.0
        # The outlier should have the largest residual.
        assert int(res.argmax()) == 5

    def test_colinear_points_fit_does_not_crash(self) -> None:
        # 5 strictly-collinear points along +x: Kasa's algebraic circle fit
        # is rank-deficient here.  We just require the call to return finite
        # results (no NaN/inf) rather than asserting small chi2 -- in
        # production the line-on-a-helix edge case is rare because real
        # tracks always have some curvature in (xy).
        xyz = np.column_stack(
            [np.linspace(0, 200, 5), np.zeros(5), np.linspace(0, 100, 5)],
        )
        helix, chi2, res = fit_helix_chi2(xyz)
        assert math.isfinite(chi2)
        assert math.isfinite(helix.R)
        assert np.isfinite(res).all()

    def test_too_few_hits_returns_inf_chi2(self) -> None:
        xyz = np.zeros((2, 3))
        _, chi2, _ = fit_helix_chi2(xyz)
        assert math.isinf(chi2)


class TestArbitrate:
    def test_clean_tracks_passthrough(self) -> None:
        xyz = _perfect_helix(n=10)
        tracks = [Track(hit_indices=np.arange(10), score=0.95)]
        out, stats = arbitrate_tracks(tracks, xyz, chi2_threshold=3.0)
        assert len(out) == 1
        assert stats.n_kept == 1
        assert stats.n_split_attempted == 0
        assert out[0].extras.get("chi2_per_dof") is not None

    def test_outlier_track_gets_split_or_pruned(self) -> None:
        helix = _perfect_helix(n=12, R=50.0, pitch=5.0)
        # Concatenate the helix with 5 random "outlier" points.
        outliers = np.random.default_rng(0).uniform(-200, 200, (5, 3))
        xyz = np.vstack([helix, outliers])
        # All 17 hits start in one track.
        tracks = [Track(hit_indices=np.arange(17), score=0.5)]
        out, stats = arbitrate_tracks(tracks, xyz, chi2_threshold=3.0,
                                       min_hits_after_split=3)
        # At minimum, the outcome differs from the input: either split into
        # two halves or one half kept; never silently passed.
        assert stats.n_split_attempted == 1
        # One of: succeeded, one_half_kept, or fell-through kept.
        assert (stats.n_split_succeeded + stats.n_one_half_kept + stats.n_kept) >= 1
        # If split succeeded, we should have 2 output tracks.
        if stats.n_split_succeeded == 1:
            assert len(out) == 2

    def test_short_tracks_dropped(self) -> None:
        xyz = _perfect_helix(n=10)
        tracks = [Track(hit_indices=np.array([0, 1]), score=0.9)]
        out, stats = arbitrate_tracks(tracks, xyz, min_hits_after_split=3)
        assert out == []
        assert stats.n_dropped == 1
