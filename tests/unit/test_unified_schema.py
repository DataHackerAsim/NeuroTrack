"""Unit tests for the unified geometric / volume-onehot feature schema."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from neurotrack.data.unified_schema import (
    NUM_VOLUMES,
    UNIFIED_FEATURES,
    Source,
    build_unified_features,
    cartesian_from_cylindrical,
    compute_conformal,
    compute_eta,
    cylindrical_from_cartesian,
    volume_onehot,
)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_cartesian_cylindrical_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(-100, 100, n)
        y = rng.uniform(-100, 100, n)
        z = rng.uniform(-100, 100, n)
        r, phi, z2 = cylindrical_from_cartesian(x, y, z)
        x2, y2, z3 = cartesian_from_cylindrical(r, phi, z2)
        np.testing.assert_allclose(x2, x, atol=1e-9)
        np.testing.assert_allclose(y2, y, atol=1e-9)
        np.testing.assert_allclose(z3, z, atol=1e-12)


class TestEta:
    def test_at_z_equals_zero_eta_is_zero(self) -> None:
        # Hits in the transverse plane (theta = pi/2) -> eta = 0.
        r = np.array([1.0, 5.0, 50.0])
        z = np.zeros_like(r)
        out = compute_eta(r, z)
        np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-9)

    def test_forward_hits_have_positive_eta(self) -> None:
        out = compute_eta(np.array([1.0]), np.array([10.0]))
        assert out[0] > 1.0  # forward, eta should be large positive

    def test_backward_hits_have_negative_eta(self) -> None:
        out = compute_eta(np.array([1.0]), np.array([-10.0]))
        assert out[0] < -1.0


class TestConformal:
    def test_helix_through_origin_maps_near_line(self) -> None:
        """A circle through the origin is parameterised by
            x = R sin(t),  y = R(1 - cos(t))
        which under (u, v) = (x/(x^2+y^2), y/(x^2+y^2)) is exactly the line
        v = 1/(2R).  Synthetic test: residuals should be tiny.
        """
        radius = 50.0
        t = np.linspace(0.05, math.pi - 0.05, 64)  # avoid the origin singularity
        x = radius * np.sin(t)
        y = radius * (1.0 - np.cos(t))
        _, v = compute_conformal(x, y)
        # v should be a constant 1/(2R) up to numerical noise.
        assert np.std(v) < 1e-8
        np.testing.assert_allclose(np.mean(v), 1.0 / (2 * radius), rtol=1e-9)

    def test_does_not_blow_up_at_origin(self) -> None:
        # Origin is clamped via epsilon -- no NaN/inf.
        u, v = compute_conformal(np.array([0.0]), np.array([0.0]))
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()


class TestVolumeOnehot:
    def test_shape_and_sum(self) -> None:
        vid = np.array([0, 7, 15, 18])
        out = volume_onehot(vid)
        assert out.shape == (4, NUM_VOLUMES)
        # Each row sums to exactly 1.0.
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4))

    def test_clip_to_last_bucket(self) -> None:
        vid = np.array([18])  # > NUM_VOLUMES - 1
        out = volume_onehot(vid)
        assert out[0, NUM_VOLUMES - 1] == 1.0
        assert out[0].sum() == 1.0


# ---------------------------------------------------------------------------
# build_unified_features
# ---------------------------------------------------------------------------
class TestBuildUnifiedFeatures:
    def test_redvid_path_produces_24_features(self) -> None:
        df = pl.DataFrame(
            {
                "event_id": [0, 0],
                "sub_detector_id": [0, 1],
                "sub_detector_type": ["a", "b"],
                "track_id": [0, 0],
                "track_type": ["t", "t"],
                "radial_const": [0.0, 0.0],
                "azimuthal_const": [0.0, 0.0],
                "pitch_const": [0.0, 0.0],
                "radial_coeff": [1.0, 1.0],
                "azimuthal_coeff": [0.0, 0.0],
                "pitch_coeff": [1.0, 1.0],
                "hit_id": [0, 1],
                "hit_r": [10.0, 20.0],
                "hit_theta": [0.5, 1.0],
                "hit_z": [5.0, 10.0],
            },
        )
        out = build_unified_features(df, Source.REDVID_HELICAL_5050)
        assert tuple(out.columns) == UNIFIED_FEATURES
        assert out.height == 2
        # No NaNs / infs in any feature.
        for c in out.columns:
            arr = out[c].to_numpy()
            assert np.isfinite(arr).all()

    def test_trackml_path_produces_24_features(self) -> None:
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [0.0, 0.0],
                "z": [3.0, -3.0],
                "volume_id": [7, 18],
                "vx": [0.0, 0.0], "vy": [0.0, 0.0], "vz": [0.0, 0.0],
                "px": [1.0, 1.0], "py": [0.0, 0.0], "pz": [0.0, 0.0],
                "q": [1, -1],
                "particle_id": [10, 20],
                "weight": [0.1, 0.1],
                "event_id": [0, 0],
            },
        )
        out = build_unified_features(df, Source.TRACKML_SMALL)
        assert tuple(out.columns) == UNIFIED_FEATURES
        assert out.height == 2
        # First row volume_id=7 -> onehot bucket 7
        assert out["volume_onehot_7"][0] == 1.0
        # Second row volume_id=18 clipped to last bucket 15
        assert out[f"volume_onehot_{NUM_VOLUMES - 1}"][1] == 1.0


def test_unified_features_count() -> None:
    """The literal list in the prompt enumerates 8 + 16 = 24 features."""
    assert len(UNIFIED_FEATURES) == 8 + NUM_VOLUMES
