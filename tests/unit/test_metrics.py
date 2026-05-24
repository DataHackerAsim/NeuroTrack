"""Tests for eval/metrics.py: recall@k, edge AUC, TrackML score, efficiency."""

from __future__ import annotations

import numpy as np
import torch

from neurotrack.eval.metrics import (
    edge_auc,
    recall_at_k,
    track_efficiency,
    trackml_score,
)


class TestRecallAtK:
    def test_perfect_separation(self) -> None:
        # Hits of particle 1 at one end of the embedding, particle 2 at the other.
        emb = torch.tensor(
            [
                [10.0, 0.0], [10.1, 0.0], [10.2, 0.0],  # particle 1
                [-10.0, 0.0], [-10.1, 0.0], [-10.2, 0.0],  # particle 2
            ],
        )
        pids = torch.tensor([1, 1, 1, 2, 2, 2])
        r = recall_at_k(emb, pids, k=2)
        assert r == 1.0

    def test_no_separation(self) -> None:
        # Random emb -- some recall, but not necessarily 1.0.
        torch.manual_seed(0)
        emb = torch.randn(30, 6)
        pids = torch.zeros(30, dtype=torch.long)  # all noise -> drop_noise excludes everything.
        r = recall_at_k(emb, pids, k=5)
        assert r == 0.0


class TestEdgeAUC:
    def test_perfect_separation(self) -> None:
        scores = torch.tensor([0.1, 0.2, 0.9, 0.95])
        labels = torch.tensor([0, 0, 1, 1])
        assert edge_auc(scores, labels) == 1.0

    def test_inverse_predictions_zero_auc(self) -> None:
        scores = torch.tensor([0.9, 0.95, 0.1, 0.2])
        labels = torch.tensor([0, 0, 1, 1])
        assert edge_auc(scores, labels) == 0.0

    def test_degenerate_returns_half(self) -> None:
        scores = torch.tensor([0.5, 0.5, 0.5])
        labels = torch.tensor([1, 1, 1])
        assert edge_auc(scores, labels) == 0.5


class TestTrackMLScore:
    def test_perfect_reconstruction(self) -> None:
        # Two particles, 3 hits each.  Perfect tracks.
        pids = np.array([1, 1, 1, 2, 2, 2])
        w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        tracks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        score = trackml_score(tracks, pids, w)
        assert abs(score - 1.0) < 1e-6

    def test_zero_when_no_tracks(self) -> None:
        pids = np.array([1, 1, 2, 2])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert trackml_score([], pids, w) == 0.0

    def test_double_majority_required(self) -> None:
        # Particle 1: 4 hits.  Track contains 2 hits of particle 1 + 2 hits of
        # particle 2.  Track's dominant particle is tied; with strict >50% it fails.
        pids = np.array([1, 1, 1, 1, 2, 2])
        w = np.array([0.1] * 6)
        bad_track = np.array([0, 1, 4, 5])  # 50/50 split -> majority fails.
        assert trackml_score([bad_track], pids, w) == 0.0

    def test_partial_credit(self) -> None:
        # Particle 1 has 4 hits; track has 3 of them + 0 noise.  Should be credited.
        pids = np.array([1, 1, 1, 1, 2, 2])
        w = np.array([0.1] * 6)
        good = np.array([0, 1, 2])  # 3 of 4 -> dominant + double-majority
        # The score is sum of weights of the 3 hits credited / total weight = 0.3/0.6 = 0.5
        s = trackml_score([good], pids, w)
        assert abs(s - 0.5) < 1e-6


class TestEfficiency:
    def test_perfect(self) -> None:
        pids = np.array([1, 1, 1, 2, 2, 2])
        tracks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        m = track_efficiency(tracks, pids, min_hits=3, purity=0.5)
        assert m["efficiency"] == 1.0
        assert m["fake_rate"] == 0.0
        assert m["duplicate_rate"] == 0.0

    def test_extra_fake_track(self) -> None:
        # One real track + one entirely-noise track.
        pids = np.array([1, 1, 1, 0, 0, 0])
        tracks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        m = track_efficiency(tracks, pids, min_hits=3, purity=0.5)
        assert m["efficiency"] == 1.0
        # Half the predicted tracks are fake.
        assert m["fake_rate"] == 0.5
