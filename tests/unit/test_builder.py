"""Tests for tracking/builder.py."""

from __future__ import annotations

import torch

from neurotrack.tracking.builder import Track, build_tracks


class TestBuildTracks:
    def test_basic_two_components(self) -> None:
        # 6 hits, edges forming two chains.
        # Chain A: 0-1, 1-2  (high scores)
        # Chain B: 3-4, 4-5  (high scores)
        # Cross edge 2-3: low score, should be dropped.
        edge_index = torch.tensor(
            [
                [0, 1, 3, 4, 2],
                [1, 2, 4, 5, 3],
            ],
            dtype=torch.long,
        )
        edge_score = torch.tensor([0.95, 0.9, 0.95, 0.9, 0.1])
        tracks = build_tracks(edge_index, edge_score, n_hits=6, threshold=0.5, min_hits=3)
        assert len(tracks) == 2
        ids = [set(t.hit_indices.tolist()) for t in tracks]
        assert {0, 1, 2} in ids
        assert {3, 4, 5} in ids

    def test_min_hits_filter(self) -> None:
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        edge_score = torch.tensor([0.9, 0.9])
        tracks = build_tracks(edge_index, edge_score, n_hits=4, threshold=0.5, min_hits=3)
        # Two components of size 2 each -> all filtered.
        assert len(tracks) == 0

    def test_score_threshold_drops_everything(self) -> None:
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_score = torch.tensor([0.3])
        tracks = build_tracks(edge_index, edge_score, n_hits=3, threshold=0.7, min_hits=2)
        assert tracks == []

    def test_track_has_mean_score(self) -> None:
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_score = torch.tensor([0.8, 0.9, 0.7])
        tracks = build_tracks(edge_index, edge_score, n_hits=3, threshold=0.5, min_hits=2)
        assert len(tracks) == 1
        assert 0.5 < tracks[0].score < 1.0

    def test_track_dataclass_default_extras(self) -> None:
        t = Track(hit_indices=torch.tensor([0, 1, 2]).numpy(), score=0.9)
        assert t.extras == {}
