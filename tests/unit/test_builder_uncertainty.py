"""Tests for tracking/builder_uncertainty.py."""

from __future__ import annotations

import torch

from neurotrack.tracking.builder_uncertainty import build_tracks_uncertainty


def _ei(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class TestBuilderUncertainty:
    def test_single_chain_all_high_scores(self) -> None:
        # 4-hit chain, all edges >= 0.95 -> 1 track of size 4.
        ei = _ei([(0, 1), (1, 2), (2, 3), (1, 0), (2, 1), (3, 2)])
        sc = torch.tensor([0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
        tracks = build_tracks_uncertainty(ei, sc, n_hits=4, min_hits=3)
        assert len(tracks) == 1
        assert set(tracks[0].hit_indices.tolist()) == {0, 1, 2, 3}

    def test_chain_protection_re_binds_split_chain(self) -> None:
        # Two halves [0,1,2] and [3,4,5] each connected with score 0.95.
        # Bridge edge (2, 3) at score 0.45 (chain-protection zone).
        # Expected: re-bound into single 6-hit track because both halves
        # are size >= 2 chains.
        ei = _ei(
            [
                (0, 1), (1, 0),
                (1, 2), (2, 1),
                (3, 4), (4, 3),
                (4, 5), (5, 4),
                (2, 3), (3, 2),  # the borderline bridge
            ],
        )
        sc = torch.tensor(
            [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.45, 0.45],
        )
        tracks = build_tracks_uncertainty(ei, sc, n_hits=6, min_hits=3)
        # Chain protection should re-bind into a single track.
        assert len(tracks) == 1
        assert set(tracks[0].hit_indices.tolist()) == {0, 1, 2, 3, 4, 5}

    def test_low_score_alone_does_not_form_track(self) -> None:
        # Two isolated hits with only a single 0.45 edge -- no chain on either side.
        # Chain protection blocks the merge, so 0 tracks of size >= min_hits.
        ei = _ei([(0, 1), (1, 0)])
        sc = torch.tensor([0.45, 0.45])
        tracks = build_tracks_uncertainty(ei, sc, n_hits=2, min_hits=2)
        assert tracks == []

    def test_two_disconnected_chains(self) -> None:
        # [0,1,2] and [10,11,12] -- never bridged.
        ei = _ei(
            [
                (0, 1), (1, 0),
                (1, 2), (2, 1),
                (10, 11), (11, 10),
                (11, 12), (12, 11),
            ],
        )
        sc = torch.tensor([0.95] * 8)
        tracks = build_tracks_uncertainty(ei, sc, n_hits=13, min_hits=3)
        assert len(tracks) == 2
        sets = {frozenset(t.hit_indices.tolist()) for t in tracks}
        assert frozenset({0, 1, 2}) in sets
        assert frozenset({10, 11, 12}) in sets

    def test_all_below_hard_threshold(self) -> None:
        ei = _ei([(0, 1), (1, 0), (1, 2), (2, 1)])
        sc = torch.tensor([0.20, 0.20, 0.20, 0.20])
        tracks = build_tracks_uncertainty(ei, sc, n_hits=3, min_hits=2)
        assert tracks == []

    def test_min_hits_filter(self) -> None:
        # 2-hit chain -- below min_hits=3.
        ei = _ei([(0, 1), (1, 0)])
        sc = torch.tensor([0.95, 0.95])
        tracks = build_tracks_uncertainty(ei, sc, n_hits=2, min_hits=3)
        assert tracks == []

    def test_track_extras_populated(self) -> None:
        ei = _ei([(0, 1), (1, 0), (1, 2), (2, 1)])
        sc = torch.tensor([0.95, 0.95, 0.80, 0.80])
        tracks = build_tracks_uncertainty(ei, sc, n_hits=3, min_hits=3)
        assert len(tracks) == 1
        e = tracks[0].extras
        assert e["size"] == 3
        assert abs(e["min_edge_score"] - 0.80) < 1e-6
        assert abs(e["max_edge_score"] - 0.95) < 1e-6

    def test_chain_protection_rejects_isolated_endpoint(self) -> None:
        # Chain [0,1,2] plus an isolated hit 3 connected to hit 2 only via
        # a low-confidence edge (0.45).  Hit 3 has NO high-confidence
        # incident edge (max_inc[3] = 0.45), so chain protection rejects
        # the bridge -- we don't want to extend a chain by attaching an
        # isolated weakly-connected hit.
        ei = _ei(
            [
                (0, 1), (1, 0),
                (1, 2), (2, 1),
                (2, 3), (3, 2),
            ],
        )
        sc = torch.tensor([0.95, 0.95, 0.95, 0.95, 0.45, 0.45])
        tracks = build_tracks_uncertainty(ei, sc, n_hits=4, min_hits=3)
        # Only the {0,1,2} chain survives; hit 3 is excluded.
        assert len(tracks) == 1
        assert set(tracks[0].hit_indices.tolist()) == {0, 1, 2}
