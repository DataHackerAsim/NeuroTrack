"""Tests for graph/truth.py: truth edges and label lookup."""

from __future__ import annotations

import numpy as np
import torch

from neurotrack.graph.truth import build_truth_edges, edge_label_from_truth


class TestBuildTruthEdges:
    def test_two_particles_three_hits_each(self) -> None:
        # Particle 1: hits 0, 2, 4 at r = 1, 2, 3
        # Particle 2: hits 1, 3, 5 at r = 1.5, 2.5, 3.5
        pids = np.array([1, 2, 1, 2, 1, 2], dtype=np.int64)
        r = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ei = build_truth_edges(pids, r)
        # Each particle gives 2 chain edges, symmetrised -> 4 edges per particle = 8 total.
        assert ei.shape == (2, 8)
        # Check chain ordering for particle 1: 0 -> 2 -> 4 (and reverse).
        edges_p1 = {
            (s, d) for s, d in zip(ei[0].tolist(), ei[1].tolist(), strict=True)
            if s in (0, 2, 4) or d in (0, 2, 4)
        }
        assert (0, 2) in edges_p1
        assert (2, 4) in edges_p1
        assert (2, 0) in edges_p1
        assert (4, 2) in edges_p1

    def test_drop_noise_skips_particle_zero(self) -> None:
        pids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        r = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        ei = build_truth_edges(pids, r, drop_noise=True)
        # Only the particle-1 chain (2 fwd + 2 rev = 4 edges).
        assert ei.shape == (2, 4)
        # All edge endpoints come from indices {3, 4, 5}.
        endpoints = set(ei[0].tolist()) | set(ei[1].tolist())
        assert endpoints == {3, 4, 5}

    def test_single_hit_particle_skipped(self) -> None:
        pids = np.array([1, 2, 2, 2], dtype=np.int64)
        r = np.array([1.0, 1.0, 2.0, 3.0])
        ei = build_truth_edges(pids, r)
        # Particle 1 has only 1 hit -> skipped.  Particle 2 gives 2 fwd + 2 rev edges.
        assert ei.shape == (2, 4)
        assert set(ei[0].tolist()) | set(ei[1].tolist()) == {1, 2, 3}

    def test_empty_event(self) -> None:
        ei = build_truth_edges(np.array([], dtype=np.int64), np.array([]))
        assert ei.shape == (2, 0)


class TestEdgeLabel:
    def test_label_is_one_when_same_nonzero_particle(self) -> None:
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        pids = torch.tensor([1, 1, 2], dtype=torch.long)
        labels = edge_label_from_truth(ei, pids)
        # (0,1) same -> 1; (1,2) different -> 0; (2,0) different -> 0
        assert labels.tolist() == [1.0, 0.0, 0.0]

    def test_label_is_zero_for_noise_pair(self) -> None:
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        pids = torch.tensor([0, 0], dtype=torch.long)
        labels = edge_label_from_truth(ei, pids)
        assert labels.tolist() == [0.0, 0.0]
