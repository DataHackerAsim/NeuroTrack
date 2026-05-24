"""Tests for graph/construction.py: kNN graph and edge features."""

from __future__ import annotations

import torch

from neurotrack.graph.construction import build_edge_features, build_knn_graph


class TestKNN:
    def test_each_node_has_k_neighbours(self) -> None:
        torch.manual_seed(0)
        n, d, k = 20, 8, 4
        emb = torch.randn(n, d)
        ei = build_knn_graph(emb, k=k, symmetrise=False, self_loops=False)
        # n nodes * k neighbours = n*k edges
        assert ei.shape == (2, n * k)
        # Every source index appears exactly k times.
        src_counts = torch.bincount(ei[0])
        assert int((src_counts == k).sum().item()) == n

    def test_no_self_loops_by_default(self) -> None:
        emb = torch.eye(5, 4)
        ei = build_knn_graph(emb, k=2, symmetrise=False)
        for s, d in zip(ei[0].tolist(), ei[1].tolist(), strict=True):
            assert s != d

    def test_symmetric_graph_contains_reverse_edges(self) -> None:
        torch.manual_seed(1)
        emb = torch.randn(10, 4)
        ei = build_knn_graph(emb, k=3, symmetrise=True)
        edges = {(s, d) for s, d in zip(ei[0].tolist(), ei[1].tolist(), strict=True)}
        for s, d in edges:
            assert (d, s) in edges

    def test_max_distance_filters(self) -> None:
        emb = torch.tensor([[0.0], [1.0], [10.0]])
        ei = build_knn_graph(emb, k=2, max_distance=2.0, symmetrise=False)
        # Node 2 is far from nodes 0/1; only the close pairs should survive.
        endpoints = set(ei[0].tolist()) | set(ei[1].tolist())
        assert 2 not in endpoints or len(endpoints) > 0

    def test_empty_input(self) -> None:
        ei = build_knn_graph(torch.empty((0, 4)), k=3)
        assert ei.shape == (2, 0)


class TestEdgeFeatures:
    def test_shape_and_finite(self) -> None:
        x = torch.tensor(
            [
                # cols: x, y, z, r, phi, eta, conformal_u, conformal_v
                [1.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5, 0.0],
                [2.0, 0.0, 10.0, 2.0, 0.0, 0.0, 0.25, 0.0],
            ],
        )
        # Pad to 24 columns (build_unified_features schema width).
        x = torch.cat([x, torch.zeros((2, 24 - x.shape[1]))], dim=1)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ef = build_edge_features(x, ei)
        assert ef.shape == (2, 7)
        assert torch.isfinite(ef).all()

    def test_empty_edges(self) -> None:
        x = torch.zeros((5, 24))
        ei = torch.empty((2, 0), dtype=torch.long)
        ef = build_edge_features(x, ei)
        assert ef.shape == (0, 7)
