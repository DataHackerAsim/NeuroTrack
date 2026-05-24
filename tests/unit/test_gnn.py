"""Tests for the InteractionNetwork edge classifier."""

from __future__ import annotations

import torch

from neurotrack.models.gnn import InteractionNetwork


class TestInteractionNetwork:
    def test_forward_shape(self) -> None:
        gnn = InteractionNetwork(
            node_dim=24, edge_dim=7, hidden_dim=16, num_iter=2, use_checkpoint=False,
        )
        n_nodes = 20
        n_edges = 50
        x = torch.randn(n_nodes, 24)
        ei = torch.randint(0, n_nodes, (2, n_edges))
        ea = torch.randn(n_edges, 7)
        out = gnn(x, ei, ea)
        assert out.shape == (n_edges,)

    def test_empty_edges(self) -> None:
        gnn = InteractionNetwork(hidden_dim=16, num_iter=2)
        x = torch.randn(5, 24)
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, 7))
        out = gnn(x, ei, ea)
        assert out.shape == (0,)

    def test_grad_flows(self) -> None:
        gnn = InteractionNetwork(hidden_dim=16, num_iter=2, use_checkpoint=False)
        x = torch.randn(10, 24, requires_grad=True)
        ei = torch.randint(0, 10, (2, 30))
        ea = torch.randn(30, 7, requires_grad=True)
        out = gnn(x, ei, ea)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert ea.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(ea.grad).all()

    def test_checkpoint_path_runs(self) -> None:
        gnn = InteractionNetwork(hidden_dim=8, num_iter=3, use_checkpoint=True)
        gnn.train()
        x = torch.randn(6, 24, requires_grad=True)
        ei = torch.randint(0, 6, (2, 12))
        ea = torch.randn(12, 7)
        out = gnn(x, ei, ea)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
