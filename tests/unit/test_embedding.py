"""Tests for the HitEmbedNet MLP."""

from __future__ import annotations

import torch

from neurotrack.models.embedding import HitEmbedNet


class TestHitEmbedNet:
    def test_shape_and_l2_norm(self) -> None:
        model = HitEmbedNet(in_dim=24, hidden_dim=64, out_dim=12, num_layers=4)
        x = torch.randn(50, 24)
        out = model(x)
        assert out.shape == (50, 12)
        norms = torch.linalg.norm(out, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_no_l2_norm_optional(self) -> None:
        model = HitEmbedNet(l2_normalise=False)
        x = torch.randn(10, 24)
        out = model(x)
        norms = torch.linalg.norm(out, dim=1)
        # Unconstrained -- in general not unit length.
        assert not torch.allclose(norms, torch.ones_like(norms))

    def test_grad_flows(self) -> None:
        model = HitEmbedNet()
        x = torch.randn(8, 24, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
