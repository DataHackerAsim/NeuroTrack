"""Tests for losses: hinge embedding mining + focal BCE."""

from __future__ import annotations

import torch

from neurotrack.models.losses import (
    focal_bce_with_logits,
    hinge_embedding_loss,
    mine_pairs,
)


class TestMinePairs:
    def test_positives_share_particle_negatives_dont(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(20, 8)
        pids = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        a, p, n = mine_pairs(
            emb, pids, n_anchors=10, n_pos_per_anchor=2, n_neg_per_anchor=4,
        )
        if a.numel() == 0:
            return
        for ai, pi, ni in zip(a.tolist(), p.tolist(), n.tolist(), strict=True):
            assert pids[ai].item() == pids[pi].item()
            assert pids[ai].item() != pids[ni].item() or pids[ni].item() == 0

    def test_drops_noise_anchors(self) -> None:
        emb = torch.randn(10, 4)
        pids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        a, _p, _n = mine_pairs(emb, pids, n_anchors=5)
        for ai in a.tolist():
            assert pids[ai].item() != 0


class TestHingeLoss:
    def test_positive_finite_grad(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(30, 8, requires_grad=True)
        pids = torch.randint(1, 4, (30,))
        loss, info = hinge_embedding_loss(emb, pids, margin=0.4)
        if loss.requires_grad:
            loss.backward()
            assert emb.grad is not None
            assert torch.isfinite(emb.grad).all()
        assert info["n_pairs"] >= 0


class TestFocalBCE:
    def test_zero_loss_on_perfect_predictions(self) -> None:
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = focal_bce_with_logits(logits, labels)
        assert float(loss.item()) < 1e-3

    def test_high_loss_on_inverted_predictions(self) -> None:
        logits = torch.tensor([-10.0, 10.0])
        labels = torch.tensor([1.0, 0.0])
        loss = focal_bce_with_logits(logits, labels)
        # Focal exponent (1-p)^gamma is ~1 when prediction is wrong; loss should
        # be roughly cross-entropy * alpha.
        assert float(loss.item()) > 1.0
