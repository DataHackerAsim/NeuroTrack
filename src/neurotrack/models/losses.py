"""Loss functions for the embedding and GNN-edge-classifier stages.

* :func:`hinge_embedding_loss` -- pairwise margin loss with semi-hard
  negative mining for metric learning.  Pulls hits of the same particle
  together (squared distance) and pushes hits of different particles
  apart by at least ``margin`` (margin-squared on the negative side).
* :func:`focal_bce_with_logits` -- focal binary cross-entropy for the
  heavy class imbalance in candidate-edge classification (truth edges
  are 1-5 % of kNN edges in a dense event).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _sample_indices(
    n: int, max_samples: int, generator: torch.Generator | None = None,
) -> Tensor:
    if n <= max_samples:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:max_samples]


def mine_pairs(
    emb: Tensor,
    particle_ids: Tensor,
    *,
    n_anchors: int = 256,
    n_pos_per_anchor: int = 4,
    n_neg_per_anchor: int = 16,
    hard_neg_ratio: float = 0.5,
    drop_noise: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample anchor / positive / negative index pairs from one event.

    Returns three index tensors -- ``(anchor_idx, pos_idx, neg_idx)`` --
    each of shape ``(n_anchors * n_per_anchor,)``.  Positive pairs share
    the anchor's ``particle_id``; negative pairs do not.

    For ``hard_neg_ratio`` of the negative slots, we pick the in-batch
    negative with the smallest embedding distance to the anchor (= hardest
    negative).  The remaining slots are random.
    """
    device = emb.device
    n = emb.shape[0]
    pids = particle_ids.long()

    # Eligible anchors: non-noise and with at least one other same-particle hit.
    eligible_mask = (
        pids > 0 if drop_noise else torch.ones_like(pids, dtype=torch.bool)
    )
    # Count per-particle occurrences to ensure at least 2 hits.
    unique, counts = torch.unique(pids[eligible_mask], return_counts=True)
    multi = unique[counts >= 2]
    multi_mask = torch.isin(pids, multi) & eligible_mask
    anchor_pool = torch.nonzero(multi_mask, as_tuple=False).squeeze(-1)
    if anchor_pool.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty

    anchors = anchor_pool[_sample_indices(anchor_pool.numel(), n_anchors, generator).to(device)]

    # Build a (M,) -> (M, K) sampling of positives + negatives per anchor.
    pos_idx_list: list[Tensor] = []
    neg_idx_list: list[Tensor] = []
    anchor_rep_list: list[Tensor] = []

    # Pre-compute pairwise distances once (M anchors by N hits) for hard-neg mining.
    a_emb = emb[anchors]  # (M, D)
    all_dists = torch.cdist(a_emb, emb)  # (M, N)
    # For each anchor: distances to all hits.

    # Random positives.
    for i, a_idx in enumerate(anchors.tolist()):
        a_pid = int(pids[a_idx].item())
        same = torch.nonzero((pids == a_pid) & (torch.arange(n, device=device) != a_idx),
                             as_tuple=False).squeeze(-1)
        if same.numel() == 0:
            continue
        pos_pick = same[_sample_indices(same.numel(), n_pos_per_anchor, generator).to(device)]

        diff_mask = pids != a_pid
        if drop_noise:
            # noise hits are still valid negatives -- they exercise the model
            # to push noise away from real tracks.
            pass
        diff_idx = torch.nonzero(diff_mask, as_tuple=False).squeeze(-1)
        if diff_idx.numel() == 0:
            continue

        n_hard = int(n_neg_per_anchor * hard_neg_ratio)
        n_rand = n_neg_per_anchor - n_hard

        d_diff = all_dists[i, diff_idx]
        if n_hard > 0:
            hard_picks = diff_idx[torch.topk(
                d_diff, min(n_hard, diff_idx.numel()), largest=False,
            ).indices]
        else:
            hard_picks = diff_idx.new_empty((0,))
        if n_rand > 0 and diff_idx.numel() > n_hard:
            rand_pool = diff_idx
            rand_picks = rand_pool[_sample_indices(
                rand_pool.numel(), n_rand, generator,
            ).to(device)]
        else:
            rand_picks = diff_idx.new_empty((0,))
        neg_picks = torch.cat([hard_picks, rand_picks])
        if neg_picks.numel() == 0:
            continue

        k = min(pos_pick.numel(), neg_picks.numel())
        pos_pick = pos_pick[:k]
        neg_picks = neg_picks[:k]
        pos_idx_list.append(pos_pick)
        neg_idx_list.append(neg_picks)
        anchor_rep_list.append(torch.full((k,), a_idx, dtype=torch.long, device=device))

    if not anchor_rep_list:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty
    return (
        torch.cat(anchor_rep_list),
        torch.cat(pos_idx_list),
        torch.cat(neg_idx_list),
    )


def hinge_embedding_loss(
    emb: Tensor,
    particle_ids: Tensor,
    *,
    margin: float = 0.4,
    n_anchors: int = 256,
    n_pos_per_anchor: int = 4,
    n_neg_per_anchor: int = 16,
    hard_neg_ratio: float = 0.5,
    drop_noise: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Squared hinge metric-learning loss with hard-negative mining.

    Returns ``(loss, info)`` where info is a dict of diagnostics
    (positive/negative distances, fraction of active negatives) -- the
    trainer logs these.
    """
    a, p, n = mine_pairs(
        emb, particle_ids,
        n_anchors=n_anchors,
        n_pos_per_anchor=n_pos_per_anchor,
        n_neg_per_anchor=n_neg_per_anchor,
        hard_neg_ratio=hard_neg_ratio,
        drop_noise=drop_noise,
        generator=generator,
    )
    if a.numel() == 0:
        zero = emb.new_zeros((), requires_grad=True)
        return zero, {"n_pairs": 0.0, "d_pos": 0.0, "d_neg": 0.0, "active_frac": 0.0}

    d_pos = torch.linalg.norm(emb[a] - emb[p], dim=1)
    d_neg = torch.linalg.norm(emb[a] - emb[n], dim=1)
    loss_pos = (d_pos ** 2).mean()
    margin_violation = torch.clamp(margin - d_neg, min=0.0)
    loss_neg = (margin_violation ** 2).mean()
    loss = loss_pos + loss_neg

    info = {
        "n_pairs": float(a.numel()),
        "d_pos": float(d_pos.mean().item()),
        "d_neg": float(d_neg.mean().item()),
        "active_frac": float((margin_violation > 0).float().mean().item()),
    }
    return loss, info


def focal_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Focal binary cross-entropy on logits.

    ``alpha`` re-weights the positive class; ``gamma`` is the focusing
    exponent that down-weights easy examples.  Falls back to plain BCE
    when ``gamma == 0``.
    """
    if logits.shape != labels.shape:
        raise ValueError(f"shape mismatch: logits {logits.shape} vs labels {labels.shape}")
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none",
        pos_weight=(
            torch.tensor(pos_weight, device=logits.device)
            if pos_weight is not None else None
        ),
    )
    p_t = p * labels + (1 - p) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    focal: Tensor = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "mean":
        out_mean: Tensor = focal.mean()
        return out_mean
    if reduction == "sum":
        out_sum: Tensor = focal.sum()
        return out_sum
    return focal


__all__ = [
    "focal_bce_with_logits",
    "hinge_embedding_loss",
    "mine_pairs",
]
