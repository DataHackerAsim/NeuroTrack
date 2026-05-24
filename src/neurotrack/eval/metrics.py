"""Evaluation metrics: TrackML score, efficiency, fake rate, edge AUC.

The TrackML score follows the official competition definition.  For each
predicted track we look at the dominant particle (the one whose hits make
up the majority of the track); the track is *credited* with a hit's
weight iff **both** of these majority conditions hold:

* the predicted track contains > 50 % of the hits of that particle, AND
* > 50 % of the hits of the predicted track belong to that particle.

The score is the sum of credited weights divided by the total weight of
all hits in the event.  Noise hits (``particle_id == 0``) carry weight 0
in TrackML, so they cannot contribute.

See: https://arxiv.org/abs/1904.06778 (Amrouche et al., 2019).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


# ---------------------------------------------------------------------------
# Embedding -- recall @ k
# ---------------------------------------------------------------------------
def recall_at_k(
    emb: torch.Tensor,
    particle_ids: torch.Tensor,
    k: int = 10,
    *,
    drop_noise: bool = True,
) -> float:
    """For each non-noise hit, fraction whose top-k neighbours include at
    least one hit from the same particle.
    """
    n = emb.shape[0]
    if n < 2:
        return 0.0
    pids = particle_ids.long()
    mask = pids > 0 if drop_noise else torch.ones_like(pids, dtype=torch.bool)
    if not mask.any():
        return 0.0

    d = torch.cdist(emb.float(), emb.float())
    d.fill_diagonal_(float("inf"))
    nn_idx = torch.topk(d, k=min(k, n - 1), largest=False).indices  # (N, k)
    nn_pids = pids[nn_idx]                                    # (N, k)

    has_match = (nn_pids == pids.unsqueeze(1)).any(dim=1)
    return float(has_match[mask].float().mean().item())


# ---------------------------------------------------------------------------
# GNN edge AUC
# ---------------------------------------------------------------------------
def edge_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary AUC via the Mann-Whitney U formula (torch-only, no sklearn).

    Returns 0.5 if either class is missing (degenerate case).
    """
    y = labels.detach().to(torch.int64).cpu()
    s = scores.detach().to(torch.float64).cpu()
    pos_mask = y == 1
    n_pos = int(pos_mask.sum().item())
    n_neg = int((~pos_mask).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks of the positive class (handle ties by mean rank).
    order = torch.argsort(s)
    s_sorted = s[order]
    y_sorted = pos_mask[order].to(torch.float64)
    # Compute average ranks for ties.
    ranks = torch.arange(1, s.numel() + 1, dtype=torch.float64)
    # For runs of equal scores, replace their ranks with the mean.
    i = 0
    n = s.numel()
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        if j > i + 1:
            mean_rank = ranks[i:j].mean()
            ranks[i:j] = mean_rank
        i = j
    sum_ranks_pos = float((ranks * y_sorted).sum().item())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# TrackML score
# ---------------------------------------------------------------------------
def trackml_score(
    predicted_tracks: list[npt.NDArray[Any]],
    particle_ids: npt.NDArray[Any],
    weights: npt.NDArray[Any],
) -> float:
    """Official double-majority TrackML score.

    Parameters
    ----------
    predicted_tracks
        List of arrays; each array contains the hit indices of one
        predicted track.
    particle_ids
        Per-hit truth particle ids (``0`` = noise; weight 0).
    weights
        Per-hit weights.
    """
    pids = np.asarray(particle_ids).astype(np.int64)
    w = np.asarray(weights).astype(np.float64)
    total_weight = w[pids > 0].sum()
    if total_weight <= 0:
        return 0.0

    # Total hits per truth particle.
    truth_counts: Counter[int] = Counter(pids[pids > 0].tolist())

    credited = 0.0
    for track in predicted_tracks:
        if len(track) == 0:
            continue
        track_pids = pids[track]
        track_w = w[track]
        # Dominant particle in the track (excluding noise).
        non_noise = track_pids > 0
        if not non_noise.any():
            continue
        c = Counter(track_pids[non_noise].tolist())
        dom_pid, dom_cnt = c.most_common(1)[0]

        # Condition 1: dominant particle is > 50 % of the track.
        if dom_cnt * 2 <= len(track):
            continue
        # Condition 2: track holds > 50 % of the dominant particle's hits.
        if dom_cnt * 2 <= truth_counts[dom_pid]:
            continue

        # Credit: sum the weights of hits in the track that come from the
        # dominant particle.
        match_mask = (track_pids == dom_pid)
        credited += float(track_w[match_mask].sum())
    return float(credited / total_weight)


# ---------------------------------------------------------------------------
# Track-level efficiency / fake / duplicate
# ---------------------------------------------------------------------------
def track_efficiency(
    predicted_tracks: list[npt.NDArray[Any]],
    particle_ids: npt.NDArray[Any],
    *,
    min_hits: int = 3,
    purity: float = 0.5,
) -> dict[str, float]:
    """Coarse physics-style efficiency / fake-rate / duplicate metrics.

    A predicted track is *matched* to a truth particle if at least
    ``purity`` of its hits come from that particle and at least
    ``min_hits`` of that particle's hits are in the track.

    Returns a dict with::

        n_truth_particles      -- unique non-noise particles in the event
        n_tracks_predicted     -- number of predicted tracks
        n_matched              -- predicted tracks that match a truth particle
        n_truth_matched        -- distinct truth particles that got matched
        efficiency             -- n_truth_matched / n_truth_particles
        fake_rate              -- 1 - (n_matched / n_tracks_predicted)
        duplicate_rate         -- (n_matched - n_truth_matched) / n_matched
    """
    pids = np.asarray(particle_ids).astype(np.int64)
    truth_pids = np.unique(pids[pids > 0])
    truth_counts: Counter[int] = Counter(pids[pids > 0].tolist())

    n_truth = int(truth_pids.size)
    n_pred = len(predicted_tracks)
    matched: list[int] = []
    matched_to_truth: set[int] = set()
    for track in predicted_tracks:
        if len(track) < min_hits:
            continue
        tp = pids[track]
        c = Counter(tp[tp > 0].tolist())
        if not c:
            continue
        dom_pid, dom_cnt = c.most_common(1)[0]
        if dom_cnt < min_hits:
            continue
        if dom_cnt / len(track) < purity:
            continue
        # double-majority: also need purity on the truth side.
        if dom_cnt / max(1, truth_counts[dom_pid]) < 0.5:
            continue
        matched.append(dom_pid)
        matched_to_truth.add(dom_pid)

    n_matched = len(matched)
    n_truth_matched = len(matched_to_truth)
    eff = (n_truth_matched / n_truth) if n_truth else 0.0
    fake = (1.0 - n_matched / n_pred) if n_pred else 0.0
    dup = ((n_matched - n_truth_matched) / n_matched) if n_matched else 0.0
    return {
        "n_truth_particles": float(n_truth),
        "n_tracks_predicted": float(n_pred),
        "n_matched": float(n_matched),
        "n_truth_matched": float(n_truth_matched),
        "efficiency": float(eff),
        "fake_rate": float(fake),
        "duplicate_rate": float(dup),
    }


__all__ = [
    "edge_auc",
    "recall_at_k",
    "track_efficiency",
    "trackml_score",
]
