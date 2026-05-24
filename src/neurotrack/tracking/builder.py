"""Track building: thresholded edges -> connected components.

Given the GNN's per-edge scores, we keep only those above a threshold
(default ``0.7``), treat the survivors as an undirected graph, and
extract connected components.  Each component with ``>= min_hits`` is
emitted as a predicted track; smaller ones are discarded as noise.

The implementation uses :func:`scipy.sparse.csgraph.connected_components`
which is O((N + E) alpha(N)) on the disjoint-set side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Track:
    """One predicted track.

    Attributes
    ----------
    hit_indices
        Indices into the event's ``x`` tensor of the hits assigned to this
        track.  Sorted ascending.
    score
        Mean edge score of the surviving edges that fall inside this track.
    """

    hit_indices: npt.NDArray[np.int64]
    score: float
    extras: dict[str, Any] = field(default_factory=dict)


def build_tracks(
    edge_index: torch.Tensor,
    edge_score: torch.Tensor,
    n_hits: int,
    *,
    threshold: float = 0.7,
    min_hits: int = 3,
) -> list[Track]:
    """Threshold edges, find connected components, return predicted tracks.

    Parameters
    ----------
    edge_index
        ``(2, E)`` candidate edges (typically the symmetric kNN graph).
    edge_score
        ``(E,)`` per-edge scores in [0, 1] (sigmoid of the GNN logits).
    n_hits
        Total number of hits in the event (used to size the adjacency).
    threshold
        Edges with score below this are dropped.
    min_hits
        Components with fewer hits than this are discarded as noise.
    """
    if edge_index.numel() == 0:
        return []
    src_t = edge_index[0].cpu().numpy().astype(np.int64)
    dst_t = edge_index[1].cpu().numpy().astype(np.int64)
    scores_np = edge_score.detach().cpu().numpy().astype(np.float32)

    keep = scores_np >= threshold
    if not keep.any():
        return []
    src = src_t[keep]
    dst = dst_t[keep]
    kept_scores = scores_np[keep]

    # Build symmetric sparse adjacency.
    data = np.ones_like(src, dtype=np.float32)
    adj = coo_matrix(
        (data, (src, dst)), shape=(n_hits, n_hits),
    ).tocsr()
    _, labels = connected_components(
        csgraph=adj, directed=False, return_labels=True,
    )

    tracks: list[Track] = []
    # Build a hit -> component map and a per-component mean score.
    comp_to_hits: dict[int, list[int]] = {}
    for hit, comp in enumerate(labels.tolist()):
        comp_to_hits.setdefault(comp, []).append(hit)
    # Per-edge component (each kept edge has src == dst component label).
    edge_comp = labels[src]
    score_sum: dict[int, float] = {}
    score_cnt: dict[int, int] = {}
    for c, s in zip(edge_comp.tolist(), kept_scores.tolist(), strict=True):
        score_sum[c] = score_sum.get(c, 0.0) + s
        score_cnt[c] = score_cnt.get(c, 0) + 1

    for comp, hits in comp_to_hits.items():
        if len(hits) < min_hits:
            continue
        mean_score = (
            score_sum.get(comp, 0.0) / max(1, score_cnt.get(comp, 1))
        )
        tracks.append(
            Track(
                hit_indices=np.array(sorted(hits), dtype=np.int64),
                score=float(mean_score),
            ),
        )
    return tracks


__all__ = ["Track", "build_tracks"]
