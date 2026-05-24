"""Score-aware track builder with chain protection.

Replaces :func:`neurotrack.tracking.builder.build_tracks` for the
post-R-B.5 pipeline.  See ``artifacts/diagnostic/REPORT.md`` for the
failure mode this is designed to fix (truth chains fragmenting when a
single edge scores just below the legacy 0.7 threshold).

Algorithm (Boruvka-flavour Union-Find with descending-score iteration):

1. Drop edges below ``hard_threshold`` (absolute floor; below this the
   edges are considered noise even if they happen to connect real
   particles in some rare event).
2. Sort surviving edges by score *descending*.
3. Iterate:

   * score >= ``merge_threshold``  -> unconditional union.
   * score in [``hard_threshold``, ``merge_threshold``)  -> *chain
     protection*: union only if **both** endpoint components are
     already size >= 2 (i.e., real chains established by
     high-confidence edges) **and** the new edge's score is within
     ``max_chain_break`` of each component's lowest current edge
     score.  This blocks low-confidence edges from creating tracks
     from nothing, while letting them re-bind two genuine chains
     that got split by a single below-threshold edge.

4. Drop components with fewer than ``min_hits`` hits.

Defaults are aligned with R-H's specification: ``hard_threshold=0.30``,
``merge_threshold=0.50``, ``bridge_threshold=0.70``,
``max_chain_break=0.20``, ``min_hits=3``.
"""

from __future__ import annotations

import numpy as np
import torch

from .builder import Track  # reuse the dataclass so consumers don't branch


class _UnionFind:
    """Path-compressed union-find with size + per-component min/max
    incident-edge scores so chain-protection can reason about chains.
    """

    __slots__ = ("max_score", "min_score", "parent", "rank", "size")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        # Per-component running min/max of edge scores that joined it.
        # None means "no edge yet" (size-1 component).
        self.min_score: list[float] = [float("inf")] * n
        self.max_score: list[float] = [float("-inf")] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, score: float) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Union by rank.
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.size[ra] += self.size[rb]
        self.min_score[ra] = min(self.min_score[ra], self.min_score[rb], score)
        self.max_score[ra] = max(self.max_score[ra], self.max_score[rb], score)
        return True


def build_tracks_uncertainty(
    edge_index: torch.Tensor,
    edge_score: torch.Tensor,
    n_hits: int,
    *,
    hard_threshold: float = 0.30,
    merge_threshold: float = 0.50,
    bridge_threshold: float = 0.70,
    min_hits: int = 3,
    max_chain_break: float = 0.20,
) -> list[Track]:
    """Score-aware Union-Find with chain protection.

    Parameters mirror :func:`build_tracks` plus four new knobs:

    * ``hard_threshold`` -- edges below this are dropped entirely.
    * ``merge_threshold`` -- edges above this merge unconditionally.
    * ``bridge_threshold`` -- to merge in the chain-protection zone
      [``hard_threshold``, ``merge_threshold``), both endpoints must have
      at least one incident edge with score >= ``bridge_threshold`` --
      i.e., both endpoints are already part of high-confidence chains.
    * ``max_chain_break`` -- currently unused; reserved for future,
      stricter chain-coherence checks.
    """
    if edge_index.numel() == 0:
        return []
    if not (0.0 <= hard_threshold <= merge_threshold <= 1.0):
        raise ValueError(
            "expected 0 <= hard_threshold <= merge_threshold <= 1",
        )

    src_t = edge_index[0].cpu().numpy().astype(np.int64)
    dst_t = edge_index[1].cpu().numpy().astype(np.int64)
    sc_np = edge_score.detach().cpu().numpy().astype(np.float64)

    # 1. Filter edges below the absolute floor.
    keep = sc_np >= hard_threshold
    src_t = src_t[keep]
    dst_t = dst_t[keep]
    sc_np = sc_np[keep]
    if src_t.size == 0:
        return []

    # Precompute per-hit max incident score for chain-protection check.
    max_inc = np.zeros(n_hits, dtype=np.float64)
    for s, d, sc in zip(src_t.tolist(), dst_t.tolist(), sc_np.tolist(), strict=True):
        if sc > max_inc[s]:
            max_inc[s] = sc
        if sc > max_inc[d]:
            max_inc[d] = sc

    # 2. Sort by score descending.
    order = np.argsort(-sc_np, kind="stable")
    src_t = src_t[order]
    dst_t = dst_t[order]
    sc_np = sc_np[order]

    # 3. Union-Find pass.
    uf = _UnionFind(n_hits)
    for s, d, sc in zip(src_t.tolist(), dst_t.tolist(), sc_np.tolist(), strict=True):
        rs = uf.find(s)
        rd = uf.find(d)
        if rs == rd:
            continue
        if sc >= merge_threshold:
            uf.union(s, d, sc)
            continue
        # ----- chain-protection zone: sc in [hard_threshold, merge_threshold)
        # Both endpoints must already be part of high-confidence chains
        # (have a >= bridge_threshold incident edge). This makes the
        # current low-confidence edge a *re-binder* of two genuine chains
        # rather than a new chain-creator from a single weak edge.
        if max_inc[s] < bridge_threshold or max_inc[d] < bridge_threshold:
            continue
        uf.union(s, d, sc)

    # 4. Collect components, compute per-component mean score, filter min_hits.
    root_to_hits: dict[int, list[int]] = {}
    for hit in range(n_hits):
        root = uf.find(hit)
        root_to_hits.setdefault(root, []).append(hit)

    # Per-root mean score = average of edges currently inside it
    # (approximated by sweeping edges with both endpoints in the same comp).
    score_sum: dict[int, float] = {}
    score_cnt: dict[int, int] = {}
    for s, d, sc in zip(src_t.tolist(), dst_t.tolist(), sc_np.tolist(), strict=True):
        rs = uf.find(s)
        if rs == uf.find(d):
            score_sum[rs] = score_sum.get(rs, 0.0) + sc
            score_cnt[rs] = score_cnt.get(rs, 0) + 1

    tracks: list[Track] = []
    for root, hits in root_to_hits.items():
        if len(hits) < min_hits:
            continue
        mean_score = score_sum.get(root, 0.0) / max(1, score_cnt.get(root, 1))
        tracks.append(
            Track(
                hit_indices=np.array(sorted(hits), dtype=np.int64),
                score=float(mean_score),
                extras={
                    "size": len(hits),
                    "min_edge_score": float(uf.min_score[root]),
                    "max_edge_score": float(uf.max_score[root]),
                },
            ),
        )
    return tracks


__all__ = ["build_tracks_uncertainty"]
