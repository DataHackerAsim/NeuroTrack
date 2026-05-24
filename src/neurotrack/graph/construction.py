"""Candidate edge construction.

Given a per-hit embedding ``emb`` of shape ``(N, D)``, build a sparse
candidate edge set ``(2, E)`` by finding each hit's ``k`` nearest
neighbours in embedding space.

We use a pure PyTorch implementation (works on CPU and CUDA), which is
fast for the event sizes in this dataset (median ~225 hits for TrackML
small, ~400 for REDVID).  For events with O(10^5) hits a FAISS-based
backend would be a worthwhile drop-in -- the function signature is
deliberately compatible.
"""

from __future__ import annotations

import torch


def build_knn_graph(
    emb: torch.Tensor,
    k: int = 8,
    *,
    max_distance: float | None = None,
    symmetrise: bool = True,
    self_loops: bool = False,
) -> torch.Tensor:
    """Return ``edge_index`` of shape ``(2, E)``.

    Parameters
    ----------
    emb
        Embedding tensor ``(N, D)``.  Should be L2-normalised for the
        cosine-like behaviour; the function works either way.
    k
        Number of neighbours per node.  If ``N <= k`` we fall back to a
        fully connected graph.
    max_distance
        Optional distance cap; edges longer than this in L2 space are
        dropped.  ``None`` keeps all kNN edges.
    symmetrise
        If True, the returned edge set is the union of ``i -> j`` and
        ``j -> i``.  Required for undirected message passing.
    self_loops
        If False (default), drop ``i -> i`` edges.
    """
    n, _ = emb.shape
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long, device=emb.device)

    # Available neighbours per node: n if self-loops allowed, else n-1.
    k_eff = min(k, n if self_loops else max(1, n - 1))
    # cdist works in float32 on CUDA; cast for safety.
    dists = torch.cdist(emb.float(), emb.float())  # (N, N)
    if not self_loops:
        # Mask the diagonal with +inf so it never appears in top-k.
        dists = dists.masked_fill(
            torch.eye(n, dtype=torch.bool, device=emb.device),
            float("inf"),
        )
    # Top-k smallest distances per row.
    topk = torch.topk(dists, k_eff, largest=False)
    nbr = topk.indices  # (N, k_eff)
    dst = nbr.reshape(-1)
    src = torch.arange(n, device=emb.device).unsqueeze(1).expand(-1, k_eff).reshape(-1)

    if max_distance is not None:
        d_flat = topk.values.reshape(-1)
        keep = d_flat <= max_distance
        src = src[keep]
        dst = dst[keep]

    if symmetrise:
        edge_index = torch.stack(
            [
                torch.cat([src, dst]),
                torch.cat([dst, src]),
            ],
            dim=0,
        )
        # Deduplicate (i,j) pairs after symmetrisation.
        edge_index = _dedup_edges(edge_index)
    else:
        edge_index = torch.stack([src, dst], dim=0)
    return edge_index.long()


def _dedup_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Drop duplicate edges via row-wise unique on the transposed (E, 2) view."""
    if edge_index.numel() == 0:
        return edge_index
    uniq: torch.Tensor = torch.unique(edge_index.t(), dim=0)
    return uniq.t().long()


def build_edge_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct per-edge attributes from per-node features.

    Returns a tensor ``(E, F_edge)`` with the following columns:

    * ``dx, dy, dz``   -- cartesian deltas (cols 0, 1, 2 of ``x``)
    * ``dr, dphi, dz_norm`` -- cylindrical deltas (cols 3, 4, 2)
    * ``distance``     -- L2 distance in the cartesian xy-z subspace
    * ``emb_distance`` -- L2 distance in embedding space (if ``emb`` given;
                          otherwise zeros, so F_edge is constant)
    """
    src = edge_index[0].long()
    dst = edge_index[1].long()
    if src.numel() == 0:
        return torch.zeros((0, 7), dtype=x.dtype, device=x.device)

    dx = x[dst, 0] - x[src, 0]
    dy = x[dst, 1] - x[src, 1]
    dz = x[dst, 2] - x[src, 2]
    dr = x[dst, 3] - x[src, 3]
    dphi = x[dst, 4] - x[src, 4]
    # Wrap dphi into (-pi, pi]
    import math
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    distance = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)

    if emb is not None:
        emb_dist = torch.linalg.norm(emb[dst] - emb[src], dim=1)
    else:
        emb_dist = torch.zeros_like(distance)

    return torch.stack([dx, dy, dz, dr, dphi, distance, emb_dist], dim=1)


__all__ = ["build_edge_features", "build_knn_graph"]
