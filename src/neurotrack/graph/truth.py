"""Truth-edge construction from per-event particle labels.

For each event we build the **expected sequential edges** between hits of
the same particle, sorted by radial distance ``r`` (a proxy for layer
order).  This matches the standard ATLAS / Exa.TrkX truth-graph
construction and feeds the GNN as positive supervision.

Output is a sparse COO edge list ``(2, E)`` plus a per-edge binary label
(always 1 for truth edges).  The full edge-classifier truth is then
computed at training time by looking up which candidate edges from the
kNN graph match this truth set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def build_truth_edges(
    particle_ids: npt.NDArray[Any],
    r: npt.NDArray[Any],
    *,
    drop_noise: bool = True,
    min_hits_per_particle: int = 2,
) -> torch.Tensor:
    """Return ``edge_index`` of shape ``(2, E)`` containing the truth edges.

    Each particle's hits are sorted by ``r`` and connected pairwise between
    consecutive hits, yielding a chain.  We also include the reverse edge
    so the graph is undirected (matches the kNN graph topology).

    Parameters
    ----------
    particle_ids
        Shape ``(N,)`` int array.  ``0`` is treated as noise when
        ``drop_noise=True`` (TrackML convention).
    r
        Shape ``(N,)`` float array of radial distances used for sorting.
    drop_noise
        If True, hits with ``particle_id == 0`` never participate in truth
        edges.
    min_hits_per_particle
        Particles with fewer hits than this are skipped (no useful edges).
    """
    if particle_ids.shape != r.shape:
        raise ValueError("particle_ids and r must have matching shape")
    if particle_ids.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    pids = np.asarray(particle_ids).astype(np.int64)
    rr = np.asarray(r).astype(np.float64)

    src_list: list[npt.NDArray[np.int64]] = []
    dst_list: list[npt.NDArray[np.int64]] = []

    # Group hits by particle_id.  Use np.unique + indices for O(N log N).
    sort_idx = np.argsort(pids, kind="stable")
    pids_sorted = pids[sort_idx]
    # Boundary indices where particle_id changes.
    change = np.flatnonzero(np.diff(pids_sorted)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(pids_sorted)]])

    for s, e in zip(starts, ends, strict=False):
        pid = int(pids_sorted[s])
        if drop_noise and pid == 0:
            continue
        if e - s < min_hits_per_particle:
            continue
        group = sort_idx[s:e]
        # Sort this particle's hits by r ascending (inner -> outer layer).
        order = group[np.argsort(rr[group])]
        # Chain edges: order[i] -> order[i+1]
        src_list.append(order[:-1])
        dst_list.append(order[1:])

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    # Symmetrise.
    ei = np.stack(
        [
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ],
        axis=0,
    )
    return torch.from_numpy(ei).long()


def edge_label_from_truth(
    candidate_edge_index: torch.Tensor,
    particle_ids: torch.Tensor,
) -> torch.Tensor:
    """For each candidate edge, return 1 iff src/dst share a non-zero particle.

    This is the supervision target for the edge classifier: positive edges
    are those that connect two hits of the same real particle.  Noise hits
    (``particle_id == 0``) yield label 0 regardless of their neighbour.
    """
    if candidate_edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    src = candidate_edge_index[0].long()
    dst = candidate_edge_index[1].long()
    p_src = particle_ids[src]
    p_dst = particle_ids[dst]
    same = (p_src == p_dst) & (p_src > 0)
    return same.float()


__all__ = ["build_truth_edges", "edge_label_from_truth"]
