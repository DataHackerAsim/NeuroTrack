"""Minimal pipeline orchestrator: embedding -> kNN -> GNN -> tracks.

Per R-H, this exposes a ``tracking_mode`` flag with three modes:

* ``"baseline"``            -- legacy connected-components builder (R-G).
* ``"uncertainty"``         -- score-aware Union-Find with chain protection.
* ``"uncertainty_kalman"``  -- ``uncertainty`` + Kalman chi2 arbitration.

Default is ``"baseline"`` to preserve reproducibility of old runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import torch

from neurotrack.graph.construction import build_edge_features, build_knn_graph
from neurotrack.models.embedding import HitEmbedNet
from neurotrack.models.gnn import InteractionNetwork
from neurotrack.tracking.arbitrate import arbitrate_tracks
from neurotrack.tracking.builder import Track, build_tracks
from neurotrack.tracking.builder_uncertainty import build_tracks_uncertainty


class TrackingMode(StrEnum):
    BASELINE = "baseline"
    UNCERTAINTY = "uncertainty"
    UNCERTAINTY_KALMAN = "uncertainty_kalman"


@dataclass
class PipelineConfig:
    knn_k: int = 8
    threshold: float = 0.7        # baseline only
    min_hits: int = 3
    precision: str = "bf16"
    tracking_mode: TrackingMode = TrackingMode.BASELINE

    # Uncertainty-builder knobs (used when tracking_mode != BASELINE).
    hard_threshold: float = 0.30
    merge_threshold: float = 0.50
    bridge_threshold: float = 0.70
    max_chain_break: float = 0.20

    # Kalman arbitration knobs (used only for UNCERTAINTY_KALMAN).
    chi2_threshold: float = 3.0
    min_hits_after_split: int = 3


def _autocast(precision: str) -> torch.autocast:
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def run_event(
    x: torch.Tensor,
    embedding: HitEmbedNet,
    gnn: InteractionNetwork,
    cfg: PipelineConfig,
    *,
    device: torch.device,
) -> tuple[list[Track], torch.Tensor, torch.Tensor]:
    """Run the pipeline on one event's feature tensor.

    Returns ``(tracks, edge_index, edge_scores)`` so the caller can also
    compute diagnostics from the candidate graph.
    """
    with torch.no_grad():
        with _autocast(cfg.precision):
            emb = embedding(x).float()
        ei = build_knn_graph(emb, k=cfg.knn_k)
        if ei.numel() == 0:
            return [], ei, x.new_empty((0,))
        ea = build_edge_features(x, ei, emb=emb)
        with _autocast(cfg.precision):
            logits = gnn(x, ei, ea).float()
        scores = torch.sigmoid(logits)

    n_hits = x.shape[0]
    if cfg.tracking_mode == TrackingMode.BASELINE:
        tracks = build_tracks(
            ei, scores, n_hits=n_hits,
            threshold=cfg.threshold, min_hits=cfg.min_hits,
        )
    else:
        tracks = build_tracks_uncertainty(
            ei, scores, n_hits=n_hits,
            hard_threshold=cfg.hard_threshold,
            merge_threshold=cfg.merge_threshold,
            bridge_threshold=cfg.bridge_threshold,
            min_hits=cfg.min_hits,
            max_chain_break=cfg.max_chain_break,
        )
        if cfg.tracking_mode == TrackingMode.UNCERTAINTY_KALMAN:
            # Kalman post-fit + chi2 arbitration.
            xyz: npt.NDArray[np.float64] = x[:, :3].cpu().numpy().astype(np.float64)
            tracks, _stats = arbitrate_tracks(
                tracks, xyz,
                chi2_threshold=cfg.chi2_threshold,
                min_hits_after_split=cfg.min_hits_after_split,
            )
    return tracks, ei, scores


__all__ = ["PipelineConfig", "TrackingMode", "run_event"]
