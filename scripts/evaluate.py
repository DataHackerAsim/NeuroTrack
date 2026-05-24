"""End-to-end evaluation: embedding -> kNN -> GNN -> tracks -> TrackML score.

Usage::

    python scripts/evaluate.py \\
        --emb-ckpt artifacts/checkpoints/embedding.pt \\
        --gnn-ckpt artifacts/checkpoints/gnn.pt \\
        --shard data/processed/trackml_small \\
        --limit 200 --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from neurotrack.data.dataset import EventBatch, EventParquetDataset
from neurotrack.eval.metrics import (
    edge_auc,
    recall_at_k,
    track_efficiency,
    trackml_score,
)
from neurotrack.graph.construction import build_edge_features, build_knn_graph
from neurotrack.graph.truth import edge_label_from_truth
from neurotrack.models.embedding import HitEmbedNet
from neurotrack.models.gnn import InteractionNetwork
from neurotrack.tracking.arbitrate import arbitrate_tracks
from neurotrack.tracking.builder import build_tracks
from neurotrack.tracking.builder_uncertainty import build_tracks_uncertainty
from neurotrack.utils.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-ckpt", type=Path, required=True)
    p.add_argument("--gnn-ckpt", type=Path, required=True)
    p.add_argument("--shard", type=Path, action="append", default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument(
        "--tracking-mode",
        choices=("baseline", "uncertainty", "uncertainty_kalman"),
        default="baseline",
    )
    p.add_argument("--hard-threshold", type=float, default=0.30)
    p.add_argument("--merge-threshold", type=float, default=0.50)
    p.add_argument("--bridge-threshold", type=float, default=0.70)
    p.add_argument("--max-chain-break", type=float, default=0.20)
    p.add_argument("--chi2-threshold", type=float, default=3.0)
    p.add_argument("--report", type=Path, default=Path("artifacts/eval_report.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args(argv)


def _autocast(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    shards: list[Path] = args.shard or [Path("data/processed/trackml_small")]
    ds = EventParquetDataset(shards, limit=args.limit)
    print(f"[eval] events={len(ds)}", flush=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu",
    )

    # Load models.
    emb_ckpt = torch.load(args.emb_ckpt, map_location="cpu", weights_only=False)
    emb_cfg = emb_ckpt.get("cfg") or {}
    emb = HitEmbedNet(
        in_dim=int(emb_cfg.get("in_dim", 24)),
        hidden_dim=int(emb_cfg.get("hidden_dim", 128)),
        out_dim=int(emb_cfg.get("out_dim", 12)),
        num_layers=int(emb_cfg.get("num_layers", 4)),
        dropout=float(emb_cfg.get("dropout", 0.0)),
    )
    emb.load_state_dict(emb_ckpt["model"])
    emb.to(device).eval()

    gnn_ckpt = torch.load(args.gnn_ckpt, map_location="cpu", weights_only=False)
    gnn_cfg = gnn_ckpt.get("cfg") or {}
    gnn = InteractionNetwork(
        node_dim=int(gnn_cfg.get("node_dim", 24)),
        edge_dim=int(gnn_cfg.get("edge_dim", 7)),
        hidden_dim=int(gnn_cfg.get("hidden_dim", 64)),
        num_iter=int(gnn_cfg.get("num_iter", 8)),
        use_checkpoint=False,
    )
    gnn.load_state_dict(gnn_ckpt["model"])
    gnn.to(device).eval()

    per_event: list[dict[str, float]] = []
    score_sum = 0.0
    auc_sum = 0.0
    recall_sum = 0.0
    eff_sum = 0.0
    fake_sum = 0.0
    dup_sum = 0.0
    n = 0
    t0 = time.time()
    for idx in range(len(ds)):
        ev: EventBatch = ds[idx].to(device)
        if ev.n < args.min_hits:
            continue
        with torch.no_grad():
            with _autocast(args.precision):
                emb_out = emb(ev.x).float()
            edge_index = build_knn_graph(emb_out, k=args.knn_k)
            if edge_index.numel() == 0:
                continue
            edge_attr = build_edge_features(ev.x, edge_index, emb=emb_out)
            with _autocast(args.precision):
                logits = gnn(ev.x, edge_index, edge_attr).float()
            scores = torch.sigmoid(logits)
            labels = edge_label_from_truth(edge_index, ev.particle_ids).to(device)

        if args.tracking_mode == "baseline":
            tracks = build_tracks(
                edge_index, scores, n_hits=ev.n,
                threshold=args.threshold, min_hits=args.min_hits,
            )
        else:
            tracks = build_tracks_uncertainty(
                edge_index, scores, n_hits=ev.n,
                hard_threshold=args.hard_threshold,
                merge_threshold=args.merge_threshold,
                bridge_threshold=args.bridge_threshold,
                max_chain_break=args.max_chain_break,
                min_hits=args.min_hits,
            )
            if args.tracking_mode == "uncertainty_kalman":
                xyz = ev.x[:, :3].cpu().numpy().astype(np.float64)
                tracks, _arb_stats = arbitrate_tracks(
                    tracks, xyz,
                    chi2_threshold=args.chi2_threshold,
                    min_hits_after_split=args.min_hits,
                )
        track_hit_arrays = [t.hit_indices for t in tracks]

        pids_np = ev.particle_ids.cpu().numpy()
        w_np = ev.weights.cpu().numpy()

        s = trackml_score(track_hit_arrays, pids_np, w_np)
        a = edge_auc(scores, labels)
        r = recall_at_k(emb_out, ev.particle_ids, k=10)
        e = track_efficiency(track_hit_arrays, pids_np, min_hits=args.min_hits)

        per_event.append(
            {
                "event_id": ev.event_id,
                "n_hits": ev.n,
                "n_edges": int(edge_index.shape[1]),
                "n_tracks": len(tracks),
                "trackml_score": s,
                "edge_auc": a,
                "recall@10": r,
                "efficiency": e["efficiency"],
                "fake_rate": e["fake_rate"],
                "duplicate_rate": e["duplicate_rate"],
            },
        )
        score_sum += s
        auc_sum += a
        recall_sum += r
        eff_sum += e["efficiency"]
        fake_sum += e["fake_rate"]
        dup_sum += e["duplicate_rate"]
        n += 1
        if n % 25 == 0:
            print(
                f"  [eval] {n}/{len(ds)}  mean_score={score_sum/n:.4f}  "
                f"mean_auc={auc_sum/n:.4f}",
                flush=True,
            )

    wall = time.time() - t0
    if n == 0:
        print("[eval] no events evaluated", file=sys.stderr)
        return 1

    summary = {
        "events_evaluated": n,
        "wall_time_s": wall,
        "p95_latency_s_per_event": wall / n,
        "trackml_score_mean": score_sum / n,
        "trackml_score_median": float(np.median([d["trackml_score"] for d in per_event])),
        "edge_auc_mean": auc_sum / n,
        "recall@10_mean": recall_sum / n,
        "efficiency_mean": eff_sum / n,
        "fake_rate_mean": fake_sum / n,
        "duplicate_rate_mean": dup_sum / n,
        "config": {
            "knn_k": args.knn_k,
            "threshold": args.threshold,
            "min_hits": args.min_hits,
            "precision": args.precision,
            "tracking_mode": args.tracking_mode,
            "hard_threshold": args.hard_threshold,
            "merge_threshold": args.merge_threshold,
            "bridge_threshold": args.bridge_threshold,
            "max_chain_break": args.max_chain_break,
            "chi2_threshold": args.chi2_threshold,
            "shards": [str(s) for s in shards],
            "limit": args.limit,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "per_event": per_event}, indent=2))

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<25s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
