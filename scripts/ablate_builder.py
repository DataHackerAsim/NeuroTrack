"""R-H ablation: baseline vs uncertainty vs uncertainty_kalman.

Runs all three tracking modes against the same 1500-event held-out
slice of trackml_small with the same checkpoints, collects per-event
scores, and writes both a JSON aggregate and a markdown report.
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

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-ckpt", type=Path, default=ROOT / "artifacts/checkpoints/embedding.pt")
    p.add_argument("--gnn-ckpt", type=Path, default=ROOT / "artifacts/checkpoints/gnn_v2.pt")
    p.add_argument("--shard", type=Path, default=ROOT / "data/processed/trackml_small")
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--threshold-baseline", type=float, default=0.7)
    p.add_argument("--hard-threshold", type=float, default=0.30)
    p.add_argument("--merge-threshold", type=float, default=0.50)
    p.add_argument("--bridge-threshold", type=float, default=0.70)
    p.add_argument("--chi2-threshold", type=float, default=3.0)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/ablation_builder.json")
    p.add_argument("--report", type=Path, default=ROOT / "artifacts/ablation_builder_REPORT.md")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args(argv)


def _autocast() -> torch.autocast:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def evaluate_mode(
    mode: str,
    ds: EventParquetDataset,
    emb: HitEmbedNet,
    gnn: InteractionNetwork,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    score_list: list[float] = []
    eff_list: list[float] = []
    fake_list: list[float] = []
    dup_list: list[float] = []
    auc_list: list[float] = []
    n_tracks_list: list[int] = []
    n_edges_list: list[int] = []
    per_event: list[dict] = []

    t0 = time.time()
    n_used = 0
    for idx in range(len(ds)):
        ev: EventBatch = ds[idx].to(device)
        if ev.n < args.min_hits:
            continue
        with torch.no_grad():
            with _autocast():
                emb_out = emb(ev.x).float()
            ei = build_knn_graph(emb_out, k=args.knn_k)
            if ei.numel() == 0:
                continue
            ea = build_edge_features(ev.x, ei, emb=emb_out)
            with _autocast():
                logits = gnn(ev.x, ei, ea).float()
            scores = torch.sigmoid(logits)
            labels = edge_label_from_truth(ei, ev.particle_ids).to(device)

        if mode == "baseline":
            tracks = build_tracks(
                ei, scores, n_hits=ev.n,
                threshold=args.threshold_baseline, min_hits=args.min_hits,
            )
        else:
            tracks = build_tracks_uncertainty(
                ei, scores, n_hits=ev.n,
                hard_threshold=args.hard_threshold,
                merge_threshold=args.merge_threshold,
                bridge_threshold=args.bridge_threshold,
                min_hits=args.min_hits,
            )
            if mode == "uncertainty_kalman":
                xyz = ev.x[:, :3].cpu().numpy().astype(np.float64)
                tracks, _stats = arbitrate_tracks(
                    tracks, xyz,
                    chi2_threshold=args.chi2_threshold,
                    min_hits_after_split=args.min_hits,
                )

        track_arrays = [t.hit_indices for t in tracks]
        pids = ev.particle_ids.cpu().numpy()
        w = ev.weights.cpu().numpy()
        s = trackml_score(track_arrays, pids, w)
        e = track_efficiency(track_arrays, pids, min_hits=args.min_hits)
        a = edge_auc(scores, labels)
        score_list.append(s)
        eff_list.append(e["efficiency"])
        fake_list.append(e["fake_rate"])
        dup_list.append(e["duplicate_rate"])
        auc_list.append(a)
        n_tracks_list.append(len(tracks))
        n_edges_list.append(int(ei.shape[1]))
        per_event.append(
            {
                "event_id": ev.event_id,
                "n_hits": ev.n,
                "n_tracks": len(tracks),
                "trackml_score": s,
                "edge_auc": a,
                "efficiency": e["efficiency"],
                "fake_rate": e["fake_rate"],
            },
        )
        n_used += 1
        if n_used % 100 == 0:
            print(f"  [{mode}] {n_used}/{len(ds)}  mean_score={np.mean(score_list):.4f}", flush=True)

    wall = time.time() - t0
    sc = np.array(score_list)
    p10_thr = float(np.quantile(sc, 0.10))
    bottom_mean = float(sc[sc <= p10_thr].mean()) if (sc <= p10_thr).any() else 0.0
    return {
        "mode": mode,
        "events": n_used,
        "wall_s": wall,
        "p95_latency_s_per_event": wall / max(1, n_used),
        "trackml_score_mean": float(sc.mean()),
        "trackml_score_median": float(np.median(sc)),
        "trackml_score_p10_threshold": p10_thr,
        "trackml_score_p10_mean": bottom_mean,
        "efficiency_mean": float(np.mean(eff_list)),
        "fake_rate_mean": float(np.mean(fake_list)),
        "duplicate_rate_mean": float(np.mean(dup_list)),
        "edge_auc_mean": float(np.mean(auc_list)),
        "n_tracks_mean": float(np.mean(n_tracks_list)),
        "per_event": per_event,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_ckpt = torch.load(args.emb_ckpt, map_location=device, weights_only=False)
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

    gnn_ckpt = torch.load(args.gnn_ckpt, map_location=device, weights_only=False)
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

    ds = EventParquetDataset([args.shard], limit=args.limit)
    print(f"[ablate] events={len(ds)}", flush=True)

    results = {}
    for mode in ("baseline", "uncertainty", "uncertainty_kalman"):
        print(f"\n=== {mode} ===", flush=True)
        results[mode] = evaluate_mode(mode, ds, emb, gnn, device, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Write markdown report.
    base = results["baseline"]
    unc = results["uncertainty"]
    kal = results["uncertainty_kalman"]
    lines = []
    lines.append("# R-H ablation: builder change on 1500-event trackml_small\n")
    lines.append(f"Checkpoints: {args.emb_ckpt.name}, {args.gnn_ckpt.name}\n")
    lines.append("| metric | baseline | uncertainty | uncertainty_kalman |")
    lines.append("|---|---:|---:|---:|")
    for key, label in (
        ("trackml_score_mean", "TrackML score (mean)"),
        ("trackml_score_median", "TrackML score (median)"),
        ("trackml_score_p10_mean", "TrackML score (bottom decile mean)"),
        ("efficiency_mean", "Efficiency (mean)"),
        ("fake_rate_mean", "Fake rate (mean)"),
        ("duplicate_rate_mean", "Duplicate rate (mean)"),
        ("edge_auc_mean", "Edge AUC (mean)"),
        ("n_tracks_mean", "Predicted tracks / event"),
        ("p95_latency_s_per_event", "Latency s/event"),
    ):
        b = base[key]
        u = unc[key]
        k = kal[key]
        lines.append(
            f"| {label} | {b:.4f} | {u:.4f} (d={u - b:+.4f}) | {k:.4f} (d={k - b:+.4f}) |",
        )
    lines.append("")
    winner_score = max(
        ("uncertainty", unc["trackml_score_mean"]),
        ("uncertainty_kalman", kal["trackml_score_mean"]),
        key=lambda x: x[1],
    )
    winner_lift = winner_score[1] - base["trackml_score_mean"]
    lines.append("## Verdict\n")
    lines.append(
        f"Winner: **{winner_score[0]}** with mean score {winner_score[1]:.4f} "
        f"(baseline {base['trackml_score_mean']:.4f}, lift "
        f"d = {winner_lift:+.4f}).",
    )
    target = 0.02
    if winner_lift >= target:
        lines.append(f"Lift >= {target:.2f} acceptance target -- **PASS**.")
    else:
        lines.append(f"Lift < {target:.2f} acceptance target -- **FAIL**.")
    lines.append("")
    lines.append("## Bottom-decile uplift\n")
    for key, label in (("trackml_score_p10_mean", "bottom-decile mean"),):
        b = base[key]
        u = unc[key]
        k = kal[key]
        lines.append(
            f"- {label}: {b:.4f} -> uncertainty {u:.4f} (d={u-b:+.4f}) -> "
            f"+kalman {k:.4f} (d={k-b:+.4f})",
        )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {args.report}")
    print(f"JSON:   {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
