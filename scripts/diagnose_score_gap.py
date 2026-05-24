"""Diagnostic: why does mean lag median by ~0.04 on trackml_small?

Reads ``artifacts/eval_final_trackml_small.json`` plus the per-event Parquet
files and the trained checkpoints; produces:

    artifacts/diagnostic/score_histogram.png
    artifacts/diagnostic/noise_correlation.png
    artifacts/diagnostic/score_vs_n_tracks.png
    artifacts/diagnostic/edge_scores_bad_events.png
    artifacts/diagnostic/edge_auc_distribution.png
    artifacts/diagnostic/bottom_decile.json
    artifacts/diagnostic/event_features.csv
    artifacts/diagnostic/track_failures.json
    artifacts/diagnostic/REPORT.md
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from neurotrack.data.dataset import EventParquetDataset
from neurotrack.eval.metrics import edge_auc, recall_at_k
from neurotrack.graph.construction import build_edge_features, build_knn_graph
from neurotrack.graph.truth import edge_label_from_truth
from neurotrack.models.embedding import HitEmbedNet
from neurotrack.models.gnn import InteractionNetwork
from neurotrack.tracking.builder import build_tracks

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "artifacts" / "eval_final_trackml_small.json"
SHARD = ROOT / "data" / "processed" / "trackml_small"
DIAG = ROOT / "artifacts" / "diagnostic"
DIAG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def step1_distribution(per_event: list[dict]) -> dict:
    scores = np.array([e["trackml_score"] for e in per_event], dtype=np.float64)
    qs = {q: float(np.quantile(scores, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=40, range=(0.0, 1.0), edgecolor="black", alpha=0.8)
    plt.axvline(scores.mean(), color="red", linestyle="--", label=f"mean={scores.mean():.4f}")
    plt.axvline(np.median(scores), color="green", linestyle="--", label=f"median={np.median(scores):.4f}")
    plt.axvline(qs[0.10], color="orange", linestyle=":", label=f"p10={qs[0.10]:.4f}")
    plt.xlabel("TrackML score")
    plt.ylabel("events")
    plt.title(f"Score distribution (n={len(scores)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIAG / "score_histogram.png", dpi=110)
    plt.close()
    return {"quantiles": qs, "mean": float(scores.mean()), "n": len(scores)}


# ---------------------------------------------------------------------------
def step2_bottom_decile(per_event: list[dict], p10: float) -> list[dict]:
    bottom = [e for e in per_event if e["trackml_score"] <= p10]
    (DIAG / "bottom_decile.json").write_text(json.dumps(bottom, indent=2))
    return bottom


# ---------------------------------------------------------------------------
def _safe_pT(px: float, py: float) -> float:
    return math.sqrt(px * px + py * py)


def _safe_eta(px: float, py: float, pz: float) -> float:
    pT = _safe_pT(px, py)
    if pT < 1.0e-9:
        return 10.0 * (1 if pz > 0 else -1)
    theta = math.atan2(pT, pz)
    return -math.log(math.tan(theta / 2.0))


def step3_event_features(per_event: list[dict]) -> pl.DataFrame:
    """Augment per-event records with truth-table features by reading parquets."""
    events_dir = SHARD / "events"
    parts_dir = SHARD / "particles"

    rows: list[dict] = []
    for e in per_event:
        eid = int(e["event_id"])
        hits = pl.read_parquet(events_dir / f"{eid}.parquet")
        parts = pl.read_parquet(parts_dir / f"{eid}.parquet")

        pids = hits["particle_id"].to_numpy()
        n_hits = hits.height
        n_tracks_truth = int(parts.height)
        n_noise = int((pids == 0).sum())
        noise_frac = n_noise / max(1, n_hits)

        px = parts["px"].to_numpy()
        py = parts["py"].to_numpy()
        pz = parts["pz"].to_numpy()
        pT = np.sqrt(px * px + py * py)
        # Particle eta: exclude particles with pT == 0 / particle_id == 0.
        mask = (pT > 0) & (parts["particle_id"].to_numpy() > 0)
        if mask.any():
            pT_v = pT[mask]
            eta_v = np.array([_safe_eta(px[i], py[i], pz[i]) for i in np.flatnonzero(mask)])
        else:
            pT_v = np.array([])
            eta_v = np.array([])
        rows.append(
            {
                "event_id": eid,
                "trackml_score": e["trackml_score"],
                "edge_auc": e["edge_auc"],
                "recall@10": e["recall@10"],
                "efficiency": e["efficiency"],
                "fake_rate": e["fake_rate"],
                "n_hits": n_hits,
                "n_tracks_truth": n_tracks_truth,
                "n_noise_hits": n_noise,
                "noise_frac": noise_frac,
                "mean_pT": float(pT_v.mean()) if pT_v.size else 0.0,
                "median_pT": float(np.median(pT_v)) if pT_v.size else 0.0,
                "n_tracks_low_pT": int((pT_v < 1.0).sum()),
                "n_tracks_high_eta": int((np.abs(eta_v) > 2.0).sum()) if eta_v.size else 0,
                "n_tracks_predicted": int(e["n_tracks"]),
            },
        )
    df = pl.DataFrame(rows)
    df.write_csv(DIAG / "event_features.csv")
    return df


# ---------------------------------------------------------------------------
def step3_ratios(df: pl.DataFrame, p10: float) -> dict:
    bottom = df.filter(pl.col("trackml_score") <= p10)
    rest = df.filter(pl.col("trackml_score") > p10)
    cols = (
        "n_hits", "n_tracks_truth", "n_noise_hits", "noise_frac",
        "mean_pT", "median_pT", "n_tracks_low_pT", "n_tracks_high_eta",
        "n_tracks_predicted", "edge_auc", "recall@10", "efficiency", "fake_rate",
    )
    out = {"n_bottom": bottom.height, "n_rest": rest.height, "features": {}}
    for c in cols:
        b_mean = float(bottom[c].mean() or 0.0)
        r_mean = float(rest[c].mean() or 0.0)
        ratio = b_mean / r_mean if r_mean else float("inf")
        out["features"][c] = {"bottom": b_mean, "rest": r_mean, "ratio_b_over_r": ratio}
    return out


# ---------------------------------------------------------------------------
def step4_track_failures(
    bottom: list[dict], n_pick: int = 5, *, seed: int = 7,
) -> list[dict]:
    """For 5 random bad events, classify each truth track as
    recovered / missed / merged / split.
    """
    random.seed(seed)
    picked = random.sample(bottom, min(n_pick, len(bottom)))

    out: list[dict] = []
    for e in picked:
        eid = int(e["event_id"])
        # We need the prediction here -- regenerate quickly using checkpoints.
        # Heavy: reload models per event isn't, since we batch all 5 below.
        out.append({"event_id": eid, "trackml_score": e["trackml_score"]})
    return out


def step4_full(bottom: list[dict], n_pick: int, *, embedding, gnn, device, knn_k, threshold, min_hits) -> list[dict]:
    """Rich per-event classification."""
    random.seed(7)
    picked = random.sample(bottom, min(n_pick, len(bottom)))
    events_dir = SHARD / "events"
    breakdown: list[dict] = []

    ds_paths = [events_dir / f"{int(e['event_id'])}.parquet" for e in picked]

    for path, e_meta in zip(ds_paths, picked, strict=True):
        df = pl.read_parquet(path)
        x = np.stack(
            [df[c].to_numpy().astype(np.float32) for c in EventParquetDataset(
                [SHARD], limit=1,
            ).feature_cols],
            axis=1,
        )
        x_t = torch.from_numpy(x).to(device)
        pids = df["particle_id"].to_numpy()
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                enabled=device.type == "cuda"):
                emb = embedding(x_t).float()
            ei = build_knn_graph(emb, k=knn_k)
            ea = build_edge_features(x_t, ei, emb=emb)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                enabled=device.type == "cuda"):
                logits = gnn(x_t, ei, ea).float()
            scores = torch.sigmoid(logits)

        tracks = build_tracks(
            ei, scores, n_hits=df.height, threshold=threshold, min_hits=min_hits,
        )

        # Classify truth tracks: for each non-noise truth particle, find which
        # predicted tracks contain its hits. Majority predicted track is its
        # "match".  Then label as recovered / missed / merged / split.
        truth_pids = np.unique(pids[pids > 0])
        truth_to_pred_count: dict[int, Counter] = {}
        for pid in truth_pids.tolist():
            truth_to_pred_count[pid] = Counter()
        # Map hit -> predicted-track index (or -1 if unassigned).
        hit_to_pred = np.full(df.height, -1, dtype=np.int64)
        for t_idx, t in enumerate(tracks):
            for h in t.hit_indices:
                hit_to_pred[h] = t_idx
        for h, pid in enumerate(pids.tolist()):
            if pid > 0 and hit_to_pred[h] >= 0:
                truth_to_pred_count[pid][int(hit_to_pred[h])] += 1

        # Reverse: pred -> truth majority.
        pred_to_truth_count: list[Counter] = [Counter() for _ in tracks]
        for h, pid in enumerate(pids.tolist()):
            if hit_to_pred[h] >= 0 and pid > 0:
                pred_to_truth_count[int(hit_to_pred[h])][pid] += 1

        truth_counts = Counter(pids[pids > 0].tolist())

        recovered = 0
        missed = 0
        merged_truth_ids: set[int] = set()
        split_truth_ids: set[int] = set()
        for pid in truth_pids.tolist():
            preds = truth_to_pred_count[pid]
            if not preds:
                missed += 1
                continue
            top_pred, top_cnt = preds.most_common(1)[0]
            # Split: this truth particle's hits are spread across >=2 preds with non-trivial count.
            non_trivial = [c for c in preds.values() if c >= 2]
            if len(non_trivial) >= 2:
                split_truth_ids.add(pid)
            # Recovered double-majority?
            t_count = truth_counts[pid]
            if top_cnt * 2 > t_count and top_cnt * 2 > len(tracks[top_pred].hit_indices):
                # Also confirm the predicted track is not predominantly another particle.
                pred_top, _ = pred_to_truth_count[top_pred].most_common(1)[0]
                if pred_top == pid:
                    recovered += 1
                else:
                    merged_truth_ids.add(pid)
            else:
                # Top prediction doesn't reach the majority threshold -- this is a split/merge case.
                if top_cnt < t_count:
                    split_truth_ids.add(pid)

        # Predicted tracks that have zero hits from any non-noise truth particle = full fakes.
        full_fakes = 0
        for t in tracks:
            tp = pids[t.hit_indices]
            if (tp > 0).sum() == 0:
                full_fakes += 1

        # Merged: when one predicted track's hits come from >= 2 distinct truth particles
        # (each with >=2 hits in the track), treat both truth particles as "merged together".
        merged_pred = 0
        for c in pred_to_truth_count:
            high = [v for v in c.values() if v >= 2]
            if len(high) >= 2:
                merged_pred += 1

        breakdown.append(
            {
                "event_id": int(e_meta["event_id"]),
                "trackml_score": float(e_meta["trackml_score"]),
                "n_truth_tracks": len(truth_pids),
                "n_predicted_tracks": len(tracks),
                "n_full_fake_predicted": int(full_fakes),
                "n_merged_predicted_tracks": int(merged_pred),
                "n_truth_recovered": int(recovered),
                "n_truth_missed": int(missed),
                "n_truth_split": len(split_truth_ids),
                "n_truth_merged": len(merged_truth_ids),
                "n_hits": int(df.height),
            },
        )

    (DIAG / "track_failures.json").write_text(json.dumps(breakdown, indent=2))
    return breakdown


# ---------------------------------------------------------------------------
def step5_edge_diagnostic(
    bottom: list[dict], n_pick: int, *, embedding, gnn, device, knn_k,
) -> dict:
    random.seed(7)
    picked = random.sample(bottom, min(n_pick, len(bottom)))
    events_dir = SHARD / "events"
    ds = EventParquetDataset([SHARD], limit=1)
    feat_cols = ds.feature_cols

    truth_scores: list[float] = []
    false_scores: list[float] = []
    per_event_metrics: list[dict] = []

    for e_meta in picked:
        eid = int(e_meta["event_id"])
        df = pl.read_parquet(events_dir / f"{eid}.parquet")
        x = np.stack(
            [df[c].to_numpy().astype(np.float32) for c in feat_cols],
            axis=1,
        )
        x_t = torch.from_numpy(x).to(device)
        pids_t = torch.from_numpy(df["particle_id"].to_numpy().astype(np.int64)).to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                enabled=device.type == "cuda"):
                emb = embedding(x_t).float()
            ei = build_knn_graph(emb, k=knn_k)
            labels = edge_label_from_truth(ei, pids_t).to(device)
            ea = build_edge_features(x_t, ei, emb=emb)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                enabled=device.type == "cuda"):
                logits = gnn(x_t, ei, ea).float()
            scores = torch.sigmoid(logits)

        s_np = scores.cpu().numpy()
        l_np = labels.cpu().numpy()
        truth_scores.extend(s_np[l_np > 0.5].tolist())
        false_scores.extend(s_np[l_np <= 0.5].tolist())

        auc = edge_auc(scores, labels)
        rec = recall_at_k(emb, pids_t, k=10)
        per_event_metrics.append(
            {
                "event_id": eid,
                "trackml_score": float(e_meta["trackml_score"]),
                "edge_auc": float(auc),
                "recall@10": float(rec),
                "n_truth_edges": int((l_np > 0.5).sum()),
                "n_total_edges": int(l_np.size),
                "truth_edge_score_mean": float(s_np[l_np > 0.5].mean() if (l_np > 0.5).any() else 0.0),
                "false_edge_score_mean": float(s_np[l_np <= 0.5].mean() if (l_np <= 0.5).any() else 0.0),
            },
        )

    # Histogram comparison.
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 41)
    plt.hist(truth_scores, bins=bins, alpha=0.6, label=f"truth edges (n={len(truth_scores)})", color="green")
    plt.hist(false_scores, bins=bins, alpha=0.6, label=f"false edges (n={len(false_scores)})", color="red")
    plt.axvline(0.7, linestyle="--", color="black", label="threshold=0.7")
    plt.xlabel("edge score")
    plt.ylabel("edges")
    plt.title("Edge scores on 5 bottom-decile events (pre-threshold)")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(DIAG / "edge_scores_bad_events.png", dpi=110)
    plt.close()

    return {"per_event_metrics": per_event_metrics}


# ---------------------------------------------------------------------------
def step6_noise_correlation(df: pl.DataFrame) -> dict:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["noise_frac"].to_numpy(), df["trackml_score"].to_numpy(),
                alpha=0.4, s=10)
    plt.xlabel("noise hit fraction")
    plt.ylabel("TrackML score")
    plt.title("Score vs noise fraction")
    plt.tight_layout()
    plt.savefig(DIAG / "noise_correlation.png", dpi=110)
    plt.close()

    # Pearson correlation.
    x = df["noise_frac"].to_numpy()
    y = df["trackml_score"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0

    plt.figure(figsize=(8, 5))
    plt.scatter(df["n_tracks_truth"].to_numpy(), df["trackml_score"].to_numpy(),
                alpha=0.4, s=10)
    plt.xlabel("n truth tracks")
    plt.ylabel("TrackML score")
    plt.title("Score vs n truth tracks")
    plt.tight_layout()
    plt.savefig(DIAG / "score_vs_n_tracks.png", dpi=110)
    plt.close()

    r_ntrk = float(np.corrcoef(df["n_tracks_truth"].to_numpy(),
                                df["trackml_score"].to_numpy())[0, 1])

    # Edge AUC distribution across all events.
    plt.figure(figsize=(8, 5))
    plt.hist(df["edge_auc"].to_numpy(), bins=40, alpha=0.8, edgecolor="black")
    plt.axvline(0.99, linestyle="--", color="red", label="0.99")
    plt.xlabel("edge AUC per event")
    plt.ylabel("events")
    plt.title("Edge AUC distribution (1500 events)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIAG / "edge_auc_distribution.png", dpi=110)
    plt.close()

    return {
        "pearson_score_vs_noise_frac": r,
        "pearson_score_vs_n_tracks": r_ntrk,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    print("[diag] loading", REPORT_JSON)
    data = json.loads(REPORT_JSON.read_text())
    per_event = data["per_event"]
    print(f"[diag] {len(per_event)} per-event records")

    print("[diag] Step 1: distribution + histogram")
    d1 = step1_distribution(per_event)
    print(f"  quantiles: {d1['quantiles']}")

    p10 = d1["quantiles"][0.10]
    print("[diag] Step 2: bottom decile <=", p10)
    bottom = step2_bottom_decile(per_event, p10)
    print(f"  {len(bottom)} events <= p10")

    print("[diag] Step 3: per-event features from parquets")
    df = step3_event_features(per_event)
    ratios = step3_ratios(df, p10)
    print(f"  n_bottom={ratios['n_bottom']}  n_rest={ratios['n_rest']}")

    # Load checkpoints once for steps 4 + 5.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_ckpt = torch.load(
        ROOT / "artifacts" / "checkpoints" / "embedding.pt",
        map_location=device, weights_only=False,
    )
    emb_cfg = emb_ckpt.get("cfg") or {}
    emb_model = HitEmbedNet(
        in_dim=int(emb_cfg.get("in_dim", 24)),
        hidden_dim=int(emb_cfg.get("hidden_dim", 128)),
        out_dim=int(emb_cfg.get("out_dim", 12)),
        num_layers=int(emb_cfg.get("num_layers", 4)),
        dropout=float(emb_cfg.get("dropout", 0.0)),
    )
    emb_model.load_state_dict(emb_ckpt["model"])
    emb_model.to(device).eval()

    gnn_ckpt = torch.load(
        ROOT / "artifacts" / "checkpoints" / "gnn_v2.pt",
        map_location=device, weights_only=False,
    )
    gnn_cfg = gnn_ckpt.get("cfg") or {}
    gnn_model = InteractionNetwork(
        node_dim=int(gnn_cfg.get("node_dim", 24)),
        edge_dim=int(gnn_cfg.get("edge_dim", 7)),
        hidden_dim=int(gnn_cfg.get("hidden_dim", 64)),
        num_iter=int(gnn_cfg.get("num_iter", 8)),
        use_checkpoint=False,
    )
    gnn_model.load_state_dict(gnn_ckpt["model"])
    gnn_model.to(device).eval()

    print("[diag] Step 4: per-track failure breakdown on 5 random bad events")
    tf = step4_full(
        bottom, n_pick=5,
        embedding=emb_model, gnn=gnn_model, device=device,
        knn_k=8, threshold=0.7, min_hits=3,
    )
    print(f"  wrote {len(tf)} event breakdowns")

    print("[diag] Step 5: edge-level diagnostic on same 5 bad events")
    e5 = step5_edge_diagnostic(
        bottom, n_pick=5,
        embedding=emb_model, gnn=gnn_model, device=device, knn_k=8,
    )

    print("[diag] Step 6: noise correlation + n_tracks correlation")
    corr = step6_noise_correlation(df)

    # ----- write the markdown report
    print("[diag] writing REPORT.md")
    write_report(d1, ratios, tf, e5, corr, df, bottom)
    print("[diag] done.")
    return 0


# ---------------------------------------------------------------------------
def write_report(
    d1: dict,
    ratios: dict,
    track_failures: list[dict],
    e5: dict,
    corr: dict,
    df: pl.DataFrame,
    bottom: list[dict],
) -> None:
    qs = d1["quantiles"]
    bot_auc_mean = float(df.filter(pl.col("trackml_score") <= qs[0.10])["edge_auc"].mean() or 0.0)
    bot_rec_mean = float(df.filter(pl.col("trackml_score") <= qs[0.10])["recall@10"].mean() or 0.0)
    rest_auc_mean = float(df.filter(pl.col("trackml_score") > qs[0.10])["edge_auc"].mean() or 0.0)
    rest_rec_mean = float(df.filter(pl.col("trackml_score") > qs[0.10])["recall@10"].mean() or 0.0)

    # Aggregate Step 4 numbers.
    sum_missed = sum(t["n_truth_missed"] for t in track_failures)
    sum_split = sum(t["n_truth_split"] for t in track_failures)
    sum_merged = sum(t["n_truth_merged"] for t in track_failures)
    sum_recovered = sum(t["n_truth_recovered"] for t in track_failures)
    sum_truth = sum(t["n_truth_tracks"] for t in track_failures)
    sum_full_fake = sum(t["n_full_fake_predicted"] for t in track_failures)
    sum_merged_pred = sum(t["n_merged_predicted_tracks"] for t in track_failures)
    sum_pred = sum(t["n_predicted_tracks"] for t in track_failures)

    lines = []
    lines.append("# Score-gap diagnostic (trackml_small, 1500 events)\n")
    lines.append("Run config: k=8, threshold=0.7, min_hits=3, GNN v2.\n")

    lines.append("## 1. Score distribution\n")
    lines.append(f"- mean:        **{d1['mean']:.4f}**")
    lines.append(f"- median (p50):**{qs[0.50]:.4f}**")
    lines.append(f"- p5:           {qs[0.05]:.4f}")
    lines.append(f"- p10:          {qs[0.10]:.4f}")
    lines.append(f"- p25:          {qs[0.25]:.4f}")
    lines.append(f"- p75:          {qs[0.75]:.4f}")
    lines.append(f"- p90:          {qs[0.90]:.4f}")
    lines.append(f"- p95:          {qs[0.95]:.4f}\n")
    lines.append(f"Mean - median gap = **{d1['mean'] - qs[0.50]:+.4f}** (long left tail).")
    lines.append("Histogram: `artifacts/diagnostic/score_histogram.png`\n")

    lines.append("## 2. Bottom decile\n")
    lines.append(f"Threshold (p10): **{qs[0.10]:.4f}**.")
    lines.append(f"Bottom decile: **{len(bottom)} events**.")
    lines.append("List: `artifacts/diagnostic/bottom_decile.json`\n")

    lines.append("## 3. Bottom-decile feature ratios (bottom / rest)\n")
    lines.append("| feature | bottom | rest | ratio |")
    lines.append("|---|---:|---:|---:|")
    for k, v in ratios["features"].items():
        flag = ""
        if v["ratio_b_over_r"] >= 1.5 or v["ratio_b_over_r"] <= 0.5:
            flag = "  **SUSPECT**"
        lines.append(
            f"| {k} | {v['bottom']:.4f} | {v['rest']:.4f} | "
            f"{v['ratio_b_over_r']:.3f}{flag} |",
        )
    lines.append("")
    lines.append("Per-event CSV: `artifacts/diagnostic/event_features.csv`\n")

    lines.append("## 4. Per-track failure breakdown (5 random bad events)\n")
    lines.append("| event_id | score | truth | pred | recovered | missed | split | merged | full_fakes | merged_pred |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in track_failures:
        lines.append(
            f"| {t['event_id']} | {t['trackml_score']:.3f} | "
            f"{t['n_truth_tracks']} | {t['n_predicted_tracks']} | "
            f"{t['n_truth_recovered']} | {t['n_truth_missed']} | "
            f"{t['n_truth_split']} | {t['n_truth_merged']} | "
            f"{t['n_full_fake_predicted']} | {t['n_merged_predicted_tracks']} |",
        )
    lines.append("")
    lines.append(
        f"Across the 5 events: **{sum_recovered}/{sum_truth}** truth recovered "
        f"({sum_recovered/max(1,sum_truth):.0%}), "
        f"**{sum_missed}** missed, "
        f"**{sum_split}** split, "
        f"**{sum_merged}** merged, "
        f"**{sum_full_fake}/{sum_pred}** predicted are full fakes "
        f"({sum_full_fake/max(1,sum_pred):.0%}), "
        f"**{sum_merged_pred}** predicted tracks merge >=2 truth particles.\n",
    )
    lines.append("Raw: `artifacts/diagnostic/track_failures.json`\n")

    lines.append("## 5. Edge-level diagnostic on the same 5 events\n")
    lines.append("| event_id | score | edge AUC | recall@10 | n_truth_edges | n_total_edges | mean truth score | mean false score |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in e5["per_event_metrics"]:
        lines.append(
            f"| {m['event_id']} | {m['trackml_score']:.3f} | "
            f"{m['edge_auc']:.4f} | {m['recall@10']:.4f} | "
            f"{m['n_truth_edges']} | {m['n_total_edges']} | "
            f"{m['truth_edge_score_mean']:.4f} | {m['false_edge_score_mean']:.4f} |",
        )
    lines.append("")
    lines.append("Edge-score histogram: `artifacts/diagnostic/edge_scores_bad_events.png`")
    lines.append("Per-event-AUC histogram: `artifacts/diagnostic/edge_auc_distribution.png`\n")

    lines.append("## 6. Bottom-decile vs rest -- edge AUC + recall\n")
    lines.append(f"- Edge AUC on bottom: **{bot_auc_mean:.4f}** (rest: {rest_auc_mean:.4f})")
    lines.append(f"- recall@10 on bottom: **{bot_rec_mean:.4f}** (rest: {rest_rec_mean:.4f})\n")

    lines.append("## 7. Correlation with topology\n")
    lines.append(f"- Pearson r(score, noise_frac):  {corr['pearson_score_vs_noise_frac']:+.4f}")
    lines.append(f"- Pearson r(score, n_tracks):    {corr['pearson_score_vs_n_tracks']:+.4f}\n")
    lines.append("Scatter: `artifacts/diagnostic/noise_correlation.png` ; `artifacts/diagnostic/score_vs_n_tracks.png`\n")

    lines.append("## VERDICT\n")
    # Decision logic (data-driven).
    # If edge AUC on bottom decile is still > 0.99 and recall@10 on bottom > 0.95,
    # the GNN/embedding are fine; the track builder is the failure surface.
    # If recall@10 drops noticeably below 0.95 on the bottom decile, embedding
    # capacity is the bottleneck.
    # If edge AUC drops below 0.99 but recall@10 stays high, GNN training is.
    if bot_rec_mean < 0.95:
        verdict = "C"
    elif bot_auc_mean < 0.99:
        verdict = "A_or_D"
    else:
        verdict = "B"

    if verdict == "B":
        lines.append("**Primary fix: option B -- uncertainty Union-Find in the track builder.**\n")
        lines.append(
            f"Justification: even on the bottom decile, edge AUC = "
            f"**{bot_auc_mean:.4f}** (rest {rest_auc_mean:.4f}) and recall@10 = "
            f"**{bot_rec_mean:.4f}** (rest {rest_rec_mean:.4f}). The GNN and "
            f"embedding are NOT under-performing on these events -- they are "
            f"essentially as good as on the easy events. What changes is "
            "purely topology: more truth tracks per event (correlation "
            f"r(score, n_tracks) = {corr['pearson_score_vs_n_tracks']:+.4f}) "
            "and possibly noise (r(score, noise_frac) = "
            f"{corr['pearson_score_vs_noise_frac']:+.4f}). In the bad events "
            f"we see **{sum_merged}** truth tracks merged and **{sum_merged_pred}** "
            "predicted tracks merging >=2 truth particles -- which is exactly the "
            "failure mode that a confidence-aware Union-Find (refuse to merge "
            "across low-score chains, score-weighted majority vote) is designed "
            "to fix.\n",
        )
    elif verdict == "C":
        lines.append("**Primary fix: option C -- higher embedding dim (or wider/deeper HitEmbedNet).**\n")
        lines.append(
            f"Justification: recall@10 drops to **{bot_rec_mean:.4f}** on the "
            f"bottom decile (vs {rest_rec_mean:.4f} on the rest). The kNN "
            "candidate graph is already missing truth edges before the GNN ever "
            "sees them, so no amount of edge-classifier or builder work can "
            "recover the lost recall. Lift the embedding capacity first.\n",
        )
    else:
        lines.append("**Primary fix: option A or D -- retrain on a curriculum that exposes the failure mode.**\n")
        lines.append(
            f"Justification: bottom-decile edge AUC = **{bot_auc_mean:.4f}** "
            f"(rest {rest_auc_mean:.4f}) -- the GNN ranks edges noticeably worse "
            "on the bad events, while recall@10 stays acceptable. That points to "
            "GNN under-fitting on whichever topology these events share. The "
            "feature-ratio table above identifies which axis (low-pT tracks, "
            "high-eta, high n_tracks, ...) to weight upwards in the curriculum.\n",
        )

    lines.append("### Why NOT the other options\n")
    if verdict == "B":
        lines.append(
            f"- **A (REDVID curriculum):** REDVID is helical+noiseless and would "
            "teach a different prior; the bad-event signal here is dense-track "
            "interference, not low-pT separation.\n"
            f"- **C (higher embedding dim):** recall@10 on bottom decile is "
            f"{bot_rec_mean:.4f} (>=0.95); kNN candidate set already contains the "
            "truth edges. More dims would not be the bottleneck.\n"
            f"- **D (low-pT augmentation):** mean_pT ratio bottom/rest = "
            f"{ratios['features']['mean_pT']['ratio_b_over_r']:.3f}. If close to "
            "1.0, pT is not the discriminator.\n",
        )
    elif verdict == "C":
        lines.append(
            "- **B (uncertainty Union-Find):** the candidate graph is already "
            "missing truth edges; no clever builder can re-introduce them.\n"
            "- **A / D (retraining):** addressing the wrong stage; recall@10 "
            "tracks the embedding stage's coverage.\n",
        )
    else:
        lines.append(
            "- **B (Union-Find):** edge AUC is the bottleneck, not the builder.\n"
            "- **C (higher emb dim):** recall@10 on bad events is acceptable.\n",
        )

    (DIAG / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import sys
    sys.exit(main())
