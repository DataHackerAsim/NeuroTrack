"""Trainer for the InteractionNetwork edge classifier.

Pipeline per training step:

1. Pull one event from disk.
2. Run the (frozen) embedding model to get hit embeddings.
3. Build the kNN candidate graph from those embeddings.
4. Label each candidate edge as true / false using truth (same particle).
5. Run the GNN on raw features + candidate graph + edge features.
6. Focal BCE loss; backprop; step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from neurotrack.data.dataset import EventBatch, EventParquetDataset
from neurotrack.eval.metrics import edge_auc
from neurotrack.graph.construction import build_edge_features, build_knn_graph
from neurotrack.graph.truth import edge_label_from_truth
from neurotrack.models.embedding import HitEmbedNet
from neurotrack.models.gnn import InteractionNetwork
from neurotrack.models.losses import focal_bce_with_logits


# ---------------------------------------------------------------------------
@dataclass
class GnnTrainConfig:
    node_dim: int = 24
    edge_dim: int = 7
    hidden_dim: int = 64
    num_iter: int = 8
    use_checkpoint: bool = True

    knn_k: int = 8
    knn_max_distance: float | None = None

    lr_max: float = 1.0e-3
    weight_decay: float = 1.0e-4
    alpha: float = 0.25
    gamma: float = 2.0
    pct_start: float = 0.1
    max_epochs: int = 5
    grad_clip: float = 1.0
    precision: str = "bf16"
    device: str = "cuda"
    log_every: int = 50

    ckpt_path: Path = Path("artifacts/checkpoints/gnn.pt")
    metrics_path: Path = Path("artifacts/checkpoints/gnn_metrics.json")


# ---------------------------------------------------------------------------
def _autocast(precision: str) -> torch.autocast:
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def _truth_recall_in_knn(
    knn_ei: torch.Tensor, labels: torch.Tensor,
) -> float:
    """Fraction of truth edges (label==1) actually present in the kNN graph.

    Diagnostic for whether the candidate-graph construction is finding the
    truth edges in the first place; an upper bound on what the GNN can
    achieve.
    """
    if labels.numel() == 0:
        return 0.0
    return float(labels.float().mean().item()) * 100.0  # %% positive in the candidate set


def train_gnn(
    train_ds: EventParquetDataset,
    val_ds: EventParquetDataset,
    cfg: GnnTrainConfig,
    *,
    embedding_model: HitEmbedNet,
    gnn: InteractionNetwork | None = None,
) -> tuple[InteractionNetwork, dict[str, list[float]]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    embedding_model.to(device).eval()
    for p in embedding_model.parameters():
        p.requires_grad_(False)

    if gnn is None:
        gnn = InteractionNetwork(
            node_dim=cfg.node_dim,
            edge_dim=cfg.edge_dim,
            hidden_dim=cfg.hidden_dim,
            num_iter=cfg.num_iter,
            use_checkpoint=cfg.use_checkpoint,
        )
    gnn.to(device)

    optimizer = AdamW(gnn.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.max_epochs * len(train_ds))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr_max,
        total_steps=total_steps,
        pct_start=cfg.pct_start,
    )

    history: dict[str, list[float]] = {
        "epoch": [], "step": [],
        "train_loss": [], "train_auc": [], "train_pos_frac": [],
        "val_loss": [], "val_auc": [], "val_pos_frac": [],
        "epoch_time_s": [],
    }

    global_step = 0
    cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(cfg.max_epochs):
        epoch_t0 = time.time()
        gnn.train()
        loss_acc = 0.0
        auc_acc = 0.0
        pos_acc = 0.0
        n_steps = 0

        for idx in range(len(train_ds)):
            ev: EventBatch = train_ds[idx].to(device)
            if ev.n < 4:
                continue

            with torch.no_grad():
                with _autocast(cfg.precision):
                    emb = embedding_model(ev.x).detach().float()
                edge_index = build_knn_graph(
                    emb, k=cfg.knn_k, max_distance=cfg.knn_max_distance,
                )
                if edge_index.numel() == 0:
                    continue
                labels = edge_label_from_truth(edge_index, ev.particle_ids).to(device)
                edge_attr = build_edge_features(ev.x, edge_index, emb=emb)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg.precision):
                logits = gnn(ev.x, edge_index, edge_attr).float()
                loss = focal_bce_with_logits(
                    logits, labels, alpha=cfg.alpha, gamma=cfg.gamma,
                )
            if not torch.isfinite(loss):
                continue
            loss.backward()  # type: ignore[no-untyped-call]
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(gnn.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                scores = torch.sigmoid(logits)
                auc = edge_auc(scores, labels)
                pos_frac = float(labels.mean().item())

            loss_acc += float(loss.item())
            auc_acc += auc
            pos_acc += pos_frac
            n_steps += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                print(
                    f"  [gnn e{epoch} s{global_step}] "
                    f"loss={loss.item():.4f} auc={auc:.4f} pos_frac={pos_frac:.3f}",
                    flush=True,
                )

        if n_steps == 0:
            print(f"  [gnn e{epoch}] no training steps", flush=True)
            continue
        train_loss = loss_acc / n_steps
        train_auc = auc_acc / n_steps
        train_pos = pos_acc / n_steps

        # Validation.
        gnn.eval()
        v_loss = 0.0
        v_auc = 0.0
        v_pos = 0.0
        v_steps = 0
        with torch.no_grad():
            for vidx in range(min(len(val_ds), 50)):
                ev = val_ds[vidx].to(device)
                if ev.n < 4:
                    continue
                with _autocast(cfg.precision):
                    emb = embedding_model(ev.x).detach().float()
                edge_index = build_knn_graph(emb, k=cfg.knn_k)
                if edge_index.numel() == 0:
                    continue
                labels = edge_label_from_truth(edge_index, ev.particle_ids).to(device)
                edge_attr = build_edge_features(ev.x, edge_index, emb=emb)
                with _autocast(cfg.precision):
                    logits = gnn(ev.x, edge_index, edge_attr).float()
                    loss = focal_bce_with_logits(logits, labels, alpha=cfg.alpha, gamma=cfg.gamma)
                v_loss += float(loss.item())
                v_auc += edge_auc(torch.sigmoid(logits), labels)
                v_pos += float(labels.mean().item())
                v_steps += 1
        v_loss = v_loss / max(1, v_steps)
        v_auc = v_auc / max(1, v_steps)
        v_pos = v_pos / max(1, v_steps)

        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["train_pos_frac"].append(train_pos)
        history["val_loss"].append(v_loss)
        history["val_auc"].append(v_auc)
        history["val_pos_frac"].append(v_pos)
        history["epoch_time_s"].append(time.time() - epoch_t0)

        print(
            f"[gnn e{epoch}] train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
            f"val_loss={v_loss:.4f} val_auc={v_auc:.4f} pos_frac={v_pos:.3f} "
            f"({history['epoch_time_s'][-1]:.1f}s)",
            flush=True,
        )

        if v_auc > best_auc:
            best_auc = v_auc
            torch.save(
                {
                    "model": gnn.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_auc": v_auc,
                },
                cfg.ckpt_path,
            )

    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(history, indent=2))
    return gnn, history


__all__ = ["GnnTrainConfig", "train_gnn"]
