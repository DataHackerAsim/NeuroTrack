"""Trainer for the hit-embedding metric-learning model.

Single-event-per-step training (events vary in size from a few hundred to
several thousand hits, so a static batch size doesn't fit cleanly).  We
loop one event at a time, mine triplets, compute the hinge loss, and
step.

bf16 autocast and ``torch.cuda.amp.GradScaler`` are off because bf16
doesn't need a loss scaler -- the dynamic range is already sufficient.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurotrack.data.dataset import EventBatch, EventParquetDataset
from neurotrack.eval.metrics import recall_at_k
from neurotrack.models.embedding import HitEmbedNet
from neurotrack.models.losses import hinge_embedding_loss


# ---------------------------------------------------------------------------
@dataclass
class EmbeddingTrainConfig:
    in_dim: int = 24
    hidden_dim: int = 128
    out_dim: int = 12
    num_layers: int = 4
    dropout: float = 0.0
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    margin: float = 0.4
    n_anchors: int = 256
    n_pos_per_anchor: int = 4
    n_neg_per_anchor: int = 16
    hard_neg_ratio: float = 0.5
    max_epochs: int = 10
    grad_clip: float = 1.0
    precision: str = "bf16"   # 'fp32' | 'bf16'
    device: str = "cuda"
    log_every: int = 50
    val_recall_k: int = 10
    ckpt_path: Path = Path("artifacts/checkpoints/embedding.pt")
    metrics_path: Path = Path("artifacts/checkpoints/embedding_metrics.json")


# ---------------------------------------------------------------------------
def _autocast(precision: str) -> torch.autocast:
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def train_embedding(
    train_ds: EventParquetDataset,
    val_ds: EventParquetDataset,
    cfg: EmbeddingTrainConfig,
    *,
    model: HitEmbedNet | None = None,
) -> tuple[HitEmbedNet, dict[str, list[float]]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if model is None:
        model = HitEmbedNet(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.max_epochs * len(train_ds)))

    history: dict[str, list[float]] = {
        "epoch": [], "step": [],
        "train_loss": [], "train_d_pos": [], "train_d_neg": [], "train_active_frac": [],
        "val_loss": [], "val_recall_at_k": [], "epoch_time_s": [],
    }

    global_step = 0
    cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0

    for epoch in range(cfg.max_epochs):
        epoch_t0 = time.time()
        model.train()
        loss_acc = 0.0
        dpos_acc = 0.0
        dneg_acc = 0.0
        active_acc = 0.0
        n_steps = 0
        for idx in range(len(train_ds)):
            ev: EventBatch = train_ds[idx]
            ev = ev.to(device)
            if ev.n < 4:
                continue
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg.precision):
                emb = model(ev.x)
                loss, info = hinge_embedding_loss(
                    emb, ev.particle_ids,
                    margin=cfg.margin,
                    n_anchors=cfg.n_anchors,
                    n_pos_per_anchor=cfg.n_pos_per_anchor,
                    n_neg_per_anchor=cfg.n_neg_per_anchor,
                    hard_neg_ratio=cfg.hard_neg_ratio,
                )
            if not torch.isfinite(loss):
                continue
            loss.backward()  # type: ignore[no-untyped-call]
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_acc += float(loss.item())
            dpos_acc += info["d_pos"]
            dneg_acc += info["d_neg"]
            active_acc += info["active_frac"]
            n_steps += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                print(
                    f"  [emb e{epoch} s{global_step}] "
                    f"loss={loss.item():.4f} d_pos={info['d_pos']:.3f} "
                    f"d_neg={info['d_neg']:.3f} active={info['active_frac']:.3f}",
                    flush=True,
                )

        if n_steps == 0:
            print(f"  [emb e{epoch}] no training steps -- empty dataset?", flush=True)
            continue
        train_loss = loss_acc / n_steps
        train_dpos = dpos_acc / n_steps
        train_dneg = dneg_acc / n_steps
        train_active = active_acc / n_steps

        # Validation: loss + recall@k on a subset.
        model.eval()
        val_loss = 0.0
        val_recall = 0.0
        v_steps = 0
        with torch.no_grad():
            for vidx in range(min(len(val_ds), 50)):
                ev = val_ds[vidx]
                ev = ev.to(device)
                if ev.n < 4:
                    continue
                with _autocast(cfg.precision):
                    emb = model(ev.x)
                    loss, _ = hinge_embedding_loss(
                        emb, ev.particle_ids,
                        margin=cfg.margin,
                        n_anchors=cfg.n_anchors,
                        n_pos_per_anchor=cfg.n_pos_per_anchor,
                        n_neg_per_anchor=cfg.n_neg_per_anchor,
                        hard_neg_ratio=cfg.hard_neg_ratio,
                    )
                val_loss += float(loss.item())
                val_recall += recall_at_k(emb, ev.particle_ids, k=cfg.val_recall_k)
                v_steps += 1
        val_loss = val_loss / max(1, v_steps)
        val_recall = val_recall / max(1, v_steps)

        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["train_loss"].append(train_loss)
        history["train_d_pos"].append(train_dpos)
        history["train_d_neg"].append(train_dneg)
        history["train_active_frac"].append(train_active)
        history["val_loss"].append(val_loss)
        history["val_recall_at_k"].append(val_recall)
        history["epoch_time_s"].append(time.time() - epoch_t0)

        print(
            f"[emb e{epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_recall@{cfg.val_recall_k}={val_recall:.4f}  "
            f"d_pos={train_dpos:.3f}/d_neg={train_dneg:.3f}  "
            f"({history['epoch_time_s'][-1]:.1f}s)",
            flush=True,
        )

        # Save best by recall.
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_recall": val_recall,
                },
                cfg.ckpt_path,
            )

    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(history, indent=2))
    return model, history


__all__ = ["EmbeddingTrainConfig", "train_embedding"]
