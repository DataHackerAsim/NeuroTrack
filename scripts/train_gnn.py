"""Train the GNN edge classifier on top of a pre-trained embedding.

Usage::

    python scripts/train_gnn.py --emb-ckpt artifacts/checkpoints/embedding.pt \\
        --shard data/processed/trackml_small --limit 1000 --epochs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from neurotrack.data.dataset import EventParquetDataset, split_dataset
from neurotrack.models.embedding import HitEmbedNet
from neurotrack.train.gnn_trainer import GnnTrainConfig, train_gnn
from neurotrack.utils.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-ckpt", type=Path, required=True)
    p.add_argument("--shard", type=Path, action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--knn-max-distance", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-iter", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("artifacts/checkpoints/gnn.pt"))
    p.add_argument("--metrics", type=Path, default=Path("artifacts/checkpoints/gnn_metrics.json"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr-max", type=float, default=1.0e-3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    shards: list[Path] = args.shard or [Path("data/processed/trackml_small")]
    full = EventParquetDataset(shards, limit=args.limit)
    train_idx, val_idx, _test_idx = split_dataset(full, train_frac=0.85, val_frac=0.1)
    train_ds = _Subset(full, train_idx)
    val_ds = _Subset(full, val_idx)
    print(
        f"[train_gnn] total_events={len(full)} train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    # Load embedding model.
    ckpt = torch.load(args.emb_ckpt, map_location="cpu", weights_only=False)
    emb_cfg = ckpt.get("cfg") or {}
    emb_model = HitEmbedNet(
        in_dim=int(emb_cfg.get("in_dim", 24)),
        hidden_dim=int(emb_cfg.get("hidden_dim", 128)),
        out_dim=int(emb_cfg.get("out_dim", 12)),
        num_layers=int(emb_cfg.get("num_layers", 4)),
        dropout=float(emb_cfg.get("dropout", 0.0)),
    )
    emb_model.load_state_dict(ckpt["model"])
    emb_model.eval()
    for p in emb_model.parameters():
        p.requires_grad_(False)

    cfg = GnnTrainConfig(
        knn_k=args.knn_k,
        knn_max_distance=args.knn_max_distance,
        hidden_dim=args.hidden_dim,
        num_iter=args.num_iter,
        max_epochs=args.epochs,
        precision=args.precision,
        device=args.device,
        lr_max=args.lr_max,
        ckpt_path=args.out,
        metrics_path=args.metrics,
    )
    train_gnn(train_ds, val_ds, cfg, embedding_model=emb_model)  # type: ignore[arg-type]
    return 0


class _Subset:
    def __init__(self, parent: EventParquetDataset, indices: list[int]) -> None:
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
        return self.parent[self.indices[i]]


if __name__ == "__main__":
    sys.exit(main())
