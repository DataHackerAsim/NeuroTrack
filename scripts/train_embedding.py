"""Train the hit-embedding metric-learning model on processed events.

Usage::

    python scripts/train_embedding.py --shard data/processed/trackml_small \\
        --limit 1000 --epochs 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurotrack.data.dataset import EventParquetDataset, split_dataset
from neurotrack.train.embedding_trainer import (
    EmbeddingTrainConfig,
    train_embedding,
)
from neurotrack.utils.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shard",
        type=Path,
        action="append",
        default=None,
        help="processed shard dir; can be repeated. Default: data/processed/trackml_small",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("artifacts/checkpoints/embedding.pt"))
    p.add_argument("--metrics", type=Path, default=Path("artifacts/checkpoints/embedding_metrics.json"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--in-dim", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--out-dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--margin", type=float, default=0.4)
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
        f"[train_embedding] shards={[str(s) for s in shards]} "
        f"total_events={len(full)} train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    cfg = EmbeddingTrainConfig(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        margin=args.margin,
        max_epochs=args.epochs,
        precision=args.precision,
        device=args.device,
        ckpt_path=args.out,
        metrics_path=args.metrics,
    )
    train_embedding(train_ds, val_ds, cfg)  # type: ignore[arg-type]
    return 0


class _Subset:
    """Lightweight subset (we keep this local to avoid importing torch.utils.data.Subset
    which would force its more rigid contract on us).
    """

    def __init__(self, parent: EventParquetDataset, indices: list[int]) -> None:
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
        return self.parent[self.indices[i]]


if __name__ == "__main__":
    sys.exit(main())
