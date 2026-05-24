"""Per-event Parquet dataset.

Yields one ``EventBatch`` per Parquet file.  Each event is a variable-size
point cloud (hundreds to thousands of hits), so we batch one event per
training step rather than concatenating events.  Heavy artefacts
(per-event Parquets) are read lazily via Polars.

Schema produced by the Zenodo ingestion (R-B):

    24 unified features (UNIFIED_FEATURES)
    + event_id, hit_id, particle_id, weight, source

A note on shard semantics
-------------------------
For REDVID shards we treat ``track_id`` (renamed to ``particle_id`` during
ingestion) as the particle label.  REDVID has no noise hits, so every
``particle_id`` corresponds to a real track.

For TrackML shards ``particle_id == 0`` denotes noise / unassociated hits.
The dataset preserves this verbatim; the downstream truth-graph builder
filters them out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from .unified_schema import UNIFIED_FEATURES


# ---------------------------------------------------------------------------
@dataclass
class EventBatch:
    """One event, ready for PyTorch consumption.

    All tensors live on CPU by default; the trainer moves them to GPU.
    """

    event_id: int
    source: str
    x: torch.Tensor          # (N, F)  -- float32 features
    particle_ids: torch.Tensor  # (N,)  -- int64
    weights: torch.Tensor    # (N,)    -- float32
    hit_ids: torch.Tensor    # (N,)    -- int64
    n: int                   # number of hits

    def to(self, device: torch.device | str) -> EventBatch:
        return EventBatch(
            event_id=self.event_id,
            source=self.source,
            x=self.x.to(device),
            particle_ids=self.particle_ids.to(device),
            weights=self.weights.to(device),
            hit_ids=self.hit_ids.to(device),
            n=self.n,
        )


# ---------------------------------------------------------------------------
class EventParquetDataset(Dataset[EventBatch]):
    """Loads one event Parquet per ``__getitem__``.

    Parameters
    ----------
    shard_dirs
        One or more ``<processed_root>/<shard>`` directories.  Their
        ``events/*.parquet`` files are concatenated to form the dataset.
    limit
        Optional cap on the number of events; useful for smoke runs.
    sort_by_id
        If True, sort events by their integer id for determinism.
    """

    def __init__(
        self,
        shard_dirs: list[Path],
        *,
        limit: int | None = None,
        sort_by_id: bool = True,
    ) -> None:
        self.paths: list[Path] = []
        for shard in shard_dirs:
            ev_dir = Path(shard) / "events"
            if not ev_dir.is_dir():
                raise FileNotFoundError(f"missing events dir: {ev_dir}")
            shard_files = list(ev_dir.glob("*.parquet"))
            if sort_by_id:
                shard_files.sort(key=lambda p: int(p.stem))
            self.paths.extend(shard_files)
        if not self.paths:
            raise RuntimeError(f"no parquet events under {shard_dirs}")
        if limit is not None:
            self.paths = self.paths[:limit]
        self.feature_cols: tuple[str, ...] = UNIFIED_FEATURES

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> EventBatch:
        p = self.paths[idx]
        df = pl.read_parquet(p)
        x = np.stack(
            [df[c].to_numpy().astype(np.float32) for c in self.feature_cols],
            axis=1,
        )
        return EventBatch(
            event_id=int(df["event_id"][0]),
            source=str(df["source"][0]),
            x=torch.from_numpy(x),
            particle_ids=torch.from_numpy(
                df["particle_id"].to_numpy().astype(np.int64),
            ),
            weights=torch.from_numpy(
                df["weight"].to_numpy().astype(np.float32),
            ),
            hit_ids=torch.from_numpy(
                df["hit_id"].to_numpy().astype(np.int64),
            ),
            n=df.height,
        )


# ---------------------------------------------------------------------------
def split_dataset(
    ds: EventParquetDataset,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 1337,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train, val, test) index lists deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    n = len(ds)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train : n_train + n_val].tolist()
    test_idx = perm[n_train + n_val :].tolist()
    return train_idx, val_idx, test_idx


__all__ = ["EventBatch", "EventParquetDataset", "split_dataset"]
