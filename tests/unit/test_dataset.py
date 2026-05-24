"""Tests for data/dataset.py."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from neurotrack.data.dataset import EventParquetDataset, split_dataset
from neurotrack.data.unified_schema import UNIFIED_FEATURES


def _make_event(tmp: Path, eid: int, n: int = 6) -> None:
    ev_dir = tmp / "events"
    ev_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, list] = {c: [float(i + eid) for i in range(n)] for c in UNIFIED_FEATURES}
    data["event_id"] = [eid] * n
    data["hit_id"] = list(range(n))
    data["particle_id"] = [(i % 3) for i in range(n)]
    data["weight"] = [0.1] * n
    data["source"] = ["test"] * n
    pl.DataFrame(data).write_parquet(ev_dir / f"{eid}.parquet")


class TestDataset:
    @pytest.fixture
    def shard(self, tmp_path: Path) -> Path:
        for eid in [1, 2, 3]:
            _make_event(tmp_path, eid)
        return tmp_path

    def test_loads_all_events(self, shard: Path) -> None:
        ds = EventParquetDataset([shard])
        assert len(ds) == 3
        ev = ds[0]
        assert ev.x.shape == (6, len(UNIFIED_FEATURES))
        assert ev.x.dtype == torch.float32
        assert ev.particle_ids.dtype == torch.int64

    def test_to_device_roundtrip(self, shard: Path) -> None:
        ds = EventParquetDataset([shard])
        ev = ds[0].to("cpu")
        assert ev.x.device == torch.device("cpu")

    def test_split_dataset(self, shard: Path) -> None:
        ds = EventParquetDataset([shard])
        # Force fractions so all three splits get >= 1 with n=3.
        train, val, test = split_dataset(ds, train_frac=1.0 / 3, val_frac=1.0 / 3)
        all_idx = sorted(train + val + test)
        assert all_idx == list(range(3))

    def test_missing_shard_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            EventParquetDataset([tmp_path / "nope"])

    def test_limit_caps(self, shard: Path) -> None:
        ds = EventParquetDataset([shard], limit=2)
        assert len(ds) == 2
