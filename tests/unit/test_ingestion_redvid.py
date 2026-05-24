"""Unit tests for ``data.ingestion_redvid.ingest_redvid_tarball``.

A synthetic 1-event REDVID tarball is built in ``tmp_path`` with the exact
column names and ``;`` separator the real Zenodo tarballs use.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from neurotrack.data.ingestion_redvid import (
    EXPECTED_COLUMNS,
    ingest_redvid_tarball,
)
from neurotrack.data.unified_schema import UNIFIED_FEATURES, Source


def _build_redvid_csv(n_rows: int = 5, event_id: int = 42) -> bytes:
    header = ";".join(f'"{c}"' for c in EXPECTED_COLUMNS)
    lines = [header]
    for i in range(n_rows):
        # Quoted strings, semicolon delimited.
        lines.append(
            f'{event_id};{i % 3};"strip";{i};"helical_expanding";'
            f"0;0;0;{0.5 + i * 0.01};{1 if i % 2 else -1};{1.5 + i * 0.01};"
            f"{i};{1.0 + i * 0.1};{0.5 + i * 0.05};{2.0 + i * 0.02}",
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_tarball(tmp_path: Path, csv_bytes: bytes) -> Path:
    tar_path = tmp_path / "fake_redvid.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="fake_redvid/events_all/data.csv")
        info.size = len(csv_bytes)
        tf.addfile(info, io.BytesIO(csv_bytes))
    return tar_path


class TestIngestRedvid:
    @pytest.fixture
    def tar_path(self, tmp_path: Path) -> Path:
        return _make_tarball(tmp_path, _build_redvid_csv(n_rows=5, event_id=42))

    def test_writes_event_parquet(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        stats = ingest_redvid_tarball(
            tar_path, out_dir, Source.REDVID_HELICAL_5050, validate_sample=-1,
        )
        assert stats["n_events"] == 1
        assert stats["n_events_written"] == 1
        assert stats["n_rows"] == 5
        ev_path = (
            out_dir / Source.REDVID_HELICAL_5050.value / "events" / "42.parquet"
        )
        assert ev_path.exists()

    def test_parquet_has_unified_features(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        ingest_redvid_tarball(
            tar_path, out_dir, Source.REDVID_HELICAL_5050, validate_sample=0,
        )
        df = pl.read_parquet(
            out_dir / Source.REDVID_HELICAL_5050.value / "events" / "42.parquet",
        )
        # All 24 unified features present, plus bookkeeping columns.
        for c in UNIFIED_FEATURES:
            assert c in df.columns
        for c in ("event_id", "hit_id", "particle_id", "weight", "source"):
            assert c in df.columns
        assert df.height == 5
        # No null / NaN in features.
        for c in UNIFIED_FEATURES:
            arr = df[c].to_numpy()
            assert np.isfinite(arr).all(), c

    def test_idempotent_skip(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        s1 = ingest_redvid_tarball(
            tar_path, out_dir, Source.REDVID_HELICAL_5050, validate_sample=0,
        )
        s2 = ingest_redvid_tarball(
            tar_path, out_dir, Source.REDVID_HELICAL_5050, validate_sample=0,
        )
        assert s1["n_events_written"] == 1
        assert s2["n_events_written"] == 0
        assert s2["n_events_skipped"] == 1

    def test_metadata_json_written(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        ingest_redvid_tarball(
            tar_path, out_dir, Source.REDVID_HELICAL_5050, validate_sample=0,
        )
        meta = out_dir / Source.REDVID_HELICAL_5050.value / "_metadata.json"
        assert meta.exists()
        import json
        m = json.loads(meta.read_text(encoding="utf-8"))
        assert m["source"] == "redvid_helical_5050"
        assert m["n_events"] == 1


class TestIngestRedvidMultiEvent:
    def test_two_events_split_correctly(self, tmp_path: Path) -> None:
        csv = _build_redvid_csv(n_rows=3, event_id=0)
        csv += _build_redvid_csv(n_rows=4, event_id=1).split(b"\n", 1)[1]  # drop header
        tar = _make_tarball(tmp_path, csv)
        out_dir = tmp_path / "processed"
        stats = ingest_redvid_tarball(
            tar, out_dir, Source.REDVID_HELICAL_5050, validate_sample=0,
        )
        assert stats["n_events"] == 2
        assert stats["n_rows"] == 7
        events_dir = out_dir / Source.REDVID_HELICAL_5050.value / "events"
        assert (events_dir / "0.parquet").exists()
        assert (events_dir / "1.parquet").exists()
        assert pl.read_parquet(events_dir / "0.parquet").height == 3
        assert pl.read_parquet(events_dir / "1.parquet").height == 4
