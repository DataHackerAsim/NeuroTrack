"""Unit tests for ``data.ingestion_trackml_reduced.ingest_trackml_tarball``."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from neurotrack.data.ingestion_trackml_reduced import (
    EXPECTED_COLUMNS,
    ingest_trackml_tarball,
)
from neurotrack.data.unified_schema import UNIFIED_FEATURES, Source


def _build_trackml_csv(rows: list[dict[str, str]]) -> bytes:
    header = ",".join(EXPECTED_COLUMNS)
    body = "\n".join(",".join(str(r[c]) for c in EXPECTED_COLUMNS) for r in rows)
    return (header + "\n" + body + "\n").encode("utf-8")


def _make_tarball(tmp_path: Path, csv_bytes: bytes) -> Path:
    tar_path = tmp_path / "fake_trackml.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="fake_trackml.csv")
        info.size = len(csv_bytes)
        tf.addfile(info, io.BytesIO(csv_bytes))
    return tar_path


def _row(
    event_id: int, particle_id: int, q: int, x: float, y: float, z: float,
    vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
    px: float = 1.0, py: float = 0.0, pz: float = 0.0,
    weight: float = 0.1, volume_id: int = 8,
) -> dict[str, str]:
    return {
        "x": str(x), "y": str(y), "z": str(z),
        "volume_id": str(volume_id),
        "vx": str(vx), "vy": str(vy), "vz": str(vz),
        "px": str(px), "py": str(py), "pz": str(pz),
        "q": str(q),
        "particle_id": str(particle_id),
        "weight": str(weight),
        "event_id": str(event_id),
    }


class TestIngestTrackML:
    @pytest.fixture
    def tar_path(self, tmp_path: Path) -> Path:
        rows = [
            _row(0, 1, +1, 1.0, 0.0, 5.0),
            _row(0, 1, +1, 2.0, 0.0, 10.0),
            _row(0, 2, -1, 0.0, 1.0, 5.0),
            _row(0, 0, 0, 0.5, 0.5, 0.0, weight=0.0),  # noise hit
            _row(1, 5, +1, 3.0, 0.0, 5.0),
            _row(1, 5, +1, 6.0, 0.0, 10.0),
            _row(1, 5, +1, 9.0, 0.0, 15.0),
        ]
        return _make_tarball(tmp_path, _build_trackml_csv(rows))

    def test_writes_event_and_particle_parquets(
        self, tar_path: Path, tmp_path: Path,
    ) -> None:
        out_dir = tmp_path / "processed"
        stats = ingest_trackml_tarball(
            tar_path, out_dir, Source.TRACKML_SMALL, validate_sample=-1,
        )
        assert stats["n_events"] == 2
        ev_dir = out_dir / Source.TRACKML_SMALL.value / "events"
        pt_dir = out_dir / Source.TRACKML_SMALL.value / "particles"
        assert (ev_dir / "0.parquet").exists()
        assert (ev_dir / "1.parquet").exists()
        assert (pt_dir / "0.parquet").exists()
        assert (pt_dir / "1.parquet").exists()

    def test_unified_features_present(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        ingest_trackml_tarball(
            tar_path, out_dir, Source.TRACKML_SMALL, validate_sample=0,
        )
        df = pl.read_parquet(
            out_dir / Source.TRACKML_SMALL.value / "events" / "0.parquet",
        )
        for c in UNIFIED_FEATURES:
            assert c in df.columns
        for c in ("event_id", "hit_id", "particle_id", "weight", "source"):
            assert c in df.columns
        # Both signal (particle_id != 0) and noise (== 0) present.
        ids = df["particle_id"].to_list()
        assert 0 in ids
        assert any(i != 0 for i in ids)

    def test_particles_dedup_and_n_hits(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        ingest_trackml_tarball(
            tar_path, out_dir, Source.TRACKML_SMALL, validate_sample=0,
        )
        # event 0: particle 1 has 2 hits, particle 2 has 1, particle 0 has 1.
        ev0 = pl.read_parquet(
            out_dir / Source.TRACKML_SMALL.value / "events" / "0.parquet",
        )
        pt0 = pl.read_parquet(
            out_dir / Source.TRACKML_SMALL.value / "particles" / "0.parquet",
        )
        # The dedup is one row per particle_id.
        assert pt0.height == ev0["particle_id"].n_unique()
        # n_hits column sums to total hits in the event.
        assert int(pt0["n_hits"].sum()) == ev0.height

        # event 1: single particle, 3 hits.
        pt1 = pl.read_parquet(
            out_dir / Source.TRACKML_SMALL.value / "particles" / "1.parquet",
        )
        assert pt1.height == 1
        assert int(pt1["n_hits"][0]) == 3

    def test_idempotent_skip(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        s1 = ingest_trackml_tarball(
            tar_path, out_dir, Source.TRACKML_SMALL, validate_sample=0,
        )
        s2 = ingest_trackml_tarball(
            tar_path, out_dir, Source.TRACKML_SMALL, validate_sample=0,
        )
        assert s1["n_events_written"] == 2
        assert s2["n_events_written"] == 0
        assert s2["n_events_skipped"] == 2

    def test_features_finite(self, tar_path: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "processed"
        ingest_trackml_tarball(
            tar_path, out_dir, Source.TRACKML_SMALL, validate_sample=0,
        )
        df = pl.read_parquet(
            out_dir / Source.TRACKML_SMALL.value / "events" / "0.parquet",
        )
        for c in UNIFIED_FEATURES:
            assert np.isfinite(df[c].to_numpy()).all(), c
