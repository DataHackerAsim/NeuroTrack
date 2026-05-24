"""Unit tests for ``data.ingestion.ingest_event``.

These exercise the per-event-directory layout against the synthetic event
in ``tests/fixtures/event_000001000/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from neurotrack.data.ingestion_legacy import SCHEMA_VERSION, ingest_event
from neurotrack.data.schemas import EventData

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EVENT_DIR = FIXTURES / "event_000001000"
EVENT_ID = 1000


def test_fixture_event_present() -> None:
    """Fail loudly if the test fixture is not where we expect it."""
    assert EVENT_DIR.is_dir(), f"missing fixture {EVENT_DIR}"
    for required in ("hits.csv", "particles.csv", "truth.csv", "cells.csv"):
        assert (EVENT_DIR / required).exists(), f"missing {required}"


# ---------------------------------------------------------------------------
class TestIngestEventHappyPath:
    @pytest.fixture
    def out_dir(self, tmp_path: Path) -> Path:
        out = tmp_path / "events"
        out.mkdir()
        return out

    def test_returns_event_data(self, out_dir: Path) -> None:
        ev = ingest_event(EVENT_DIR, out_dir)
        assert isinstance(ev, EventData)
        assert ev.event_id == EVENT_ID
        assert ev.n_hits == 7
        assert ev.n_particles == 2
        assert ev.n_noise_hits == 1
        assert ev.has_cells is True
        assert ev.schema_version == SCHEMA_VERSION

    def test_writes_atomic_parquet(self, out_dir: Path) -> None:
        ev = ingest_event(EVENT_DIR, out_dir)
        assert ev.parquet_path.name == "event_000001000.parquet"
        assert ev.parquet_path.exists()
        # No leftover temp files.
        for p in out_dir.iterdir():
            assert not p.name.startswith(".event_"), f"leaked tmp file: {p.name}"

    def test_parquet_columns_and_join(self, out_dir: Path) -> None:
        ev = ingest_event(EVENT_DIR, out_dir)
        df = pl.read_parquet(ev.parquet_path)

        # Hits + truth + particles all denormalised.
        for col in (
            "hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id",
            "particle_id", "hit_weight",
            "particle_vx", "particle_px", "particle_q", "particle_nhits",
            "n_cells", "cell_value_sum",
        ):
            assert col in df.columns, col

        # 7 hits, sorted by hit_id.
        assert df.height == 7
        assert df["hit_id"].to_list() == sorted(df["hit_id"].to_list())

        # Particle 1 has 3 hits; particle 2 has 3 hits; one noise hit.
        assert (df["particle_id"] == 1).sum() == 3
        assert (df["particle_id"] == 2).sum() == 3
        assert (df["particle_id"] == 0).sum() == 1

        # Charges joined correctly: particle 1 -> +1, particle 2 -> -1.
        q_p1 = df.filter(pl.col("particle_id") == 1)["particle_q"].unique().to_list()
        assert q_p1 == [1]
        q_p2 = df.filter(pl.col("particle_id") == 2)["particle_q"].unique().to_list()
        assert q_p2 == [-1]

    def test_cell_aggregation(self, out_dir: Path) -> None:
        ev = ingest_event(EVENT_DIR, out_dir)
        df = pl.read_parquet(ev.parquet_path)
        # hit 1 has 2 cells with values 0.5 + 0.3 = 0.8
        row = df.filter(pl.col("hit_id") == 1).to_dicts()[0]
        assert row["n_cells"] == 2
        assert row["cell_value_sum"] == pytest.approx(0.8)
        # hit 7 (noise) has one cell with value 0.05
        row = df.filter(pl.col("hit_id") == 7).to_dicts()[0]
        assert row["n_cells"] == 1
        assert row["cell_value_sum"] == pytest.approx(0.05)


class TestIngestEventIdempotency:
    def test_second_call_is_skip(self, tmp_path: Path) -> None:
        out = tmp_path / "events"
        out.mkdir()
        first = ingest_event(EVENT_DIR, out)
        mtime1 = first.parquet_path.stat().st_mtime_ns

        # Second call should NOT rewrite the file.
        second = ingest_event(EVENT_DIR, out)
        assert second.parquet_path == first.parquet_path
        assert second.parquet_path.stat().st_mtime_ns == mtime1
        # Manifest fields should still be correct on the skip path.
        assert second.n_hits == first.n_hits
        assert second.n_particles == first.n_particles

    def test_force_rewrites(self, tmp_path: Path) -> None:
        out = tmp_path / "events"
        out.mkdir()
        first = ingest_event(EVENT_DIR, out)
        mtime1 = first.parquet_path.stat().st_mtime_ns

        # Force a rewrite: the file should change.
        second = ingest_event(EVENT_DIR, out, force=True)
        assert second.parquet_path.stat().st_mtime_ns >= mtime1


class TestIngestEventErrors:
    def test_unknown_event_dir_name_requires_event_id(self, tmp_path: Path) -> None:
        # Copy fixture to a directory whose name does NOT encode an event id.
        weird = tmp_path / "something_else"
        shutil.copytree(EVENT_DIR, weird)
        out = tmp_path / "events"
        out.mkdir()

        with pytest.raises(ValueError, match="Cannot derive event_id"):
            ingest_event(weird, out)

        # But works if event_id is passed explicitly.
        ev = ingest_event(weird, out, event_id=42)
        assert ev.event_id == 42
        assert ev.parquet_path.name == "event_000000042.parquet"

    def test_missing_required_csv_raises(self, tmp_path: Path) -> None:
        broken = tmp_path / "event_000002000"
        shutil.copytree(EVENT_DIR, broken)
        (broken / "hits.csv").unlink()
        out = tmp_path / "events"
        out.mkdir()
        with pytest.raises(FileNotFoundError):
            ingest_event(broken, out)

    def test_optional_cells_csv_missing_is_ok(self, tmp_path: Path) -> None:
        no_cells = tmp_path / "event_000003000"
        shutil.copytree(EVENT_DIR, no_cells)
        (no_cells / "cells.csv").unlink()
        out = tmp_path / "events"
        out.mkdir()
        ev = ingest_event(no_cells, out)
        assert ev.has_cells is False
        df = pl.read_parquet(ev.parquet_path)
        assert "n_cells" not in df.columns
