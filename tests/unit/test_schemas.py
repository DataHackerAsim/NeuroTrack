"""Unit tests for the Pydantic v2 schemas."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from neurotrack.data.schemas import (
    Cell,
    EventData,
    Hit,
    Particle,
    SchemaValidationError,
    Truth,
)


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------
class TestHit:
    def test_accepts_valid_row(self) -> None:
        h = Hit(hit_id=1, x=10.0, y=0.0, z=5.0, volume_id=8, layer_id=2, module_id=1)
        assert h.hit_id == 1
        assert h.x == 10.0

    def test_rejects_zero_hit_id(self) -> None:
        with pytest.raises(ValidationError):
            Hit(hit_id=0, x=0, y=0, z=0, volume_id=8, layer_id=2, module_id=1)

    def test_rejects_negative_volume_id(self) -> None:
        with pytest.raises(ValidationError):
            Hit(hit_id=1, x=0, y=0, z=0, volume_id=-1, layer_id=2, module_id=1)


class TestParticle:
    def test_accepts_valid_row(self) -> None:
        p = Particle(particle_id=1, vx=0, vy=0, vz=0, px=1, py=0, pz=0, q=1, nhits=3)
        assert p.q == 1

    def test_rejects_charge_outside_set(self) -> None:
        with pytest.raises(ValidationError):
            Particle(particle_id=1, vx=0, vy=0, vz=0, px=0, py=0, pz=0, q=2, nhits=0)


class TestTruth:
    def test_noise_hit_has_zero_particle_id(self) -> None:
        t = Truth(hit_id=1, particle_id=0, tx=0, ty=0, tz=0, tpx=0, tpy=0, tpz=0, weight=0.0)
        assert t.particle_id == 0

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Truth(hit_id=1, particle_id=1, tx=0, ty=0, tz=0, tpx=0, tpy=0, tpz=0, weight=-0.1)


class TestCell:
    def test_accepts_valid_row(self) -> None:
        Cell(hit_id=1, ch0=10, ch1=20, value=0.5)

    def test_negative_channel_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Cell(hit_id=1, ch0=-1, ch1=0, value=0.1)


# ---------------------------------------------------------------------------
# Dataframe-level validation
# ---------------------------------------------------------------------------
class TestValidateDataframe:
    def test_passes_on_valid_frame(self) -> None:
        df = pl.DataFrame(
            {
                "hit_id": [1, 2],
                "x": [1.0, 2.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "volume_id": [8, 8],
                "layer_id": [2, 4],
                "module_id": [1, 1],
            },
        )
        Hit.validate_dataframe(df, sample_size=-1)

    def test_missing_column_raises(self) -> None:
        df = pl.DataFrame({"hit_id": [1], "x": [1.0]})
        with pytest.raises(SchemaValidationError):
            Hit.validate_dataframe(df)

    def test_polars_schema_returns_dtypes(self) -> None:
        schema = Hit.polars_schema()
        assert set(schema.keys()) == set(Hit.csv_columns)
        assert schema["hit_id"] == pl.Int64()
        assert schema["x"] == pl.Float64()


class TestEventData:
    def test_accepts_minimal(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ev = EventData(
            event_id=1000,
            parquet_path=tmp_path / "event.parquet",
            n_hits=7,
            n_particles=2,
            n_noise_hits=1,
        )
        assert ev.has_cells is True
        assert ev.schema_version == "1"
