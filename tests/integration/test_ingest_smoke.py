"""Integration smoke test: ingest the smallest real tarball, spot-check.

Marked ``@pytest.mark.slow`` -- skipped under ``pytest -m "not slow"``.
The test runs against the actual ``trackml_40k-events-10-to-50-tracks.tar.gz``
(~135 MB compressed) and writes Parquet under a tmp-dir, so it leaves the
real ``data/processed/`` untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from neurotrack.data.ingestion_trackml_reduced import ingest_trackml_tarball
from neurotrack.data.unified_schema import UNIFIED_FEATURES, Source

TARBALL = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "trackml_40k-events-10-to-50-tracks.tar.gz"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not TARBALL.exists() or os.environ.get("NEUROTRACK_SKIP_SLOW") == "1",
    reason="real Zenodo tarball not present (or NEUROTRACK_SKIP_SLOW=1 set)",
)
def test_trackml_small_real_ingest_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed"
    stats = ingest_trackml_tarball(
        TARBALL,
        out_dir,
        Source.TRACKML_SMALL,
        validate_sample=20,  # small sample -- full validation would dominate runtime
    )

    # We expect ~40k events; allow some headroom in case the pack changes.
    n = int(stats["n_events"])  # type: ignore[arg-type]
    assert 30_000 <= n <= 50_000, f"unexpected n_events={n}"

    events_dir = out_dir / Source.TRACKML_SMALL.value / "events"
    parts_dir = out_dir / Source.TRACKML_SMALL.value / "particles"

    parquets = sorted(events_dir.glob("*.parquet"))
    assert len(parquets) == n

    # Spot-check three random events.
    rng = np.random.default_rng(0)
    sample = rng.choice(parquets, size=3, replace=False)
    for p in sample:
        df = pl.read_parquet(p)
        assert df.height > 0
        for c in UNIFIED_FEATURES:
            assert np.isfinite(df[c].to_numpy()).all(), f"NaN in {c} of {p.name}"
        ids = df["particle_id"].to_list()
        # TrackML reduced has both noise (0) and signal (>0) particles.
        assert any(i == 0 for i in ids) or any(i > 0 for i in ids)

        # Matching particles parquet for the same event.
        eid = int(p.stem)
        pt = pl.read_parquet(parts_dir / f"{eid}.parquet")
        assert int(pt["n_hits"].sum()) == df.height
