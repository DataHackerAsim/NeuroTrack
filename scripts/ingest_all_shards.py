"""Ingest every Zenodo shard's tarball into per-event Parquet under data/processed.

One process per shard (so REDVID-50/100 -- the slowest -- runs in parallel
with the smaller TrackML shards).  Within a shard the CSV is streamed
sequentially; Polars does its own multithreading inside ``write_parquet``.

Usage::

    python scripts/ingest_all_shards.py
    python scripts/ingest_all_shards.py --shards trackml_small
    python scripts/ingest_all_shards.py --workers 4 --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from neurotrack.data.ingestion_redvid import ingest_redvid_tarball
from neurotrack.data.ingestion_trackml_reduced import ingest_trackml_tarball
from neurotrack.data.unified_schema import Source

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# Free-space requirement on the processed-dir filesystem.  35 GiB matches
# the prompt's headroom guidance (Parquet output is roughly 20-30 % of the
# CSV size, so worst-case ~25 GB plus overhead).
DEFAULT_DISK_REQUIRED_GIB = 35


# ---------------------------------------------------------------------------
# Shard table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Shard:
    name: str
    tarball: str
    source: Source
    handler_name: str  # 'redvid' | 'trackml'


SHARDS: dict[str, Shard] = {
    "redvid_helical_5050": Shard(
        name="redvid_helical_5050",
        tarball="redvid_3d_noisy-100k-events-10-to-50-helical-tracks.tar.gz",
        source=Source.REDVID_HELICAL_5050,
        handler_name="redvid",
    ),
    "redvid_helical_100": Shard(
        name="redvid_helical_100",
        tarball="redvid_3d_noisy-100k-events-50-to-100-helical-tracks.tar.gz",
        source=Source.REDVID_HELICAL_100,
        handler_name="redvid",
    ),
    "trackml_small": Shard(
        name="trackml_small",
        tarball="trackml_40k-events-10-to-50-tracks.tar.gz",
        source=Source.TRACKML_SMALL,
        handler_name="trackml",
    ),
    "trackml_large": Shard(
        name="trackml_large",
        tarball="trackml_40k-events-200-to-500-tracks.tar.gz",
        source=Source.TRACKML_LARGE,
        handler_name="trackml",
    ),
}


# ---------------------------------------------------------------------------
# Argparse / preflight
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument(
        "--shards",
        nargs="+",
        choices=list(SHARDS.keys()),
        default=list(SHARDS.keys()),
    )
    import os
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--validate-sample",
        type=int,
        default=200,
        help="rows per event to validate via Pydantic (0 = skip, -1 = all)",
    )
    p.add_argument(
        "--disk-required-gib",
        type=int,
        default=DEFAULT_DISK_REQUIRED_GIB,
    )
    return p.parse_args(argv)


def preflight(args: argparse.Namespace) -> int | None:
    """Refuse to start if tarballs are missing or disk space is short."""
    missing: list[str] = []
    for sname in args.shards:
        p = args.raw_dir / SHARDS[sname].tarball
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(
            "[ingest_all] missing tarballs (run scripts/download_zenodo.py first):",
            file=sys.stderr,
        )
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        return 2

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.processed_dir).free
    needed = args.disk_required_gib * (1 << 30)
    if free < needed:
        print(
            f"[ingest_all] only {free / (1<<30):.1f} GiB free at "
            f"{args.processed_dir}, need >= {args.disk_required_gib} GiB. STOP.",
            file=sys.stderr,
        )
        return 2
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(payload: tuple[str, str, str, str, bool, int]) -> dict[str, object]:
    shard_name, tar_path_s, processed_dir_s, source_value, force, sample = payload
    shard = SHARDS[shard_name]
    handler: Callable[..., dict[str, object]] = (
        ingest_redvid_tarball
        if shard.handler_name == "redvid"
        else ingest_trackml_tarball
    )
    try:
        stats = handler(
            Path(tar_path_s),
            Path(processed_dir_s),
            Source(source_value),
            force=force,
            validate_sample=sample,
        )
        stats["status"] = "OK"
        stats["shard"] = shard_name
        return stats
    except Exception as e:
        return {
            "status": "ERR",
            "shard": shard_name,
            "error": f"{type(e).__name__}: {e}",
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rc = preflight(args)
    if rc is not None:
        return rc

    payloads = [
        (
            sname,
            str(args.raw_dir / SHARDS[sname].tarball),
            str(args.processed_dir),
            SHARDS[sname].source.value,
            args.force,
            args.validate_sample,
        )
        for sname in args.shards
    ]

    print(
        f"[ingest_all] {len(args.shards)} shards "
        f"(workers={args.workers}, force={args.force})",
    )
    t0 = time.time()
    results: list[dict[str, object]] = []
    n_workers = min(args.workers, len(payloads))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_worker, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            stats = fut.result()
            results.append(stats)
            tag = "OK   " if stats.get("status") == "OK" else "ERR  "
            print(
                f"[ingest_all] {tag} shard {stats.get('shard')} "
                f"events={stats.get('n_events', '?')} "
                f"rows={stats.get('n_rows', '?')} "
                f"wall={stats.get('wall_time_s', '?')} s",
            )

    total_wall = round(time.time() - t0, 2)

    # Final summary table.
    print()
    print(f"{'shard':<22s}  {'events':>8s}  {'rows':>12s}  {'particles':>10s}  {'wall_s':>8s}  size_mb")
    print("-" * 90)
    overall_events = 0
    overall_rows = 0
    overall_particles = 0
    for r in sorted(results, key=lambda d: str(d.get("shard"))):
        if r.get("status") != "OK":
            err_label = str(r.get("shard"))
            print(f"{err_label:<22s}  ERROR: {r.get('error')}")
            continue
        sd = Path(str(r.get("shard_dir")))
        size_mb = (
            sum(p.stat().st_size for p in sd.rglob("*.parquet")) / (1 << 20)
            if sd.exists() else 0.0
        )
        n_ev = int(r.get("n_events", 0) or 0)  # type: ignore[arg-type]
        n_rw = int(r.get("n_rows", 0) or 0)  # type: ignore[arg-type]
        n_pt = int(r.get("n_particles", 0) or 0)  # type: ignore[arg-type]
        overall_events += n_ev
        overall_rows += n_rw
        overall_particles += n_pt
        shard_label = str(r.get("shard"))
        wall = float(r.get("wall_time_s", 0) or 0)  # type: ignore[arg-type]
        print(
            f"{shard_label:<22s}  {n_ev:>8d}  {n_rw:>12d}  "
            f"{n_pt:>10d}  {wall:>8.1f}  {size_mb:>7.1f}",
        )
    print("-" * 90)
    total_label = "TOTAL"
    print(
        f"{total_label:<22s}  {overall_events:>8d}  {overall_rows:>12d}  "
        f"{overall_particles:>10d}  {total_wall:>8.1f}",
    )

    # Aggregate manifest at the processed-dir root.
    (args.processed_dir / "_zenodo_ingest_summary.json").write_text(
        json.dumps(
            {"results": results, "total_wall_s": total_wall},
            indent=2,
        ),
        encoding="utf-8",
    )

    n_err = sum(1 for r in results if r.get("status") != "OK")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
