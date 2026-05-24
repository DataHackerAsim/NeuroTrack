# LEGACY. 4-CSV Kaggle layout. Replaced by scripts/ingest_all_shards.py.
"""Ingest every TrackML event under ``data/raw/train_sample/`` into Parquet.

Discovers events by scanning for ``event{event_id:09d}-hits.csv`` files (the
flat layout shipped by Kaggle).  Each event is processed in a child process;
the pool is sized to ``cpu_count() // 2`` to leave headroom for the Polars /
pyarrow threads spawned inside each worker.

Usage
-----
    python scripts/ingest.py
    python scripts/ingest.py --raw data/raw/train_sample --out data/processed/events
    python scripts/ingest.py --workers 6 --force
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from neurotrack.data.ingestion_legacy import ingest_event

_HITS_RE = re.compile(r"event(\d{9})-hits\.csv$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/train_sample"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/events"))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="process pool size (default cpu_count() // 2)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing parquets")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="ingest at most N events (useful for smoke tests)",
    )
    parser.add_argument("--validate-sample", type=int, default=100)
    return parser.parse_args(argv)


def discover_events(raw_dir: Path) -> list[int]:
    """Return sorted list of event_ids found under ``raw_dir``."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dir {raw_dir} does not exist")
    eids: list[int] = []
    for entry in raw_dir.iterdir():
        m = _HITS_RE.match(entry.name)
        if m:
            eids.append(int(m.group(1)))
    return sorted(set(eids))


def _worker(args: tuple[Path, Path, int, bool, int]) -> tuple[int, str]:
    raw, out, eid, force, sample = args
    try:
        ev = ingest_event(raw, out, event_id=eid, force=force, validate_sample=sample)
        return ev.event_id, f"OK n_hits={ev.n_hits} n_part={ev.n_particles}"
    except Exception as e:  # pragma: no cover  -- surfaced through the pool
        return eid, f"ERROR {type(e).__name__}: {e}"


_DOWNLOAD_HINT = "Run 'python scripts/download_zenodo.py' first."


def _preflight(raw_dir: Path) -> int | None:
    """Return an exit code if the raw dir is unusable, else None to proceed.

    The Zenodo distribution lands tarballs in ``data/raw/`` (plus a renamed
    ``trackml_README.html``).  Ingestion proper still expects the legacy flat
    ``event*-hits.csv`` layout -- Prompt R-B will rewire ingestion to stream
    from tarballs.  Until then the preflight just checks the Zenodo artefacts
    are present so we fail fast with an actionable hint.
    """
    if not raw_dir.exists():
        print(f"[ingest] raw dir does not exist: {raw_dir}", file=sys.stderr)
        print(_DOWNLOAD_HINT, file=sys.stderr)
        return 2
    if not raw_dir.is_dir():
        print(f"[ingest] raw path is not a directory: {raw_dir}", file=sys.stderr)
        return 2

    tarballs = [p for p in raw_dir.iterdir() if p.suffixes[-2:] == [".tar", ".gz"]]
    readme = raw_dir / "trackml_README.html"
    if not tarballs or not readme.exists():
        missing: list[str] = []
        if not tarballs:
            missing.append("*.tar.gz")
        if not readme.exists():
            missing.append("trackml_README.html")
        print(
            f"[ingest] missing under {raw_dir}: {', '.join(missing)}",
            file=sys.stderr,
        )
        print(_DOWNLOAD_HINT, file=sys.stderr)
        return 2
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rc = _preflight(args.raw)
    if rc is not None:
        return rc

    eids = discover_events(args.raw)
    if args.limit is not None:
        eids = eids[: args.limit]
    if not eids:
        print(f"[ingest] no events found under {args.raw}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"[ingest] {len(eids)} events from {args.raw} -> {args.out} "
        f"(workers={args.workers}, force={args.force})",
    )

    payload = [(args.raw, args.out, eid, args.force, args.validate_sample) for eid in eids]

    n_ok = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed(pool.submit(_worker, p) for p in payload):
            eid, msg = fut.result()
            tag = "OK   " if msg.startswith("OK") else "ERR  "
            print(f"[ingest] {tag} event {eid:09d}  {msg}")
            if msg.startswith("OK"):
                n_ok += 1
            else:
                n_err += 1

    print(f"[ingest] done: {n_ok} ok, {n_err} errors")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
