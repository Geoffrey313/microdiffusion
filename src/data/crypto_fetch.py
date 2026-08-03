#!/usr/bin/env python3
"""
crypto_fetch.py - download the external cryptocurrency quotes feed (Appendix E).

Fetches the Coinbase best-quote archive for the screened instrument panel over a
list of first-of-month days, from the Tardis public datasets service. The free
tier serves the first calendar day of each month at no cost and without a key,
which is exactly the walk-forward day-block granularity the external check uses.
Each response is a gzip-compressed CSV of the best bid and ask, one row per
change of the top of book. Files land in CRYPTO_DIR as <SYM>_<YYYY-MM-DD>.csv.gz,
which is where crypto_panel.py reads them.

This step needs network access and is therefore kept out of the deterministic
reproduction chain: analysis/crypto_transfer.py self-skips when the feed is
absent. Provenance and licence terms are documented in data/README.md.

Usage
    python src/data/crypto_fetch.py                       # default panel and dates
    python src/data/crypto_fetch.py --dates 2026-05-01,2026-06-01
    MICRODIFFUSION_CRYPTO_DIR=/path python src/data/crypto_fetch.py
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CRYPTO_DIR
from data.crypto_panel import PANEL

DATASET = "https://datasets.tardis.dev/v1/coinbase/quotes/{y}/{m}/{d}/{sym}.csv.gz"
# First-of-month days used as walk-forward blocks; every one is served free.
DEFAULT_DATES = (
    "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01",
    "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01",
)


def fetch_one(sym: str, date: str, out_dir: Path) -> tuple[str, int]:
    """Download one (symbol, day) archive; return (status, bytes). Skips if present."""
    out = out_dir / f"{sym}_{date}.csv.gz"
    if out.exists() and out.stat().st_size > 0:
        return "cached", out.stat().st_size
    y, m, d = date.split("-")
    url = DATASET.format(y=y, m=m, d=d, sym=sym)
    r = subprocess.run(["curl", "-s", "--compressed", "--max-time", "300", url],
                       capture_output=True)
    # An error or an empty window comes back as a single JSON/HTML line, not gzip.
    if not r.stdout or r.stdout[:2] != b"\x1f\x8b":
        return "no data", 0
    out.write_bytes(r.stdout)
    return "ok", len(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", default=",".join(DEFAULT_DATES),
                    help="comma-separated first-of-month days (YYYY-MM-DD)")
    ap.add_argument("--pairs", default=",".join(PANEL),
                    help="comma-separated Coinbase products")
    ap.add_argument("--out", default=str(CRYPTO_DIR), help="destination directory")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    total = len(dates) * len(pairs)
    print(f"{total} files -> {out_dir}")
    done = 0
    for sym in pairs:
        for date in dates:
            done += 1
            status, size = fetch_one(sym, date, out_dir)
            print(f"  [{done}/{total}] {sym}_{date}: {status} ({size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
