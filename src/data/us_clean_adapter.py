#!/usr/bin/env python3
"""
us_clean_adapter.py - stage the external United States level-one panel into the
QSE clean layout so the primary battery can run on it unchanged (Appendix F).

The engine, robustness, benchmark and figure stages read per-session L1 parquet
from CLEAN_DIR under the layout CLEAN_DIR/<session>/<INSTRUMENT>_l1.parquet with
columns ts (wall-clock second), ns (sub-second nanoseconds), bid, bid_sz, ask,
ask_sz, mid, and recompute the book state on the fly. The United States feed
built by us_itch_fetch.py is instead a flat US_DIR/<SYM>_<YYYY-MM-DD>.parquet with
columns ts (nanoseconds since ET midnight), bid, bid_sz, ask, ask_sz. This adapter
rewrites the second form into the first, so that pointing MICRODIFFUSION_DATA_DIR
at the destination root lets every CLEAN_DIR-reading stage run on the United
States sample with no code change.

It reads the real US_DIR (resolved from common.paths, i.e. before any data-dir
override) and writes <out>/<session>/<SYM>_l1.parquet, session in MM-DD-YY form to
match the QSE session-folder convention. Only well-formed rows are kept (uncrossed
book, positive finite best sizes and prices), mirroring us_panel.load_clean.

Usage
    python src/data/us_clean_adapter.py                 # US_DIR -> data_us/clean
    python src/data/us_clean_adapter.py --out /path/clean
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import US_DIR, REPO_ROOT, ensure
from data.us_panel import PANEL, _symbol_session

_NS = 1_000_000_000


def stage_file(path: Path, out_root: Path) -> Path | None:
    """Rewrite one flat US panel into the QSE clean layout; return the written path."""
    sym, sess = _symbol_session(path)              # sess is MM-DD-YY
    y, m, d = path.name[: -len(".parquet")].rsplit("_", 1)[1].split("-")
    day_midnight = pd.Timestamp(f"{y}-{m}-{d}")

    df = pd.read_parquet(path)
    bid = df["bid"].astype(float); ask = df["ask"].astype(float)
    bsz = df["bid_sz"].astype(float); asz = df["ask_sz"].astype(float)
    # Same cleaning rule as us_panel.load_clean: drop crossed or empty books.
    valid = (ask > bid) & (bsz > 0) & (asz > 0) & ask.notna() & bid.notna()

    ts_ns = df["ts"].astype("int64").to_numpy()    # nanoseconds since ET midnight
    wall = day_midnight + pd.to_timedelta(ts_ns, unit="ns")
    out = pd.DataFrame({
        "ts": wall.floor("s"),                     # wall-clock second (datetime64[ns])
        "ns": (ts_ns % _NS).astype("int64"),       # sub-second nanoseconds
        "bid": bid, "bid_sz": df["bid_sz"].astype("int64"),
        "ask": ask, "ask_sz": df["ask_sz"].astype("int64"),
        "mid": 0.5 * (ask + bid),
    })[valid.to_numpy()].reset_index(drop=True)

    dest_dir = ensure(out_root / sess)
    dest = dest_dir / f"{sym}_l1.parquet"
    out.to_parquet(dest, index=False)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(US_DIR),
                    help="flat US panel directory (default: US_DIR)")
    ap.add_argument("--out", default=str(REPO_ROOT / "data_us" / "clean"),
                    help="destination clean root (default: data_us/clean)")
    args = ap.parse_args()

    src = Path(args.src); out_root = ensure(Path(args.out))
    files = sorted(p for p in src.glob("*_*.parquet") if _symbol_session(p)[0] in PANEL)
    if not files:
        print(f"[us_clean_adapter] no US panels under {src}; nothing to stage")
        return 0
    print(f"[us_clean_adapter] staging {len(files)} file(s): {src} -> {out_root}")
    n = 0
    for p in files:
        dest = stage_file(p, out_root)
        rows = len(pd.read_parquet(dest))
        print(f"    {p.name} -> {dest.relative_to(out_root)}  ({rows} rows)", flush=True)
        n += 1
    print(f"[us_clean_adapter] staged {n} session-symbol file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
