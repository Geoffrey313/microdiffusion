#!/usr/bin/env python3
"""
crypto_panel.py - external cryptocurrency level-one adapter and state builder.

Assembles the book state used by the external-sample robustness check
(Appendix E of the manuscript) from an event-time best-quote feed for the five
screened instruments (Bitcoin, Ether, Zcash, Ondo, Stellar). Each source file is
a Tardis "quotes" archive of the Coinbase best bid and ask -- one row per change
of the top of book, price or size -- so the sampling clock is the book event
itself, not a fixed wall-clock bin. The raw feed is NOT part of the repository:
its location is resolved from CRYPTO_DIR (see common.paths), and each file is
named CRYPTO_DIR/<SYM>_<YYYY-MM-DD>.csv.gz with the columns timestamp,
ask_amount, ask_price, bid_price, bid_amount.

Two steps, both in memory (no intermediate files are written):
  clean   keep well-formed rows (uncrossed book, positive finite best sizes and
          prices), map to the (bid, ask, sizes, mid) shape used by the primary
          pipeline, and key each file by its instrument and calendar day.
  state   per session, x = log(mid), the best-level imbalance
          I = (Vb - Va)/(Vb + Va), and S_tick, the within-instrument global
          relative-spread tercile in {1, 2, 3} (1 = tight, 3 = wide).

One first-of-month day is one walk-forward day-block. The spread state is a
within-instrument tercile rather than a fixed tick count because the panel spans
several tick regimes and no common tick exists; terciles keep the spread
dimension non-degenerate and comparable across day-blocks. The instruments were
fixed by an out-of-sample screen (see crypto_screen.py) that keeps only pairs
whose spread varies materially and whose imbalance is dispersed on every screened
date, so the panel is not chosen ex post on one day.

Inputs : CRYPTO_DIR/<SYM>_<YYYY-MM-DD>.csv.gz for SYM in PANEL (external,
         git-ignored; fetched by crypto_fetch.py).
Outputs: in-memory dictionaries consumed by analysis/crypto_transfer.py.
Serves : Appendix E (external-sample robustness).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import CRYPTO_DIR

# Screened instrument panel (see crypto_screen.py); ADA and other tick-pinned or
# unstable names are excluded because their spread does not vary.
PANEL = ("BTC-USD", "ETH-USD", "ZEC-USD", "ONDO-USD", "XLM-USD")

# Columns read from each quotes archive: the exchange timestamp and the best
# bid/ask price and size. The best sizes give the imbalance, the prices the spread.
USECOLS = ["timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]
# Contiguous per-session event cap. It bounds memory and run time; a first-of-month
# day holds one to four million events, far more than the surface needs, so the
# leading MAX_EVENTS events (from the session start) are kept. This is a
# computational cap, not a sampling choice: it keeps whole one-step event
# increments intact and applies the same rule to every session.
MAX_EVENTS = 400_000
MIN_ROWS = 5_000           # minimum events to keep a (symbol, day) session


def _symbol_session(path: Path) -> tuple[str, str]:
    """Split <SYM>_<YYYY-MM-DD>.csv.gz into (symbol, session) with session MM-DD-YY."""
    stem = path.name[: -len(".csv.gz")] if path.name.endswith(".csv.gz") else path.stem
    sym, date = stem.rsplit("_", 1)
    y, m, d = date.split("-")
    return sym, f"{m}-{d}-{y[2:]}"


def available(crypto_dir: Path = CRYPTO_DIR) -> list[Path]:
    """Return the sorted panel source files, empty if the feed is absent."""
    d = Path(crypto_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*_*.csv.gz") if _symbol_session(p)[0] in PANEL)


def load_clean(crypto_dir: Path = CRYPTO_DIR) -> dict:
    """Return {(symbol, session): cleaned event-time frame} for every source file.

    Each frame carries ts, bid, ask, bid_sz, ask_sz, mid in event order; the
    session key is the calendar day of the file in MM-DD-YY form.
    """
    clean = {}
    for path in available(crypto_dir):
        sym, sess = _symbol_session(path)
        d = pd.read_csv(path, usecols=USECOLS, nrows=MAX_EVENTS)
        ts = pd.to_datetime(d["timestamp"], unit="us", utc=True)
        ap = d["ask_price"].astype(float); bp = d["bid_price"].astype(float)
        aa = d["ask_amount"].astype(float); ba = d["bid_amount"].astype(float)
        # A crossed or empty book is a sequencing artefact, not a quote; drop it.
        valid = (ap > bp) & (aa > 0) & (ba > 0) & ap.notna() & bp.notna()
        df = pd.DataFrame({
            "ts": ts, "bid": bp, "bid_sz": ba,
            "ask": ap, "ask_sz": aa, "mid": 0.5 * (ap + bp),
        })[valid.to_numpy()].reset_index(drop=True)
        if len(df) >= MIN_ROWS:
            clean[(sym, sess)] = df
    return clean


def build_states(clean: dict) -> dict:
    """Return {(symbol, session): state frame} with columns ts_event, x, I, S_tick, mid.

    S_tick is the within-instrument global relative-spread tercile in {1, 2, 3},
    computed over every retained event of that instrument so the spread state is
    comparable across the day-blocks the transfer test iterates.
    """
    states = {}
    for sym in sorted({s for s, _ in clean}):
        keys = [k for k in clean if k[0] == sym]
        alls = np.concatenate([
            ((clean[k]["ask"] - clean[k]["bid"]) / (clean[k]["ask"] + clean[k]["bid"])).to_numpy()
            for k in keys])
        q33, q67 = np.nanquantile(alls, [1 / 3, 2 / 3])
        for k in keys:
            l1 = clean[k]
            mid = l1["mid"].to_numpy(float)
            bsz = l1["bid_sz"].to_numpy(float)
            asz = l1["ask_sz"].to_numpy(float)
            s = (l1["ask"].to_numpy(float) - l1["bid"].to_numpy(float)) / \
                (l1["ask"].to_numpy(float) + l1["bid"].to_numpy(float))
            st = pd.DataFrame({
                "ts_event": pd.to_datetime(l1["ts"]),
                "x": np.log(mid),
                "I": np.divide(bsz - asz, bsz + asz,
                               out=np.full_like(bsz, np.nan), where=(bsz + asz) > 0),
                "S_tick": np.where(s <= q33, 1, np.where(s <= q67, 2, 3)).astype(int),
                "mid": mid,
            })
            states[k] = st[np.isfinite(st["x"]) & np.isfinite(st["I"])].reset_index(drop=True)
    return states
