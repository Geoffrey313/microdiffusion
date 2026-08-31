#!/usr/bin/env python3
"""
us_panel.py - external United States large-cap level-one adapter and state builder.

Assembles the book state used by the United States equity robustness check
(Appendix F of the manuscript) from a reconstructed event-time top-of-book feed
for a fixed panel of large Nasdaq-listed instruments. Each source file is a
compact level-one panel written by data/us_itch_fetch.py from the free
Nasdaq TotalView-ITCH 5.0 sample: one row per change of the top of book, so the
sampling clock is the book event itself, not a fixed wall-clock bin. The raw
message feed is NOT part of the repository, and neither is the reconstructed
panel: its location is resolved from US_DIR (see common.paths), and each file is
named US_DIR/<SYM>_<YYYY-MM-DD>.parquet with the columns ts, bid, ask, bid_sz,
ask_sz, mid.

Two steps, both in memory (no intermediate files are written here):
  clean   keep well-formed rows (uncrossed book, positive finite best sizes and
          prices) and key each file by its instrument and calendar day.
  state   per session, x = log(mid), the best-level imbalance
          I = (Vb - Va)/(Vb + Va), and S_tick, the within-instrument global
          relative-spread tercile in {1, 2, 3} (1 = tight, 3 = wide).

One sample day is one walk-forward day-block. The spread state is a
within-instrument tercile rather than a fixed tick count so the spread dimension
stays comparable with the primary sample and across day-blocks, even though the
instruments share a one-cent tick. The instrument panel is fixed ex ante and
contains large, actively traded Nasdaq-listed capitalisations over the sample
window; United States capitalisations listed on other venues do not appear in
this feed and are therefore out of scope, a limitation stated with the result.

Inputs : US_DIR/<SYM>_<YYYY-MM-DD>.parquet for SYM in PANEL (external,
         git-ignored; built by us_itch_fetch.py).
Outputs: in-memory dictionaries consumed by analysis/us_transfer.py.
Serves : Appendix F (United States large-cap robustness).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import US_DIR

# A fixed ex-ante panel of large, actively traded capitalisations listed on the
# Nasdaq Stock Market over the sample window. Names whose primary listing is
# another venue are not carried by this single-venue feed and are out of scope;
# this is stated with the result. FB is the canonical key for the Facebook/Meta
# instrument; its traded ticker becomes META on 2022-06-09, resolved in
# data/us_itch_fetch.py (traded_ticker) so the instrument stays continuous across
# the extended sample.
PANEL = ("AAPL", "MSFT", "AMZN", "GOOGL", "FB", "INTC", "CSCO", "PEP", "NVDA", "CMCSA")

# Contiguous per-session event cap. It bounds memory and run time; one regular
# session of a mega-cap holds several million top-of-book changes, far more than
# the surface needs, so the leading MAX_EVENTS events (from the session open) are
# kept. This is a computational cap, not a sampling choice: it keeps whole
# one-step event increments intact and applies the same rule to every session.
MAX_EVENTS = 400_000
MIN_ROWS = 5_000           # minimum events to keep a (symbol, day) session


def _symbol_session(path: Path) -> tuple[str, str]:
    """Split <SYM>_<YYYY-MM-DD>.parquet into (symbol, session) with session MM-DD-YY."""
    stem = path.name[: -len(".parquet")] if path.name.endswith(".parquet") else path.stem
    sym, date = stem.rsplit("_", 1)
    y, m, d = date.split("-")
    return sym, f"{m}-{d}-{y[2:]}"


def available(us_dir: Path = US_DIR) -> list[Path]:
    """Return the sorted panel source files, empty if the feed is absent."""
    d = Path(us_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*_*.parquet") if _symbol_session(p)[0] in PANEL)


def load_clean(us_dir: Path = US_DIR) -> dict:
    """Return {(symbol, session): cleaned event-time frame} for every source file.

    Each frame carries ts, bid, ask, bid_sz, ask_sz, mid in event order; the
    session key is the calendar day of the file in MM-DD-YY form.
    """
    clean = {}
    for path in available(us_dir):
        sym, sess = _symbol_session(path)
        d = pd.read_parquet(path).head(MAX_EVENTS)
        ap = d["ask"].astype(float); bp = d["bid"].astype(float)
        aa = d["ask_sz"].astype(float); ba = d["bid_sz"].astype(float)
        # A crossed or empty book is a sequencing artefact, not a quote; drop it.
        valid = (ap > bp) & (aa > 0) & (ba > 0) & ap.notna() & bp.notna()
        df = pd.DataFrame({
            "ts": d["ts"].astype("int64"), "bid": bp, "bid_sz": ba,
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
                "ts_event": l1["ts"].to_numpy("int64"),
                "x": np.log(mid),
                "I": np.divide(bsz - asz, bsz + asz,
                               out=np.full_like(bsz, np.nan), where=(bsz + asz) > 0),
                "S_tick": np.where(s <= q33, 1, np.where(s <= q67, 2, 3)).astype(int),
                "mid": mid,
            })
            states[k] = st[np.isfinite(st["x"]) & np.isfinite(st["I"])].reset_index(drop=True)
    return states
