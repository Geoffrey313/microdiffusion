#!/usr/bin/env python3
"""
crypto_panel.py - external cryptocurrency L1 adapter and reduced-state builder.

Assembles the book state used by the external-sample robustness check
(Appendix E of the manuscript) from a one-second aggregated level-one feed for
three cryptocurrencies (Bitcoin, Ether, Cardano). The raw feed is NOT part of
the repository: its location is resolved from CRYPTO_DIR (an environment
override, see common.paths), and each source file CRYPTO_DIR/<SYM>_1sec.csv
carries the columns system_time, midpoint, spread, buys, sells, and per-level
notional depth.

Two steps, both in memory (no intermediate files are written):
  clean   keep well-formed rows (positive, finite mid, spread, and top-of-book
          notional), map to the (bid, ask, sizes, mid) shape used by the primary
          pipeline, split into UTC calendar-day sessions, and drop sessions with
          fewer than MIN_ROWS rows.
  state   per session, x = log(mid), the best-level imbalance
          I = (Vb - Va)/(Vb + Va), and S_tick, the within-instrument global
          relative-spread tercile in {1, 2, 3} (1 = tight, 3 = wide).

The market trades continuously, so a "session" is an arbitrary UTC-day partition
of a single uninterrupted tape, used only so the walk-forward transfer test has
day-blocks to iterate. The spread state is a within-instrument tercile rather
than a fixed tick count because the sample has no common tick and a tick-based
state degenerates for coarse names; terciles keep the spread dimension
non-degenerate and comparable across day-blocks.

Inputs : CRYPTO_DIR/{BTC,ETH,ADA}_1sec.csv (external, git-ignored).
Outputs: in-memory dictionaries consumed by analysis/crypto_transfer.py.
Serves : Appendix E (external-sample robustness).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import CRYPTO_DIR

# Columns read from each source file; midpoint and spread give the touch, the
# top-level notional columns give the best-level sizes, buys/sells give flow.
USECOLS = ["system_time", "midpoint", "spread", "buys", "sells",
           "bids_notional_0", "asks_notional_0"]
MIN_ROWS = 5000            # minimum one-second rows to keep a (symbol, UTC-day) session


def available(crypto_dir: Path = CRYPTO_DIR) -> list[Path]:
    """Return the sorted list of source files, empty if the feed is absent."""
    d = Path(crypto_dir)
    return sorted(d.glob("*_1sec.csv")) if d.exists() else []


def load_clean(crypto_dir: Path = CRYPTO_DIR) -> dict:
    """Return {(symbol, session): cleaned L1 frame} for every source file.

    Each frame carries ts, bid, ask, bid_sz, ask_sz, mid, flow with the original
    one-second row order preserved; the session key is the UTC calendar day in
    MM-DD-YY form.
    """
    clean = {}
    for path in available(crypto_dir):
        sym = path.name.split("_")[0]
        d = pd.read_csv(path, usecols=USECOLS)
        ts = pd.to_datetime(d["system_time"], utc=True)
        mid = d["midpoint"].astype(float)
        spread = d["spread"].astype(float)
        bsz = d["bids_notional_0"].astype(float)
        asz = d["asks_notional_0"].astype(float)
        flow = d["buys"].astype(float) - d["sells"].astype(float)
        valid = (mid > 0) & (spread > 0) & (bsz > 0) & (asz > 0) & mid.notna() & spread.notna()
        df = pd.DataFrame({
            "ts": ts, "bid": mid - spread / 2.0, "bid_sz": bsz,
            "ask": mid + spread / 2.0, "ask_sz": asz, "mid": mid, "flow": flow,
            "session": ts.dt.strftime("%m-%d-%y"),
        })[valid.to_numpy()].reset_index(drop=True)
        for sess, g in df.groupby("session"):
            if len(g) < MIN_ROWS:
                continue
            clean[(sym, sess)] = g.drop(columns="session").reset_index(drop=True)
    return clean


def build_states(clean: dict) -> dict:
    """Return {(symbol, session): state frame} with columns ts_event, x, I, S_tick, mid.

    S_tick is the within-instrument global relative-spread tercile in {1, 2, 3},
    computed over every retained row of that instrument so the spread state is
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
