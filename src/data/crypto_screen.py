#!/usr/bin/env python3
"""
crypto_screen.py - instrument selection rule for the external panel (Appendix E).

Records, reproducibly, why the external cryptocurrency panel is the one it is.
The diffusion surface a_xx(I, S) has two state dimensions, so an instrument is
useful for the external check only if BOTH the relative spread S varies (its
per-instrument terciles do not collapse) and the best-level imbalance I is
dispersed, while the book stays liquid. A pair whose spread never leaves one
tick (for example Cardano, pinned at one tick on more than four fifths of quote
updates) makes the S axis degenerate and is rejected.

To guard against picking instruments ex post on a single lucky day, the rule is
evaluated on several first-of-month days and an instrument is retained only if
it is state-informative on EVERY screened date. The candidate universe is frozen
in this script, so rerunning the screen does not depend on Coinbase's current
product ranking or twenty-four-hour volume. The screen reads only the free
Tardis daily quotes archive (best bid and ask, price and size) for each screened
day; the tick size is inferred from the minimum positive spread observed in the
sampled quotes.

Criterion per (instrument, date): the spread exceeds one tick on at least 20% of
quote updates AND round(q67) > round(q33) of the spread in ticks (terciles
separate), the book is liquid (median spread <= 25 bps and at least 20k sampled
quotes), and std(I) >= 0.30.

Outputs: results/diagnostics/tables/crypto_pair_screen.csv (per instrument and
date), and the retained panel printed to stdout. Diagnostic only; not part of the
digest gate and not required by the reproduction chain.
"""
from __future__ import annotations
import csv
import gzip
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DIAG_DIR, ensure

QUOTES = "https://datasets.tardis.dev/v1/coinbase/quotes/{y}/{m}/{d}/{sym}.csv.gz"
SCREEN_DATES = ("2026-04-01", "2026-05-01", "2026-06-01")
CANDIDATES = (
    "BTC-USD", "ETH-USD", "ZEC-USD", "ONDO-USD", "XLM-USD",
    "ACH-USD", "LTC-USD", "DOGE-USD", "NEAR-USD", "HYPE-USD",
    "ADA-USD", "XRP-USD", "SOL-USD", "LINK-USD", "USDT-USD", "COTI-USD",
)
RETAINED_PANEL = ("BTC-USD", "ETH-USD", "ZEC-USD", "ONDO-USD", "XLM-USD")
SAMPLE_EVERY = 7            # one sampled quote in SAMPLE_EVERY; the median needs no more
GE2_MIN = 0.20             # spread must exceed one tick on this share of updates
MAX_MED_BPS = 25.0
MIN_QUOTES = 20_000
MIN_I_STD = 0.30


def _quotes(sym: str, date: str) -> bytes | None:
    y, m, d = date.split("-")
    url = QUOTES.format(y=y, m=m, d=d, sym=sym)
    r = subprocess.run(["curl", "-s", "--max-time", "120", url], capture_output=True)
    return r.stdout if r.stdout[:2] == b"\x1f\x8b" else None


def measure(sym: str, date: str) -> dict | None:
    payload = _quotes(sym, date)
    if not payload:
        return None
    spread_px, mid_px, imb = [], [], []
    stream = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(payload)), encoding="utf-8")
    for i, row in enumerate(csv.DictReader(stream)):
        if i % SAMPLE_EVERY:
            continue
        try:
            ap = float(row["ask_price"]); bp = float(row["bid_price"])
            aa = float(row["ask_amount"]); ba = float(row["bid_amount"])
        except (KeyError, TypeError, ValueError):
            continue
        if ap <= bp or (aa + ba) <= 0:
            continue
        spread_px.append(ap - bp)
        mid_px.append(0.5 * (ap + bp))
        imb.append((ba - aa) / (ba + aa))
    if len(spread_px) < 1_000:
        return None
    spread_px = np.array(spread_px)
    mid_px = np.array(mid_px)
    imb = np.array(imb)
    tick = float(np.min(spread_px[spread_px > 0]))
    stick = spread_px / tick
    q33, q67 = np.quantile(stick, [1 / 3, 2 / 3])
    return {"symbol": sym, "date": date, "tick": tick,
            "quotes": len(stick), "ge2": float(np.mean(stick >= 1.5)),
            "q33_ticks": float(q33), "q67_ticks": float(q67),
            "median_bps": float(np.median(spread_px / mid_px) * 1e4),
            "I_std": float(np.std(imb))}


def informative(m: dict) -> bool:
    s_ok = m["ge2"] >= GE2_MIN and round(m["q67_ticks"]) > round(m["q33_ticks"])
    liq = m["median_bps"] <= MAX_MED_BPS and m["quotes"] >= MIN_QUOTES
    return bool(s_ok and liq and m["I_std"] >= MIN_I_STD)


def main() -> int:
    print(f"screening {len(CANDIDATES)} frozen candidate pairs over {len(SCREEN_DATES)} dates")
    rows = []
    for sym in CANDIDATES:
        for date in SCREEN_DATES:
            m = measure(sym, date)
            if m:
                m["informative"] = informative(m)
                m["retained_panel"] = sym in RETAINED_PANEL
                rows.append(m)
                print(f"  {sym:<12} {date}  ge2 {m['ge2']*100:>3.0f}%  "
                      f"Istd {m['I_std']:.2f}  {'INFO' if m['informative'] else '--'}")

    df = pd.DataFrame(rows)
    out = ensure(DIAG_DIR / "tables") / "crypto_pair_screen.csv"
    df.to_csv(out, index=False)

    passed = df.groupby("symbol")["informative"].agg(["sum", "count"])
    stable = sorted(passed.index[(passed["sum"] == passed["count"]) &
                                 (passed["count"] == len(SCREEN_DATES))])
    print(f"\nwrote {out}")
    print("stable state-informative on every screened date: " + ", ".join(stable))
    print("retained panel: " + ", ".join(RETAINED_PANEL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
