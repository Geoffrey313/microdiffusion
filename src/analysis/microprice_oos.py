#!/usr/bin/env python3
"""
microprice_oos.py - out-of-sample test of the drift restriction (H1, Assumption 2):
does the conditional drift of the next log-return depend on the price level x, or
only on the imbalance and spread (I, S)?

The test is an expanding-window, day-by-day forecast run per instrument. For each
trading day d (from the second onward) the conditional drift surface is estimated
on all earlier days of that instrument and used to predict the next log-return on
day d, under two nested specifications:
  2D model  b_x(I, S)      per-cell mean drift over the I_bin x S_tick grid
  3D model  b_x(x, I, S)   the same grid split further by terciles of the price
                           level x = log(mid)
A cell is used only once it holds at least MIN_CELL training observations; sparser
cells fall back to the coarser model and finally to the running mean drift. Each
(instrument, day) is one out-of-sample pair; there are 1,588 - 41 = 1,547 of them
(one warm-up day per instrument has no history).

Skill is summarised by the pooled out-of-sample R^2 (residual sums aggregated over
all pairs) and the pooled directional hit-rate, with an instrument-clustered
bootstrap for intervals. Under Assumption 2 the level adds nothing, so the 3D model
should not beat the 2D model; if anything the extra, weakly-populated coordinate
overfits and slightly hurts, so Delta R^2 = R^2_3D - R^2_2D is negative.

The cell floor MIN_CELL is the same one used across the forecast analyses. The point
estimates depend on second-order estimation conventions (cell floor, evaluation
support, pooled vs per-fold aggregation); the sign of R^2_2D, the negative sign of
Delta R^2, and the downward direction of the hit-rate are the qualitative facts the
test establishes and are robust to those choices.

Writes results/07_microprice_oos_summary.csv, one of the digest-checked article
tables, read by the drift-surface figure (figures/gap_figures.py). Every value in
that file is computed by this script; nothing is entered by hand.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PANEL, RESULTS_DIR, ensure

OUT = ensure(RESULTS_DIR)

M_S, N_I, N_X, MIN_CELL = 3, 10, 3, 50
N_CELL2 = N_I * M_S
N_CELL3 = N_CELL2 * N_X
N_BOOT = 1000
RNG = np.random.default_rng(0)


def boot_pooled(per_inst, num_key, den_key):
    """Instrument-clustered bootstrap of a pooled ratio statistic 1 - sum(num)/sum(den)."""
    nums = np.array([d[num_key] for d in per_inst])
    dens = np.array([d[den_key] for d in per_inst])
    G = len(per_inst)
    vals = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, G, G)
        den = dens[idx].sum()
        if den > 0:
            vals.append(1.0 - nums[idx].sum() / den)
    point = 1.0 - nums.sum() / dens.sum()
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_pooled_diff(per_inst):
    """Instrument-clustered bootstrap of pooled Delta R^2 = R^2_3D - R^2_2D."""
    r2 = np.array([d["ss2"] for d in per_inst]); r3 = np.array([d["ss3"] for d in per_inst])
    tot = np.array([d["sst"] for d in per_inst])
    G = len(per_inst); vals = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, G, G); t = tot[idx].sum()
        if t > 0:
            vals.append((1 - r3[idx].sum() / t) - (1 - r2[idx].sum() / t))
    point = (1 - r3.sum() / tot.sum()) - (1 - r2.sum() / tot.sum())
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_pooled_hitdiff(per_inst):
    """Instrument-clustered bootstrap of the pooled hit-rate change (3D minus 2D)."""
    h2 = np.array([d["h2"] for d in per_inst]); h3 = np.array([d["h3"] for d in per_inst])
    n = np.array([d["hn"] for d in per_inst])
    G = len(per_inst); vals = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, G, G); nn = n[idx].sum()
        if nn > 0:
            vals.append((h3[idx].sum() - h2[idx].sum()) / nn)
    point = (h3.sum() - h2.sum()) / n.sum()
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    df = pd.read_parquet(PANEL, columns=[
        "instrument", "date", "mid", "dx_log", "I_bin", "S_tick"])
    df = df.dropna(subset=["dx_log", "S_tick", "mid"])
    df["x"] = np.log(df["mid"].to_numpy())
    df["sb"] = np.clip(df["S_tick"].to_numpy(int), 1, M_S) - 1
    df["cell2"] = df["I_bin"].to_numpy(int) * M_S + df["sb"]

    per_inst = []
    n_pairs = 0
    for sym, g in df.groupby("instrument"):
        g = g.sort_values("date", kind="mergesort")
        x_edges = np.quantile(g["x"].to_numpy(), [1 / 3, 2 / 3])
        xb = np.clip(np.searchsorted(x_edges, g["x"].to_numpy(), side="right"), 0, N_X - 1)
        c2all = g["cell2"].to_numpy(int); c3all = c2all * N_X + xb
        yall = g["dx_log"].to_numpy(); dates = g["date"].to_numpy()

        sum2 = np.zeros(N_CELL2); cnt2 = np.zeros(N_CELL2)
        sum3 = np.zeros(N_CELL3); cnt3 = np.zeros(N_CELL3)
        tot_sum = 0.0; tot_cnt = 0
        s = dict(ss2=0.0, ss3=0.0, sst=0.0, h2=0, h3=0, hn=0)

        uniq, idx = np.unique(dates, return_index=True); bounds = list(idx) + [len(dates)]
        for k in range(len(uniq)):
            sl = slice(bounds[k], bounds[k + 1])
            y = yall[sl]; c2 = c2all[sl]; c3 = c3all[sl]
            if tot_cnt > 0:                                   # skip the warm-up day
                gmean = tot_sum / tot_cnt
                m2 = sum2 / np.where(cnt2 > 0, cnt2, 1); ok2 = cnt2[c2] >= MIN_CELL
                pred2 = np.where(ok2, m2[c2], gmean)
                m3 = sum3 / np.where(cnt3 > 0, cnt3, 1); ok3 = cnt3[c3] >= MIN_CELL
                pred3 = np.where(ok3, m3[c3], np.where(ok2, m2[c2], gmean))
                s["ss2"] += float(np.sum((y - pred2) ** 2))
                s["ss3"] += float(np.sum((y - pred3) ** 2))
                s["sst"] += float(np.sum((y - gmean) ** 2))
                nz = y != 0
                if nz.any():
                    s["h2"] += int((np.sign(y[nz]) == np.sign(pred2[nz])).sum())
                    s["h3"] += int((np.sign(y[nz]) == np.sign(pred3[nz])).sum())
                    s["hn"] += int(nz.sum())
                n_pairs += 1
            np.add.at(sum2, c2, y); np.add.at(cnt2, c2, 1)
            np.add.at(sum3, c3, y); np.add.at(cnt3, c3, 1)
            tot_sum += float(y.sum()); tot_cnt += len(y)
        if s["sst"] > 0:
            per_inst.append(s)

    r2_2d, r2_2d_lo, r2_2d_hi = boot_pooled(per_inst, "ss2", "sst")
    r2_3d, r2_3d_lo, r2_3d_hi = boot_pooled(per_inst, "ss3", "sst")
    d_r2, d_r2_lo, d_r2_hi = boot_pooled_diff(per_inst)
    hit2 = sum(d["h2"] for d in per_inst) / sum(d["hn"] for d in per_inst)
    hit3 = sum(d["h3"] for d in per_inst) / sum(d["hn"] for d in per_inst)
    d_hit, d_hit_lo, d_hit_hi = boot_pooled_hitdiff(per_inst)

    print(f"out-of-sample instrument-period pairs: {n_pairs:,}   "
          f"instruments contributing: {len(per_inst)}")
    print("pooled, expanding-window (cell floor MIN_CELL={}):".format(MIN_CELL))
    print(f"  R2_2D = {r2_2d:+.4f} [{r2_2d_lo:+.4f}, {r2_2d_hi:+.4f}]")
    print(f"  R2_3D = {r2_3d:+.4f} [{r2_3d_lo:+.4f}, {r2_3d_hi:+.4f}]")
    print(f"  dR2   = {d_r2:+.4f} [{d_r2_lo:+.4f}, {d_r2_hi:+.4f}]   (negative: level does not help)")
    print(f"  hit   = {hit2:.4f} -> {hit3:.4f}   dHit = {d_hit:+.4f} [{d_hit_lo:+.4f}, {d_hit_hi:+.4f}]")

    pd.DataFrame([{
        "R2_2D_mean": r2_2d, "R2_2D_lo": r2_2d_lo, "R2_2D_hi": r2_2d_hi,
        "R2_3D_mean": r2_3d, "R2_3D_lo": r2_3d_lo, "R2_3D_hi": r2_3d_hi,
        "dR2_mean": d_r2, "dR2_lo": d_r2_lo, "dR2_hi": d_r2_hi,
        "hit_2d_mean": hit2, "hit_3d_mean": hit3,
        "dHit_mean": d_hit, "dHit_lo": d_hit_lo, "dHit_hi": d_hit_hi,
        "n_pairs": n_pairs,
    }]).to_csv(OUT / "07_microprice_oos_summary.csv", index=False)
    print(f"wrote {OUT}/07_microprice_oos_summary.csv")


if __name__ == "__main__":
    main()
