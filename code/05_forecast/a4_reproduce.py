#!/usr/bin/env python3
"""
a4_reproduce.py — A4 step 1: reproduce the manuscript's structural-transfer numbers
FROM THE EXPORTED PANEL ALONE (event_panel.parquet), proving the export is faithful.

Reference protocol (10_validate_diffusion_state.py, which produced the published table
10_diffusion_state_summary.csv and the manuscript numbers 0.477 / 0.155 / 0.692 / 0.104):
  - walk-forward per symbol: train on W=5 prior sessions, test on the next session
  - one-step dx^2 with the state known at t (dx_log in the panel; anchor cancels in diffs)
  - cells: 10 uniform I-bins x spread-in-ticks clipped to [1,3]  ->  cell = I_bin*3 + (clip(S_tick,1,3)-1)
  - winsorize dx^2 at the TRAIN 0.995 quantile per fold
  - per-cell means need >= MIN_CELL=20 obs on BOTH sides to enter the correlations
  - per-fold Spearman: joint (I,S), I-only margin, S-only margin, interaction residual,
    shuffled-cell placebo; mean over folds; symbol-clustered bootstrap CI
Published targets (n=1398 folds):
  joint 0.4769 [clust 0.3952, 0.5610]  frac_pos 0.8964
  I_only 0.1546   S_only 0.6918 (n=1319)   interaction 0.1041   shuffled -0.0001
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = HERE.parents[1] / "data" / "data_export" / "event_panel.parquet"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

W = 5
WINSOR = 0.995
MIN_CELL = 20
N_I, M_S = 10, 3
N_BOOT = 1000
RNG = np.random.default_rng(0)


def sp(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def boot_cluster(df, col, by="symbol"):
    groups = [g[col].to_numpy() for _, g in df.groupby(by)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    G = len(groups); means = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, G, G)
        means.append(np.concatenate([groups[j] for j in idx]).mean())
    return (float(np.concatenate(groups).mean()),
            float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    df = pd.read_parquet(PANEL, columns=[
        "instrument", "date", "ts_event", "dx_log", "I_bin", "S_tick"])
    df = df.dropna(subset=["dx_log", "S_tick"])
    df["y"] = df["dx_log"].to_numpy() ** 2
    sb = np.clip(df["S_tick"].to_numpy(int), 1, M_S)
    df["cell"] = df["I_bin"].to_numpy() * M_S + (sb - 1)

    # cache per (symbol, session): (cell array, dx2 array)
    cache = {}
    for (sym, d), g in df.groupby(["instrument", "date"], sort=True):
        cache[(sym, d)] = (g["cell"].to_numpy(), g["y"].to_numpy())

    recs = []
    for sym in sorted({k[0] for k in cache}):
        ssl = sorted([d for (s, d) in cache if s == sym])
        if len(ssl) < W + 1:
            continue
        for i in range(W, len(ssl)):
            tr_s, te_s = ssl[i - W:i], ssl[i]
            c_tr = np.concatenate([cache[(sym, s)][0] for s in tr_s])
            y_tr = np.concatenate([cache[(sym, s)][1] for s in tr_s])
            if len(y_tr) < 200:
                continue
            cap = float(np.quantile(y_tr, WINSOR))
            y_trw = np.minimum(y_tr, cap)
            g0 = float(np.mean(y_trw))
            tdf = pd.DataFrame({"c": c_tr, "y": y_trw})
            a_cell = tdf.groupby("c")["y"].mean().to_dict()
            n_cell = tdf.groupby("c")["y"].size().to_dict()
            a_cell_sh = (pd.DataFrame({"c": RNG.permutation(c_tr), "y": y_trw})
                         .groupby("c")["y"].mean().to_dict())

            c_te, y_te = cache[(sym, te_s)]
            if len(y_te) < 100:
                continue
            y_tew = np.minimum(y_te, cap)
            g_te = float(np.mean(y_tew))

            te_cell = pd.DataFrame({"c": c_te, "y": y_tew}).groupby("c")["y"].agg(["mean", "size"])
            common = [c for c in te_cell.index if (c in a_cell) and te_cell.loc[c, "size"] >= MIN_CELL
                      and n_cell.get(c, 0) >= MIN_CELL]

            def marg(cells, yv, axis):
                idx = (cells // M_S) if axis == "I" else (cells % M_S)
                d = pd.DataFrame({"k": idx, "y": yv})
                return d.groupby("k")["y"].mean(), d.groupby("k")["y"].size()

            aI_tr, nI_tr = marg(c_tr, y_trw, "I"); aS_tr, nS_tr = marg(c_tr, y_trw, "S")
            aI_te, nI_te = marg(c_te, y_tew, "I"); aS_te, nS_te = marg(c_te, y_tew, "S")

            cI = [i for i in aI_te.index if i in aI_tr.index and nI_te[i] >= MIN_CELL and nI_tr.get(i, 0) >= MIN_CELL]
            corr_I = sp([aI_tr[i] for i in cI], [aI_te[i] for i in cI]) if len(cI) >= 4 else np.nan
            cS = [s for s in aS_te.index if s in aS_tr.index and nS_te[s] >= MIN_CELL and nS_tr.get(s, 0) >= MIN_CELL]
            corr_S = sp([aS_tr[s] for s in cS], [aS_te[s] for s in cS]) if len(cS) >= 3 else np.nan

            if len(common) >= 4:
                atr = np.array([a_cell[c] for c in common])
                ate = np.array([te_cell.loc[c, "mean"] for c in common])
                corr_s = sp(atr, ate)
                atr_sh = np.array([a_cell_sh.get(c, g0) for c in common])
                corr_s_sh = sp(atr_sh, ate)
                rtr = np.array([a_cell[c] - aI_tr.get(c // M_S, g0) - aS_tr.get(c % M_S, g0) + g0 for c in common])
                rte = np.array([te_cell.loc[c, "mean"] - aI_te.get(c // M_S, g_te) - aS_te.get(c % M_S, g_te) + g_te
                                for c in common])
                corr_inter = sp(rtr, rte)
            else:
                corr_s = corr_s_sh = corr_inter = np.nan

            recs.append({"symbol": sym, "test": te_s, "n_common": len(common),
                         "joint": corr_s, "I_only": corr_I, "S_only": corr_S,
                         "interaction": corr_inter, "shuffled": corr_s_sh})

    r = pd.DataFrame(recs)
    print(f"folds: {len(r)}  (published: 1398/1399)")
    targets = {"joint": (0.4769, 0.8964), "I_only": (0.1546, None), "S_only": (0.6918, None),
               "interaction": (0.1041, None), "shuffled": (-0.0001, None)}
    rows = []
    for col, (tgt, fp_tgt) in targets.items():
        m, lo, hi = boot_cluster(r, col)
        fp = float((r[col] > 0).mean())
        n = int(r[col].notna().sum())
        rows.append({"metric": col, "panel_mean": round(m, 4), "clust_lo": round(lo, 4),
                     "clust_hi": round(hi, 4), "frac_pos": round(fp, 4), "n": n,
                     "published": tgt, "abs_dev": round(abs(m - tgt), 4)})
        print(f"{col:12s}  panel {m:+.4f} [{lo:+.4f},{hi:+.4f}]  frac_pos {fp:.4f}  n={n}"
              f"   published {tgt:+.4f}   |dev| {abs(m - tgt):.4f}")
    pd.DataFrame(rows).to_csv(OUT / "a4_reproduction.csv", index=False)
    print(f"\nwrote {OUT}/a4_reproduction.csv")


if __name__ == "__main__":
    main()
