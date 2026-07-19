#!/usr/bin/env python3
"""
a2_twopart.py — A2: the two-part zero/nonzero model, on the paper's 60/40 split.

Decomposition:  E[dx^2 | state] = P(move | state) * E[dx^2 | move, state]

Part 1 (occurrence): P(zero | I,S) per state cell, ladder constant -> spread-only ->
  imbalance-only -> additive -> full. Losses: binomial log-loss and Brier — both proper
  and NON-degenerate here (this is where the 48% zero mass belongs).
Part 2 (magnitude): E[dx^2 | nonzero, I,S], same ladder, fit on train nonzero rows.
  Losses: QLIKE (legitimate now, y>0 strictly) and winsorized MSE.
Recombined: two-part forecast p_move * m vs the one-part A3 models under MSE on ALL test obs.

Key question for Article B: does imbalance survive in the OCCURRENCE part (whether the
price moves) even though A3 demoted it in the magnitude part (how much it moves)?

Inference: per-(instrument, date) fold means, instrument-clustered bootstrap (1000 draws).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = HERE.parents[1] / "data" / "data_export" / "event_panel.parquet"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

M_S, MIN_CELL, WINSOR = 3, 50, 0.995
N_BOOT = 1000
EPS = 1e-4
RNG = np.random.default_rng(0)


def boot_cluster_mean(fold_df, col):
    groups = [g[col].to_numpy() for _, g in fold_df.groupby("instrument")]
    groups = [g[np.isfinite(g)] for g in groups if len(g)]
    G = len(groups); means = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, G, G)
        means.append(np.concatenate([groups[j] for j in idx]).mean())
    allv = np.concatenate(groups)
    return float(allv.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ladder_fit(tr, val_col, ib, sb, cell, min_cell, g0):
    """Return dict of per-model test predictions given train frame tr with columns ib/sb/cell/v."""
    aS = tr.groupby("sb")["v"].mean().to_dict()
    aI = tr.groupby("ib")["v"].mean().to_dict()
    cs = tr.groupby("cell")["v"].agg(["mean", "size"])
    a_full = cs.loc[cs["size"] >= min_cell, "mean"].to_dict()
    fS = np.array([aS.get(s, g0) for s in sb])
    fI = np.array([aI.get(i, g0) for i in ib])
    fAdd = fI + fS - g0
    fFull = np.array([a_full.get(c, np.nan) for c in cell])
    fFull = np.where(np.isfinite(fFull), fFull, fAdd)
    return {"M0_const": np.full(len(ib), g0), "M1_spread": fS, "M2_imb": fI,
            "M3_additive": fAdd, "M4_full": fFull}


def report(folds, models, prefix, label, out_rows, pair_rows, base=None):
    print(f"\n== {label} ==")
    for m in models:
        v, lo, hi = boot_cluster_mean(folds, f"{prefix}_{m}")
        extra = ""
        if base is not None:
            skill = 1 - folds[f"{prefix}_{m}"].mean() / folds[f"{prefix}_{base}"].mean()
            extra = f"   skill vs const {skill:+.4f}"
        out_rows.append({"part": label, "model": m, "loss": v, "lo": lo, "hi": hi})
        print(f"{m:12s}  {v:+.6f} [{lo:+.6f},{hi:+.6f}]{extra}")
    print(f"-- pairwise (positive => second better) --")
    pairs = [("M0_const", "M1_spread"), ("M0_const", "M2_imb"), ("M1_spread", "M4_full"),
             ("M2_imb", "M4_full"), ("M1_spread", "M3_additive"), ("M0_const", "M4_full")]
    for a, b in pairs:
        folds["_d"] = folds[f"{prefix}_{a}"] - folds[f"{prefix}_{b}"]
        d, lo, hi = boot_cluster_mean(folds, "_d")
        sig = "*" if (lo > 0 or hi < 0) else " "
        pair_rows.append({"part": label, "pair": f"{a}->{b}", "d": d, "lo": lo, "hi": hi})
        print(f"{a:12s}->{b:12s}  d {d:+.3e} [{lo:+.3e},{hi:+.3e}]{sig}")


def main():
    df = pd.read_parquet(PANEL, columns=[
        "instrument", "date", "dx_log", "zero", "I_bin", "S_tick", "split_original"])
    df = df.dropna(subset=["dx_log", "S_tick", "zero"])
    df["y"] = df["dx_log"] ** 2
    df["sb"] = np.clip(df["S_tick"].to_numpy(int), 1, M_S) - 1
    df["cell"] = df["I_bin"] * M_S + df["sb"]
    df["move"] = 1 - df["zero"].to_numpy(int)

    models = ["M0_const", "M1_spread", "M2_imb", "M3_additive", "M4_full"]
    preds = []
    for sym, g in df.groupby("instrument"):
        tr = g[g["split_original"] == "train"]
        te = g[g["split_original"] == "test"]
        if len(tr) < 2000 or len(te) < 1000:
            continue
        ib_te, sb_te, cell_te = (te[c].to_numpy() for c in ("I_bin", "sb", "cell"))

        # --- part 1: occurrence P(move | state), fit on ALL train rows
        t1 = pd.DataFrame({"ib": tr["I_bin"].to_numpy(), "sb": tr["sb"].to_numpy(),
                           "cell": tr["cell"].to_numpy(), "v": tr["move"].to_numpy(float)})
        p0 = float(t1["v"].mean())
        P1 = ladder_fit(t1, "v", ib_te, sb_te, cell_te, MIN_CELL, p0)
        P1 = {m: np.clip(p, EPS, 1 - EPS) for m, p in P1.items()}

        # --- part 2: magnitude E[dx^2 | move, state], fit on train NONZERO rows
        trn = tr[tr["move"] == 1]
        if len(trn) < 1000:
            continue
        cap = float(np.quantile(trn["y"], WINSOR))
        t2 = pd.DataFrame({"ib": trn["I_bin"].to_numpy(), "sb": trn["sb"].to_numpy(),
                           "cell": trn["cell"].to_numpy(),
                           "v": np.minimum(trn["y"].to_numpy(), cap)})
        m0 = float(t2["v"].mean())
        P2 = ladder_fit(t2, "v", ib_te, sb_te, cell_te, MIN_CELL, m0)
        P2 = {m: np.maximum(f, EPS * m0) for m, f in P2.items()}

        yte = np.minimum(te["y"].to_numpy(), cap)
        mv = te["move"].to_numpy(float)
        p = pd.DataFrame({"instrument": sym, "date": te["date"].to_numpy(),
                          "y": yte, "move": mv})
        for m in models:
            p[f"p_{m}"] = P1[m]      # occurrence forecast
            p[f"m_{m}"] = P2[m]      # magnitude forecast
        preds.append(p)

    P = pd.concat(preds, ignore_index=True)
    mv = P["move"].to_numpy()
    nz = mv == 1
    print(f"instruments kept: {P['instrument'].nunique()}   test obs: {len(P):,}   "
          f"move share (test): {mv.mean():.3f}")

    # losses per observation
    for m in models:
        pm = P[f"p_{m}"].to_numpy()
        P[f"ll_{m}"] = -(mv * np.log(pm) + (1 - mv) * np.log(1 - pm))     # binomial log-loss
        P[f"br_{m}"] = (mv - pm) ** 2                                     # Brier
        fm = P[f"m_{m}"].to_numpy()
        P[f"ql_{m}"] = np.where(nz, P["y"] / fm + np.log(fm), np.nan)     # QLIKE, nonzero only
        P[f"m2_{m}"] = np.where(nz, (P["y"] - fm) ** 2, np.nan)           # MSE, nonzero only
        P[f"tp_{m}"] = (P["y"] - pm * fm) ** 2                            # recombined two-part, all obs

    agg = {}
    for pref in ("ll", "br", "ql", "m2", "tp"):
        for m in models:
            agg[f"{pref}_{m}"] = "mean"
    folds = P.groupby(["instrument", "date"]).agg(agg).reset_index()
    print(f"folds: {len(folds):,}")

    out_rows, pair_rows = [], []
    report(folds, models, "ll", "Part 1 occurrence — log-loss", out_rows, pair_rows, base="M0_const")
    report(folds, models, "br", "Part 1 occurrence — Brier", out_rows, pair_rows, base="M0_const")
    report(folds, models, "ql", "Part 2 magnitude — QLIKE (nonzero, non-degenerate)", out_rows, pair_rows)
    report(folds, models, "m2", "Part 2 magnitude — winsorized MSE (nonzero)", out_rows, pair_rows, base="M0_const")
    report(folds, models, "tp", "Recombined p*m — MSE on all test obs", out_rows, pair_rows, base="M0_const")

    pd.DataFrame(out_rows).to_csv(OUT / "a2_twopart_levels.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "a2_twopart_pairs.csv", index=False)
    print(f"\nwrote {OUT}/a2_twopart_levels.csv, a2_twopart_pairs.csv")


if __name__ == "__main__":
    main()
