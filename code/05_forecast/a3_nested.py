#!/usr/bin/env python3
"""
a3_nested.py — A3: the nested model ladder, judged by forecast loss on the held-out split.

Models for the one-step conditional variance E[dx^2 | state], fit on split_original=="train"
per instrument, evaluated on "test" (the paper's chronological 60/40 split):
  M0 constant          g
  M1 spread-only       a(S)          spread-in-ticks clipped to [1,3]
  M2 imbalance-only    a(I)          10 uniform I-bins
  M3 additive          a(I) + a(S) - g   (floored at 1e-4 * g)
  M4 full surface      a(I,S) per cell, MIN_CELL=50, fallback -> additive -> g

Losses on test:
  primary   MSE on winsorized dx^2 (train 0.995 cap per instrument) — the proper loss for the
            conditional MEAN of dx^2; one-step QLIKE/log-score are degenerate here because 48%
            of moves are exactly zero (the two-part model A2 handles that mass separately)
  secondary QLIKE = y/f + log f on the NONZERO-move subset

Inference: per-(instrument, date) mean loss differences vs the FULL surface; instrument-
clustered bootstrap (1000 draws) on the fold means. Positive dMSE => full surface better.
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


def main():
    df = pd.read_parquet(PANEL, columns=[
        "instrument", "date", "dx_log", "zero", "I_bin", "S_tick", "split_original"])
    df = df.dropna(subset=["dx_log", "S_tick"])
    df["y"] = df["dx_log"] ** 2
    df["sb"] = np.clip(df["S_tick"].to_numpy(int), 1, M_S) - 1
    df["cell"] = df["I_bin"] * M_S + df["sb"]

    preds, fold_rows = [], []
    for sym, g in df.groupby("instrument"):
        tr = g[g["split_original"] == "train"]
        te = g[g["split_original"] == "test"]
        if len(tr) < 2000 or len(te) < 1000:
            continue
        cap = float(np.quantile(tr["y"], WINSOR))
        ytr = np.minimum(tr["y"].to_numpy(), cap)
        g0 = float(ytr.mean())
        t = pd.DataFrame({"ib": tr["I_bin"].to_numpy(), "sb": tr["sb"].to_numpy(),
                          "cell": tr["cell"].to_numpy(), "y": ytr})
        aS = t.groupby("sb")["y"].mean().to_dict()
        aI = t.groupby("ib")["y"].mean().to_dict()
        cell_stats = t.groupby("cell")["y"].agg(["mean", "size"])
        a_full = cell_stats.loc[cell_stats["size"] >= MIN_CELL, "mean"].to_dict()

        ib_te = te["I_bin"].to_numpy(); sb_te = te["sb"].to_numpy(); cell_te = te["cell"].to_numpy()
        fS = np.array([aS.get(s, g0) for s in sb_te])
        fI = np.array([aI.get(i, g0) for i in ib_te])
        floor = 1e-4 * g0
        fAdd = np.maximum(fI + fS - g0, floor)
        fFull = np.array([a_full.get(c, np.nan) for c in cell_te])
        fFull = np.where(np.isfinite(fFull), fFull, fAdd)          # fallback: additive
        f0 = np.full(len(te), g0)

        yte = np.minimum(te["y"].to_numpy(), cap)
        nz = te["zero"].to_numpy() == 0                             # nonzero moves (QLIKE domain)
        p = pd.DataFrame({"instrument": sym, "date": te["date"].to_numpy(), "y": yte, "nz": nz,
                          "M0_const": f0, "M1_spread": fS, "M2_imb": fI,
                          "M3_additive": fAdd, "M4_full": fFull})
        preds.append(p)

    P = pd.concat(preds, ignore_index=True)
    models = ["M0_const", "M1_spread", "M2_imb", "M3_additive", "M4_full"]

    # per-(instrument, date) fold losses
    for m in models:
        P[f"mse_{m}"] = (P["y"] - P[m]) ** 2
        P[f"ql_{m}"] = np.where(P["nz"], P["y"] / P[m] + np.log(P[m]), np.nan)
    folds = P.groupby(["instrument", "date"]).agg(
        {**{f"mse_{m}": "mean" for m in models}, **{f"ql_{m}": "mean" for m in models}}).reset_index()

    print(f"instruments: {P['instrument'].nunique()}   test obs: {len(P):,}   "
          f"folds: {len(folds):,}   nonzero share: {P['nz'].mean():.3f}\n")

    rows = []
    print("== Ladder: loss level and skill vs constant (instrument-clustered bootstrap) ==")
    base_mse = folds["mse_M0_const"]
    for m in models:
        mse_m, mse_lo, mse_hi = boot_cluster_mean(folds, f"mse_{m}")
        ql_m, ql_lo, ql_hi = boot_cluster_mean(folds, f"ql_{m}")
        skill = 1 - folds[f"mse_{m}"].mean() / base_mse.mean()
        rows.append({"model": m, "mse": mse_m, "mse_lo": mse_lo, "mse_hi": mse_hi,
                     "skill_vs_const": skill, "qlike_nz": ql_m, "ql_lo": ql_lo, "ql_hi": ql_hi})
        print(f"{m:12s}  MSE {mse_m:.3e} [{mse_lo:.3e},{mse_hi:.3e}]  "
              f"skill {skill:+.4f}   QLIKE(nz) {ql_m:+.4f} [{ql_lo:+.4f},{ql_hi:+.4f}]")

    print("\n== Pairwise loss differences (positive => second model better) ==")
    pairs = [("M0_const", "M1_spread"), ("M1_spread", "M4_full"), ("M2_imb", "M4_full"),
             ("M1_spread", "M3_additive"), ("M3_additive", "M4_full"), ("M0_const", "M4_full")]
    prow = []
    for a, b in pairs:
        folds["d_mse"] = folds[f"mse_{a}"] - folds[f"mse_{b}"]
        folds["d_ql"] = folds[f"ql_{a}"] - folds[f"ql_{b}"]
        dm, dlo, dhi = boot_cluster_mean(folds, "d_mse")
        dq, qlo, qhi = boot_cluster_mean(folds, "d_ql")
        sig_m = "*" if (dlo > 0 or dhi < 0) else " "
        sig_q = "*" if (qlo > 0 or qhi < 0) else " "
        prow.append({"pair": f"{a}->{b}", "d_mse": dm, "d_mse_lo": dlo, "d_mse_hi": dhi,
                     "d_qlike": dq, "d_ql_lo": qlo, "d_ql_hi": qhi})
        print(f"{a:12s}->{b:12s}  dMSE {dm:+.3e} [{dlo:+.3e},{dhi:+.3e}]{sig_m}  "
              f"dQLIKE {dq:+.5f} [{qlo:+.5f},{qhi:+.5f}]{sig_q}")

    pd.DataFrame(rows).to_csv(OUT / "a3_ladder_levels.csv", index=False)
    pd.DataFrame(prow).to_csv(OUT / "a3_ladder_pairs.csv", index=False)
    print(f"\nwrote {OUT}/a3_ladder_levels.csv, a3_ladder_pairs.csv")


if __name__ == "__main__":
    main()
