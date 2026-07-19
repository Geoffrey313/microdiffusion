#!/usr/bin/env python3
"""
a5_losses_garch.py — A5: forecast losses vs GARCH under the paper's exact protocol,
with instrument-day block-bootstrap inference, run from event_panel.parquet alone.

Protocol replicated from conditional_sde_benchmarks.py (the script behind the published
GARCH comparison): r = 1e4 * dx_log demeaned by the train mean; cells = 10 uniform I-bins
x per-instrument S_mid train terciles (the panel's I_bin/S_bin); surface a_xx = raw train
cell means, MIN_CELL=50; Gaussian GARCH(1,1) and JOINT GARCH-t fitted once on train
(FIT_CAP=12000, Nelder-Mead), variance filtered through the concatenated series; rows in
cells without a surface value are masked out on both sides; GH innovation law fitted on
POOLED train standardised residuals (genhyperbolic, fb=0, floc=0, subsample 120k).

Models compared on test:
  GARCH+Gaussian | joint GARCH-t | surface a_xx + GH | spread-only + GH
Losses:
  predictive log-score (all test obs)  — the paper's headline metric
  QLIKE on nonzero moves               — variance loss where non-degenerate
  winsorized MSE on dx^2 (train 0.995) — variance loss, A3-consistent
Inference: per-(instrument, date) fold means; instrument-clustered bootstrap (1000).

Also emits Jinliang's fast-path files into data_export/:
  predictions_test.parquet — per test row: all four sigma^2 forecasts + state + target
  residuals_var.parquet    — per train row: standardised residual z = r / sigma_axx
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
PANEL = HERE.parents[1] / "data" / "data_export" / "event_panel.parquet"
EXPORT = HERE.parents[1] / "data" / "data_export"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

N_I, M_S = 10, 3
MIN_CELL = 50
FIT_CAP = 12000
WINSOR = 0.995
BPS = 1e4
N_BOOT = 1000
RNG = np.random.default_rng(0)


# ---- GARCH machinery, verbatim from conditional_sde_benchmarks.py ----
def garch_filter(r2, w, a, b, s0):
    n = len(r2); s2 = np.empty(n); s2[0] = s0
    for t in range(1, n):
        s2[t] = w + a * r2[t - 1] + b * s2[t - 1]
    return s2


def fit_garch_gauss(r):
    rr = r if len(r) <= FIT_CAP else r[-FIT_CAP:]
    r2 = rr * rr; v = float(np.var(rr)) + 1e-9

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        s2 = np.maximum(garch_filter(r2, w, a, b, v), 1e-12)
        return 0.5 * np.sum(np.log(s2) + r2 / s2)
    best = None
    for init in [(0.1 * v, 0.05, 0.90), (0.3 * v, 0.10, 0.80)]:
        res = optimize.minimize(nll, init, method="Nelder-Mead",
                                options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-2})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x)


def std_t_logpdf(z, nu):
    c = np.sqrt(nu / (nu - 2.0))
    return stats.t.logpdf(z * c, nu) + np.log(c)


def fit_garch_t_joint(r):
    rr = r if len(r) <= FIT_CAP else r[-FIT_CAP:]
    r2 = rr * rr; v = float(np.var(rr)) + 1e-9

    def nll(p):
        w, a, b, nu = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999 or nu <= 2.05 or nu > 100:
            return 1e12
        s2 = np.maximum(garch_filter(r2, w, a, b, v), 1e-12)
        z = rr / np.sqrt(s2)
        return -np.sum(std_t_logpdf(z, nu) - 0.5 * np.log(s2))
    best = None
    for init in [(0.1 * v, 0.05, 0.90, 6.0), (0.3 * v, 0.10, 0.80, 4.0)]:
        res = optimize.minimize(nll, init, method="Nelder-Mead",
                                options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-2})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x)


def fit_gh(z):
    zz = z if len(z) <= 120_000 else RNG.choice(z, 120_000, replace=False)
    p, a, b, loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    return stats.genhyperbolic(p, a, 0.0, 0.0, s)


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
        "instrument", "date", "ts_event", "dx_log", "zero", "I_bin", "S_bin", "split_original"])
    df = df.dropna(subset=["dx_log"])
    df = df[df["S_bin"] >= 0]                       # thin names carry S_bin = -1
    df["cell"] = df["I_bin"] * M_S + df["S_bin"]

    per_sym, ztr_axx, ztr_sp = [], [], []
    res_rows = []
    for sym, g in df.groupby("instrument"):
        g = g.sort_values(["date", "ts_event"], kind="mergesort")
        tr = g[g["split_original"] == "train"]
        te = g[g["split_original"] == "test"]
        if len(tr) < 3000 or len(te) < 1000:
            continue
        mu = float((tr["dx_log"] * BPS).mean())
        r_tr = tr["dx_log"].to_numpy() * BPS - mu
        r_te = te["dx_log"].to_numpy() * BPS - mu

        # surface and spread-only sigma^2 from raw train cell means (bps^2)
        t = pd.DataFrame({"cell": tr["cell"].to_numpy(), "sb": tr["S_bin"].to_numpy(),
                          "y": r_tr * r_tr})
        cs = t.groupby("cell")["y"].agg(["mean", "size"])
        axx = cs.loc[cs["size"] >= MIN_CELL, "mean"].to_dict()
        asp = t.groupby("sb")["y"].mean().to_dict()
        if not axx:
            continue

        ax_tr = np.array([axx.get(c, np.nan) for c in tr["cell"]])
        ax_te = np.array([axx.get(c, np.nan) for c in te["cell"]])
        sp_tr = np.array([asp.get(s, np.nan) for s in tr["S_bin"]])
        sp_te = np.array([asp.get(s, np.nan) for s in te["S_bin"]])

        # GARCH fits once on train; filter through concatenated series (paper protocol)
        w, a, b = fit_garch_gauss(r_tr)
        wt, at, bt, nu = fit_garch_t_joint(r_tr)
        r_all = np.concatenate([r_tr, r_te])
        s0 = float(np.var(r_tr)) + 1e-9
        g_all = garch_filter(r_all * r_all, w, a, b, s0)
        gt_all = garch_filter(r_all * r_all, wt, at, bt, s0)
        g_te = g_all[len(r_tr):]; gt_te = gt_all[len(r_tr):]
        g_tr = g_all[:len(r_tr)]

        mtr = np.isfinite(ax_tr) & (g_tr > 0)
        mte = np.isfinite(ax_te) & (g_te > 0)
        if mtr.sum() < 1000 or mte.sum() < 500:
            continue

        ztr_axx.append(r_tr[mtr] / np.sqrt(ax_tr[mtr]))
        ztr_sp.append(r_tr[mtr] / np.sqrt(sp_tr[mtr]))
        res_rows.append(pd.DataFrame({
            "instrument": sym, "date": tr["date"].to_numpy()[mtr],
            "ts_event": tr["ts_event"].to_numpy()[mtr],
            "z_axx": r_tr[mtr] / np.sqrt(ax_tr[mtr])}))

        cap = float(np.quantile(r_tr[mtr] ** 2, WINSOR))
        per_sym.append({
            "sym": sym, "nu": nu,
            "date": te["date"].to_numpy()[mte], "ts": te["ts_event"].to_numpy()[mte],
            "r": r_te[mte], "zero": te["zero"].to_numpy()[mte],
            "s2": {"garch_g": g_te[mte], "garch_t": gt_te[mte],
                   "axx": ax_te[mte], "spread": sp_te[mte]},
            "cap": cap})
        print(f"[fit] {sym:6s} n_te={mte.sum():6d}  GARCH-t nu={nu:5.2f}  persist={at+bt:.3f}",
              flush=True)

    print("\nfitting pooled GH innovation laws (train residuals)...", flush=True)
    gh_axx = fit_gh(np.concatenate(ztr_axx))
    gh_sp = fit_gh(np.concatenate(ztr_sp))
    print("GH fits done.", flush=True)

    MODELS = ["garch_g", "garch_t", "axx_gh", "spread_gh"]
    obs, pred_rows = [], []
    for d in per_sym:
        r = d["r"]; s2 = d["s2"]; n = len(r)
        ls = {
            "garch_g": -0.5 * np.log(s2["garch_g"]) + stats.norm.logpdf(r / np.sqrt(s2["garch_g"])),
            "garch_t": -0.5 * np.log(s2["garch_t"]) + std_t_logpdf(r / np.sqrt(s2["garch_t"]), d["nu"]),
            "axx_gh": -0.5 * np.log(s2["axx"]) + gh_axx.logpdf(r / np.sqrt(s2["axx"])),
            "spread_gh": -0.5 * np.log(s2["spread"]) + gh_sp.logpdf(r / np.sqrt(s2["spread"])),
        }
        y = np.minimum(r * r, d["cap"])
        nz = d["zero"] == 0
        o = pd.DataFrame({"instrument": d["sym"], "date": d["date"]})
        for m in MODELS:
            f = s2[m.replace("_gh", "")] if m.endswith("_gh") else s2[m]
            o[f"ls_{m}"] = ls[m]
            o[f"mse_{m}"] = (y - f) ** 2
            o[f"ql_{m}"] = np.where(nz, r * r / f + np.log(f), np.nan)
        obs.append(o)
        pred_rows.append(pd.DataFrame({
            "instrument": d["sym"], "date": d["date"], "ts_event": d["ts"],
            "r_bps": r, "zero": d["zero"],
            "sigma2_garch_gauss": s2["garch_g"], "sigma2_garch_t": s2["garch_t"],
            "sigma2_surface": s2["axx"], "sigma2_spread": s2["spread"],
            "nu_t": d["nu"]}))

    P = pd.concat(obs, ignore_index=True)
    agg = {c: "mean" for c in P.columns if c not in ("instrument", "date")}
    folds = P.groupby(["instrument", "date"]).agg(agg).reset_index()
    print(f"\ninstruments: {P['instrument'].nunique()}   test obs: {len(P):,}   folds: {len(folds):,}\n")

    rows, prow = [], []
    for pref, label, better in [("ls", "predictive log-score (higher better)", "higher"),
                                ("ql", "QLIKE nonzero (lower better)", "lower"),
                                ("mse", "winsorized MSE (lower better)", "lower")]:
        print(f"== {label} ==")
        for m in MODELS:
            v, lo, hi = boot_cluster_mean(folds, f"{pref}_{m}")
            rows.append({"loss": pref, "model": m, "value": v, "lo": lo, "hi": hi})
            print(f"{m:10s}  {v:+.6f} [{lo:+.6f},{hi:+.6f}]")
        print("-- pairwise differences (sign convention: positive => SECOND model better) --")
        for a2, b2 in [("garch_t", "axx_gh"), ("garch_t", "spread_gh"), ("axx_gh", "spread_gh"),
                       ("garch_g", "garch_t"), ("garch_g", "axx_gh")]:
            if better == "higher":
                folds["_d"] = folds[f"{pref}_{b2}"] - folds[f"{pref}_{a2}"]
            else:
                folds["_d"] = folds[f"{pref}_{a2}"] - folds[f"{pref}_{b2}"]
            dv, lo, hi = boot_cluster_mean(folds, "_d")
            sig = "*" if (lo > 0 or hi < 0) else " "
            prow.append({"loss": pref, "pair": f"{a2}->{b2}", "d": dv, "lo": lo, "hi": hi})
            print(f"{a2:10s}->{b2:10s}  d {dv:+.4e} [{lo:+.4e},{hi:+.4e}]{sig}")
        print()

    pd.DataFrame(rows).to_csv(OUT / "a5_losses_levels.csv", index=False)
    pd.DataFrame(prow).to_csv(OUT / "a5_losses_pairs.csv", index=False)

    pred = pd.concat(pred_rows, ignore_index=True)
    pred.to_parquet(EXPORT / "predictions_test.parquet", index=False)
    res = pd.concat(res_rows, ignore_index=True)
    res.to_parquet(EXPORT / "residuals_var.parquet", index=False)
    print(f"wrote {OUT}/a5_losses_levels.csv, a5_losses_pairs.csv")
    print(f"wrote {EXPORT}/predictions_test.parquet ({len(pred):,} rows), "
          f"residuals_var.parquet ({len(res):,} rows)")


if __name__ == "__main__":
    main()
