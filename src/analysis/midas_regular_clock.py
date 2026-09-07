#!/usr/bin/env python3
"""
midas_regular_clock.py - regular-clock (60-second) volatility forecasting: does the
event-time microdiffusion carry out-of-sample content on a calendar clock?

The paper's headline forecasting result lives on the EVENT clock (per book update). A
referee interested in mixed-frequency / MIDAS work will ask the natural next question:
does the same book state help predict volatility on a standard CALENDAR clock? This
script answers it as a mixed-frequency (MIDAS-type) forecasting experiment.

Design (per instrument, same 60/40 chronological train/test split as the rest of the paper):
  1. Fit the diffusion surface a_xx(I,S) on the TRAIN sessions (identical construction to
     var_backtest.py / es_evt_scoring.py).
  2. Resample each session's event-time L1 stream into 60-second calendar bars. For bar t:
        RV_t   = sum of squared per-update mid log-returns in the bar   (realised variance)
        axx_t  = mean of the trained a_xx(I,S) over the bar's updates    (book-state signal,
                 the microdiffusion aggregated from event time to the 60s clock)
        S_t    = mean relative spread over the bar
     Only bars with at least MIN_UPD updates are kept (frontier-market bars are thin).
  3. One-step-ahead forecasting of RV_{t+1} (bars ordered within a session; no target crosses
     a session boundary). Predictors known at the end of bar t:
        HAR-RV baseline : RV_t and the mean RV over the last RV_WIN bars (a parsimonious,
                          Corsi-style mixed-horizon lag - itself a restricted MIDAS).
        book-state add  : axx_t (and, separately, S_t).
     Models are fitted by OLS on log-RV (stabilises the right skew) on TRAIN and evaluated
     out-of-sample on TEST.

Reported (out of sample): log-RV R^2, QLIKE (scale-free), and, for the key contrast
HAR+book vs HAR, a Diebold-Mariano / Giacomini-White loss-differential test clustered by
session (per-session mean QLIKE differential, one-sample t across sessions).

This is a REGULAR-CLOCK (60s) HAR-RV forecasting contest with a mixed-frequency book-state
regressor; it is NOT a fully parametrised Beta-weight MIDAS regression (no Beta lag is
estimated), so it is labelled "regular-clock (60s) forecasting", not "MIDAS estimation".

Outputs (NOT digest-gated, NOT written into the manuscript):
  results/diagnostics/tables/midas_regular_clock_summary.csv   per-instrument OOS metrics
  results/diagnostics/tables/midas_regular_clock_pooled.csv    pooled headline numbers
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_config, CLEAN_DIR, DIAG_DIR, SECTIONS_DIR
from common.grid import N_I, M_S, cell_ids
from analysis.var_backtest import parse_date

TRAIN_FRAC = 0.60
MIN_CELL   = 50
BAR_SECONDS = 60
MIN_UPD    = 3        # minimum updates in a 60s bar to compute a usable RV
RV_WIN     = 5        # bars in the HAR-style medium lag (5 min)
MIN_TEST_BARS = 60    # minimum usable one-step test pairs to keep an instrument
EPS = 1e-12


# ---------------------------------------------------------------------------
# per-update reader that KEEPS timestamps (var_backtest.session_increments drops them)
# ---------------------------------------------------------------------------

def session_updates(path):
    l1 = pd.read_parquet(path, columns=["ts", "bid", "ask", "bid_sz", "ask_sz", "mid"])
    mid = pd.to_numeric(l1["mid"], errors="coerce").to_numpy(float)
    bid = pd.to_numeric(l1["bid"], errors="coerce").to_numpy(float)
    ask = pd.to_numeric(l1["ask"], errors="coerce").to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float)
    asz = l1["ask_sz"].to_numpy(float)
    ts  = pd.to_datetime(l1["ts"]).to_numpy()
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz) & (bsz + asz > 0)
    if good.sum() < 200:
        return None
    x = np.log(mid)
    dx = np.diff(x)                       # increment i -> i+1
    I = (bsz - asz) / (bsz + asz)
    S = (ask - bid) / mid
    # Canonical alignment (identical to var_backtest.py / es_evt_scoring.py): pair each
    # increment dx[i] with the book state BEFORE it, I[:-1] / S[:-1]; the increment is
    # timestamped by its start ts[:-1] for the 60s binning.
    df = pd.DataFrame({"ts": ts[:-1], "dx": dx, "I": I[:-1], "S": S[:-1]})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df if len(df) >= 50 else None


def fit_surface(tr_paths):
    frames = [session_updates(p) for p in tr_paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    tr = pd.concat(frames, ignore_index=True)
    if len(tr) < 2000:
        return None
    s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
    ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
    dxtr = tr["dx"].to_numpy()
    axx = {}
    for c in range(N_I * M_S):
        m = ctr == c
        if m.sum() >= MIN_CELL:
            axx[c] = float((dxtr[m] ** 2).mean())
    return s_edges, axx


def session_bars(path, s_edges, axx):
    """60s bars for one session: RV, mean book-state a_xx, mean spread, n_upd."""
    df = session_updates(path)
    if df is None:
        return None
    c = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
    df = df.assign(axx_c=[axx.get(cc, np.nan) for cc in c])
    bar = pd.to_datetime(df["ts"]).dt.floor(f"{BAR_SECONDS}s")
    g = df.groupby(bar)
    out = pd.DataFrame({
        "rv":    g["dx"].apply(lambda s: float(np.sum(s.to_numpy() ** 2))),
        "axx":   g["axx_c"].mean(),
        "spread": g["S"].mean(),
        "n_upd": g["dx"].count(),
    }).reset_index(drop=True)
    out = out[(out["n_upd"] >= MIN_UPD) & (out["rv"] > 0) & np.isfinite(out["axx"])]
    return out if len(out) >= (RV_WIN + 3) else None


# ---------------------------------------------------------------------------
# forecasting dataset: one-step-ahead pairs, reset per session
# ---------------------------------------------------------------------------

def build_pairs(bars_list):
    """From a list of per-session bar frames, build one-step X/y with HAR lags.
    Returns a DataFrame with target y=logRV_{t+1}, and features, plus a 'sess' id."""
    rows = []
    for sid, b in enumerate(bars_list):
        rv = b["rv"].to_numpy()
        axx = b["axx"].to_numpy()
        spr = b["spread"].to_numpy()
        lrv = np.log(rv + EPS)
        n = len(rv)
        # medium lag: trailing mean of RV over the last RV_WIN bars (inclusive of t)
        rv_win = np.array([rv[max(0, i - RV_WIN + 1):i + 1].mean() for i in range(n)])
        for t in range(RV_WIN - 1, n - 1):     # need history and a next bar
            rows.append(dict(
                sess=sid,
                y=lrv[t + 1],
                rv_t=lrv[t],
                rv_win=np.log(rv_win[t] + EPS),
                axx_t=np.log(axx[t] + EPS),
                spread_t=spr[t],
                rv_true=rv[t + 1],
            ))
    return pd.DataFrame(rows)


def ols_fit(X, y):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), X.T @ y)
    return beta


def evaluate(train, test, feats):
    """Fit OLS(log-RV) on train feats, return dict of OOS metrics on test."""
    Xtr = np.column_stack([np.ones(len(train))] + [train[f].to_numpy() for f in feats])
    ytr = train["y"].to_numpy()
    beta = ols_fit(Xtr, ytr)
    Xte = np.column_stack([np.ones(len(test))] + [test[f].to_numpy() for f in feats])
    pred = Xte @ beta                      # predicted log-RV
    yte = test["y"].to_numpy()
    sse = float(np.sum((yte - pred) ** 2))
    sst = float(np.sum((yte - ytr.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    rv_true = test["rv_true"].to_numpy()
    rv_pred = np.exp(pred)
    ratio = rv_true / np.maximum(rv_pred, EPS)
    qlike = float(np.mean(ratio - np.log(np.maximum(ratio, EPS)) - 1.0))
    # per-observation QLIKE loss (for DM), and the session id
    qloss = ratio - np.log(np.maximum(ratio, EPS)) - 1.0
    return dict(r2=r2, qlike=qlike), qloss


def oos_r2(train, test, feats, target):
    """Out-of-sample R^2 for OLS(target ~ feats), fit on train, evaluated on test."""
    Xtr = np.column_stack([np.ones(len(train))] + [train[f].to_numpy() for f in feats])
    ytr = train[target].to_numpy()
    beta = ols_fit(Xtr, ytr)
    Xte = np.column_stack([np.ones(len(test))] + [test[f].to_numpy() for f in feats])
    pred = Xte @ beta
    yte = test[target].to_numpy()
    sse = float(np.sum((yte - pred) ** 2))
    sst = float(np.sum((yte - ytr.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


def dm_clustered(qloss_base, qloss_alt, sess):
    """Session-clustered DM/Giacomini-White: per-session mean loss differential
    (base - alt), one-sample t across sessions. Positive => alt improves."""
    d = qloss_base - qloss_alt
    df = pd.DataFrame({"sess": sess, "d": d})
    per = df.groupby("sess")["d"].mean().to_numpy()
    per = per[np.isfinite(per)]
    if len(per) < 5:
        return np.nan, np.nan, len(per)
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)) + EPS)
    p = 2 * stats.t.sf(abs(t), df=len(per) - 1)
    return float(t), float(p), len(per)


def main():
    load_config()
    venue = os.environ.get("MICRODIFFUSION_VENUE", "QSE")  # relabelled by reproduce.py --us
    files: dict[str, list] = {}
    for p in CLEAN_DIR.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    MODELS = {
        "HAR":         ["rv_t", "rv_win"],
        "HAR+axx":     ["rv_t", "rv_win", "axx_t"],
        "HAR+spread":  ["rv_t", "rv_win", "spread_t"],
        "axx_only":    ["axx_t"],
        "spread_only": ["spread_t"],
    }

    rows, dm_rows = [], []
    kept_bars_frac = []
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr_paths, te_paths = paths[:ntr], paths[ntr:]
        if not te_paths:
            continue
        surf = fit_surface(tr_paths)
        if surf is None:
            continue
        s_edges, axx = surf
        tr_bars = [b for b in (session_bars(p, s_edges, axx) for p in tr_paths) if b is not None]
        te_bars = [b for b in (session_bars(p, s_edges, axx) for p in te_paths) if b is not None]
        if not tr_bars or not te_bars:
            continue
        train = build_pairs(tr_bars)
        test  = build_pairs(te_bars)
        if len(train) < 100 or len(test) < MIN_TEST_BARS:
            continue

        res, qloss = {}, {}
        for name, feats in MODELS.items():
            m, q = evaluate(train, test, feats)
            res[name] = m
            qloss[name] = q
        # DM: does adding the book state to HAR help?
        t_axx, p_axx, nse = dm_clustered(qloss["HAR"], qloss["HAR+axx"], test["sess"].to_numpy())
        t_spr, p_spr, _   = dm_clustered(qloss["HAR"], qloss["HAR+spread"], test["sess"].to_numpy())
        row = dict(sym=sym, n_test=len(test),
                   dm_t_axx=t_axx, dm_p_axx=p_axx, dm_t_spread=t_spr, dm_p_spread=p_spr)
        for name in MODELS:
            row[f"r2_{name}"] = res[name]["r2"]
            row[f"qlike_{name}"] = res[name]["qlike"]
        # contemporaneous (same-bar nowcast): does the book state describe RV_t itself?
        row["r2_axx_contemp"]    = oos_r2(train, test, ["axx_t"], "rv_t")
        row["r2_spread_contemp"] = oos_r2(train, test, ["spread_t"], "rv_t")
        row["r2_both_contemp"]   = oos_r2(train, test, ["axx_t", "spread_t"], "rv_t")
        row["r2_both_onestep"]   = oos_r2(train, test, ["axx_t", "spread_t"], "y")
        rows.append(row)
        print(f"  {sym:8s} n_test={len(test):>5d}  "
              f"R2 HAR={res['HAR']['r2']:+.3f} +axx={res['HAR+axx']['r2']:+.3f} "
              f"+spread={res['HAR+spread']['r2']:+.3f}  "
              f"DM(axx) t={t_axx:+.2f}  |  "
              f"axx contemp(RV_t)={row['r2_axx_contemp']:+.3f} vs 1step(RV_t+1)={res['axx_only']['r2']:+.3f}")

    if not rows:
        print("No results - check data path / bar sparsity.")
        return
    df = pd.DataFrame(rows)
    (DIAG_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(DIAG_DIR / "tables" / "midas_regular_clock_summary.csv", index=False)

    # ---- pooled headline ----
    N = len(df)
    def med(col): return float(df[col].median())
    def improves(base, alt):   # QLIKE lower is better
        return int((df[f"qlike_{alt}"] < df[f"qlike_{base}"]).sum())
    pooled = dict(
        n_instruments=N,
        r2_HAR=med("r2_HAR"), r2_HARaxx=med("r2_HAR+axx"), r2_HARspread=med("r2_HAR+spread"),
        r2_axx_only=med("r2_axx_only"), r2_spread_only=med("r2_spread_only"),
        qlike_HAR=med("qlike_HAR"), qlike_HARaxx=med("qlike_HAR+axx"), qlike_HARspread=med("qlike_HAR+spread"),
        n_axx_improves_HAR=improves("HAR", "HAR+axx"),
        n_spread_improves_HAR=improves("HAR", "HAR+spread"),
        n_axx_DM_sig_pos=int(((df["dm_p_axx"] < 0.05) & (df["dm_t_axx"] > 0)).sum()),
        n_spread_DM_sig_pos=int(((df["dm_p_spread"] < 0.05) & (df["dm_t_spread"] > 0)).sum()),
        r2_axx_contemp=med("r2_axx_contemp"), r2_spread_contemp=med("r2_spread_contemp"),
        r2_both_contemp=med("r2_both_contemp"), r2_both_onestep=med("r2_both_onestep"),
    )
    pd.DataFrame([pooled]).to_csv(DIAG_DIR / "tables" / "midas_regular_clock_pooled.csv", index=False)

    print("\n=== Regular-clock (60s) RV forecasting, out of sample, N=%d instruments ===" % N)
    print("median OOS log-RV R^2:  HAR=%.3f  HAR+axx=%.3f  HAR+spread=%.3f  |  axx_only=%.3f  spread_only=%.3f"
          % (pooled["r2_HAR"], pooled["r2_HARaxx"], pooled["r2_HARspread"],
             pooled["r2_axx_only"], pooled["r2_spread_only"]))
    print("median OOS QLIKE:       HAR=%.3f  HAR+axx=%.3f  HAR+spread=%.3f  (lower better)"
          % (pooled["qlike_HAR"], pooled["qlike_HARaxx"], pooled["qlike_HARspread"]))
    print("adding book state beats HAR (QLIKE): axx %d/%d, spread %d/%d instruments"
          % (pooled["n_axx_improves_HAR"], N, pooled["n_spread_improves_HAR"], N))
    print("session-clustered DM significant improvement (p<0.05, t>0): axx %d/%d, spread %d/%d"
          % (pooled["n_axx_DM_sig_pos"], N, pooled["n_spread_DM_sig_pos"], N))
    print("median OOS R^2 -- book state CONTEMPORANEOUS (RV_t) vs ONE-STEP (RV_{t+1}):")
    print("   axx:    contemp %.3f  vs  one-step %.3f" % (pooled["r2_axx_contemp"], pooled["r2_axx_only"]))
    print("   spread: contemp %.3f  vs  one-step %.3f" % (pooled["r2_spread_contemp"], pooled["r2_spread_only"]))
    print("   both (axx+spread) contemp %.3f  one-step %.3f" % (pooled["r2_both_contemp"], pooled["r2_both_onestep"]))

    # ---- LaTeX table fragment for the Robustness section (auto-generated, bilingual) ----
    fr = os.environ.get("MICRODIFFUSION_LANG", "en") == "fr"
    P = pooled
    def f3(x): return "%.3f" % x
    a_cnt = "$%d/%d\\ (%d)$" % (P["n_axx_improves_HAR"], N, P["n_axx_DM_sig_pos"])
    s_cnt = "$%d/%d\\ (%d)$" % (P["n_spread_improves_HAR"], N, P["n_spread_DM_sig_pos"])
    if fr:
        caption = ("Vérification de prévision de volatilité sur grille calendaire de 60 secondes, sessions %s "
                   "de test. Chaque session est rééchantillonnée en barres de 60 secondes ; $RV$ est la variance "
                   "réalisée de la barre, $\\bar{a}_{xx}$ et $\\bar{S}$ la diffusion d'état du carnet et l'écart "
                   "relatif moyens de la barre (caractéristiques en temps-événement agrégées à la grille de "
                   "60 secondes). Tous les chiffres sont des médianes sur les $%d$ instruments, hors échantillon ; "
                   "le $R^2$ porte sur le log de la variance réalisée. Panneau~A : prévision à un pas de $RV_{t+1}$ "
                   "par une référence HAR-RV et par HAR-RV augmentée de l'état du carnet ; le QLIKE est sans échelle "
                   "(plus bas est meilleur), et la dernière colonne compte les instruments où le modèle augmenté "
                   "améliore le QLIKE face à HAR (test de différentiel de perte clusterisé par session, significatif "
                   "à 5\\%% entre parenthèses). Panneau~B : $R^2$ médian hors échantillon sur le log-$RV$ de l'état "
                   "du carnet pour la barre \\emph{contemporaine} ($\\log RV_t$) contre la barre \\emph{à un pas} "
                   "($\\log RV_{t+1}$). Vérification à fréquence mixte sur grille calendaire de 60 secondes, non une estimation "
                   "MIDAS à poids Beta.") % (venue, N)
        pA = "\\emph{Panneau A : prévision à un pas de $RV_{t+1}$ (médiane sur les instruments)}"
        hA = "Modèle & $R^2$ & QLIKE & Bat HAR (sig.)"
        pB = "\\emph{Panneau B : $R^2$ de l'état du carnet, contemporain vs à un pas}"
        hB = "Prédicteur & Contemp.\\ ($\\log RV_t$) & À un pas ($\\log RV_{t+1}$) &"
    else:
        caption = ("Regular-clock (60-second) volatility-forecasting check on held-out %s sessions. Each session "
                   "is resampled into 60-second bars; $RV$ is the bar realised variance, $\\bar{a}_{xx}$ and "
                   "$\\bar{S}$ the bar-mean book-state diffusion and relative spread (event-time features aggregated "
                   "to the 60-second clock). All figures are medians across the $%d$ instruments, out of sample; "
                   "$R^2$ is on log realised variance. Panel~A: one-step-ahead forecast of $RV_{t+1}$ by a HAR-RV "
                   "baseline and by HAR-RV augmented with the book state; QLIKE is scale-free (lower is better), and "
                   "the last column counts instruments where the augmented model improves QLIKE over HAR "
                   "(session-clustered loss-differential test significant at 5\\%% in parentheses). Panel~B: median "
                   "out-of-sample log-$RV$ $R^2$ of the book state for the \\emph{contemporaneous} bar ($\\log RV_t$) "
                   "versus the \\emph{one-step-ahead} bar ($\\log RV_{t+1}$). A regular-clock mixed-frequency "
                   "forecasting check, not a Beta-weight MIDAS estimation.") % (venue, N)
        pA = "\\emph{Panel A: one-step-ahead $RV_{t+1}$ forecast (median across instruments)}"
        hA = "Model & $R^2$ & QLIKE & Beats HAR (sig.)"
        pB = "\\emph{Panel B: book-state $R^2$, contemporaneous vs one-step-ahead}"
        hB = "Predictor & Contemp.\\ ($\\log RV_t$) & One-step ($\\log RV_{t+1}$) &"
    tex = (
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{" + caption + "}\n"
        "\\label{tab:regular-clock}\n\\small\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\multicolumn{4}{@{}l}{" + pA + "} \\\\\n\\midrule\n"
        + hA + " \\\\\n\\midrule\n"
        + "HAR-RV & $%s$ & $%s$ & --- \\\\\n" % (f3(P["r2_HAR"]), f3(P["qlike_HAR"]))
        + "\\quad $+\\,\\bar{a}_{xx}$ & $%s$ & $%s$ & %s \\\\\n" % (f3(P["r2_HARaxx"]), f3(P["qlike_HARaxx"]), a_cnt)
        + "\\quad $+\\,\\bar{S}$ & $%s$ & $%s$ & %s \\\\\n\\midrule\n" % (f3(P["r2_HARspread"]), f3(P["qlike_HARspread"]), s_cnt)
        + "\\multicolumn{4}{@{}l}{" + pB + "} \\\\\n\\midrule\n"
        + hB + " \\\\\n\\midrule\n"
        + "$\\bar{a}_{xx}$ & $%s$ & $%s$ & \\\\\n" % (f3(P["r2_axx_contemp"]), f3(P["r2_axx_only"]))
        + "$\\bar{S}$ & $%s$ & $%s$ & \\\\\n" % (f3(P["r2_spread_contemp"]), f3(P["r2_spread_only"]))
        + "$\\bar{a}_{xx}+\\bar{S}$ & $%s$ & $%s$ & \\\\\n" % (f3(P["r2_both_contemp"]), f3(P["r2_both_onestep"]))
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (SECTIONS_DIR / "tab_regular_clock.tex").write_text(tex)
    print("wrote %s" % (SECTIONS_DIR / "tab_regular_clock.tex"))
    print("\n[midas] wrote %s/tables/midas_regular_clock_summary.csv and _pooled.csv" % DIAG_DIR)


if __name__ == "__main__":
    main()
