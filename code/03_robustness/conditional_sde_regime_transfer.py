#!/usr/bin/env python3
"""
conditional_sde_regime_transfer.py — cross-regime transfer of the a_xx(I,S) surface and GH tail law.

Regime key: calendar trading day.
  For each day, pool |Delta x| across all instruments that traded that day and compute the
  cross-instrument median |Delta x|. Sort days by this measure; Calm = bottom half, Stress = top half.

Three experiments:
  A. Train on calm days  -> test on stress days
  B. Train on stress days -> test on calm days
  C. Train on first 60% of days -> test on last 40%  (adjacent-block baseline)

For each experiment, measure:
  1. Surface transfer: Spearman rank correlation of per-cell a_xx means (train vs test cells)
  2. Tail transfer: mean per-observation OOS log-likelihood gain of GH over Gaussian

Output:
  - paper/paper/figures/fig_regime_transfer.png
  - stdout: summary table
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUTF = CODE / "paper" / "figures"
OUTF.mkdir(parents=True, exist_ok=True)

N_I, M_S = 10, 3
MIN_CELL = 50
MIN_TR = 2000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def parse_date(s):
    mm, dd, yy = s.split("-")
    return (int(yy), int(mm), int(dd))


def session_increments(path):
    l1 = pd.read_parquet(path)
    mid = l1["mid"].to_numpy(float)
    bid = l1["bid"].to_numpy(float)
    ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float)
    asz = l1["ask_sz"].to_numpy(float)
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz)
    if good.sum() < 200:
        return None
    x = np.log(mid)
    dx = np.diff(x)
    I = (bsz - asz) / (bsz + asz)
    S = (ask - bid) / mid
    df = pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1]}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    return df if len(df) >= 30 else None


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


# ---------------------------------------------------------------------------
# load all data indexed by (symbol, date_str)
# ---------------------------------------------------------------------------

def load_all(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    # files[(sym, date_str)] = path
    files: dict[tuple, Path] = {}
    for p in clean.glob("*/*_l1.parquet"):
        sym = p.name.replace("_l1.parquet", "")
        date_str = p.parent.name          # mm-dd-yy
        files[(sym, date_str)] = p
    return files


# ---------------------------------------------------------------------------
# regime split: calendar day key
# ---------------------------------------------------------------------------

def compute_regime(files: dict) -> tuple[list, list, float]:
    """
    For each calendar day, pool |dx| across all instruments and compute median.
    Returns (calm_dates, stress_dates, cutoff).
    """
    # group paths by date
    by_date: dict[str, list] = {}
    for (sym, date_str), path in files.items():
        by_date.setdefault(date_str, []).append(path)

    day_vol: dict[str, float] = {}
    for date_str, paths in by_date.items():
        dxs = []
        for p in paths:
            df = session_increments(p)
            if df is not None:
                dxs.append(np.abs(df["dx"].to_numpy()))
        if dxs:
            pooled = np.concatenate(dxs)
            day_vol[date_str] = float(np.median(pooled))

    dates_sorted = sorted(day_vol.keys(), key=lambda d: day_vol[d])
    n = len(dates_sorted)
    n_calm = n // 2
    calm_dates = set(dates_sorted[:n_calm])
    stress_dates = set(dates_sorted[n_calm:])
    cutoff = day_vol[dates_sorted[n_calm]]

    print(f"\nRegime split: {n} trading days total")
    print(f"  Calm:   {len(calm_dates)} days  (median |dx| <= {cutoff*1e4:.3f} bps)")
    print(f"  Stress: {len(stress_dates)} days  (median |dx| > {cutoff*1e4:.3f} bps)")
    stress_sorted = sorted(stress_dates, key=lambda d: parse_date(d))
    print(f"  Stress days (sorted): {stress_sorted[:10]} ... {stress_sorted[-5:]}")
    return sorted(calm_dates), sorted(stress_dates), cutoff


# ---------------------------------------------------------------------------
# build surface from a set of (sym, date_str) sessions
# ---------------------------------------------------------------------------

def build_surface(files, session_keys):
    """
    Build per-instrument a_xx, bx surfaces from the given sessions.
    Returns dict: sym -> (bx, axx, s_edges)
    """
    # group by sym
    by_sym: dict[str, list] = {}
    for (sym, date_str) in session_keys:
        if (sym, date_str) in files:
            by_sym.setdefault(sym, []).append((sym, date_str))

    surfaces = {}
    for sym, keys in by_sym.items():
        frames = [session_increments(files[(sym, d)]) for (sym, d) in keys]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        tr = pd.concat(frames, ignore_index=True)
        if len(tr) < MIN_TR:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy()
        bx, axx = {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = float(dxtr[m].mean())
                axx[c] = float((dxtr[m] ** 2).mean())
        if axx:
            surfaces[sym] = (bx, axx, s_edges)
    return surfaces


# ---------------------------------------------------------------------------
# measure surface transfer: rank correlation of cell-level axx across splits
# ---------------------------------------------------------------------------

def surface_rank_corr(surfaces_tr, surfaces_te, files, te_keys):
    """
    For each instrument present in both train and test surfaces, compute
    per-cell axx in test data; Spearman corr of train cell axx vs test cell axx.
    """
    by_sym_te: dict[str, list] = {}
    for (sym, date_str) in te_keys:
        if (sym, date_str) in files:
            by_sym_te.setdefault(sym, []).append((sym, date_str))

    tr_vals, te_vals = [], []
    for sym, keys in by_sym_te.items():
        if sym not in surfaces_tr:
            continue
        bx_tr, axx_tr, s_edges = surfaces_tr[sym]
        frames = [session_increments(files[(sym, d)]) for (sym, d) in keys]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        te = pd.concat(frames, ignore_index=True)
        if len(te) < 500:
            continue
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxte = te["dx"].to_numpy()
        # build test-side cell axx
        axx_te = {}
        for c in range(N_I * M_S):
            m = cte == c
            if m.sum() >= MIN_CELL:
                axx_te[c] = float((dxte[m] ** 2).mean())
        # only cells present in both
        common = set(axx_tr.keys()) & set(axx_te.keys())
        for c in common:
            tr_vals.append(axx_tr[c])
            te_vals.append(axx_te[c])

    if len(tr_vals) < 10:
        return np.nan
    rho = stats.spearmanr(tr_vals, te_vals).correlation
    return float(rho)


# ---------------------------------------------------------------------------
# measure tail transfer: OOS log-likelihood gain GH over Gaussian
# ---------------------------------------------------------------------------

def fit_gh(surfaces_tr, files, tr_keys):
    """Fit a pooled GH on train-split standardised residuals."""
    z_all = []
    by_sym: dict[str, list] = {}
    for (sym, date_str) in tr_keys:
        if (sym, date_str) in files:
            by_sym.setdefault(sym, []).append((sym, date_str))
    for sym, keys in by_sym.items():
        if sym not in surfaces_tr:
            continue
        bx, axx, s_edges = surfaces_tr[sym]
        frames = [session_increments(files[(sym, d)]) for (sym, d) in keys]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        tr = pd.concat(frames, ignore_index=True)
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy()
        for i, c in enumerate(ctr):
            if c in axx and axx[c] > 0:
                z_all.append((dxtr[i] - bx[c]) / np.sqrt(axx[c]))
    if len(z_all) < 500:
        return None
    z = np.array(z_all, dtype=float)
    z = z[np.isfinite(z)]
    zz = z if len(z) <= 150_000 else RNG.choice(z, 150_000, replace=False)
    try:
        p_, a_, _b, _loc, sc_ = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
        gh = stats.genhyperbolic(p_, a_, 0.0, 0.0, sc_)
        sd_gh = float(np.sqrt(gh.var()))
        return (p_, a_, sc_, sd_gh, gh)
    except Exception:
        return None


def oosllik_gain(surfaces_tr, gh_params, files, te_keys):
    """
    Mean per-observation OOS log-likelihood gain: log p_GH(z) - log p_Gauss(z)
    on test sessions using train surface and train-fitted GH.
    """
    if gh_params is None:
        return np.nan
    _p, _a, _sc, sd_gh, gh = gh_params
    gains = []
    by_sym: dict[str, list] = {}
    for (sym, date_str) in te_keys:
        if (sym, date_str) in files:
            by_sym.setdefault(sym, []).append((sym, date_str))
    for sym, keys in by_sym.items():
        if sym not in surfaces_tr:
            continue
        bx, axx, s_edges = surfaces_tr[sym]
        frames = [session_increments(files[(sym, d)]) for (sym, d) in keys]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        te = pd.concat(frames, ignore_index=True)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxte = te["dx"].to_numpy()
        for i, c in enumerate(cte):
            if c not in axx or axx[c] <= 0:
                continue
            sigma = np.sqrt(axx[c])
            z = (dxte[i] - bx[c]) / sigma
            if not np.isfinite(z):
                continue
            # unit-variance GH density: rescale by sd_gh
            ll_gh = float(gh.logpdf(z * sd_gh) + np.log(sd_gh) - np.log(sigma))
            ll_g  = float(stats.norm.logpdf(z) - np.log(sigma))
            gains.append(ll_gh - ll_g)
    return float(np.mean(gains)) if gains else np.nan


# ---------------------------------------------------------------------------
# run one experiment
# ---------------------------------------------------------------------------

def run_experiment(label, files, tr_keys, te_keys):
    print(f"\n--- Experiment {label} ---")
    print(f"  Train: {len(set(d for _, d in tr_keys))} days, "
          f"{len(set(s for s, _ in tr_keys))} instruments")
    print(f"  Test:  {len(set(d for _, d in te_keys))} days, "
          f"{len(set(s for s, _ in te_keys))} instruments")

    surfaces_tr = build_surface(files, tr_keys)
    print(f"  Surfaces built: {len(surfaces_tr)} instruments")

    rho = surface_rank_corr(surfaces_tr, surfaces_tr, files, te_keys)
    # re-run with te surfaces
    rho = surface_rank_corr(surfaces_tr, {}, files, te_keys)

    # compute rho properly: train cell axx vs test cell axx
    by_sym_te: dict[str, list] = {}
    for (sym, date_str) in te_keys:
        if (sym, date_str) in files:
            by_sym_te.setdefault(sym, []).append((sym, date_str))
    tr_v, te_v = [], []
    for sym, keys in by_sym_te.items():
        if sym not in surfaces_tr:
            continue
        bx_tr, axx_tr, s_edges = surfaces_tr[sym]
        frames = [session_increments(files[(sym, d)]) for (sym, d) in keys]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        te_df = pd.concat(frames, ignore_index=True)
        if len(te_df) < 500:
            continue
        cte = cell_ids(te_df["I"].to_numpy(), te_df["S"].to_numpy(), s_edges)
        dxte = te_df["dx"].to_numpy()
        axx_te_local = {}
        for c in range(N_I * M_S):
            m = cte == c
            if m.sum() >= MIN_CELL:
                axx_te_local[c] = float((dxte[m] ** 2).mean())
        common = set(axx_tr.keys()) & set(axx_te_local.keys())
        for c in common:
            tr_v.append(axx_tr[c])
            te_v.append(axx_te_local[c])
    rho = float(stats.spearmanr(tr_v, te_v).correlation) if len(tr_v) >= 10 else np.nan

    gh_params = fit_gh(surfaces_tr, files, tr_keys)
    gain = oosllik_gain(surfaces_tr, gh_params, files, te_keys)

    print(f"  Surface rank-corr (Spearman): {rho:.3f}  (n_cells={len(tr_v)})")
    print(f"  GH OOS log-lik gain vs Gaussian: {gain:+.4f} nats/obs")
    return {"label": label, "rho": rho, "gain": gain, "n_cells": len(tr_v)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cfg_path = CODE / "config.yaml"
    files = load_all(cfg_path)
    print(f"Loaded {len(files)} (sym, date) pairs across "
          f"{len(set(s for s,_ in files))} instruments and "
          f"{len(set(d for _,d in files))} trading days")

    calm_dates, stress_dates, cutoff = compute_regime(files)
    all_dates_sorted = sorted(set(d for _, d in files), key=parse_date)
    n_all = len(all_dates_sorted)
    n_adj_tr = int(round(0.60 * n_all))
    adj_tr_dates = set(all_dates_sorted[:n_adj_tr])
    adj_te_dates = set(all_dates_sorted[n_adj_tr:])

    calm_set  = set(calm_dates)
    stress_set = set(stress_dates)

    def keys_for_dates(date_set):
        return [(sym, d) for (sym, d) in files if d in date_set]

    # Experiment A: calm -> stress
    res_A = run_experiment("A (Calm→Stress)",
                           files,
                           keys_for_dates(calm_set),
                           keys_for_dates(stress_set))

    # Experiment B: stress -> calm
    res_B = run_experiment("B (Stress→Calm)",
                           files,
                           keys_for_dates(stress_set),
                           keys_for_dates(calm_set))

    # Experiment C: adjacent 60/40 baseline
    res_C = run_experiment("C (Adjacent baseline, first 60%→last 40%)",
                           files,
                           keys_for_dates(adj_tr_dates),
                           keys_for_dates(adj_te_dates))

    results = [res_A, res_B, res_C]

    print("\n=== Cross-Regime Transfer Summary ===")
    print(f"{'Experiment':40s}  {'Surface rho':>12s}  {'GH gain (nats)':>14s}")
    for r in results:
        print(f"  {r['label']:38s}  {r['rho']:>12.3f}  {r['gain']:>+14.4f}")

    # --- figure ---
    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, WARN, MUTED
    plt = setup_mpl()

    labels = [r"Calm$\to$Stress", r"Stress$\to$Calm", r"Adjacent\n(baseline)"]
    rhos  = [res_A["rho"],  res_B["rho"],  res_C["rho"]]
    gains = [res_A["gain"], res_B["gain"], res_C["gain"]]
    colors = [ACCENT, WARN, MUTED]
    x = np.arange(3)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel (a): surface rank correlation
    bars0 = ax0.bar(x, rhos, color=colors, width=0.55, edgecolor=INK, linewidth=0.6)
    ax0.axhline(res_C["rho"], color=INK, ls="--", lw=1.0,
                label=rf"Adjacent baseline $\rho={res_C['rho']:.2f}$")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel(r"Spearman $\rho$: train $a_{xx}$ vs test $a_{xx}$")
    ax0.set_title(r"(a) $a_{xx}(I,S)$ surface transfer across regimes")
    ax0.set_ylim(0, 1)
    for bar, v in zip(bars0, rhos):
        ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax0.legend(fontsize=7); despine(ax0)

    # Panel (b): GH OOS log-likelihood gain
    bars1 = ax1.bar(x, gains, color=colors, width=0.55, edgecolor=INK, linewidth=0.6)
    ax1.axhline(0, color=INK, ls="-", lw=0.7, alpha=0.4)
    ax1.axhline(res_C["gain"], color=INK, ls="--", lw=1.0,
                label=rf"Adjacent baseline gain$={res_C['gain']:+.3f}$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"Mean OOS log-likelihood gain (nats/obs), GH over Gaussian")
    ax1.set_title(r"(b) GH tail law: OOS gain across regimes")
    for bar, v in zip(bars1, gains):
        ypos = v + 0.0005 if v >= 0 else v - 0.0015
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{v:+.3f}", ha="center", va="bottom", fontsize=8)
    ax1.legend(fontsize=7); despine(ax1)

    fig.suptitle(
        r"Cross-regime transfer of $a_{xx}(I,S)$ surface and GH tail law "
        r"(calm/stress split by cross-instrument median $|\Delta x|$ per trading day)",
        y=1.02
    )
    finish(fig, OUTF / "fig_regime_transfer.png")
    print("\n[regime_transfer] wrote paper/figures/fig_regime_transfer.png")


if __name__ == "__main__":
    main()
