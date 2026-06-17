#!/usr/bin/env python3
"""
conditional_sde_var_backtest.py — Kupiec/Christoffersen VaR backtest for the state-dependent SDE.

For each instrument, train the a_xx(I,S) surface and GH innovation on the first TRAIN_FRAC of
sessions, then evaluate on held-out sessions:

  Predictive interval at level (1-alpha):
      [b_x(c_n) - sigma_n * q_{1-alpha/2},  b_x(c_n) + sigma_n * q_{1-alpha/2}]

  where sigma_n = sqrt(a_xx(c_n)) and q_{1-alpha/2} is the upper (1-alpha/2) quantile of the
  fitted symmetric GH innovation (unit-variance rescaled).

  Violation: I_n = 1{|Delta x_n - b_x(c_n)| > sigma_n * q_{1-alpha/2}}

Statistical tests (implemented from scratch):
  - Kupiec LR_uc: unconditional coverage, chi2(1)
  - Christoffersen LR_ind: independence of violations, chi2(1)
    * Transition sequence RESETS at every session boundary to avoid cross-session contamination.
  - Joint LR_cc = LR_uc + LR_ind, chi2(2)

Outputs:
  - figures/fig_var_backtest.png  (two panels)
  - stdout: summary table per instrument x alpha
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
TRAIN_FRAC = 0.60
MIN_CELL = 50
ALPHAS = [0.01, 0.05]
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# data helpers (identical to sibling scripts)
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
    return df if len(df) >= 50 else None


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


# ---------------------------------------------------------------------------
# statistical tests
# ---------------------------------------------------------------------------

def lr_uc(n_viol, T, alpha):
    """Kupiec (1995) unconditional coverage LR, chi2(1)."""
    pi_hat = n_viol / T
    if pi_hat <= 0 or pi_hat >= 1:
        return np.nan, np.nan
    lr = -2.0 * (
        n_viol * np.log(alpha / pi_hat)
        + (T - n_viol) * np.log((1 - alpha) / (1 - pi_hat))
    )
    return float(lr), float(stats.chi2.sf(lr, df=1))


def lr_ind(viol_sequences):
    """Christoffersen (1998) independence LR, chi2(1).
    viol_sequences: list of 1-D int arrays, one per (instrument, session).
    Transitions are counted WITHIN each sequence only (no cross-sequence contamination).
    """
    n00 = n01 = n10 = n11 = 0
    for seq in viol_sequences:
        if len(seq) < 2:
            continue
        v0 = seq[:-1]
        v1 = seq[1:]
        n00 += int(((v0 == 0) & (v1 == 0)).sum())
        n01 += int(((v0 == 0) & (v1 == 1)).sum())
        n10 += int(((v0 == 1) & (v1 == 0)).sum())
        n11 += int(((v0 == 1) & (v1 == 1)).sum())
    n0 = n00 + n01
    n1 = n10 + n11
    if n0 == 0 or n1 == 0:
        return np.nan, np.nan
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi2  = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi01 <= 0 or pi01 >= 1 or pi11 <= 0 or pi11 >= 1 or pi2 <= 0 or pi2 >= 1:
        return np.nan, np.nan
    log_a = (n00 * np.log(1 - pi2) + n01 * np.log(pi2)
             + n10 * np.log(1 - pi2) + n11 * np.log(pi2))
    log_b = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
             + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    lr = -2.0 * (log_a - log_b)
    return float(lr), float(stats.chi2.sf(lr, df=1))


# ---------------------------------------------------------------------------
# per-instrument backtest
# ---------------------------------------------------------------------------

def backtest_instrument(sym, paths, alpha):
    paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
    ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
    tr_paths, te_paths = paths[:ntr], paths[ntr:]
    if not te_paths:
        return None

    # --- train surface ---
    tr_frames = [session_increments(p) for p in tr_paths]
    tr_frames = [f for f in tr_frames if f is not None]
    if not tr_frames:
        return None
    tr = pd.concat(tr_frames, ignore_index=True)
    if len(tr) < 2000:
        return None

    s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
    ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
    dxtr = tr["dx"].to_numpy()
    bx, axx = {}, {}
    for c in range(N_I * M_S):
        m = ctr == c
        if m.sum() >= MIN_CELL:
            bx[c] = float(dxtr[m].mean())
            axx[c] = float((dxtr[m] ** 2).mean())

    # --- GH fit on train standardised residuals ---
    z_tr = np.array(
        [(dxtr[i] - bx[c]) / np.sqrt(axx[c])
         for i, c in enumerate(ctr) if c in axx and axx[c] > 0],
        dtype=float
    )
    z_tr = z_tr[np.isfinite(z_tr)]
    if len(z_tr) < 500:
        return None
    zz = z_tr if len(z_tr) <= 150_000 else RNG.choice(z_tr, 150_000, replace=False)
    try:
        p_, a_, _b, _loc, sc_ = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
        gh = stats.genhyperbolic(p_, a_, 0.0, 0.0, sc_)
        sd_gh = float(np.sqrt(gh.var()))
        # unit-variance upper quantile at (1-alpha/2): two-sided threshold
        q_gh  = float(gh.ppf(1 - alpha / 2) / sd_gh)
        q_gauss = float(stats.norm.ppf(1 - alpha / 2))
    except Exception:
        return None

    # --- test sessions: collect violations per session (for Christoffersen reset) ---
    gh_seqs  = []
    gau_seqs = []
    T_total  = 0

    for p in te_paths:
        df = session_increments(p)
        if df is None or len(df) < 20:
            continue
        cte = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
        dx  = df["dx"].to_numpy()
        gh_viol_sess  = []
        gau_viol_sess = []
        for i, c in enumerate(cte):
            if c not in axx or axx[c] <= 0:
                continue
            sigma = np.sqrt(axx[c])
            mu    = bx[c]
            resid = abs(dx[i] - mu)
            gh_viol_sess.append(1 if resid > sigma * q_gh  else 0)
            gau_viol_sess.append(1 if resid > sigma * q_gauss else 0)
        if gh_viol_sess:
            gh_seqs.append(np.array(gh_viol_sess, dtype=int))
            gau_seqs.append(np.array(gau_viol_sess, dtype=int))
            T_total += len(gh_viol_sess)

    if T_total < 100:
        return None

    gh_all  = np.concatenate(gh_seqs)
    gau_all = np.concatenate(gau_seqs)

    def run_tests(viol_seqs, viol_all, al):
        n_v = int(viol_all.sum())
        T   = len(viol_all)
        lr_uc_stat, p_uc  = lr_uc(n_v, T, al)
        lr_ind_stat, p_ind = lr_ind(viol_seqs)
        return dict(T=T, violations=n_v, viol_rate=n_v / T,
                    LR_uc=lr_uc_stat, p_uc=p_uc,
                    LR_ind=lr_ind_stat, p_ind=p_ind)

    gh_res  = run_tests(gh_seqs,  gh_all,  alpha)
    gau_res = run_tests(gau_seqs, gau_all, alpha)
    return {"sym": sym, "alpha": alpha,
            "gh": gh_res, "gauss": gau_res}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files: dict[str, list] = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    results = []
    for sym, paths in sorted(files.items()):
        for alpha in ALPHAS:
            r = backtest_instrument(sym, paths, alpha)
            if r is not None:
                results.append(r)
                print(f"  {sym:8s}  alpha={alpha:.2f}  "
                      f"GH viol={r['gh']['viol_rate']:.4f}  p_uc={r['gh']['p_uc']:.3f}  |  "
                      f"Gauss viol={r['gauss']['viol_rate']:.4f}  p_uc={r['gauss']['p_uc']:.3f}")

    if not results:
        print("No results — check data path.")
        return

    # --- summary statistics ---
    print("\n=== VaR Backtest Summary ===")
    header = f"{'sym':8s}  {'alpha':5s}  {'T':>7s}  {'GH_viol':>8s}  {'GH_puc':>7s}  "
    header += f"{'GH_pind':>7s}  {'Ga_viol':>8s}  {'Ga_puc':>7s}  {'Ga_pind':>7s}"
    print(header)
    for r in results:
        g, ga = r["gh"], r["gauss"]
        print(f"{r['sym']:8s}  {r['alpha']:.2f}  {g['T']:>7d}  "
              f"{g['viol_rate']:>8.4f}  {g['p_uc']:>7.3f}  {g['p_ind']:>7.3f}  "
              f"{ga['viol_rate']:>8.4f}  {ga['p_uc']:>7.3f}  {ga['p_ind']:>7.3f}")

    for alpha in ALPHAS:
        sub = [r for r in results if r["alpha"] == alpha]
        gh_rates  = [r["gh"]["viol_rate"] for r in sub if np.isfinite(r["gh"]["viol_rate"])]
        gh_pucs   = [r["gh"]["p_uc"]      for r in sub if np.isfinite(r["gh"]["p_uc"])]
        gau_rates = [r["gauss"]["viol_rate"] for r in sub if np.isfinite(r["gauss"]["viol_rate"])]
        gau_pucs  = [r["gauss"]["p_uc"]    for r in sub if np.isfinite(r["gauss"]["p_uc"])]
        gh_pass   = sum(1 for p in gh_pucs  if p > 0.05)
        gau_pass  = sum(1 for p in gau_pucs if p > 0.05)
        print(f"\nalpha={alpha}: N={len(sub)} instruments")
        print(f"  GH   median viol={np.median(gh_rates):.4f}  (nominal={alpha:.4f})  "
              f"LR_uc pass (p>0.05): {gh_pass}/{len(gh_pucs)}")
        print(f"  Gauss median viol={np.median(gau_rates):.4f}  "
              f"LR_uc pass (p>0.05): {gau_pass}/{len(gau_pucs)}")

    # --- figure ---
    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, WARN, MUTED
    plt = setup_mpl()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel (a): violin of violation rates GH vs Gaussian, both alpha levels
    positions = [1, 2, 4, 5]
    colors    = [ACCENT, WARN, ACCENT, WARN]
    labels    = [r"GH, $\alpha{=}0.01$", r"Gauss, $\alpha{=}0.01$",
                 r"GH, $\alpha{=}0.05$", r"Gauss, $\alpha{=}0.05$"]
    data_viol = []
    nom_lines = []
    for alpha in ALPHAS:
        sub = [r for r in results if r["alpha"] == alpha]
        data_viol.append([r["gh"]["viol_rate"]    for r in sub if np.isfinite(r["gh"]["viol_rate"])])
        data_viol.append([r["gauss"]["viol_rate"]  for r in sub if np.isfinite(r["gauss"]["viol_rate"])])
        nom_lines.append(alpha)

    parts = ax0.violinplot(data_viol, positions=positions, showmedians=True, widths=0.7)
    for i, (body, col) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(col); body.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(INK); parts[key].set_linewidth(1.0)

    ax0.axhline(ALPHAS[0], color=INK,  ls="--", lw=1.0, label=r"nominal $\alpha{=}0.01$")
    ax0.axhline(ALPHAS[1], color=MUTED, ls=":",  lw=1.0, label=r"nominal $\alpha{=}0.05$")
    ax0.set_xticks(positions)
    ax0.set_xticklabels(labels, fontsize=7)
    ax0.set_ylabel(r"empirical violation rate $\hat{\pi}$")
    ax0.set_title(r"(a) Violation rates: GH vs Gaussian by $\alpha$")
    ax0.legend(fontsize=7); despine(ax0)

    # Panel (b): scatter LR_uc p-value GH vs Gaussian, per instrument, both alpha levels
    markers = ["o", "s"]
    for k, alpha in enumerate(ALPHAS):
        sub = [r for r in results if r["alpha"] == alpha
               and np.isfinite(r["gh"]["p_uc"]) and np.isfinite(r["gauss"]["p_uc"])]
        xv = [r["gauss"]["p_uc"] for r in sub]
        yv = [r["gh"]["p_uc"]    for r in sub]
        ax1.scatter(xv, yv, s=22, marker=markers[k],
                    color=ACCENT if k == 0 else ACCENT_DARK, alpha=0.75,
                    label=rf"$\alpha={alpha}$")

    lo, hi = 0.0, 1.0
    ax1.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    ax1.axhline(0.05, color=INK,  lw=0.8, ls=":", alpha=0.6)
    ax1.axvline(0.05, color=WARN, lw=0.8, ls=":", alpha=0.6)
    ax1.set_xlabel(r"Kupiec $p$-value: Gaussian VaR")
    ax1.set_ylabel(r"Kupiec $p$-value: GH VaR")
    ax1.set_title(r"(b) Kupiec LR$_{\mathrm{uc}}$ $p$-values, GH vs Gaussian")
    ax1.legend(fontsize=7); despine(ax1)

    fig.suptitle(
        r"Intraday VaR backtest: $|\Delta x_n - b_x(I_n,S_n)| > \sqrt{a_{xx}(I_n,S_n)}\,q^{\,}_{1-\alpha/2}$"
        r" — GH vs Gaussian innovation, Kupiec/Christoffersen, QSE held-out sessions",
        y=1.02
    )
    finish(fig, OUTF / "fig_var_backtest.png")
    print("\n[var_backtest] wrote paper/figures/fig_var_backtest.png")


if __name__ == "__main__":
    main()
