#!/usr/bin/env python3
"""
conditional_sde_prototype.py — does the state-dependent diffusion model reproduce the conditional move
distribution on QSE? (Part III of fokker_planck_plain_language.md.)

Model (per book-state cell c=(I-bin, S-tercile)):
    Delta x_t = b_x(c) + sqrt(a_xx(c)) * eps_t,   eps_t ~ N(0,1)
with b_x(c)=E[Delta x|c] (the Stoikov-style drift) and a_xx(c)=E[(Delta x)^2|c] (our diffusion). Both are
estimated on TRAIN sessions (first 60% per symbol) and evaluated on HELD-OUT test sessions, so nothing is
in-sample.

Tests on held-out data:
  (1) Variance calibration: per cell, does the train a_xx predict the realised test E[(Delta x)^2]? (slope ~1)
  (2) Standardised residual z=(Delta x - b_x(c))/sqrt(a_xx(c)): if the model captures the conditional
      distribution, z ~ N(0,1). std(z)~1 tests the SCALE (the a_xx working); kurtosis and P(|z|>k) test the
      SHAPE (are moves Gaussian once conditioned, or still fat-tailed?).
  (3) Homoskedasticity: does conditioning on the state make std(z) ~1 in EVERY cell, vs a state-independent
      (single global) standardisation that leaves heteroskedasticity?
  (4) Tail reproduction: simulate Delta x from the model and compare P(|Delta x|/sigma_local > k) to data.

Outputs: tables/sde_prototype_summary.csv ; figures/sde_prototype.png
Run: python3 conditional_sde_prototype.py
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50
RNG = np.random.default_rng(0)


def parse_date(s):
    mm, dd, yy = s.split("-"); return (int(yy), int(mm), int(dd))


def session_increments(l1):
    mid = l1["mid"].to_numpy(float); bid = l1["bid"].to_numpy(float); ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float); asz = l1["ask_sz"].to_numpy(float)
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz)
    if good.sum() < 200:
        return None
    x = np.log(mid)
    dx = np.diff(x)                                   # one-step increment, realised over (t, t+1)
    I = (bsz - asz) / (bsz + asz)
    S = (ask - bid) / mid
    # state known at t aligns with increment to t+1
    return pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1]}).replace([np.inf, -np.inf], np.nan).dropna()


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    Z, Z0, DXr, DXs, SIGloc = [], [], [], [], []      # pooled residuals / moves across symbols
    cal_rows = []                                     # per-(symbol,cell) variance calibration
    cellstd_dep, cellstd_ind, cellvol = [], [], []    # homoskedasticity by cell
    for sym, paths in files.items():
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        te = [session_increments(pd.read_parquet(p)) for p in paths[ntr:]]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        te = pd.concat([t for t in te if t is not None], ignore_index=True) if any(t is not None for t in te) else None
        if tr is None or te is None or len(tr) < 2000 or len(te) < 1000:
            continue
        s_edges = np.quantile(tr["S"], [1/3, 2/3])               # within-symbol spread terciles (train)
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy(); dxte = te["dx"].to_numpy()

        # train cell estimates: b_x (drift) and a_xx (diffusion = mean squared increment)
        bx, axx, ncell = {}, {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = dxtr[m].mean(); axx[c] = (dxtr[m] ** 2).mean(); ncell[c] = int(m.sum())
        # state-independent baseline (one global cell per symbol)
        b0 = dxtr.mean(); a0 = (dxtr ** 2).mean()

        keep = np.array([c in axx and axx[c] > 0 for c in cte])
        cte_k = cte[keep]; dxte_k = dxte[keep]
        if keep.sum() < 500:
            continue
        bx_v = np.array([bx[c] for c in cte_k]); axx_v = np.array([axx[c] for c in cte_k])
        z = (dxte_k - bx_v) / np.sqrt(axx_v)                      # state-dependent standardisation
        z0 = (dxte_k - b0) / np.sqrt(a0)                          # state-independent
        sigloc = np.sqrt(axx_v)
        dxsim = bx_v + np.sqrt(axx_v) * RNG.standard_normal(len(dxte_k))   # model simulation
        Z.append(z); Z0.append(z0); DXr.append(dxte_k); DXs.append(dxsim); SIGloc.append(sigloc)

        # per-cell variance calibration + homoskedasticity (test)
        for c in np.unique(cte_k):
            mc = cte_k == c
            if mc.sum() >= MIN_CELL:
                cal_rows.append({"sym": sym, "cell": int(c), "train_axx": axx[c],
                                 "test_msq": float((dxte_k[mc] ** 2).mean()), "n": int(mc.sum())})
                cellstd_dep.append(float(z[mc].std())); cellstd_ind.append(float(z0[mc].std()))
                cellvol.append(float(np.sqrt(axx[c])))

    z = np.concatenate(Z); z0 = np.concatenate(Z0)
    dxr = np.concatenate(DXr); dxs = np.concatenate(DXs); sigloc = np.concatenate(SIGloc)
    cal = pd.DataFrame(cal_rows)

    def kurt(a):
        a = a[np.isfinite(a)]; return float(((a - a.mean()) ** 4).mean() / a.var() ** 2)
    summ = {
        "n_test": len(z), "n_symbols_cells": len(cal),
        "z_std_state_dep": float(z.std()), "z_std_state_indep": float(z0.std()),
        "z_kurtosis_state_dep": kurt(z), "z_kurtosis_state_indep": kurt(z0),
        "frac_|z|>3_dep": float(np.mean(np.abs(z) > 3)), "frac_|z|>3_indep": float(np.mean(np.abs(z0) > 3)),
        "frac_|z|>5_dep": float(np.mean(np.abs(z) > 5)), "normal_ref_|z|>3": 0.0027, "normal_ref_|z|>5": 5.7e-7,
        "var_calib_spearman": float(pd.Series(cal["train_axx"]).corr(pd.Series(cal["test_msq"]), method="spearman")),
        "cellstd_dep_mean": float(np.mean(cellstd_dep)), "cellstd_dep_sd": float(np.std(cellstd_dep)),
        "cellstd_indep_sd": float(np.std(cellstd_ind)),
    }
    pd.DataFrame([summ]).to_csv(HERE / "output" / "tables" / "sde_prototype_summary.csv", index=False)

    print("=== Conditional SDE prototype on QSE (train 60% / test 40% per symbol) ===")
    for k, v in summ.items():
        print(f"  {k:22s}: {v:.4g}" if isinstance(v, float) else f"  {k:22s}: {v}")
    print("\nReading:")
    print(f"  - VARIANCE captured: std(z) state-dep = {summ['z_std_state_dep']:.3f} (target 1) vs "
          f"state-indep {summ['z_std_state_indep']:.3f}; per-cell std(z) = "
          f"{summ['cellstd_dep_mean']:.3f}+-{summ['cellstd_dep_sd']:.3f} (flat ~1 => homoskedastic).")
    print(f"  - SHAPE missed: kurtosis(z) = {summ['z_kurtosis_state_dep']:.1f} (Gaussian=3); "
          f"P(|z|>3) = {summ['frac_|z|>3_dep']:.4f} vs Gaussian 0.0027 => moves stay FAT-TAILED after "
          f"conditioning.")

    # ---- figure ----
    import sys
sys.path.insert(0, str(CODE))
from plot_style import ACCENT, ACCENT_DARK, WARN, MUTED, INK, POS, finish, setup_mpl, despine
    plt = setup_mpl()
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # A: standardised-residual density (log y) vs N(0,1)
    a = ax[0, 0]; grid = np.linspace(-8, 8, 200)
    a.hist(z[np.abs(z) < 8], bins=grid, density=True, histtype="step", color=ACCENT_DARK, lw=1.4,
           label=r"state-dependent $z$")
    a.hist(z0[np.abs(z0) < 8], bins=grid, density=True, histtype="step", color=MUTED, lw=1.1,
           label=r"state-independent")
    a.plot(grid, np.exp(-grid**2/2)/np.sqrt(2*np.pi), color=WARN, ls="--", lw=1.3, label=r"$N(0,1)$")
    a.set_yscale("log"); a.set_ylim(1e-5, 1)
    a.set_xlabel(r"standardised increment $z=(\Delta x-b_x)/\sqrt{a_{xx}}$")
    a.set_ylabel(r"density (log)"); a.set_title(r"Conditioning fixes the scale, not the tails")
    a.legend(loc="lower center"); despine(a)

    # B: variance calibration (train a_xx vs test mean square), per cell
    b = ax[0, 1]
    b.scatter(cal["train_axx"], cal["test_msq"], s=8, c=ACCENT, alpha=0.5, edgecolors="none")
    lim = [min(cal["train_axx"].min(), cal["test_msq"].min()), max(cal["train_axx"].max(), cal["test_msq"].max())]
    b.plot(lim, lim, color=INK, lw=1, ls="--"); b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"train $a_{xx}(c)$"); b.set_ylabel(r"test $\mathbb{E}[(\Delta x)^2\mid c]$")
    b.set_title(rf"Variance calibration (Spearman {summ['var_calib_spearman']:.2f})"); despine(b)

    # C: per-cell std(z) vs cell volatility — homoskedasticity
    c_ = ax[1, 0]
    order = np.argsort(cellvol)
    c_.scatter(np.array(cellvol)[order]*1e4, np.array(cellstd_ind)[order], s=8, c=MUTED, alpha=0.6,
               edgecolors="none", label=r"state-independent")
    c_.scatter(np.array(cellvol)[order]*1e4, np.array(cellstd_dep)[order], s=8, c=ACCENT_DARK, alpha=0.7,
               edgecolors="none", label=r"state-dependent")
    c_.axhline(1, color=WARN, ls="--", lw=1.2)
    c_.set_xscale("log"); c_.set_xlabel(r"cell volatility $\sqrt{a_{xx}}$ (bps)")
    c_.set_ylabel(r"$\mathrm{std}(z)$ within cell"); c_.set_title(r"State model removes heteroskedasticity")
    c_.legend(loc="best"); despine(c_)

    # D: tail reproduction — P(|dx|/sigma_local > k), data vs model
    d = ax[1, 1]
    ks = np.arange(1, 7)
    zr = np.abs(dxr / sigloc); zs = np.abs((dxs - 0) / sigloc)   # standardise both by local sigma
    pr = [np.mean(zr > k) for k in ks]; ps = [np.mean(zs > k) for k in ks]
    pn = [2*(1-0.5*(1+math.erf(k/np.sqrt(2)))) for k in ks]
    d.plot(ks, pr, "-o", color=ACCENT_DARK, ms=4, label=r"data")
    d.plot(ks, ps, "-s", color=POS, ms=4, label=r"Gaussian SDE (sim)")
    d.plot(ks, pn, "--", color=WARN, lw=1.2, label=r"$N(0,1)$")
    d.set_yscale("log"); d.set_xlabel(r"threshold $k$ (local std units)")
    d.set_ylabel(r"$P(|\Delta x|/\sigma_{\mathrm{loc}}>k)$"); d.set_title(r"Tails: data heavier than the model")
    d.legend(loc="best"); despine(d)

    fig.suptitle(r"Conditional state-dependent diffusion model vs.\ QSE data (held-out)", y=1.0)
    finish(fig, HERE / "output" / "figures" / "sde_prototype.png")
    print("\n[sde] wrote figures/sde_prototype.png and tables/sde_prototype_summary.csv")


if __name__ == "__main__":
    main()
