#!/usr/bin/env python3
"""
us_appendix_figures.py - the two Appendix F figures for the United States sample.

Mirrors, on the event-time United States large-cap panel, the two QSE views used
to read the external check:

  fig_us_axx        (a) the pooled diffusion surface a_xx(I,S) as a heatmap
                        (each instrument normalised to its own median, so only
                        the shape over the imbalance-spread grid is shown), and
                    (b) the walk-forward transfer as a scatter of the per-cell
                        training a_xx against the held-out test a_xx, which is
                        the object behind the joint (I,S) rank correlation
                        reported in the appendix. Parallels Figure fig:axxheatmap.

  fig_us_heavytail  the pooled held-out standardised increment
                    z = (dx - b_x)/sqrt(a_xx): its density on a log scale and
                    its tail-exceedance curve, against a best-fit Gaussian, a
                    Student-t (ML) and a symmetric Generalized Hyperbolic (ML).
                    Parallels Figure fig:heavytail.

The cell geometry, walk-forward window and winsorisation match
analysis/us_transfer.py exactly, so the figures and the reported numbers are the
same estimator. The panel is external (see common.paths.US_DIR); if it is absent
the script prints a notice and exits without error, so the chain still completes.

Inputs : US_DIR/<SYM>_<YYYY-MM-DD>.parquet via data/us_panel.py.
Outputs: paper/figures/fig_us_axx.{png,pdf},
         paper/figures/fig_us_heavytail.{png,pdf}.
Serves : Appendix F (United States large-cap robustness).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIG_DIR, load_config
from common.plot_style import (ACCENT, ACCENT_DARK, ACCENT_LIGHT, WARN, MUTED, INK,
                               POS, finish, setup_mpl, despine)
from data.us_panel import available, build_states, load_clean
from analysis.us_transfer import W, WINSOR, MIN_CELL

RNG = np.random.default_rng(0)          # per-module stream (see repo dedup rule)
# One colour per instrument for the transfer scatter, within the restrained
# manuscript palette plus a few muted tones for the additional series. The colour
# only distinguishes instruments in the legend; it carries no other meaning.
SYM_COLOURS = {"AAPL": ACCENT, "MSFT": WARN, "AMZN": INK, "GOOGL": ACCENT_LIGHT,
               "FB": POS, "INTC": ACCENT_DARK, "CSCO": "#4f7a4f", "PEP": "#8a6d3b",
               "NVDA": "#7a4f7a", "CMCSA": "#4f6d8a"}


def _sorted_sessions(states):
    return sorted({s for _, s in states},
                  key=lambda x: tuple(int(v) for v in (x.split("-")[2], x.split("-")[0], x.split("-")[1])))


def block_cell_dx(state_df, n_I, m_S):
    """(cell at t, signed one-step increment to t+1) for one day-block, matching
    the packing cell = i_bin * m_S + (S_tick - 1) used by us_transfer.py."""
    df = state_df.sort_values("ts_event")
    x = df["x"].to_numpy(float)
    dx = np.diff(x)
    I = df["I"].to_numpy(float)[:-1]
    st = pd.to_numeric(pd.Series(df["S_tick"].to_numpy()), errors="coerce").fillna(1).astype(int).to_numpy()[:-1]
    ib = np.clip(((I + 1) / 2 * n_I).astype(int), 0, n_I - 1)
    sb = np.clip(st, 1, m_S)
    cell = ib * m_S + (sb - 1)
    ok = np.isfinite(dx)
    return cell[ok], dx[ok]


def pooled_surface(states, n_I, m_S):
    """Per-instrument per-cell mean square increment, normalised to the instrument
    median and pooled by cell median across instruments (the shape over (I,S))."""
    prof = []
    for sym in sorted({y for y, _ in states}):
        keys = [k for k in states if k[0] == sym]
        c_all, y_all = [], []
        for k in keys:
            c, dx = block_cell_dx(states[k], n_I, m_S)
            c_all.append(c); y_all.append(dx ** 2)
        c_all = np.concatenate(c_all); y_all = np.concatenate(y_all)
        a = np.full(n_I * m_S, np.nan)
        d = pd.DataFrame({"c": c_all, "y": y_all})
        for c, g in d.groupby("c"):
            if len(g) >= MIN_CELL:
                a[int(c)] = g["y"].mean()
        if np.isfinite(a).sum() >= 5:
            prof.append(a / np.nanmedian(a))
    surf = np.nanmedian(np.dstack([p.reshape(n_I, m_S) for p in prof]), axis=2)
    return surf


def transfer_points(states, n_I, m_S):
    """Walk-forward per-cell (train a_xx, test a_xx, symbol, fold), the transfer estimator.

    fold indexes one (instrument, held-out block) pair. The joint (I,S) rank
    correlation reported in the appendix is the within-fold Spearman of a_train
    against a_test, averaged over folds; carrying the fold label lets the figure
    render that same object rather than a pooled correlation that would mix the
    different diffusion scales of the instruments and read too high.
    """
    sessions = _sorted_sessions(states)
    rows = []
    fold = 0
    for sym in sorted({y for y, _ in states}):
        ssl = [s for s in sessions if (sym, s) in states]
        if len(ssl) < W + 1:
            continue
        for i in range(W, len(ssl)):
            c_tr, y_tr = [], []
            for s in ssl[i - W:i]:
                c, dx = block_cell_dx(states[(sym, s)], n_I, m_S)
                c_tr.append(c); y_tr.append(dx ** 2)
            c_tr = np.concatenate(c_tr); y_tr = np.concatenate(y_tr)
            if len(y_tr) < 200:
                continue
            cap = float(np.quantile(y_tr, WINSOR))
            tr = pd.DataFrame({"c": c_tr, "y": np.minimum(y_tr, cap)})
            a_tr = tr.groupby("c")["y"].mean(); n_tr = tr.groupby("c")["y"].size()
            c_te, dx_te = block_cell_dx(states[(sym, ssl[i])], n_I, m_S)
            te = pd.DataFrame({"c": c_te, "y": np.minimum(dx_te ** 2, cap)})
            a_te = te.groupby("c")["y"].mean(); n_te = te.groupby("c")["y"].size()
            for c in a_te.index:
                if c in a_tr.index and n_tr[c] >= MIN_CELL and n_te[c] >= MIN_CELL:
                    rows.append((a_tr[c], a_te[c], sym, fold))
            fold += 1
    return pd.DataFrame(rows, columns=["a_train", "a_test", "symbol", "fold"])


def held_out_residuals(states, n_I, m_S):
    """Pooled held-out standardised increments z = (dx - b_x)/sqrt(a_xx)."""
    sessions = _sorted_sessions(states)
    Z = []
    for sym in sorted({y for y, _ in states}):
        ssl = [s for s in sessions if (sym, s) in states]
        if len(ssl) < W + 1:
            continue
        for i in range(W, len(ssl)):
            c_tr, dx_tr = [], []
            for s in ssl[i - W:i]:
                c, dx = block_cell_dx(states[(sym, s)], n_I, m_S)
                c_tr.append(c); dx_tr.append(dx)
            c_tr = np.concatenate(c_tr); dx_tr = np.concatenate(dx_tr)
            if len(dx_tr) < 200:
                continue
            cap = float(np.quantile(dx_tr ** 2, WINSOR))
            df = pd.DataFrame({"c": c_tr, "dx": dx_tr, "y": np.minimum(dx_tr ** 2, cap)})
            bx = df.groupby("c")["dx"].mean(); axx = df.groupby("c")["y"].mean()
            n = df.groupby("c")["dx"].size()
            good = {int(c) for c in axx.index if n[c] >= MIN_CELL and axx[c] > 0}
            c_te, dx_te = block_cell_dx(states[(sym, ssl[i])], n_I, m_S)
            keep = np.array([c in good for c in c_te])
            if keep.sum() < 100:
                continue
            cc = c_te[keep]
            Z.append((dx_te[keep] - bx.loc[cc].to_numpy()) / np.sqrt(axx.loc[cc].to_numpy()))
    return np.concatenate(Z)


# --------------------------------------------------------------------------- #
#  figure 1: surface heatmap + transfer scatter
# --------------------------------------------------------------------------- #
def figure_surface(states, n_I, m_S, plt):
    surf = pooled_surface(states, n_I, m_S)
    pts = transfer_points(states, n_I, m_S).copy()

    # The appendix statistic: within-fold joint (I,S) Spearman, averaged over folds.
    # Rank correlation is invariant to the per-fold rescaling below, so the number
    # matches what the reader sees; the scatter and the caption report the same rho.
    per_fold = []
    for _, g in pts.groupby("fold"):
        if len(g) >= 4:
            per_fold.append(float(g["a_train"].corr(g["a_test"], method="spearman")))
    rho = float(np.nanmean(per_fold))
    # Rescale each fold to its own median training diffusion so the instruments share
    # a common scale in the scatter; the train/test ratio, and thus the y=x reading,
    # is preserved, and only the within-fold shape agreement is shown.
    scale = pts.groupby("fold")["a_train"].transform("median")
    pts["a_train_n"] = pts["a_train"] / scale
    pts["a_test_n"] = pts["a_test"] / scale

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    s_labels = [r"tight" + "\n" + r"$S_{(1)}$", r"mid" + "\n" + r"$S_{(2)}$", r"wide" + "\n" + r"$S_{(3)}$"]
    i_labels = [f"{e:+.1f}" for e in np.linspace(-0.9, 0.9, n_I)]

    a = ax[0]
    im = a.imshow(np.log10(surf), aspect="auto", origin="lower", cmap="viridis")
    a.set_xticks(range(m_S)); a.set_xticklabels(s_labels)
    a.set_yticks(range(n_I)); a.set_yticklabels(i_labels)
    a.set_ylabel(r"best-level imbalance $I$"); a.set_xlabel(r"relative spread $S$")
    a.set_title(r"(a) Diffusion surface $\log_{10}\,a_{xx}/\mathrm{median}$ (pooled)")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    for ib in range(n_I):
        for sb in range(m_S):
            if np.isfinite(surf[ib, sb]):
                a.text(sb, ib, f"{surf[ib, sb]:.1f}", ha="center", va="center", color="w", fontsize=6.5)

    b = ax[1]
    for sym, g in pts.groupby("symbol"):
        b.loglog(g["a_train_n"], g["a_test_n"], "o", ms=3.2, alpha=0.5,
                 color=SYM_COLOURS.get(sym, MUTED), label=sym)
    lo = float(np.nanmin(pts[["a_train_n", "a_test_n"]].to_numpy()))
    hi = float(np.nanmax(pts[["a_train_n", "a_test_n"]].to_numpy()))
    b.plot([lo, hi], [lo, hi], "-", color=INK, lw=1.0)
    b.set_xlabel(r"training cell diffusion $a_{xx}^{\mathrm{train}}$ (fold-normalised)")
    b.set_ylabel(r"held-out cell diffusion $a_{xx}^{\mathrm{test}}$ (fold-normalised)")
    b.set_title(r"(b) Walk-forward transfer per cell (fold-averaged $\rho=%.2f$)" % rho)
    b.legend(loc="upper left", ncol=2)
    despine(b)

    fig.suptitle(r"The diffusion surface on the United States large-cap panel and its walk-forward transfer", y=1.0)
    finish(fig, FIG_DIR / "fig_us_axx.png")
    return rho, len(pts)


# --------------------------------------------------------------------------- #
#  figure 2: heavy-tailed innovation
# --------------------------------------------------------------------------- #
def figure_heavytail(states, n_I, m_S, plt):
    z = held_out_residuals(states, n_I, m_S)
    z = z[np.isfinite(z)]
    sd = float(z.std())

    gauss = {"name": r"best-fit Gaussian", "ls": "--", "c": WARN,
             "pdf": lambda x: stats.norm.pdf(x / sd) / sd,
             "sf": lambda k: 2.0 * stats.norm.sf(k / sd)}
    nu, _, ts = stats.t.fit(z, floc=0.0)
    tlaw = {"name": r"Student-$t$ (ML)", "ls": "-", "c": ACCENT_DARK,
            "pdf": lambda x: stats.t.pdf(x / ts, nu) / ts,
            "sf": lambda k: 2.0 * stats.t.sf(k / ts, nu)}
    laws = [gauss, tlaw]
    try:
        zz = z if len(z) <= 200_000 else RNG.choice(z, 200_000, replace=False)
        p, al, be, loc, gs = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
        fr = stats.genhyperbolic(p, al, 0.0, 0.0, gs)
        laws.append({"name": r"Gen.\ Hyperbolic (ML)", "ls": "-", "c": POS,
                     "pdf": lambda x: fr.pdf(x), "sf": lambda k: fr.sf(k) + fr.cdf(-k)})
    except Exception as e:  # noqa: BLE001
        print("[us-fig] GH fit failed:", e)

    ks = np.arange(1, 8)
    emp = np.array([np.mean(np.abs(z) > k) for k in ks])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]; lo, hi = -8 * sd, 8 * sd
    binp = np.linspace(lo, hi, 160); gridp = np.linspace(lo, hi, 400)
    a.hist(z[(z > lo) & (z < hi)], bins=binp, density=True, histtype="step",
           color=MUTED, lw=1.5, label=r"data $z$")
    for L in laws:
        a.plot(gridp, L["pdf"](gridp), color=L["c"], ls=L["ls"], lw=1.4, label=L["name"])
    a.set_yscale("log"); a.set_ylim(1e-5, 2)
    a.set_xlabel(r"standardised increment $z=(\Delta x-b_x)/\sqrt{a_{xx}}$")
    a.set_ylabel(r"density (log scale)")
    a.set_title(r"(a) Shape of the kick: Gaussian vs heavy-tailed")
    a.legend(loc="lower center", ncol=2); despine(a); a.grid(False)

    d = ax[1]
    d.plot(ks, emp, "-o", color=MUTED, ms=4, lw=1.6, label=r"data")
    for L in laws:
        d.plot(ks, [L["sf"](k) for k in ks], ls=L["ls"], color=L["c"], lw=1.4,
               marker="s", ms=3, label=L["name"])
    d.set_yscale("log"); d.set_ylim(1e-7, 1)
    d.set_xlabel(r"threshold $k$ (in units of $\sqrt{a_{xx}}$)")
    d.set_ylabel(r"$P(|\,z\,| > k)$")
    d.set_title(r"(b) Tail exceedance on the held-out blocks")
    d.legend(loc="upper right"); despine(d); d.grid(False)

    fig.suptitle(r"Heavy-tailed innovation on the United States large-cap panel: standardised held-out "
                 r"increments vs fitted laws", y=1.02)
    finish(fig, FIG_DIR / "fig_us_heavytail.png")
    return len(z), float(nu)


def main():
    if not available():
        print("[us-fig] external United States panel not found; skipping the "
              "Appendix F figures (see data/README.md).")
        return
    cfg = load_config()
    n_I = cfg["stoikov"]["n_imbalance_bins"]
    m_S = cfg["stoikov"]["n_spread_states"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    states = build_states(load_clean())
    plt = setup_mpl()
    rho, npts = figure_surface(states, n_I, m_S, plt)
    nz, nu = figure_heavytail(states, n_I, m_S, plt)
    print(f"[us-fig] fig_us_axx: {npts} transferred cells, fold-averaged Spearman {rho:.3f}")
    print(f"[us-fig] fig_us_heavytail: {nz:,} held-out residuals, Student-t nu={nu:.1f}")


if __name__ == "__main__":
    main()
