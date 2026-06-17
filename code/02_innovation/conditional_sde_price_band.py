#!/usr/bin/env python3
"""
conditional_sde_price_band.py — illustrative time series of the SDE's one-step-ahead predictive band.

For a held-out session of one instrument we draw, at every step, the one-step-ahead predictive interval
for the price implied by the SDE
    Delta x_t = b_x(I_t,S_t) + sqrt(a_xx(I_t,S_t)) * eta_t,   eta_t ~ GH (unit variance),
centred at x_{t-1}+b_x and split into two coloured layers:
  - INNER layer  = sqrt(a_xx) * z_alpha          (the a_xx scale, with a Gaussian shape)
  - OUTER layer  = sqrt(a_xx) * (q^GH_alpha - z_alpha)   (the EXTRA width the heavy tail adds)
so the band's width comes visibly from two sources: the state-dependent scale a_xx (it breathes with the
book state) and the GH heavy tail (it extends the band at high confidence). The realised price threads
through the band; large moves that would pierce the Gaussian (inner) band are caught by the GH (outer) band.
This is an ILLUSTRATION of a genuine ex-ante predictive interval (validated in aggregate by the calibration
figure), not a directional price forecast: the centre is ~ x_{t-1} because the drift is tiny.

Output: figures/fig_price_band.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUTF = CODE / "paper" / "figures"; OUTF.mkdir(parents=True, exist_ok=True)
N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50
BPS = 1e4
RNG = np.random.default_rng(0)
ASSET = None          # None -> pick the most-traded instrument


def parse_date(s):
    mm, dd, yy = s.split("-"); return (int(yy), int(mm), int(dd))


def frame(l1):
    mid = l1["mid"].to_numpy(float); bid = l1["bid"].to_numpy(float); ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float); asz = l1["ask_sz"].to_numpy(float)
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz)
    if good.sum() < 200:
        return None
    x = np.log(mid)
    return pd.DataFrame({"x": x, "I": (bsz - asz) / (bsz + asz), "S": (ask - bid) / mid}).replace(
        [np.inf, -np.inf], np.nan)


def with_next_return(d):
    d = d.copy()
    d["dx"] = d["x"].shift(-1) - d["x"]
    return d.dropna()


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
    sym = ASSET or max(files, key=lambda s: len(files[s]))
    paths = sorted(files[sym], key=lambda p: parse_date(p.parent.name))
    ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
    tr_p, te_p = paths[:ntr], paths[ntr:]
    print(f"asset {sym}: {len(tr_p)} train / {len(te_p)} test sessions")

    # train surface + GH shape
    tr = pd.concat([with_next_return(frame(pd.read_parquet(p))) for p in tr_p],
                   ignore_index=True).dropna()
    s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
    ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
    dxtr = tr["dx"].to_numpy()
    bx = {}; axx = {}
    for c in range(N_I * M_S):
        m = ctr == c
        if m.sum() >= MIN_CELL:
            bx[c] = dxtr[m].mean(); axx[c] = (dxtr[m] ** 2).mean()
    # GH (symmetric, unit variance) fit on train standardised residuals
    z = np.array([(dxtr[i] - bx[c]) / np.sqrt(axx[c]) for i, c in enumerate(ctr) if c in axx])
    zz = z if len(z) <= 120_000 else RNG.choice(z, 120_000, replace=False)
    p_, a_, b_, loc_, sc_ = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    gh = stats.genhyperbolic(p_, a_, 0.0, 0.0, sc_)
    sd = np.sqrt(gh.var())
    qGH = lambda al: float(gh.ppf(1 - al / 2) / sd)        # unit-variance GH upper quantile
    print(f"GH p={p_:.2f} a={a_:.2f}; q_GH(99%)={qGH(0.01):.2f} vs Gaussian {stats.norm.ppf(0.995):.2f}")

    # choose a representative volatile test session: the largest single move that is still a realistic
    # intraday event (<= 120 bps), avoiding momentary bad-quote glitches that dominate the axis.
    CAP = 120 / BPS
    best = None
    for pth in te_p:
        d = with_next_return(frame(pd.read_parquet(pth))).dropna()
        if len(d) < 800:
            continue
        mx = float(np.abs(d.dx).max())
        if mx <= CAP and (best is None or mx > best[0]):
            best = (mx, pth, d)
    _, pth, d = best
    print(f"chosen session {pth.parent.name}  n={len(d)}  max|dx|={best[0]*BPS:.1f} bps")

    x = d.x.to_numpy(); cc = cell_ids(d.I.to_numpy(), d.S.to_numpy(), s_edges)
    n = len(x)
    z99 = stats.norm.ppf(0.995); qgh99 = qGH(0.01)
    P = np.exp(x)                                           # REAL mid-price
    cL = np.full(n, np.nan); hG = np.full(n, np.nan); hGH = np.full(n, np.nan)
    for t in range(n - 1):
        c = cc[t]
        if c not in axx:
            continue
        sig = np.sqrt(axx[c])                               # one-step scale (log units)
        cL[t + 1] = x[t] + bx[c]                            # predicted log-price for next step
        hG[t + 1] = sig * z99; hGH[t + 1] = sig * qgh99
    iU = np.exp(cL + hG); iL = np.exp(cL - hG)              # inner band (a_xx, Gaussian) in price units
    oU = np.exp(cL + hGH); oL = np.exp(cL - hGH)            # outer band (GH heavy tail) in price units
    tt = np.arange(n)
    dxb = np.diff(x)
    pierce = [t + 1 for t in range(n - 1)
              if cc[t] in axx and abs(dxb[t]) > np.sqrt(axx[cc[t]]) * z99]
    tstar = int(np.argmax(np.abs(dxb))) + 1
    W = 300; w0, w1 = max(1, tstar - W), min(n, tstar + W)

    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, WARN, POS, MUTED
    plt = setup_mpl()
    m = np.isfinite(cL)

    def draw(ax, marklab=False):
        ax.fill_between(tt[m], iU[m], oU[m], color=WARN, alpha=0.35, lw=0,
                        label=(r"GH heavy-tail extension (99\%)" if marklab else None))
        ax.fill_between(tt[m], oL[m], iL[m], color=WARN, alpha=0.35, lw=0)
        ax.fill_between(tt[m], iL[m], iU[m], color=ACCENT, alpha=0.32, lw=0,
                        label=(r"$a_{xx}(I,S)$ scale, Gaussian (99\%)" if marklab else None))
        ax.plot(tt[m], np.exp(cL[m]), color=ACCENT_DARK, lw=0.8, ls="--",
                label=(r"model one-step price estimate $x_t+b_x$" if marklab else None))
        ax.plot(tt, P, color=INK, lw=0.8, label=(r"realised mid-price" if marklab else None))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8.6))
    # (a) full session
    draw(ax0, marklab=True)
    ax0.axvspan(w0, w1, color="0.45", alpha=0.12, lw=0)
    ax0.set_xlim(0, n); ax0.set_ylabel(r"mid-price")
    ax0.set_title(rf"(a) Full held-out session ({sym}, {pth.parent.name}); shaded region $=$ zoom in panel (b)")
    ax0.legend(loc="best", fontsize=8, ncol=2); despine(ax0)
    # (b) zoom
    draw(ax1)
    pin = [t for t in pierce if w0 <= t < w1]
    if pin:
        ax1.scatter(pin, P[pin], s=24, color=POS, zorder=5, edgecolors="white", linewidths=0.5,
                    label=r"move beyond the Gaussian band")
    lo = float(np.nanmin(np.concatenate([P[w0:w1], oL[w0:w1]])))
    hi = float(np.nanmax(np.concatenate([P[w0:w1], oU[w0:w1]])))
    pad = 0.12 * (hi - lo)
    ax1.set_xlim(w0, w1); ax1.set_ylim(lo - pad, hi + pad)
    ax1.set_xlabel(r"event index within the held-out session"); ax1.set_ylabel(r"mid-price")
    ax1.set_title(r"(b) Zoom: the band breathes with $a_{xx}(I,S)$ and is widened by the GH tail")
    ax1.legend(loc="upper left", fontsize=8); despine(ax1)
    fig.suptitle(r"One-step-ahead predictive interval for the price from the state-dependent SDE", y=1.0)
    finish(fig, OUTF / "fig_price_band.png")
    print("[band] wrote paper/figures/fig_price_band.png")

    # ---- companion: return-shape and state-conditional-variance comparison (full held-out set) ----
    te = pd.concat([with_next_return(frame(pd.read_parquet(p))) for p in te_p],
                   ignore_index=True).dropna()
    cteA = cell_ids(te.I.to_numpy(), te.S.to_numpy(), s_edges); dxteA = te.dx.to_numpy()
    inax = np.array([c in axx for c in cteA])
    cells_obs = cteA[inax]
    zt = (dxteA[inax] - np.array([bx[c] for c in cells_obs])) / np.sqrt(np.array([axx[c] for c in cells_obs]))
    ghpdf = lambda u: sd * gh.pdf(sd * u)                          # unit-variance GH density
    # realized per-state variance on held-out data; distribution of variance over the realized state path
    rcell = {c: (dxteA[cteA == c] ** 2).mean() for c in axx if (cteA == c).sum() >= MIN_CELL}
    msk = np.array([c in rcell for c in cells_obs])
    vm = np.array([axx[c] for c in cells_obs[msk]]) * BPS * BPS    # model conditional variance per obs (bps^2)
    vr = np.array([rcell[c] for c in cells_obs[msk]]) * BPS * BPS  # realized per-state variance per obs (bps^2)

    figc, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.7))
    grid = np.linspace(-8, 8, 300); bins = np.linspace(-8, 8, 120)
    axA.hist(zt[np.abs(zt) < 8], bins=bins, density=True, histtype="step", color=INK, lw=1.4,
             label=r"realised standardised returns")
    axA.plot(grid, ghpdf(grid), color=ACCENT_DARK, lw=1.6, label=r"model: GH innovation")
    axA.plot(grid, stats.norm.pdf(grid), color=WARN, lw=1.3, ls="--", label=r"Gaussian")
    axA.set_yscale("log"); axA.set_ylim(1e-5, 1)
    axA.set_xlabel(r"standardised one-step return $(\Delta x-b_x)/\sqrt{a_{xx}}$")
    axA.set_ylabel(r"density (log scale)"); axA.set_title(r"(a) Return distribution: model vs realised")
    axA.legend(loc="lower center", fontsize=8); despine(axA)

    lo = min(vm.min(), vr.min()); hi = max(vm.max(), vr.max())
    lb = np.logspace(np.log10(lo), np.log10(hi), 36)
    axB.hist(vr, bins=lb, density=True, histtype="step", color=INK, lw=1.5,
             label=r"realised variance (by state)")
    axB.hist(vm, bins=lb, density=True, histtype="stepfilled", color=ACCENT, alpha=0.45, lw=1.2,
             edgecolor=ACCENT_DARK, label=r"model variance $a_{xx}(I,S)$")
    axB.axvline(np.median(vr), color=INK, ls=":", lw=1)
    axB.axvline(np.median(vm), color=ACCENT_DARK, ls=":", lw=1)
    axB.set_xscale("log")
    axB.set_xlabel(r"conditional variance of the one-step move  [bps$^2$]")
    axB.set_ylabel(r"density")
    axB.set_title(r"(b) Variance distribution along the realised path: model vs realised")
    axB.legend(loc="upper right", fontsize=8); despine(axB)
    figc.suptitle(rf"Model vs realised on held-out data ({sym}): the distribution of one-step returns "
                  rf"and of the conditional variance", y=1.01)
    finish(figc, OUTF / "fig_model_vs_realized.png")
    print("[band] wrote paper/figures/fig_model_vs_realized.png")


if __name__ == "__main__":
    main()
