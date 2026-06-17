#!/usr/bin/env python3
"""
generate_paper1_figures.py — produce the two gap figures for Paper 1:
  (Fig 2) drift surface b_x(I,S) heat map + 2D-vs-3D out-of-sample comparison (Assumption 2 / H1);
  (Fig 3) diffusion surface a_xx(I,S) heat map + out-of-sample transfer scatter (H2, lead result).
Outputs go to ../paper/figures/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUT = CODE / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50


def parse_date(s):
    mm, dd, yy = s.split("-"); return (int(yy), int(mm), int(dd))


def session_increments(l1):
    mid = l1["mid"].to_numpy(float); bid = l1["bid"].to_numpy(float); ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float); asz = l1["ask_sz"].to_numpy(float)
    if (np.isfinite(mid) & (mid > 0)).sum() < 200:
        return None
    x = np.log(mid); dx = np.diff(x)
    I = (bsz - asz) / (bsz + asz); S = (ask - bid) / mid
    return pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1]}).replace([np.inf, -np.inf], np.nan).dropna()


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


def collect():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)
    # per (symbol, cell): train b_x, a_xx; test realised mean-square
    rows = []
    bx_grid = np.full((len(files), M_S, N_I), np.nan)     # symbol x spread x imbalance, drift (normalised)
    axx_grid = np.full((len(files), M_S, N_I), np.nan)    # ... diffusion (normalised)
    for si, (sym, paths) in enumerate(sorted(files.items())):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        te = [session_increments(pd.read_parquet(p)) for p in paths[ntr:]]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        te = pd.concat([t for t in te if t is not None], ignore_index=True) if any(t is not None for t in te) else None
        if tr is None or te is None or len(tr) < 2000 or len(te) < 1000:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy(); dxte = te["dx"].to_numpy()
        bx, axx = {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = dxtr[m].mean(); axx[c] = (dxtr[m] ** 2).mean()
        if not axx:
            continue
        med_axx = np.median(list(axx.values()))
        typ = np.sqrt(med_axx)                            # typical move size for this symbol
        for c, a in axx.items():
            ib, sb = c // M_S, c % M_S
            axx_grid[si, sb, ib] = a / med_axx            # relative diffusion (dimensionless)
            bx_grid[si, sb, ib] = bx[c] / typ             # drift in units of a typical move
        # transfer pairs (train a_xx vs test realised mean-square)
        for c in np.unique(cte):
            if c not in axx or axx[c] <= 0:
                continue
            mc = cte == c
            if mc.sum() >= MIN_CELL:
                rows.append({"a_xx_train": axx[c], "msq_test": float((dxte[mc] ** 2).mean())})
    return (np.nanmean(bx_grid, axis=0), np.nanmean(axx_grid, axis=0), pd.DataFrame(rows))


def main():
    print("collecting ...")
    bx_grid, axx_grid, tr = collect()
    sp = stats.spearmanr(tr["a_xx_train"], tr["msq_test"]).correlation
    print(f"pooled transfer Spearman = {sp:.3f}  (n cells = {len(tr)})")
    # 2D vs 3D microprice OOS (from table 07)
    o = pd.read_csv(HERE / "output" / "tables" / "07_microprice_oos_summary.csv").iloc[0]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, MUTED
    plt = setup_mpl()
    imb_mid = (np.arange(N_I) + 0.5) / N_I * 2 - 1
    Xg, Yg = np.meshgrid(imb_mid, np.arange(M_S))          # both (M_S, N_I) = (3, 10)

    # ---------- Fig 2: 3D drift surface + 2D-vs-3D out-of-sample comparison ----------
    fig = plt.figure(figsize=(13, 4.6))
    a = fig.add_subplot(1, 2, 1, projection="3d")
    vmax = float(np.nanmax(np.abs(bx_grid)))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    a.plot_surface(Xg, Yg, bx_grid, cmap="RdBu_r", norm=norm, rstride=1, cstride=1,
                   linewidth=0.3, edgecolor="0.35", antialiased=True, alpha=0.97)
    a.set_xlabel(r"imbalance $I$", labelpad=2); a.set_ylabel(r"spread $S$", labelpad=2)
    a.set_zlabel(r"$b_x/\sqrt{a_{xx}}$", labelpad=2)
    a.set_yticks([0, 1, 2]); a.set_yticklabels([r"tight", r"mid", r"wide"])
    a.view_init(elev=24, azim=-58)
    a.set_title(r"(a) Conditional drift surface $b_x(I,S)$", pad=0)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=a, shrink=0.55, pad=0.10,
                 label=r"$b_x/\sqrt{a_{xx}}$")
    b = fig.add_subplot(1, 2, 2)
    xb = np.arange(2)
    b.bar(xb, [o.R2_2D_mean, o.R2_3D_mean], width=0.5, color=[ACCENT, MUTED])
    b.set_xticks(xb); b.set_xticklabels([r"$(I,S)$ model", r"$(x,I,S)$ model"])
    b.set_ylabel(r"out-of-sample $R^2$")
    b.set_title(r"(b) Adding the level $x$ does not help")
    b.text(0.5, 0.92, rf"$\Delta R^2={o.dR2_mean:+.4f}$" "\n" r"direction $0.57\!\to\!0.56$",
           transform=b.transAxes, ha="center", va="top", fontsize=8,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=INK, lw=0.6))
    despine(b)
    finish(fig, OUT / "fig_drift_surface.png")

    # ---------- Fig 3: 3D diffusion surface + 2D transfer scatter ----------
    fig2 = plt.figure(figsize=(13, 4.7))
    a2 = fig2.add_subplot(1, 2, 1, projection="3d")
    Z = np.log10(axx_grid)
    norm2 = Normalize(vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)))
    a2.plot_surface(Xg, Yg, Z, cmap="viridis", norm=norm2, rstride=1, cstride=1,
                    linewidth=0.3, edgecolor="0.3", antialiased=True, alpha=0.97)
    a2.set_xlabel(r"imbalance $I$", labelpad=2); a2.set_ylabel(r"spread $S$", labelpad=2)
    a2.set_zlabel(r"$\log_{10}[a_{xx}/\mathrm{med}]$", labelpad=2)
    a2.set_yticks([0, 1, 2]); a2.set_yticklabels([r"tight", r"mid", r"wide"])
    a2.view_init(elev=24, azim=-58)
    a2.set_title(r"(a) Diffusion surface $a_{xx}(I,S)$", pad=0)
    fig2.colorbar(cm.ScalarMappable(norm=norm2, cmap="viridis"), ax=a2, shrink=0.55, pad=0.10,
                  label=r"$\log_{10}[a_{xx}/\mathrm{median}]$")
    b2 = fig2.add_subplot(1, 2, 2)
    b2.scatter(tr["a_xx_train"] * 1e8, tr["msq_test"] * 1e8, s=8, c=ACCENT, alpha=0.4, edgecolors="none")
    lim = [tr["a_xx_train"].min() * 1e8, tr["a_xx_train"].max() * 1e8]
    b2.plot(lim, lim, color=INK, ls="--", lw=1, label=r"$y=x$")
    b2.set_xscale("log"); b2.set_yscale("log")
    b2.set_xlabel(r"train $a_{xx}(c)$  [bps$^2$]")
    b2.set_ylabel(r"held-out $\mathbb{E}[(\Delta x)^2\,|\,c]$  [bps$^2$]")
    b2.set_title(rf"(b) Out-of-sample transfer (Spearman {sp:.2f})")
    b2.legend(loc="upper left")
    despine(b2)
    finish(fig2, OUT / "fig_axx_surface.png")
    print("wrote fig_drift_surface.png and fig_axx_surface.png to", OUT)


if __name__ == "__main__":
    main()
