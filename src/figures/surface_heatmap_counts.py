#!/usr/bin/env python3
"""
surface_heatmap_counts.py - a heatmap-with-counts view of the 3-D surface.

A 3-D surface over a 30-cell grid with uneven occupancy can look smoother than
the data warrant; show the surface as a heatmap alongside the per-cell observation counts so the
reader sees how well each cell is populated. We pool across instruments (each normalised to its
own median a_xx so only the shape over (I,S) is shown) and annotate every cell with its total
observation count.

Outputs: ../manuscript/en/ssrn/figures/fig_axx_heatmap_counts.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_config, CLEAN_DIR, FIG_DIR
from common.grid import N_I, M_S, cell_ids
from common.plot_style import finish, setup_mpl, T

OUTF = FIG_DIR; OUTF.mkdir(parents=True, exist_ok=True)
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


def main():
    cfg = load_config()
    clean = CLEAN_DIR
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    surf = np.full((N_I, M_S), np.nan)
    cnt = np.zeros((N_I, M_S))
    prof = []                                   # per-instrument normalised surfaces for pooling
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        if tr is None or len(tr) < 3000:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        c = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        dx2 = tr["dx"].to_numpy() ** 2
        a = np.full((N_I, M_S), np.nan); n = np.zeros((N_I, M_S))
        for ib in range(N_I):
            for sb in range(M_S):
                m = c == ib * M_S + sb
                n[ib, sb] = m.sum()
                if m.sum() >= MIN_CELL:
                    a[ib, sb] = dx2[m].mean()
        cnt += n
        if np.isfinite(a).sum() >= 5:
            prof.append(a / np.nanmedian(a))    # normalise out instrument level
    surf = np.nanmedian(np.dstack(prof), axis=2)

    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    s_labels = [T("tight\n(S tercile 1)"), T("mid\n(S tercile 2)"), T("wide\n(S tercile 3)")]
    i_labels = [f"{e:+.1f}" for e in np.linspace(-0.9, 0.9, N_I)]

    a = ax[0]
    im = a.imshow(np.log10(surf), aspect="auto", origin="lower", cmap="viridis")
    a.set_xticks(range(M_S)); a.set_xticklabels(s_labels, fontsize=8)
    a.set_yticks(range(N_I)); a.set_yticklabels(i_labels, fontsize=8)
    a.set_ylabel(T(r"best-level imbalance $I$")); a.set_xlabel(T(r"relative spread $S$"))
    a.set_title("(a) " + T(r"Diffusion surface $\log_{10}\,a_{xx}/\mathrm{median}$ (pooled)"))
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    for ib in range(N_I):
        for sb in range(M_S):
            if np.isfinite(surf[ib, sb]):
                a.text(sb, ib, f"{surf[ib, sb]:.1f}", ha="center", va="center", color="w", fontsize=6.5)

    b = ax[1]
    im2 = b.imshow(np.log10(cnt + 1), aspect="auto", origin="lower", cmap="magma")
    b.set_xticks(range(M_S)); b.set_xticklabels(s_labels, fontsize=8)
    b.set_yticks(range(N_I)); b.set_yticklabels(i_labels, fontsize=8)
    b.set_xlabel(T(r"relative spread $S$"))
    b.set_title("(b) " + T(r"Total observation count per cell"))
    fig.colorbar(im2, ax=b, fraction=0.046, pad=0.04)
    for ib in range(N_I):
        for sb in range(M_S):
            v = int(cnt[ib, sb])
            b.text(sb, ib, f"{v//1000}k" if v >= 1000 else f"{v}", ha="center", va="center",
                   color="w" if cnt[ib, sb] < cnt.max() * 0.5 else "k", fontsize=6.5)

    fig.suptitle(T("The diffusion surface as a heatmap, with per-cell occupancy: well-populated cells "
                   "carry the transferable structure"), y=1.00)
    finish(fig, OUTF / "fig_axx_heatmap_counts.png")
    print("[cellcounts] wrote fig_axx_heatmap_counts.png")


if __name__ == "__main__":
    main()
