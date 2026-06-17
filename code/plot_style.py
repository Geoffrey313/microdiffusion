"""Shared publication style for empirical figures — THE SINGLE SOURCE OF TRUTH.

All font sizes / LaTeX (mathtext) settings live here and are applied via rcParams in setup_mpl(). Plot scripts
MUST NOT pass per-call `fontsize=`/`labelsize=` overrides — they inherit these defaults so every figure matches
(the reference look is intraday_reynolds.png). If a one-off size is unavoidable, use the FS_* constants below so
it is still centrally defined. To restyle every figure, change the numbers here and regenerate.
"""
from __future__ import annotations

import os

INK = "#1f2933"
GRID = "#d8d8d8"
ZERO = "#2f2f2f"
ACCENT = "#2f5d8c"
ACCENT_DARK = "#1d3f5f"
ACCENT_LIGHT = "#9bb6cf"
NEG = "#6f6f6f"
POS = "#2f5d8c"
WARN = "#8a5a2b"
MUTED = "#9a9a9a"
LIGHT = "#e8e8e8"

# Canonical font sizes (the intraday_reynolds.png look). Single source of truth — referenced by rcParams below.
FS_TITLE = 9.5     # axes titles
FS_LABEL = 8       # x/y axis labels
FS_TICK = 7        # tick labels
FS_LEGEND = 8      # legends
FS_TEXT = 8        # in-axes text / annotations (base font.size)
FS_SUPTITLE = 11   # figure-level suptitle
FIG_W = 13         # canonical figure WIDTH (inches) — keep uniform so on-page text size matches across figures
                   # (point sizes alone aren't enough: a narrower figure scaled to column width enlarges text)


def setup_mpl():
    """Return pyplot after applying a restrained LaTeX-like style."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": FS_TEXT,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.frameon": False,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_SUPTITLE,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.65,
    })
    import matplotlib.pyplot as plt

    return plt


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path)
