#!/usr/bin/env python3
"""Two-panel GARCH-comparison figure for Paper 1 (reads the event-clock comparison CSV)."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUT = CODE / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(HERE / "output" / "tables" / "sde_garch_compare.csv")
n = len(d); above = int((d.ll_combo > d.ll_garch).sum()); pos_w = int((d.w_axx > 0).sum())

import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, MUTED
plt = setup_mpl()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

# (a) combined vs GARCH OOS log-likelihood, one point per instrument
a = ax[0]
lo = float(min(d.ll_garch.min(), d.ll_combo.min())) - 0.1
hi = float(max(d.ll_garch.max(), d.ll_combo.max())) + 0.1
a.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=1, label=r"$y=x$ (no improvement)")
a.scatter(d.ll_garch, d.ll_combo, s=26, c=ACCENT, alpha=0.8, edgecolors="white", linewidths=0.4)
a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
a.set_xlabel(r"GARCH out-of-sample log-likelihood / obs.")
a.set_ylabel(r"combined (GARCH $\times$ book state) / obs.")
a.set_title(rf"(a) Combined vs GARCH ({above}/{n} above the line)")
a.legend(loc="upper left"); despine(a)

# (b) fitted weight on a_xx, one per instrument
b = ax[1]
b.hist(d.w_axx, bins=12, color=MUTED, edgecolor="white", linewidth=0.6)
b.axvline(0, color=INK, ls="--", lw=1.2, label=r"$c_2=0$ (no weight on $a_{xx}$)")
b.axvline(d.w_axx.mean(), color=ACCENT, ls="-", lw=1.6, label=rf"mean $c_2={d.w_axx.mean():.2f}$")
b.set_xlabel(r"fitted combination weight on $a_{xx}$ ($c_2$)")
b.set_ylabel(r"number of instruments")
b.set_title(rf"(b) Weight on the book state ($ {pos_w}/{n}>0$)")
b.legend(loc="upper right", fontsize=8); despine(b)

finish(fig, OUT / "fig_garch_compare.png")
print(f"wrote fig_garch_compare.png  ({above}/{n} above line, {pos_w}/{n} positive weight)")
