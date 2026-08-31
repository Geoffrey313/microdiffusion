#!/usr/bin/env python3
"""
descriptive_stats.py - pooled descriptive statistics for the QSE cross-section.

Produces the appendix census table required by the paper's descriptive-statistics
convention: for each modelled state variable it reports, pooled across all QSE
instruments, the number of observations, the mean, the standard deviation, the
minimum, the 25th, 50th and 75th percentiles, and the maximum.

Variables (all read from the exported event panel, event_panel.parquet):
  - relative spread  S_mid, reported in basis points (1e4 * S_mid)
  - best-level imbalance  I, dimensionless in [-1, 1]
  - absolute one-step log return  |dx_log|, reported in basis points

Serves table tab:descr-qse-full, the full-distribution counterpart to the per-asset
summary table tab:descr-qse. Every number the manuscript prints in that table is a
row of this script's output; nothing is entered by hand.

Output:
  results/descriptive_stats_qse.csv     one row per variable, the eight statistics
  manuscript/en/ssrn/sections/tab_descr_qse_full.tex  the LaTeX table \\input by the appendix
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PANEL, RESULTS_DIR, SECTIONS_DIR

SECTIONS = SECTIONS_DIR

VARIABLES = [
    ("Relative spread (bps)",      "S_mid",  lambda a: a * 1e4),
    ("Imbalance",                  "I",      lambda a: a),
    ("$|$one-step return$|$ (bps)", "dx_log", lambda a: np.abs(a) * 1e4),
]


def describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {"N": int(x.size), "mean": x.mean(), "sd": x.std(ddof=1),
            "min": x.min(), "p25": q25, "median": q50, "p75": q75, "max": x.max()}


def main():
    df = pd.read_parquet(PANEL, columns=["instrument", "I", "S_mid", "dx_log"])
    n_inst = df["instrument"].nunique()

    rows = []
    for name, col, transform in VARIABLES:
        s = describe(transform(df[col].to_numpy()))
        s["variable"] = name
        rows.append(s)
        print(f"{name:26s} N={s['N']:>10,}  mean={s['mean']:9.3f}  sd={s['sd']:9.3f}  "
              f"min={s['min']:9.3f}  p25={s['p25']:9.3f}  med={s['median']:9.3f}  "
              f"p75={s['p75']:9.3f}  max={s['max']:10.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)[["variable", "N", "mean", "sd", "min", "p25", "median", "p75", "max"]]
    out.to_csv(RESULTS_DIR / "descriptive_stats_qse.csv", index=False)

    # LaTeX fragment: imbalance to 3 dp, the bps variables to 2 dp
    def fmt(v, dp):
        return f"{v:,.{dp}f}".replace(",", "\\,")
    body = []
    for r in rows:
        dp = 3 if r["variable"] == "Imbalance" else 2
        cells = [r["variable"], f"{r['N']:,}".replace(",", "\\,")]
        cells += [fmt(r[k], dp) for k in ("mean", "sd", "min", "p25", "median", "p75", "max")]
        body.append(" & ".join(cells) + r" \\")
    tex = (
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Full pooled distribution of the modelled QSE state variables, across "
        f"all {n_inst} instruments in the panel. The relative spread and the absolute one-step "
        "log return are in basis points; the imbalance is dimensionless in $[-1,1]$. The count "
        "$N$ is the number of retained level-one updates; the absolute return has one fewer "
        "observation per session, the last update having no successor. No winsorisation is "
        "applied to this table.}\n"
        "\\label{tab:descr-qse-full}\n\\small\n"
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
        "Variable & $N$ & Mean & SD & Min & P25 & Median & P75 & Max \\\\\n\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    SECTIONS.mkdir(parents=True, exist_ok=True)
    (SECTIONS / "tab_descr_qse_full.tex").write_text(tex)
    print(f"\nwrote {RESULTS_DIR}/descriptive_stats_qse.csv and "
          f"{SECTIONS}/tab_descr_qse_full.tex")


if __name__ == "__main__":
    main()
