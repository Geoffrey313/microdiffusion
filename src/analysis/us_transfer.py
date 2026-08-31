#!/usr/bin/env python3
"""
us_transfer.py - external United States large-cap robustness check (Appendix F).

Repeats the diffusion-surface transfer test on the external United States equity
sample (Nasdaq TotalView-ITCH top of book; see data/us_panel.py). The principal
result is the walk-forward transfer: for each instrument the sample days are
walked forward (train on the W prior day-blocks, test on the next), the one-step
surface a_xx(I, S) is estimated on the training block, and its per-cell values are
correlated against the realised per-cell mean square of the increment on the
held-out block. Because W=4 keeps each fold's train and test days close, the
walk-forward is not dominated by the multi-year span of the free sample days. The
headline is the joint (I, S) rank correlation with an instrument-clustered
bootstrap interval; a shuffled-cell placebo provides the null.

The walk-forward is corroborated by transfer within a homogeneous era (first 60%
of an era's days train, last 40% test, for the 2019-2020 and 2025-2026 blocks) and
across volatility regimes (train on calm day-blocks, test on stress, and the
reverse). A single 60/40 split over the whole 2019-2026 span is reported only as a
transparency contrast: it is dominated by the multi-year gap and so weakens, which
reflects microstructure change over the span rather than a failure of the surface.

The one-step directional forecast decomposition (H5) is NOT evaluated on this
venue: at the one-cent tick the mega-cap mid is unchanged on most events, so the
one-step target is mechanically degenerate; H5 is assessed on the primary QSE
panel. The script also emits the per-asset descriptive statistics that feed the
appendix descriptive table.

Reported in Appendix F:
  - walk-forward joint (I, S) transfer rank correlation with its 95%
    instrument-clustered interval, and the shuffled-cell placebo near zero;
  - corroborating within-era and cross-regime surface transfer;
  - per-instrument distributions of the relative spread, imbalance, absolute
    one-step return, and best-level notional depth in thousand US dollars.

The United States panel is external to the repository (see common.paths.US_DIR
and data/README.md) and is rebuilt from the free Nasdaq sample by
data/us_itch_fetch.py. If it is not present the script prints a notice and exits
without error, so the reproduction chain still completes on the primary QSE
sample alone. Outputs are diagnostic and are not part of the digest gate, because
the external panel is not redistributed with the package.

Inputs : US_DIR/<SYM>_<YYYY-MM-DD>.parquet via data/us_panel.py.
Outputs: results/diagnostics/tables/us_transfer_summary.csv          (principal
             walk-forward transfer metrics),
         results/diagnostics/tables/us_transfer_corroboration.csv     (within-era,
             cross-regime, and the whole-span single-split transparency contrast),
         results/diagnostics/tables/us_transfer_by_symbol.csv,
         results/diagnostics/tables/us_descriptives.csv,
         paper/sections/desc_us.tex.
Serves : Appendix F (United States large-cap robustness).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DIAG_DIR, REPO_ROOT, ensure, load_config
from data.us_panel import PANEL, available, build_states, load_clean

W = 4                       # walk-forward training window over available day-blocks
L = 50                      # trailing realised-variance window (rows)
WINSOR = 0.995              # winsorise the squared increment at this train quantile
MIN_CELL = 20               # minimum count per cell (train and test) to enter the transfer
N_BOOT = 1000
RNG = np.random.default_rng(0)

PCTL = [0.25, 0.50, 0.75]


# --------------------------------------------------------------------------- #
#  state cells and increments
# --------------------------------------------------------------------------- #
def cell_ids(df, n_I, m_S):
    I = df["I"].to_numpy()
    ib = np.clip(((I + 1) / 2 * n_I).astype(int), 0, n_I - 1)
    sb = pd.to_numeric(pd.Series(df["S_tick"].to_numpy()), errors="coerce").fillna(1).astype(int)
    sb = np.clip(sb.to_numpy(), 1, m_S)
    return ib * m_S + (sb - 1)


def increments(df, n_I, m_S):
    """Return (state cell at t, squared increment to t+1) for one day-block."""
    df = df.sort_values("ts_event")
    x = df["x"].to_numpy(dtype=float)
    dx2 = np.diff(x) ** 2
    cell = cell_ids(df, n_I, m_S)[:-1]
    ok = np.isfinite(dx2)
    return cell[ok], dx2[ok]


def trailing(y):
    """Causal trailing mean of y over the last L rows (known at t)."""
    s = pd.Series(y).shift(1).rolling(L, min_periods=5).mean()
    return s.to_numpy()


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = np.sum(w * a), np.sum(w * b)
    va = np.sum(w * (a - ma) ** 2)
    vb = np.sum(w * (b - mb) ** 2)
    if va <= 0 or vb <= 0:
        return np.nan
    return float(np.sum(w * (a - ma) * (b - mb)) / np.sqrt(va * vb))


def boot_iid(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)
    bs = [RNG.choice(x, len(x), replace=True).mean() for _ in range(N_BOOT)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_cluster(df, col, by="symbol"):
    groups = [g[col].to_numpy() for _, g in df.groupby(by)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if len(groups) < 3:
        return (np.nan, np.nan, np.nan)
    G = len(groups)
    means = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, G, G)
        means.append(np.concatenate([groups[j] for j in idx]).mean())
    return (float(np.concatenate(groups).mean()),
            float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mse(y, f):
    m = np.isfinite(y) & np.isfinite(f)
    return float(np.mean((y[m] - f[m]) ** 2)) if m.any() else np.nan


def ols_predict(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta


# --------------------------------------------------------------------------- #
#  transfer test
# --------------------------------------------------------------------------- #
def run_transfer(states, n_I, m_S):
    sessions = sorted({s for _, s in states},
                      key=lambda x: tuple(int(v) for v in (x.split("-")[2], x.split("-")[0], x.split("-")[1])))
    symbols = sorted({y for y, _ in states})
    print(f"[us] {len(symbols)} instruments, {len(sessions)} day-blocks; walk-forward W={W}; "
          f"one-step a_xx(I,S); L={L}, winsor={WINSOR}")

    recs, skips = [], []
    for sym in symbols:
        ssl = [s for s in sessions if (sym, s) in states]
        if len(ssl) < W + 1:
            continue
        cache = {s: states[(sym, s)] for s in ssl}
        for i in range(W, len(ssl)):
            tr_s = ssl[i - W:i]
            te_s = ssl[i]
            try:
                c_tr, y_tr = [], []
                for s in tr_s:
                    c, y = increments(cache[s], n_I, m_S)
                    c_tr.append(c)
                    y_tr.append(y)
                c_tr = np.concatenate(c_tr)
                y_tr = np.concatenate(y_tr)
                if len(y_tr) < 200:
                    skips.append((sym, te_s, "few_train"))
                    continue
                cap = float(np.quantile(y_tr, WINSOR))
                y_trw = np.minimum(y_tr, cap)
                g = float(np.mean(y_trw))
                tdf = pd.DataFrame({"c": c_tr, "y": y_trw})
                a_cell = tdf.groupby("c")["y"].mean().to_dict()
                n_cell = tdf.groupby("c")["y"].size().to_dict()
                a_cell_sh = pd.DataFrame({"c": RNG.permutation(c_tr), "y": y_trw}).groupby("c")["y"].mean().to_dict()

                c_te, y_te = increments(cache[te_s], n_I, m_S)
                if len(y_te) < 100:
                    skips.append((sym, te_s, "few_test"))
                    continue
                y_tew = np.minimum(y_te, cap)
                f_glob = np.full(len(y_te), g)
                f_state = np.array([a_cell.get(int(c), g) for c in c_te])
                f_state_sh = np.array([a_cell_sh.get(int(c), g) for c in c_te])

                te_cell = pd.DataFrame({"c": c_te, "y": y_tew}).groupby("c")["y"].agg(["mean", "size"])
                common = [c for c in te_cell.index if (c in a_cell) and te_cell.loc[c, "size"] >= MIN_CELL
                          and n_cell.get(c, 0) >= MIN_CELL]

                def marg(cells, yv, axis):
                    idx = (cells // m_S) if axis == "I" else (cells % m_S)
                    d = pd.DataFrame({"k": idx, "y": yv})
                    return d.groupby("k")["y"].mean(), d.groupby("k")["y"].size()

                aI_tr, nI_tr = marg(c_tr, y_trw, "I")
                aS_tr, nS_tr = marg(c_tr, y_trw, "S")
                aI_te, nI_te = marg(c_te, y_tew, "I")
                aS_te, nS_te = marg(c_te, y_tew, "S")
                g_te = float(np.mean(y_tew))

                def sp(a, b):
                    return float(pd.Series(list(a)).corr(pd.Series(list(b)), method="spearman"))

                cI = [j for j in aI_te.index if j in aI_tr.index and nI_te[j] >= MIN_CELL and nI_tr.get(j, 0) >= MIN_CELL]
                corr_I = sp([aI_tr[j] for j in cI], [aI_te[j] for j in cI]) if len(cI) >= 4 else np.nan
                cS = [s for s in aS_te.index if s in aS_tr.index and nS_te[s] >= MIN_CELL and nS_tr.get(s, 0) >= MIN_CELL]
                corr_S = sp([aS_tr[s] for s in cS], [aS_te[s] for s in cS]) if len(cS) >= 3 else np.nan

                if len(common) >= 4:
                    atr = np.array([a_cell[c] for c in common])
                    ate = np.array([te_cell.loc[c, "mean"] for c in common])
                    wt = np.array([min(n_cell[c], te_cell.loc[c, "size"]) for c in common], dtype=float)
                    corr_p = wcorr(atr, ate, wt)
                    corr_s = float(pd.Series(atr).corr(pd.Series(ate), method="spearman"))
                    atr_sh = np.array([a_cell_sh.get(c, g) for c in common])
                    corr_p_sh = wcorr(atr_sh, ate, wt)
                    corr_s_sh = float(pd.Series(atr_sh).corr(pd.Series(ate), method="spearman"))
                    rtr = np.array([a_cell[c] - aI_tr.get(c // m_S, g) - aS_tr.get(c % m_S, g) + g for c in common])
                    rte = np.array([te_cell.loc[c, "mean"] - aI_te.get(c // m_S, g_te) - aS_te.get(c % m_S, g_te) + g_te
                                    for c in common])
                    corr_inter = float(pd.Series(rtr).corr(pd.Series(rte), method="spearman"))
                    n_common = len(common)
                else:
                    corr_p = corr_s = corr_p_sh = corr_s_sh = corr_inter = np.nan
                    n_common = len(common)

                sk_state = 1 - mse(y_tew, f_state) / mse(y_tew, f_glob)
                sk_state_sh = 1 - mse(y_tew, f_state_sh) / mse(y_tew, f_glob)

                tr_trail_list, tr_state_list, tr_y_list = [], [], []
                for s in tr_s:
                    c, y = increments(cache[s], n_I, m_S)
                    yw = np.minimum(y, cap)
                    tr_trail_list.append(trailing(yw))
                    tr_state_list.append(np.array([a_cell.get(int(cc), g) for cc in c]))
                    tr_y_list.append(yw)
                trail_tr = np.concatenate(tr_trail_list)
                st_tr = np.concatenate(tr_state_list)
                yy_tr = np.concatenate(tr_y_list)
                trail_te = trailing(y_tew)
                st_te = f_state
                mtr = np.isfinite(trail_tr) & np.isfinite(yy_tr)
                mte = np.isfinite(trail_te) & np.isfinite(y_tew)
                if mtr.sum() > 100 and mte.sum() > 50:
                    Xb_tr = np.column_stack([np.ones(mtr.sum()), trail_tr[mtr]])
                    Xa_tr = np.column_stack([np.ones(mtr.sum()), trail_tr[mtr], st_tr[mtr]])
                    Xb_te = np.column_stack([np.ones(mte.sum()), trail_te[mte]])
                    Xa_te = np.column_stack([np.ones(mte.sum()), trail_te[mte], st_te[mte]])
                    fb = ols_predict(Xb_tr, yy_tr[mtr], Xb_te)
                    fa = ols_predict(Xa_tr, yy_tr[mtr], Xa_te)
                    yv = y_tew[mte]
                    sk_marg = 1 - mse(yv, fa) / mse(yv, fb)
                else:
                    sk_marg = np.nan

                recs.append({"symbol": sym, "test": te_s, "n_common_cells": n_common,
                             "corr_pearson": corr_p, "corr_spearman": corr_s,
                             "corr_spearman_shuf": corr_s_sh,
                             "corr_I_only": corr_I, "corr_S_only": corr_S, "corr_interaction": corr_inter,
                             "skill_state": sk_state, "skill_state_shuf": sk_state_sh,
                             "skill_marginal": sk_marg})
            except Exception as e:  # noqa: BLE001 - a bad block is skipped, not fatal
                skips.append((sym, te_s, f"{type(e).__name__}: {e}"))
                continue

    df = pd.DataFrame(recs)
    if df.empty:
        sys.exit("[us] no folds produced")

    by_sym = df.groupby("symbol")[["n_common_cells", "corr_spearman", "corr_I_only", "corr_S_only",
                                   "corr_interaction", "skill_state", "skill_marginal"]].mean().reset_index()

    def row(col, label):
        m_i, lo_i, hi_i = boot_iid(df[col].to_numpy())
        m_c, lo_c, hi_c = boot_cluster(df, col)
        return {"metric": label, "mean": m_c, "iid_lo": lo_i, "iid_hi": hi_i,
                "clust_lo": lo_c, "clust_hi": hi_c, "frac_pos": float((df[col] > 0).mean()),
                "n": int(df[col].notna().sum())}

    summ = pd.DataFrame([
        row("corr_spearman", "transfer_JOINT_(I,S)"),
        row("corr_I_only", "transfer_I_only"),
        row("corr_S_only", "transfer_S_only"),
        row("corr_interaction", "transfer_INTERACTION_resid"),
        row("corr_spearman_shuf", "transfer_SHUFFLED"),
        row("skill_state", "skill_state_vs_global"),
        row("skill_state_shuf", "skill_state_SHUFFLED"),
        row("skill_marginal", "skill_marginal_vs_trailing"),
    ])
    summ["mean_common_cells"] = float(df["n_common_cells"].mean())
    return summ, by_sym, skips


# --------------------------------------------------------------------------- #
#  retained transfer designs: within a homogeneous era and across volatility
#  regimes. The freely available Nasdaq sample days are sparse and span materially
#  different market periods, so a single chronological walk-forward that pools the
#  whole 2019-2026 span conflates surface transfer with multi-year microstructure
#  change; it is kept only as a named secondary diagnostic. The principal designs
#  evaluate transfer where it is well posed: between comparable days.
# --------------------------------------------------------------------------- #
def _year(session: str) -> int:
    """Calendar year of a session key in MM-DD-YY form."""
    return 2000 + int(session.split("-")[2])


def _cell_axx(keys, states, n_I, m_S):
    """cell -> mean squared increment over the given (symbol, session) keys."""
    cs, ys = [], []
    for k in keys:
        c, y = increments(states[k], n_I, m_S)
        cs.append(c)
        ys.append(y)
    if not cs:
        return {}, {}
    c = np.concatenate(cs)
    y = np.concatenate(ys)
    d = pd.DataFrame({"c": c, "y": y})
    return d.groupby("c")["y"].mean().to_dict(), d.groupby("c")["y"].size().to_dict()


def surface_transfer(states, split_fn, n_I, m_S, min_cell=MIN_CELL):
    """Pooled per-(symbol, cell) transfer of the one-step surface a_xx(I, S).

    For every instrument the caller's split_fn returns (train_keys, test_keys).
    The surface is the per-cell mean squared increment on the training keys; it is
    correlated against the realised per-cell mean square on the held-out keys.
    Cells are pooled over instruments and the Spearman rank correlation is the
    transfer statistic, matching the primary diffusion-surface transfer test.
    """
    a_tr_all, a_te_all = [], []
    for sym in sorted({y for y, _ in states}):
        tr_keys, te_keys = split_fn(sym)
        if not tr_keys or not te_keys:
            continue
        a_tr, n_tr = _cell_axx(tr_keys, states, n_I, m_S)
        a_te, n_te = _cell_axx(te_keys, states, n_I, m_S)
        for c, v in a_tr.items():
            if c in a_te and n_tr[c] >= min_cell and n_te.get(c, 0) >= min_cell:
                a_tr_all.append(v)
                a_te_all.append(a_te[c])
    if len(a_tr_all) < 4:
        return np.nan, len(a_tr_all)
    rho = float(pd.Series(a_tr_all).corr(pd.Series(a_te_all), method="spearman"))
    return rho, len(a_tr_all)


def sessions_of(states, sym):
    """Chronologically sorted sessions available for one instrument."""
    ss = [s for (y, s) in states if y == sym]
    return sorted(ss, key=lambda s: (_year(s), int(s.split("-")[0]), int(s.split("-")[1])))


def intra_era_transfer(states, years, n_I, m_S):
    """Within-era transfer: per instrument, first 60% of the era's days train,
    last 40% test. Restricting to one era removes the multi-year gap."""
    def split(sym):
        ss = [s for s in sessions_of(states, sym) if _year(s) in years]
        if len(ss) < 2:
            return [], []
        ntr = max(1, int(round(0.60 * len(ss))))
        return [(sym, s) for s in ss[:ntr]], [(sym, s) for s in ss[ntr:]]
    return surface_transfer(states, split, n_I, m_S)


def crossera_single_split_transfer(states, n_I, m_S):
    """Single chronological 60/40 split over the whole 2019-2026 span, per
    instrument. Because the days are non-contiguous and span years, this trains on
    the earliest days and tests on the latest, so the estimate is dominated by the
    multi-year gap; it is reported only as a transparency contrast to the local and
    within-era designs, never as a headline."""
    def split(sym):
        ss = sessions_of(states, sym)
        if len(ss) < 2:
            return [], []
        ntr = max(1, int(round(0.60 * len(ss))))
        return [(sym, s) for s in ss[:ntr]], [(sym, s) for s in ss[ntr:]]
    return surface_transfer(states, split, n_I, m_S)


def regime_label(states):
    """Label each session calm or stress by its cross-instrument realised variance,
    the mean squared one-step increment, split at the pooled median so the two
    regimes are balanced. Realised variance is used rather than the median absolute
    increment, which is zero at the one-cent tick (the mid is unchanged on most
    events) and would leave the split degenerate."""
    rv = {}
    for s in sorted({ss for _, ss in states}):
        d2 = []
        for (y, ss) in states:
            if ss == s:
                x = states[(y, ss)].sort_values("ts_event")["x"].to_numpy(float)
                d2.append(np.diff(x) ** 2)
        if d2:
            rv[s] = float(np.mean(np.concatenate(d2)))
    if not rv:
        return {}, np.nan
    cut = float(np.median(list(rv.values())))
    return {s: ("stress" if v > cut else "calm") for s, v in rv.items()}, cut


def regime_transfer(states, reg, src, dst, n_I, m_S):
    """Cross-regime transfer: train the surface on every src-regime day, test on
    every dst-regime day, per instrument."""
    def split(sym):
        ss = sessions_of(states, sym)
        tr = [(sym, s) for s in ss if reg.get(s) == src]
        te = [(sym, s) for s in ss if reg.get(s) == dst]
        return tr, te
    return surface_transfer(states, split, n_I, m_S)


# --------------------------------------------------------------------------- #
#  per-asset descriptive statistics (Appendix F table)
# --------------------------------------------------------------------------- #
def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, min=np.nan, p25=np.nan, p50=np.nan, p75=np.nan, max=np.nan)
    q = np.quantile(x, PCTL)
    return dict(n=len(x), mean=x.mean(), sd=x.std(), min=x.min(),
                p25=q[0], p50=q[1], p75=q[2], max=x.max())


def descriptives(clean):
    """Per-asset distributions of spread, imbalance, return, and notional depth."""
    rows = []
    for sym in sorted({s for s, _ in clean}):
        keys = sorted([k for k in clean if k[0] == sym], key=lambda k: k[1])
        d = pd.concat([clean[k] for k in keys], ignore_index=True)
        mid = d["mid"].to_numpy(float)
        bid = d["bid"].to_numpy(float)
        ask = d["ask"].to_numpy(float)
        bsz = d["bid_sz"].to_numpy(float)
        asz = d["ask_sz"].to_numpy(float)
        spread_bps = 1e4 * (ask - bid) / mid
        imb = (bsz - asz) / (bsz + asz)
        # One-step return is computed within each day-block and then concatenated:
        # the day-blocks are separate sample days, so a diff across a block boundary
        # would be a spurious multi-month jump, not a move.
        ret_bps = np.concatenate([
            1e4 * np.abs(np.diff(np.log(clean[k]["mid"].to_numpy(float)),
                                 prepend=np.log(clean[k]["mid"].to_numpy(float))[0]))
            for k in keys])
        # Best-level depth as notional in thousand US dollars, comparable across
        # instruments whose share prices differ by an order of magnitude.
        depth_k = (bsz * bid + asz * ask) / 1000.0
        for var, x in [("Spread (bps)", spread_bps), ("Imbalance", imb),
                       ("|Return| (bps)", ret_bps), ("Depth ($k)", depth_k)]:
            r = describe(x)
            r.update(symbol=sym, variable=var, n_obs=len(d), n_sess=len(keys))
            rows.append(r)
    return pd.DataFrame(rows)[["symbol", "n_obs", "n_sess", "variable", "n", "mean", "sd",
                               "min", "p25", "p50", "p75", "max"]]


def _fmt(v, dp):
    return f"{v:.{dp}f}"


def write_desc_us(desc: pd.DataFrame):
    """Write the Appendix F descriptive table directly from the diagnostic CSV."""
    order = [s for s in PANEL if s in set(desc["symbol"])]
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Per-asset summary statistics for the external United States large-cap sample. For each asset the table reports the distribution of the relative spread (in basis points), the best-level imbalance, the absolute one-step log return (in basis points), and the best-level depth; the observation count for each asset is given under its name. Depth is the notional resting at the touch, in thousands of US dollars. The one-step return is computed within each day-block. The sample is the top of book reconstructed from the free Nasdaq TotalView-ITCH sample, recorded in event time over the regular session of the retained sample days.}",
        r"\label{tab:descr-us}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Asset & Variable & Mean & SD & Min & p25 & Median & p75 & Max \\",
        r"\midrule",
        r"\multicolumn{9}{@{}l}{\emph{United States large-caps (event time, retained Nasdaq sample days)}}\\",
        r"\midrule",
    ]
    variables = ["Spread (bps)", "Imbalance", "|Return| (bps)", "Depth ($k)"]
    labels = {
        "Spread (bps)": "Spread (bps)",
        "Imbalance": "Imbalance",
        "|Return| (bps)": r"$|$Return$|$ (bps)",
        "Depth ($k)": r"Depth (\$k)",
    }
    for i, sym in enumerate(order):
        block = desc[desc["symbol"] == sym].set_index("variable")
        n_obs = float(block["n_obs"].iloc[0]) / 1_000_000.0
        display_sym = sym
        for j, var in enumerate(variables):
            r = block.loc[var]
            dp = 3 if var in ("Imbalance", "Depth ($k)") else 2
            if var in ("Spread (bps)", "|Return| (bps)") and max(abs(r["mean"]), abs(r["max"])) < 1:
                dp = 3
            max_dp = 1 if abs(r["max"]) >= 100 else dp
            cells = [
                _fmt(r["mean"], dp), _fmt(r["sd"], dp), _fmt(r["min"], dp),
                _fmt(r["p25"], dp), _fmt(r["p50"], dp), _fmt(r["p75"], dp),
                _fmt(r["max"], max_dp),
            ]
            first = (r"\multirow{4}{*}{\shortstack[l]{%s\\[1pt] {\footnotesize %.2fM}}}"
                     % (display_sym, n_obs)) if j == 0 else ""
            lines.append(f"{first} & {labels[var]} & " + " & ".join(cells) + r" \\")
        if i != len(order) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    tex = "\n".join(lines)
    for path in [
        REPO_ROOT / "paper" / "sections" / "desc_us.tex",
        REPO_ROOT / "journal" / "JEF" / "sections" / "desc_us.tex",
    ]:
        if path.parent.exists():
            path.write_text(tex)
            print(f"[us] wrote {path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
def main():
    if not available():
        print("[us] external United States panel not found; skipping the "
              "Appendix F external check (see data/README.md).")
        return

    cfg = load_config()
    n_I = cfg["stoikov"]["n_imbalance_bins"]
    m_S = cfg["stoikov"]["n_spread_states"]

    clean = load_clean()
    states = build_states(clean)
    out = ensure(DIAG_DIR / "tables")

    # --- Principal result: the walk-forward surface transfer (Appendix F
    #     headline). W=4 keeps each fold's train and test days close, so it is not
    #     dominated by the multi-year span. ------------------------------------- #
    summ, by_sym, skips = run_transfer(states, n_I, m_S)
    summ.to_csv(out / "us_transfer_summary.csv", index=False)
    by_sym.to_csv(out / "us_transfer_by_symbol.csv", index=False)

    # --- Corroborating designs: transfer within a homogeneous era and across
    #     volatility regimes, plus a single whole-span 60/40 split reported only as
    #     a transparency contrast (it is dominated by the 2019-2026 gap). -------- #
    reg, cut = regime_label(states)
    n_calm = sum(v == "calm" for v in reg.values())
    n_stress = sum(v == "stress" for v in reg.values())
    designs = [
        ("intra_era_2019_2020", intra_era_transfer(states, {2019, 2020}, n_I, m_S)),
        ("intra_era_2025_2026", intra_era_transfer(states, {2025, 2026}, n_I, m_S)),
        ("regime_calm_to_stress", regime_transfer(states, reg, "calm", "stress", n_I, m_S)),
        ("regime_stress_to_calm", regime_transfer(states, reg, "stress", "calm", n_I, m_S)),
        ("crossera_single_split_2019_2026", crossera_single_split_transfer(states, n_I, m_S)),
    ]
    corrob = pd.DataFrame([{"design": lbl, "surface_rho": rho, "n_cells": npts}
                           for lbl, (rho, npts) in designs])
    corrob.to_csv(out / "us_transfer_corroboration.csv", index=False)

    desc = descriptives(clean)
    desc.to_csv(out / "us_descriptives.csv", index=False)
    write_desc_us(desc)

    print("\n=== external United States transfer of a_xx(I,S): principal walk-forward ===")
    print(summ.round(4).to_string(index=False))
    print("\ncorroborating designs (pooled surface rank correlation):")
    print(corrob.round(4).to_string(index=False))
    print(f"[us] regime split at median realised variance = {cut:.2e}: "
          f"{n_calm} calm, {n_stress} stress day-blocks")
    print("\n[us] H5 (one-step directional forecast) is NOT evaluated on this venue: "
          "at the one-cent tick the mega-cap mid is unchanged on most events "
          "(median absolute one-step return ~0 bps), so the one-step target is "
          "mechanically degenerate. H5 is assessed on the primary QSE panel, where "
          "the tick geometry yields a non-degenerate target.")
    n_obs = int(desc.groupby("symbol")["n_obs"].first().sum())
    print(f"\n[us] {desc['symbol'].nunique()} instruments, {n_obs:,} level-one observations")
    if skips:
        print(f"[us] skipped {len(skips)} folds; first few: {skips[:6]}")


if __name__ == "__main__":
    main()
