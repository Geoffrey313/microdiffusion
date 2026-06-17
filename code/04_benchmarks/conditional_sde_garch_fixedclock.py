#!/usr/bin/env python3
"""
conditional_sde_garch_fixedclock.py — robustness: repeat the GARCH-vs-book-state comparison on a
REGULAR wall-clock grid (fixed-interval bars), addressing the event-clock caveat.

For each instrument we aggregate to fixed BAR_SECONDS-second bars: the bar return is the sum of the
within-bar one-step log returns, and the bar's book-state variance forecast is the sum of the per-event
a_xx(I_t,S_t) over the events in the bar (the integrated conditional variance). We then fit GARCH(1,1)
on the regular bar-return series and compare, out of sample, the GARCH variance forecast, the
book-state forecast, and their log-linear combination -- exactly as in the event-clock test.

Outputs: tables/sde_garch_fixedclock.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50
BAR_SECONDS = 60
BPS = 1e4


def parse_date(s):
    mm, dd, yy = s.split("-"); return (int(yy), int(mm), int(dd))


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


def event_frame(l1):
    """Per-event dx, I, S, and bar index (floor of timestamp to BAR_SECONDS)."""
    mid = l1["mid"].to_numpy(float); bid = l1["bid"].to_numpy(float); ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float); asz = l1["ask_sz"].to_numpy(float)
    ts = pd.to_datetime(l1["ts"]).values.astype("datetime64[s]").astype("int64")  # epoch seconds (any input unit)
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz)
    if good.sum() < 200:
        return None
    x = np.log(mid); dx = np.diff(x)
    I = (bsz - asz) / (bsz + asz); S = (ask - bid) / mid
    bar = (ts // BAR_SECONDS)
    df = pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1], "bar": bar[:-1]})
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def bars_for(paths, s_edges, axx):
    """Aggregate a list of session files into bar (return, book-state variance) pairs, in order."""
    R, V = [], []
    for p in paths:
        df = event_frame(pd.read_parquet(p))
        if df is None or len(df) < 50:
            continue
        c = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
        a = np.array([axx[cc] if cc in axx else np.nan for cc in c])
        df = df.assign(axx=a)
        g = df.groupby("bar")
        r = g["dx"].sum()
        v = g["axx"].sum(min_count=1)          # sum of per-event a_xx; NaN if all-NaN bar
        ok = v.notna() & (v > 0)
        R.append(r[ok].to_numpy()); V.append(v[ok].to_numpy())
    if not R:
        return np.array([]), np.array([])
    return np.concatenate(R), np.concatenate(V)


def garch_filter(r2, w, a, b, s0):
    n = len(r2); s2 = np.empty(n); s2[0] = s0
    for t in range(1, n):
        s2[t] = w + a * r2[t - 1] + b * s2[t - 1]
    return s2


def fit_garch(r):
    r2 = r * r; v = float(np.var(r)) + 1e-9

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        s2 = np.maximum(garch_filter(r2, w, a, b, v), 1e-12)
        return 0.5 * np.sum(np.log(s2) + r2 / s2)

    best = None
    for init in [(0.1 * v, 0.05, 0.90), (0.3 * v, 0.10, 0.80)]:
        res = optimize.minimize(nll, init, method="Nelder-Mead",
                                options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-2})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x)


def gll(r, s2):
    s2 = np.maximum(s2, 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi) + np.log(s2) + r * r / s2)))


def fit_combo(r, u, v):
    def nll(c):
        s2 = np.minimum(np.maximum(np.exp(c[0] + c[1] * u + c[2] * v), 1e-12), 1e12)
        return 0.5 * np.sum(np.log(s2) + r * r / s2)
    return optimize.minimize(nll, [0.0, 0.5, 0.5], method="Nelder-Mead",
                             options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-3}).x


def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    rows = []
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr_p, te_p = paths[:ntr], paths[ntr:]
        # train a_xx surface (per event, from train)
        trev = [event_frame(pd.read_parquet(p)) for p in tr_p]
        trev = pd.concat([t for t in trev if t is not None], ignore_index=True) if any(t is not None for t in trev) else None
        if trev is None or len(trev) < 3000:
            continue
        s_edges = np.quantile(trev["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(trev["I"].to_numpy(), trev["S"].to_numpy(), s_edges)
        dxtr = trev["dx"].to_numpy()
        axx = {c: (dxtr[ctr == c] ** 2).mean() * BPS * BPS for c in range(N_I * M_S) if (ctr == c).sum() >= MIN_CELL}
        if not axx:
            continue
        r_tr, v_tr = bars_for(tr_p, s_edges, axx)
        r_te, v_te = bars_for(te_p, s_edges, axx)
        if len(r_tr) < 300 or len(r_te) < 150:
            continue
        r_tr = r_tr * BPS; r_te = r_te * BPS
        mu = r_tr.mean(); r_tr -= mu; r_te -= mu
        w, a, b = fit_garch(r_tr)
        r_all = np.concatenate([r_tr, r_te])
        s2_all = garch_filter(r_all * r_all, w, a, b, float(np.var(r_tr)) + 1e-9)
        s2g_tr = s2_all[:len(r_tr)]; s2g_te = s2_all[len(r_tr):]
        c0, c1, c2 = fit_combo(r_tr, np.log(np.maximum(s2g_tr, 1e-12)), np.log(v_tr))
        s2c_te = np.exp(c0 + c1 * np.log(np.maximum(s2g_te, 1e-12)) + c2 * np.log(v_te))
        rows.append({"sym": sym, "n_bars_te": len(r_te), "persist": a + b,
                     "ll_garch": gll(r_te, s2g_te), "ll_axx": gll(r_te, v_te),
                     "ll_combo": gll(r_te, s2c_te), "w_axx": c2})
        print(f"  {sym:6s} bars_te={len(r_te):5d} persist={a+b:.3f}  "
              f"LL g={rows[-1]['ll_garch']:.3f} a={rows[-1]['ll_axx']:.3f} c={rows[-1]['ll_combo']:.3f} w_axx={c2:+.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "output" / "tables" / "sde_garch_fixedclock.csv", index=False)
    n = len(df)

    def agg(x):
        x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); return m, m - 1.96 * se, m + 1.96 * se
    print(f"\n=== Fixed-clock ({BAR_SECONDS}s bars) GARCH comparison (QSE, {n} instruments) ===")
    print(f"mean persistence = {df.persist.mean():.3f}")
    for k in ["ll_garch", "ll_axx", "ll_combo"]:
        m, lo, hi = agg(df[k]); print(f"  {k:9s} = {m:.4f} [{lo:.4f},{hi:.4f}]")
    pw = int((df.w_axx > 0).sum()); pi = int((df.ll_combo > df.ll_garch).sum())
    print(f"  weight a_xx>0: {pw}/{n}  sign p={stats.binomtest(pw,n,0.5).pvalue:.2e}")
    print(f"  combo>GARCH:   {pi}/{n}  sign p={stats.binomtest(pi,n,0.5).pvalue:.2e}")
    print(f"  mean dLL(combo-garch) = {(df.ll_combo-df.ll_garch).mean():+.4f}")
    print("\n[fixedclock] wrote tables/sde_garch_fixedclock.csv")


if __name__ == "__main__":
    main()
