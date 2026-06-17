#!/usr/bin/env python3
"""
conditional_sde_benchmarks.py — T4, stronger benchmarks demanded by the referee.

Two additions the referee correctly asked for:
  (A) JOINT GARCH-t: the variance recursion and the Student-t degrees of freedom are estimated
      JOINTLY by maximum likelihood per instrument -- not the two-step "fit variance, then fit a
      residual law" of the predictive-density grid. This is the literature-standard benchmark.
  (B) HAR-RV on fixed 60-second bars: the Corsi (2009) heterogeneous-autoregressive realized-variance
      model, the standard realized-volatility forecaster, which the grid omitted.

We compare, out of sample, the paper's models against these benchmarks on matched data:
  event clock : joint GARCH-t   vs  combined(GARCH x a_xx)+GH   vs  a_xx+GH   vs  GARCH+Gaussian
  60s bars    : HAR-RV(+Gauss / +GH)   vs  integrated a_xx(+Gauss / +GH)   vs  combined(+GH)
Metric: mean out-of-sample predictive log-likelihood per held-out observation (higher better).
The claim is complementarity, not domination.

Outputs: output/tables/sde_benchmarks_event.csv, output/tables/sde_benchmarks_bars.csv
Run: python3 conditional_sde_benchmarks.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUTT = HERE / "output" / "tables"; OUTT.mkdir(parents=True, exist_ok=True)
N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50
FIT_CAP = 12000
BAR_SECONDS = 60
BPS = 1e4
RNG = np.random.default_rng(0)


def parse_date(s):
    mm, dd, yy = s.split("-"); return (int(yy), int(mm), int(dd))


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


def event_frame(l1):
    mid = l1["mid"].to_numpy(float); bid = l1["bid"].to_numpy(float); ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float); asz = l1["ask_sz"].to_numpy(float)
    ts = pd.to_datetime(l1["ts"]).values.astype("datetime64[s]").astype("int64")
    if (np.isfinite(mid) & (mid > 0)).sum() < 200:
        return None
    x = np.log(mid); dx = np.diff(x)
    I = (bsz - asz) / (bsz + asz); S = (ask - bid) / mid
    bar = ts // BAR_SECONDS
    df = pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1], "bar": bar[:-1]})
    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ----------------------- GARCH machinery -----------------------
def garch_filter(r2, w, a, b, s0):
    n = len(r2); s2 = np.empty(n); s2[0] = s0
    for t in range(1, n):
        s2[t] = w + a * r2[t - 1] + b * s2[t - 1]
    return s2


def fit_garch_gauss(r):
    rr = r if len(r) <= FIT_CAP else r[-FIT_CAP:]
    r2 = rr * rr; v = float(np.var(rr)) + 1e-9

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


def std_t_logpdf(z, nu):
    """log density of a unit-variance Student-t (nu>2)."""
    c = np.sqrt(nu / (nu - 2.0))
    return stats.t.logpdf(z * c, nu) + np.log(c)


def fit_garch_t_joint(r):
    """JOINTLY estimate (w,a,b,nu) by Student-t maximum likelihood."""
    rr = r if len(r) <= FIT_CAP else r[-FIT_CAP:]
    r2 = rr * rr; v = float(np.var(rr)) + 1e-9

    def nll(p):
        w, a, b, nu = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999 or nu <= 2.05 or nu > 100:
            return 1e12
        s2 = np.maximum(garch_filter(r2, w, a, b, v), 1e-12)
        z = rr / np.sqrt(s2)
        return -np.sum(std_t_logpdf(z, nu) - 0.5 * np.log(s2))
    best = None
    for init in [(0.1 * v, 0.05, 0.90, 6.0), (0.3 * v, 0.10, 0.80, 4.0)]:
        res = optimize.minimize(nll, init, method="Nelder-Mead",
                                options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-2})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x)


def fit_combo(r, u, v):
    def nll(c):
        s2 = np.minimum(np.maximum(np.exp(c[0] + c[1] * u + c[2] * v), 1e-12), 1e12)
        return 0.5 * np.sum(np.log(s2) + r * r / s2)
    return optimize.minimize(nll, [0.0, 0.5, 0.5], method="Nelder-Mead",
                             options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-3}).x


def fit_gh(z):
    zz = z if len(z) <= 120_000 else RNG.choice(z, 120_000, replace=False)
    p, a, b, loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    return stats.genhyperbolic(p, a, 0.0, 0.0, s)


def pred_ll(r, s2, logpdf_z):
    """mean predictive log-lik: log f(r)= -0.5 log s2 + log q(r/sqrt s2)."""
    s2 = np.maximum(s2, 1e-12); z = r / np.sqrt(s2)
    return float(np.mean(-0.5 * np.log(s2) + logpdf_z(z)))


# ----------------------- (A) event-clock benchmarks -----------------------
def run_event(files):
    MODELS = ["garch", "axx", "combo"]
    per_sym = []
    ztr = {m: [] for m in MODELS}
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        trev = [event_frame(pd.read_parquet(p)) for p in paths[:ntr]]
        teev = [event_frame(pd.read_parquet(p)) for p in paths[ntr:]]
        trev = pd.concat([t for t in trev if t is not None], ignore_index=True) if any(t is not None for t in trev) else None
        teev = pd.concat([t for t in teev if t is not None], ignore_index=True) if any(t is not None for t in teev) else None
        if trev is None or teev is None or len(trev) < 3000 or len(teev) < 1000:
            continue
        s_edges = np.quantile(trev["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(trev["I"].to_numpy(), trev["S"].to_numpy(), s_edges)
        cte = cell_ids(teev["I"].to_numpy(), teev["S"].to_numpy(), s_edges)
        axx = {c: (trev["dx"].to_numpy()[ctr == c] ** 2).mean() * BPS * BPS
               for c in range(N_I * M_S) if (ctr == c).sum() >= MIN_CELL}
        if not axx:
            continue
        r_tr = trev["dx"].to_numpy() * BPS; r_te = teev["dx"].to_numpy() * BPS
        mu = r_tr.mean(); r_tr -= mu; r_te -= mu
        # Gaussian GARCH (two-step) and JOINT GARCH-t
        w, a, b = fit_garch_gauss(r_tr)
        wt, at, bt, nu = fit_garch_t_joint(r_tr)
        r_all = np.concatenate([r_tr, r_te])
        g_all = garch_filter(r_all * r_all, w, a, b, float(np.var(r_tr)) + 1e-9)
        gt_all = garch_filter(r_all * r_all, wt, at, bt, float(np.var(r_tr)) + 1e-9)
        g_tr, g_te = g_all[:len(r_tr)], g_all[len(r_tr):]
        gt_te = gt_all[len(r_tr):]
        ax_tr = np.array([axx.get(c, np.nan) for c in ctr])
        ax_te = np.array([axx.get(c, np.nan) for c in cte])
        mtr = np.isfinite(ax_tr) & (g_tr > 0); mte = np.isfinite(ax_te) & (g_te > 0)
        if mtr.sum() < 1000 or mte.sum() < 500:
            continue
        c0, c1, c2 = fit_combo(r_tr[mtr], np.log(g_tr[mtr]), np.log(ax_tr[mtr]))
        cb_tr = np.exp(c0 + c1 * np.log(g_tr[mtr]) + c2 * np.log(ax_tr[mtr]))
        cb_te = np.exp(c0 + c1 * np.log(g_te[mte]) + c2 * np.log(ax_te[mte]))
        s2 = {"garch": (g_tr[mtr], g_te[mte]), "axx": (ax_tr[mtr], ax_te[mte]), "combo": (cb_tr, cb_te)}
        rt = r_tr[mtr]; re = r_te[mte]
        for m in MODELS:
            ztr[m].append(rt / np.sqrt(s2[m][0]))
        per_sym.append({"sym": sym, "re": re, "s2": {m: s2[m][1] for m in MODELS},
                        "gt_te": gt_te[mte], "nu": nu, "persist_t": at + bt})
        print(f"  [event] {sym:6s} n_te={mte.sum():6d} GARCH-t nu={nu:.2f}")
    # innovation laws fit on train residuals
    gh = {m: fit_gh(np.concatenate(ztr[m])) for m in MODELS}
    sd = {m: float(np.std(np.concatenate(ztr[m]))) for m in MODELS}
    rows = []
    for d in per_sym:
        r = d["re"]; row = {"sym": d["sym"], "nu": d["nu"]}
        row["GARCH+Gaussian"] = pred_ll(r, d["s2"]["garch"], lambda z, s=sd["garch"]: stats.norm.logpdf(z, 0, s))
        row["jointGARCH-t"] = pred_ll(r, d["gt_te"], lambda z, nu=d["nu"]: std_t_logpdf(z, nu))
        row["axx+GH"] = pred_ll(r, d["s2"]["axx"], lambda z, g=gh["axx"]: g.logpdf(z))
        row["combined+GH"] = pred_ll(r, d["s2"]["combo"], lambda z, g=gh["combo"]: g.logpdf(z))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------- (B) HAR-RV on 60s bars -----------------------
def bars_for(paths, s_edges, axx):
    R, V, RV = [], [], []
    for p in paths:
        df = event_frame(pd.read_parquet(p))
        if df is None or len(df) < 50:
            continue
        c = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
        a = np.array([axx.get(cc, np.nan) for cc in c])
        df = df.assign(axx=a, dx2=(df["dx"].to_numpy() * BPS) ** 2)
        g = df.groupby("bar")
        r = g["dx"].sum() * BPS
        v = g["axx"].sum(min_count=1)
        rv = g["dx2"].sum()                      # realized variance per bar (bps^2)
        ok = v.notna() & (v > 0) & (rv > 0)
        R.append(r[ok].to_numpy()); V.append(v[ok].to_numpy()); RV.append(rv[ok].to_numpy())
    if not R:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(R), np.concatenate(V), np.concatenate(RV)


def har_design(logrv):
    """HAR lags: previous bar, mean of last 5, mean of last 22 (intraday heterogeneous)."""
    n = len(logrv)
    x1 = np.full(n, np.nan); x5 = np.full(n, np.nan); x22 = np.full(n, np.nan)
    for t in range(n):
        if t >= 1:
            x1[t] = logrv[t - 1]
        if t >= 5:
            x5[t] = logrv[t - 5:t].mean()
        if t >= 22:
            x22[t] = logrv[t - 22:t].mean()
    return x1, x5, x22


def run_bars(files):
    rows = []
    ztr_har, ztr_axx, ztr_combo = [], [], []
    keep = []
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr_p, te_p = paths[:ntr], paths[ntr:]
        trev = [event_frame(pd.read_parquet(p)) for p in tr_p]
        trev = pd.concat([t for t in trev if t is not None], ignore_index=True) if any(t is not None for t in trev) else None
        if trev is None or len(trev) < 3000:
            continue
        s_edges = np.quantile(trev["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(trev["I"].to_numpy(), trev["S"].to_numpy(), s_edges)
        axx = {c: (trev["dx"].to_numpy()[ctr == c] ** 2).mean() * BPS * BPS
               for c in range(N_I * M_S) if (ctr == c).sum() >= MIN_CELL}
        if not axx:
            continue
        r_tr, v_tr, rv_tr = bars_for(tr_p, s_edges, axx)
        r_te, v_te, rv_te = bars_for(te_p, s_edges, axx)
        if len(r_tr) < 300 or len(r_te) < 150:
            continue
        mu = r_tr.mean(); r_tr = r_tr - mu; r_te = r_te - mu
        # HAR-RV fit on train log RV
        lrv_tr = np.log(rv_tr)
        x1, x5, x22 = har_design(lrv_tr)
        ok = np.isfinite(x1) & np.isfinite(x5) & np.isfinite(x22)
        if ok.sum() < 100:
            continue
        X = np.column_stack([np.ones(ok.sum()), x1[ok], x5[ok], x22[ok]])
        beta, *_ = np.linalg.lstsq(X, lrv_tr[ok], rcond=None)
        resid = lrv_tr[ok] - X @ beta
        jensen = 0.5 * np.var(resid)             # E[RV]=exp(mu+0.5 var); correct the variance forecast
        # forecast on test bars: build lags from the concatenated series (train tail + test)
        lrv_all = np.concatenate([lrv_tr, np.log(rv_te)])
        x1a, x5a, x22a = har_design(lrv_all)
        idx_te = np.arange(len(lrv_tr), len(lrv_all))
        Xte = np.column_stack([np.ones(len(idx_te)), x1a[idx_te], x5a[idx_te], x22a[idx_te]])
        okte = np.all(np.isfinite(Xte), axis=1)
        if okte.sum() < 100:
            continue
        har_te = np.exp(Xte[okte] @ beta + jensen)      # variance forecast (bps^2)
        rte = r_te[okte]; vte = v_te[okte]
        # combined HAR x a_xx (log-linear) trained on train bars
        # build train forecasts aligned to ok
        har_tr = np.exp(X @ beta + jensen)
        rtr = r_tr[ok]; vtr = v_tr[ok]
        c0, c1, c2 = fit_combo(rtr, np.log(np.maximum(har_tr, 1e-12)), np.log(vtr))
        cb_te = np.exp(c0 + c1 * np.log(np.maximum(har_te, 1e-12)) + c2 * np.log(vte))
        cb_tr = np.exp(c0 + c1 * np.log(np.maximum(har_tr, 1e-12)) + c2 * np.log(vtr))
        ztr_har.append(rtr / np.sqrt(np.maximum(har_tr, 1e-12)))
        ztr_axx.append(rtr / np.sqrt(np.maximum(vtr, 1e-12)))
        ztr_combo.append(rtr / np.sqrt(np.maximum(cb_tr, 1e-12)))
        keep.append({"sym": sym, "rte": rte, "har": har_te, "vte": vte, "cb": cb_te})
        print(f"  [bars]  {sym:6s} bars_te={okte.sum():5d}")
    gh_har = fit_gh(np.concatenate(ztr_har)); sd_har = float(np.std(np.concatenate(ztr_har)))
    gh_axx = fit_gh(np.concatenate(ztr_axx)); sd_axx = float(np.std(np.concatenate(ztr_axx)))
    gh_combo = fit_gh(np.concatenate(ztr_combo))
    rows = []
    for d in keep:
        r = d["rte"]
        rows.append({"sym": d["sym"],
                     "HAR-RV+Gaussian": pred_ll(r, d["har"], lambda z, s=sd_har: stats.norm.logpdf(z, 0, s)),
                     "HAR-RV+GH": pred_ll(r, d["har"], lambda z, g=gh_har: g.logpdf(z)),
                     "axx+Gaussian": pred_ll(r, d["vte"], lambda z, s=sd_axx: stats.norm.logpdf(z, 0, s)),
                     "axx+GH": pred_ll(r, d["vte"], lambda z, g=gh_axx: g.logpdf(z)),
                     "combined(HARxaxx)+GH": pred_ll(r, d["cb"], lambda z, g=gh_combo: g.logpdf(z))})
    return pd.DataFrame(rows)


def summarise(df, name):
    print(f"\n=== {name} ({len(df)} instruments): mean OOS predictive log-lik / obs ===")
    cols = [c for c in df.columns if c not in ("sym", "nu")]
    for c in cols:
        x = df[c].to_numpy(); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
        print(f"  {c:24s} = {m:+.4f}  [{m-1.96*se:+.4f}, {m+1.96*se:+.4f}]")
    return cols


def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    print("(A) event-clock: joint GARCH-t vs combined+GH ...")
    ev = run_event(files)
    ev.to_csv(OUTT / "sde_benchmarks_event.csv", index=False)
    cols = summarise(ev, "Event clock")
    print(f"  median joint GARCH-t nu = {ev['nu'].median():.2f}")
    # head-to-head sign tests vs combined+GH
    for c in ["jointGARCH-t", "axx+GH", "GARCH+Gaussian"]:
        d = ev["combined+GH"] - ev[c]; k = int((d > 0).sum()); n = len(d)
        print(f"  combined+GH vs {c:16s}: dLL={d.mean():+.4f}  {k}/{n}  p={stats.binomtest(k,n,0.5).pvalue:.2e}")

    print("\n(B) 60s bars: HAR-RV vs book-state ...")
    bars = run_bars(files)
    bars.to_csv(OUTT / "sde_benchmarks_bars.csv", index=False)
    summarise(bars, "60s bars")
    for c in ["HAR-RV+Gaussian", "HAR-RV+GH", "axx+GH"]:
        d = bars["combined(HARxaxx)+GH"] - bars[c]; k = int((d > 0).sum()); n = len(d)
        print(f"  combined(HARxaxx)+GH vs {c:18s}: dLL={d.mean():+.4f}  {k}/{n}  "
              f"p={stats.binomtest(k,n,0.5).pvalue:.2e}")
    print("\n[benchmarks] wrote sde_benchmarks_event.csv and sde_benchmarks_bars.csv")


if __name__ == "__main__":
    main()
