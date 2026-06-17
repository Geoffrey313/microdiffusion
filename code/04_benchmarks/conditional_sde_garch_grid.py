#!/usr/bin/env python3
"""
conditional_sde_garch_grid.py — full predictive-density grid:
   variance forecast {GARCH, a_xx(I,S), combined} x innovation {Gaussian, Student-t, GH}.

For each variance model m we standardise the move r_t by its forecast, z_{m,t}=r_t/sqrt(sigma^2_{m,t}),
fit the residual law (Gaussian / Student-t / GH; loc=0, free scale, so each variance model gets its OWN
best-fit innovation) on the pooled TRAIN residuals, and evaluate the
out-of-sample predictive log-likelihood on the held-out residuals:
   log f(r_t) = -0.5 log sigma^2_{m,t} + log q_{m,L}(z_{m,t}).
This lets the heavy tail into the contest on both sides and asks whether the full model (combined + GH)
is competitive with a GARCH variance forecast plus a fitted Student-t residual law. It is not a fully
jointly estimated GARCH-t model; the GARCH variance is estimated first, then the residual law is fitted.

Outputs: tables/sde_garch_grid.csv (3x3 mean OOS log-lik) ; tables/sde_garch_grid_bysym.csv
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
FIT_CAP = 12000
BPS = 1e4
RNG = np.random.default_rng(0)


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


def garch_filter(r2, w, a, b, s0):
    n = len(r2); s2 = np.empty(n); s2[0] = s0
    for t in range(1, n):
        s2[t] = w + a * r2[t - 1] + b * s2[t - 1]
    return s2


def fit_garch(r):
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
                                options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-2})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x)


def fit_combo(r, u, v):
    def nll(c):
        s2 = np.minimum(np.maximum(np.exp(c[0] + c[1] * u + c[2] * v), 1e-12), 1e12)
        return 0.5 * np.sum(np.log(s2) + r * r / s2)
    return optimize.minimize(nll, [0.0, 0.5, 0.5], method="Nelder-Mead",
                             options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-3}).x


def fit_laws(z):
    """Fit loc=0 free-scale Gaussian, Student-t, symmetric GH to pooled residuals z; return logpdf & ppf."""
    s_g = float(np.std(z))
    nu, _, s_t = stats.t.fit(z if len(z) <= 200_000 else RNG.choice(z, 200_000, replace=False), floc=0.0)
    zz = z if len(z) <= 120_000 else RNG.choice(z, 120_000, replace=False)
    p, a, b, loc, s_h = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    ghf = stats.genhyperbolic(p, a, 0.0, 0.0, s_h)
    return {
        "logpdf": {
            "Gaussian": lambda u: stats.norm.logpdf(u, 0.0, s_g),
            "Student-t": lambda u: stats.t.logpdf(u, nu, 0.0, s_t),
            "GH": lambda u: ghf.logpdf(u),
        },
        "ppf": {
            "Gaussian": lambda q: stats.norm.ppf(q, 0.0, s_g),
            "Student-t": lambda q: stats.t.ppf(q, nu, 0.0, s_t),
            "GH": lambda q: ghf.ppf(q),
        },
        "_nu": float(nu), "_ghp": float(p),
    }


def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    MODELS = ["garch", "axx", "combo"]
    per_sym = []                       # per-symbol test arrays: r, and sigma2 for each model
    z_train = {m: [] for m in MODELS}  # pooled train standardised residuals per model
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        te = [session_increments(pd.read_parquet(p)) for p in paths[ntr:]]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        te = pd.concat([t for t in te if t is not None], ignore_index=True) if any(t is not None for t in te) else None
        if tr is None or te is None or len(tr) < 3000 or len(te) < 1000:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        axx = {c: (tr["dx"].to_numpy()[ctr == c] ** 2).mean() * BPS * BPS
               for c in range(N_I * M_S) if (ctr == c).sum() >= MIN_CELL}
        if not axx:
            continue
        r_tr = tr["dx"].to_numpy() * BPS; r_te = te["dx"].to_numpy() * BPS
        mu = r_tr.mean(); r_tr -= mu; r_te -= mu
        w, a, b = fit_garch(r_tr)
        r_all = np.concatenate([r_tr, r_te])
        s2_all = garch_filter(r_all * r_all, w, a, b, float(np.var(r_tr)) + 1e-9)
        g_tr, g_te = s2_all[:len(r_tr)], s2_all[len(r_tr):]
        ax_tr = np.array([axx[c] if c in axx else np.nan for c in ctr])
        ax_te = np.array([axx[c] if c in axx else np.nan for c in cte])
        mtr = np.isfinite(ax_tr) & (g_tr > 0); mte = np.isfinite(ax_te) & (g_te > 0)
        if mtr.sum() < 1000 or mte.sum() < 500:
            continue
        c0, c1, c2 = fit_combo(r_tr[mtr], np.log(g_tr[mtr]), np.log(ax_tr[mtr]))
        cb_tr = np.exp(c0 + c1 * np.log(g_tr[mtr]) + c2 * np.log(ax_tr[mtr]))
        cb_te = np.exp(c0 + c1 * np.log(g_te[mte]) + c2 * np.log(ax_te[mte]))
        s2 = {"garch": (g_tr[mtr], g_te[mte]), "axx": (ax_tr[mtr], ax_te[mte]), "combo": (cb_tr, cb_te)}
        rt = r_tr[mtr]; re = r_te[mte]
        for m in MODELS:
            z_train[m].append(rt / np.sqrt(s2[m][0]))
        per_sym.append({"sym": sym, "r_te": re, "s2": {m: s2[m][1] for m in MODELS}})
        print(f"  loaded {sym:6s} n_te={mte.sum():6d}")

    laws = {m: fit_laws(np.concatenate(z_train[m])) for m in MODELS}
    for m in MODELS:
        print(f"  residual law fit [{m:6s}]: t nu={laws[m]['_nu']:.2f}, GH p={laws[m]['_ghp']:.2f}")

    INNOV = ["Gaussian", "Student-t", "GH"]
    rows = []
    for d in per_sym:
        r = d["r_te"]; row = {"sym": d["sym"]}
        for m in MODELS:
            s2m = np.maximum(d["s2"][m], 1e-12); zt = r / np.sqrt(s2m); half = -0.5 * np.log(s2m)
            for L in INNOV:
                row[f"{m}|{L}"] = float(np.mean(half + laws[m]["logpdf"][L](zt)))
        rows.append(row)
    bysym = pd.DataFrame(rows)
    bysym.to_csv(HERE / "output" / "tables" / "sde_garch_grid_bysym.csv", index=False)

    grid = pd.DataFrame({L: [bysym[f"{m}|{L}"].mean() for m in MODELS] for L in INNOV}, index=MODELS)
    grid.to_csv(HERE / "output" / "tables" / "sde_garch_grid.csv")
    print(f"\n=== Mean OOS predictive log-likelihood / obs ({len(bysym)} instruments) ===")
    print(grid.round(4).to_string())

    def sign(col_a, col_b):
        d = bysym[col_a] - bysym[col_b]; k = int((d > 0).sum()); n = len(d)
        return d.mean(), k, n, stats.binomtest(k, n, 0.5).pvalue
    print("\nKey comparisons (mean diff, #>0/n, sign-test p):")
    for ca, cb, lab in [("combo|GH", "garch|GH", "combined+GH  vs  GARCH+GH"),
                        ("combo|GH", "garch|Student-t", "combined+GH  vs  GARCH-t"),
                        ("combo|GH", "garch|Gaussian", "combined+GH  vs  GARCH+Gaussian"),
                        ("garch|GH", "garch|Gaussian", "GARCH+GH     vs  GARCH+Gaussian"),
                        ("combo|GH", "combo|Gaussian", "combined+GH  vs  combined+Gaussian")]:
        m, k, n, p = sign(ca, cb)
        print(f"  {lab:34s} dLL={m:+.4f}  {k}/{n}  p={p:.2e}")
    print("\n[grid] wrote tables/sde_garch_grid.csv and sde_garch_grid_bysym.csv")

    # ---- predictive calibration (pooled across all held-out moves) ----
    PAIRS = [("garch", "Gaussian", "GARCH + Gaussian"),
             ("garch", "Student-t", r"GARCH + Student-$t$"),
             ("axx", "GH", r"$a_{xx}(I,S)$ + GH"),
             ("combo", "GH", "Combined + GH")]
    ztest = {m: np.concatenate([d["r_te"] / np.sqrt(np.maximum(d["s2"][m], 1e-12)) for d in per_sym])
             for m in MODELS}
    nominal = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99])
    betas = np.array([0.10, 0.05, 0.02, 0.01, 0.005, 0.001])
    cov, tail, crows = {}, {}, []
    print(f"\n=== Predictive calibration (pooled n={len(ztest['garch']):,}) ===")
    for m, L, lab in PAIRS:
        z = ztest[m]; ppf = laws[m]["ppf"][L]
        cov[lab] = [float(np.mean((z >= ppf((1 - aa) / 2)) & (z <= ppf((1 + aa) / 2)))) for aa in nominal]
        tail[lab] = [float((np.mean(z > ppf(1 - bb / 2)) + np.mean(z < ppf(bb / 2))) / bb) for bb in betas]
        for aa, cc in zip(nominal, cov[lab]): crows.append({"model": lab, "kind": "coverage", "nominal": aa, "value": cc})
        for bb, tr in zip(betas, tail[lab]): crows.append({"model": lab, "kind": "tailratio", "nominal": bb, "value": tr})
        print(f"  {lab:24s} 99% cover={cov[lab][-1]:.3f}  tail-ratio@1%={tail[lab][3]:.2f}  @0.1%={tail[lab][5]:.2f}")
    pd.DataFrame(crows).to_csv(HERE / "output" / "tables" / "sde_garch_calibration.csv", index=False)

    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, MUTED, ACCENT, ACCENT_DARK
    plt = setup_mpl()
    style = {"GARCH + Gaussian": (MUTED, ":", "o"),
             r"GARCH + Student-$t$": (INK, "--", "s"),
             r"$a_{xx}(I,S)$ + GH": (ACCENT, "-", "^"),
             "Combined + GH": (ACCENT_DARK, "-", "D")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
    a = ax[0]
    a.plot([0.5, 1], [0.5, 1], color="0.6", lw=1, ls="-", zorder=0, label=r"perfect calibration")
    for _, _, lab in PAIRS:
        c, ls, mk = style[lab]
        a.plot(nominal, cov[lab], ls=ls, marker=mk, ms=4, color=c, lw=1.5, label=lab)
    a.set_xlabel(r"nominal coverage of the predictive interval")
    a.set_ylabel(r"empirical coverage (held-out)")
    a.set_title(r"(a) Are the predictive intervals calibrated?")
    a.set_xlim(0.48, 1.005); a.set_ylim(0.48, 1.005); a.legend(loc="upper left", fontsize=7.5); despine(a)
    b = ax[1]
    b.axhline(1.0, color="0.6", lw=1, label=r"perfect (ratio $=1$)")
    for _, _, lab in PAIRS:
        c, ls, mk = style[lab]
        b.plot(betas * 100, tail[lab], ls=ls, marker=mk, ms=4, color=c, lw=1.5, label=lab)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"nominal two-sided tail level (\%)")
    b.set_ylabel(r"observed / nominal tail exceedance")
    b.set_title(r"(b) Tail-risk calibration (lower tail levels $\rightarrow$)")
    b.invert_xaxis(); b.legend(loc="upper right", fontsize=7.5); despine(b)
    fig.suptitle(r"Out-of-sample predictive calibration: heavy-tailed innovations improve tail-risk "
                 r"calibration", y=1.01)
    OUT = CODE / "paper" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    finish(fig, OUT / "fig_garch_calibration.png")
    print("[grid] wrote paper/figures/fig_garch_calibration.png and tables/sde_garch_calibration.csv")


if __name__ == "__main__":
    main()
