#!/usr/bin/env python3
"""
conditional_sde_garch_compare.py — is the book-state diffusion a_xx INCREMENTAL to GARCH?

GARCH conditions the variance on the past of the return series; our a_xx(I,S) conditions it on the
contemporaneous book state. These are different, possibly complementary, information sets. We test this
out of sample with a predictive-log-likelihood comparison (Gaussian density, so the ONLY thing that
differs is the variance forecast sigma^2_t):
  (i)   GARCH(1,1)          : sigma^2_t = omega + alpha r_{t-1}^2 + beta sigma^2_{t-1}   (return history)
  (ii)  a_xx(I,S)           : sigma^2_t = a_xx(I_t,S_t)                                   (book state)
  (iii) combined (log-pool) : log sigma^2_t = c0 + c1 log sigma^2_GARCH,t + c2 log a_xx_t (weights fit on train)
If a_xx is incremental, (iii) beats (i) out of sample and the a_xx weight c2 > 0.  We do NOT claim a_xx
alone beats GARCH at one-step forecasting; the claim is complementarity.
Caveat: GARCH is applied on the event clock (the paper's native clock); stated in the write-up.

Outputs: tables/sde_garch_compare.csv
Run: python3 conditional_sde_garch_compare.py
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
FIT_CAP = 12000          # cap the series length passed to the GARCH optimiser (speed)
BPS = 1e4


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
    """Gaussian QMLE GARCH(1,1) on r (already de-meaned). Returns (omega, alpha, beta)."""
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


def gll(r, s2):
    """mean Gaussian predictive log-likelihood of r under variance s2."""
    s2 = np.maximum(s2, 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi) + np.log(s2) + r * r / s2)))


def fit_combo(r_tr, u_tr, v_tr):
    """log sigma^2 = c0 + c1 u + c2 v ; maximise train Gaussian LL. u=log s2_garch, v=log a_xx."""
    def nll(c):
        s2 = np.exp(c[0] + c[1] * u_tr + c[2] * v_tr)
        s2 = np.minimum(np.maximum(s2, 1e-12), 1e12)
        return 0.5 * np.sum(np.log(s2) + r_tr * r_tr / s2)
    res = optimize.minimize(nll, [0.0, 0.5, 0.5], method="Nelder-Mead",
                            options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-3})
    return res.x


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
        tr = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        te = [session_increments(pd.read_parquet(p)) for p in paths[ntr:]]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        te = pd.concat([t for t in te if t is not None], ignore_index=True) if any(t is not None for t in te) else None
        if tr is None or te is None or len(tr) < 3000 or len(te) < 1000:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        axx = {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                axx[c] = (tr["dx"].to_numpy()[m] ** 2).mean() * BPS * BPS     # bps^2
        if not axx:
            continue

        r_tr = tr["dx"].to_numpy() * BPS; r_te = te["dx"].to_numpy() * BPS
        mu = r_tr.mean(); r_tr = r_tr - mu; r_te = r_te - mu
        # GARCH fit on train; filter over the full ordered series, take train and test parts
        w, a, b = fit_garch(r_tr)
        r_all = np.concatenate([r_tr, r_te])
        s2_all = garch_filter(r_all * r_all, w, a, b, float(np.var(r_tr)) + 1e-9)
        s2g_tr = s2_all[:len(r_tr)]; s2g_te = s2_all[len(r_tr):]
        # a_xx per obs (valid cells only)
        keep_tr = np.array([c in axx for c in ctr]); keep_te = np.array([c in axx for c in cte])
        axx_tr = np.array([axx[c] if c in axx else np.nan for c in ctr])
        axx_te = np.array([axx[c] if c in axx else np.nan for c in cte])

        # combined weights fit on TRAIN (valid cells), evaluated on TEST
        mtr = keep_tr & np.isfinite(axx_tr) & (s2g_tr > 0)
        mte = keep_te & np.isfinite(axx_te) & (s2g_te > 0)
        if mtr.sum() < 1000 or mte.sum() < 500:
            continue
        c0, c1, c2 = fit_combo(r_tr[mtr], np.log(s2g_tr[mtr]), np.log(axx_tr[mtr]))
        s2c_te = np.exp(c0 + c1 * np.log(s2g_te[mte]) + c2 * np.log(axx_te[mte]))

        rows.append({
            "sym": sym, "n_te": int(mte.sum()), "alpha": a, "beta": b, "persist": a + b,
            "ll_garch": gll(r_te[mte], s2g_te[mte]),
            "ll_axx":   gll(r_te[mte], axx_te[mte]),
            "ll_combo": gll(r_te[mte], s2c_te),
            "w_garch": c1, "w_axx": c2,
        })
        print(f"  {sym:6s} n_te={int(mte.sum()):6d}  persist={a+b:.3f}  "
              f"LL garch={rows[-1]['ll_garch']:.3f} axx={rows[-1]['ll_axx']:.3f} combo={rows[-1]['ll_combo']:.3f}  "
              f"w_axx={c2:+.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "output" / "tables" / "sde_garch_compare.csv", index=False)

    def agg(x):
        x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
        return m, m - 1.96 * se, m + 1.96 * se
    print(f"\n=== GARCH vs book-state diffusion (QSE, {len(df)} instruments) ===")
    print(f"mean GARCH persistence alpha+beta = {df.persist.mean():.3f}")
    for k in ["ll_garch", "ll_axx", "ll_combo"]:
        m, lo, hi = agg(df[k]); print(f"  mean OOS log-lik / obs  {k:9s} = {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    dcg = df.ll_combo - df.ll_garch; m, lo, hi = agg(dcg)
    print(f"  INCREMENTAL: combined - GARCH = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"(>0 => book state adds OOS value beyond GARCH; positive for {100*np.mean(dcg>0):.0f}% of names)")
    dca = df.ll_combo - df.ll_axx; m, lo, hi = agg(dca)
    print(f"               combined - a_xx  = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    m, lo, hi = agg(df.w_axx)
    print(f"  combined weight on a_xx (c2)  = {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
          f"(positive for {100*np.mean(df.w_axx>0):.0f}% of names)")
    print("\n[garch] wrote tables/sde_garch_compare.csv")


if __name__ == "__main__":
    main()
