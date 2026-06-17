#!/usr/bin/env python3
"""
conditional_sde_gh_oos.py — fresh-split, OUT-OF-SAMPLE validation of the GH tail law.

So far the GH was fit to the held-out residuals and judged on those same residuals (a goodness-of-fit, not a
fresh OOS test). Here we go one level further: split the held-out residuals z=(Delta x-b_x)/sqrt(a_xx) into two
DISJOINT temporal blocks (per symbol: earlier test sessions -> A, later -> B), fit a symmetric GH on block A,
and ask whether it predicts block B's distribution -- especially the tail -- on data it never saw.

Gold-standard metric: out-of-sample mean log-likelihood GAIN of GH(fitted on A) over the best Gaussian(fitted on
A), evaluated on B.  Positive => the heavy-tailed law generalises and beats Gaussian on fresh data.
Also: tail exceedances P(|z|>k) on B vs GH_A-predicted vs Gaussian_A-predicted; and the in-sample GH_B as a
ceiling.  Both directions (A->B and B->A) reported.

Outputs: figures/sde_gh_oos.png ; tables/sde_gh_oos.csv
Run: python3 conditional_sde_gh_oos.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50
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


def collect_blocks():
    """Held-out residuals split into two disjoint temporal blocks A (earlier test sessions) and B (later)."""
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)
    ZA, ZB = [], []
    for sym, paths in files.items():
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr_paths, te_paths = paths[:ntr], paths[ntr:]
        if len(te_paths) < 2:
            continue
        tr = [session_increments(pd.read_parquet(p)) for p in tr_paths]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        if tr is None or len(tr) < 2000:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy()
        bx, axx = {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = dxtr[m].mean(); axx[c] = (dxtr[m] ** 2).mean()
        half = len(te_paths) // 2                       # disjoint temporal split of the TEST sessions
        for blk, plist in (("A", te_paths[:half]), ("B", te_paths[half:])):
            for p in plist:
                df = session_increments(pd.read_parquet(p))
                if df is None or len(df) < 50:
                    continue
                cc = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
                dx = df["dx"].to_numpy()
                keep = np.array([c in axx and axx[c] > 0 for c in cc])
                if keep.sum() == 0:
                    continue
                ck = cc[keep]; dk = dx[keep]
                z = (dk - np.array([bx[c] for c in ck])) / np.sqrt(np.array([axx[c] for c in ck]))
                (ZA if blk == "A" else ZB).append(z)
    return np.concatenate(ZA), np.concatenate(ZB)


def fit_symmetric_gh(z):
    """Fit symmetric GH (b=0, loc=0; free shape p,a and scale) by ML on a subsample; return frozen dist."""
    zz = z if len(z) <= 120_000 else RNG.choice(z, 120_000, replace=False)
    p, a, b, loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    return stats.genhyperbolic(p, a, 0.0, 0.0, s), (float(p), float(a), float(s))


def tail(dist_sf, ks):
    return np.array([dist_sf(k) for k in ks])


def evaluate(z_fit, z_eval, label):
    """Fit GH and Gaussian on z_fit; evaluate OOS on z_eval. Returns dict of metrics + per-obs LL diff."""
    gh, ghp = fit_symmetric_gh(z_fit)
    sd_fit = float(z_fit.std())                          # best Gaussian on the fit block (loc 0)
    ll_gh = gh.logpdf(z_eval)
    ll_n = stats.norm.logpdf(z_eval, 0.0, sd_fit)
    d = ll_gh - ll_n                                     # per-obs OOS log-likelihood gain, GH over Gaussian
    dLL = float(np.mean(d))
    # bootstrap CI on the mean gain
    rg = np.random.default_rng(7); n = len(d)
    boot = np.array([d[rg.integers(0, n, n)].mean() for _ in range(400)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    ks = np.arange(2, 7)
    emp = np.array([np.mean(np.abs(z_eval) > k) for k in ks])
    gh_pred = tail(lambda k: gh.sf(k) + gh.cdf(-k), ks)
    n_pred = np.array([2 * stats.norm.sf(k / sd_fit) for k in ks])
    print(f"\n[{label}]  GH(p={ghp[0]:.2f},a={ghp[1]:.2f},scale={ghp[2]:.3f}) fit on {len(z_fit):,}, "
          f"evaluated on {len(z_eval):,}")
    print(f"  OOS mean log-lik gain GH - Gaussian = {dLL:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"(>0 => GH generalises and beats Gaussian)")
    print("   k   B-empirical   GH_A-predicted  Gaussian_A-pred")
    for i, k in enumerate(ks):
        print(f"   {k}   {emp[i]:.3e}    {gh_pred[i]:.3e}     {n_pred[i]:.3e}")
    return {"label": label, "dLL": dLL, "dLL_lo": ci[0], "dLL_hi": ci[1],
            "p": ghp[0], "a": ghp[1], "scale": ghp[2], "n_fit": len(z_fit), "n_eval": len(z_eval),
            "ks": ks, "emp": emp, "gh": gh_pred, "norm": n_pred}


def main():
    print("collecting two disjoint temporal blocks of held-out residuals ...")
    zA, zB = collect_blocks()
    zA = zA[np.isfinite(zA)]; zB = zB[np.isfinite(zB)]
    print(f"block A: {len(zA):,}   block B: {len(zB):,}")

    rAB = evaluate(zA, zB, "A -> B")          # fit on A, validate on B
    rBA = evaluate(zB, zA, "B -> A")          # fit on B, validate on A

    pd.DataFrame([{k: v for k, v in r.items() if k not in ("ks", "emp", "gh", "norm")}
                 for r in (rAB, rBA)]).to_csv(HERE / "output" / "tables" / "sde_gh_oos.csv", index=False)

    # ---- figure: B's tail vs GH(fit on A)-predicted vs Gaussian(fit on A)-predicted ----
    import sys
sys.path.insert(0, str(CODE))
from plot_style import ACCENT_DARK, WARN, MUTED, POS, INK, finish, setup_mpl, despine
    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for axi, r, src, tgt in ((ax[0], rAB, "A", "B"), (ax[1], rBA, "B", "A")):
        axi.plot(r["ks"], r["emp"], "-o", color=MUTED, ms=5, lw=1.7, label=rf"block {tgt} (empirical)")
        axi.plot(r["ks"], r["gh"], "-s", color=POS, ms=4, lw=1.5,
                 label=rf"GH fit on {src}, predicting {tgt}")
        axi.plot(r["ks"], r["norm"], "--", color=WARN, lw=1.3, label=rf"Gaussian fit on {src}")
        axi.set_yscale("log"); axi.set_ylim(1e-5, 1e-1)
        axi.set_xlabel(r"threshold $k$ (local std units)")
        axi.set_ylabel(r"$P(|z| > k)$ on the held-out block")
        axi.set_title(rf"Fit on {src} $\rightarrow$ predict {tgt}:  "
                      rf"$\Delta\overline{{\ell\ell}}_{{\mathrm{{GH-Gauss}}}}={r['dLL']:+.3f}$")
        axi.legend(loc="upper right", fontsize=7.5); despine(axi)
    fig.suptitle(r"Out-of-sample validation of the GH tail law: fit on one temporal block, predict the disjoint "
                 r"block", y=1.01)
    finish(fig, HERE / "output" / "figures" / "sde_gh_oos.png")
    print("\n[gh-oos] wrote figures/sde_gh_oos.png and tables/sde_gh_oos.csv")


if __name__ == "__main__":
    main()
