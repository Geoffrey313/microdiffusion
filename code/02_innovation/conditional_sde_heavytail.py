#!/usr/bin/env python3
"""
conditional_sde_heavytail.py — replace the Gaussian innovation in the conditional SDE with a
heavy-tailed *normal variance-mixture* and re-check Panel A (shape) and Panel D (tails).

Model (per book-state box c = (I-bin, S-tercile)):
    Delta x_t = b_x(c) + sqrt(a_xx(c)) * eps_t,
    eps_t = sqrt(G) * Z,   Z ~ N(0,1),   G > 0 a "mixing" variable with E[G]=1  (=> Var(eps)=1).
  G = const             -> eps Gaussian              (the current prototype)
  G ~ Inverse-Gamma     -> eps Student-t             (inverse-gamma x normal)
  G ~ Gen. Inv. Gaussian-> eps Generalized Hyperbolic(generalized-inverse x normal)

Because E[G]=1, sqrt(a_xx) stays the move SIZE (Panel B unchanged); only the SHAPE of the kick changes.
Key identity for any such mixture: excess kurtosis(eps) = 3 * Var(G). So "kurtosis-match" == set Var(G)=kappa/3.

We fit both families two ways to the pooled state-standardised residuals z=(Delta x-b_x)/sqrt(a_xx):
  (1) ML  : maximum likelihood (lets the data choose the tail);
  (2) KURT: match the sample (excess) kurtosis, i.e. Var(G)=kappa/3.
All laws are compared at the empirical width of z (free scale), apples-to-apples with a best-fit Gaussian.

Outputs: tables/sde_heavytail_summary.csv ; figures/sde_heavytail.png
Run: python3 conditional_sde_heavytail.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats, optimize

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
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz)
    if good.sum() < 200:
        return None
    x = np.log(mid)
    dx = np.diff(x)
    I = (bsz - asz) / (bsz + asz)
    S = (ask - bid) / mid
    return pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1]}).replace([np.inf, -np.inf], np.nan).dropna()


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


def collect_residuals():
    """Pooled held-out state-dependent standardised residuals z (the prototype's Panel-A/D object)."""
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)
    Z = []
    for sym, paths in files.items():
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        te = [session_increments(pd.read_parquet(p)) for p in paths[ntr:]]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        te = pd.concat([t for t in te if t is not None], ignore_index=True) if any(t is not None for t in te) else None
        if tr is None or te is None or len(tr) < 2000 or len(te) < 1000:
            continue
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy(); dxte = te["dx"].to_numpy()
        bx, axx = {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = dxtr[m].mean(); axx[c] = (dxtr[m] ** 2).mean()
        keep = np.array([c in axx and axx[c] > 0 for c in cte])
        if keep.sum() < 500:
            continue
        cte_k = cte[keep]; dxte_k = dxte[keep]
        bx_v = np.array([bx[c] for c in cte_k]); axx_v = np.array([axx[c] for c in cte_k])
        Z.append((dxte_k - bx_v) / np.sqrt(axx_v))
    return np.concatenate(Z)


# ---- unit-shape laws evaluated at a free scale s (so we compare SHAPE at z's own width) ----
def t_unitvar_scale(nu):
    """multiplier a s.t. a * t_nu has unit variance."""
    return np.sqrt((nu - 2.0) / nu)


def fit_student_t_ML(z):
    nu, loc, s = stats.t.fit(z, floc=0.0)
    return {"name": "Student-$t$ (ML)", "nu": float(nu), "scale": float(s),
            "pdf": lambda x: stats.t.pdf(x / s, nu) / s,
            "sf": lambda k: 2.0 * stats.t.sf(k / s, nu)}


def fit_student_t_kurt(z, kappa, sd):
    nu = 4.0 + 6.0 / max(kappa, 1e-9)          # excess kurtosis of t_nu = 6/(nu-4)
    nu = float(np.clip(nu, 4.05, 1e6))
    s = sd * t_unitvar_scale(nu)               # set width to z's std
    return {"name": "Student-$t$ (kurtosis)", "nu": nu, "scale": float(s),
            "pdf": lambda x: stats.t.pdf(x / s, nu) / s,
            "sf": lambda k: 2.0 * stats.t.sf(k / s, nu)}


def fit_gh_ML(z):
    """Symmetric Generalized Hyperbolic by ML (b=0, loc=0); subsample for speed."""
    zz = z if len(z) <= 200_000 else RNG.choice(z, 200_000, replace=False)
    p, a, b, loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    fr = stats.genhyperbolic(p, a, 0.0, 0.0, s)
    return {"name": "Gen.\\ Hyperbolic (ML)", "p": float(p), "a": float(a), "scale": float(s),
            "pdf": lambda x: fr.pdf(x), "sf": lambda k: fr.sf(k) + fr.cdf(-k)}


def fit_gh_kurt(z, kappa, sd):
    """GIG x normal, kurtosis-matched via NIG (Inverse-Gaussian mixing, E[G]=1, Var(G)=kappa/3).
    Empirical unit-shape law by large simulation; report tail/pdf via a smooth grid (KDE-free histogram)."""
    varG = max(kappa, 1e-9) / 3.0
    # Inverse-Gaussian with mean mu=1 and variance varG: scipy invgauss(mu=lam... ) param: var = mu^3/lambda
    lam = 1.0 / varG                              # mean=1 => var = 1/lam
    n = 2_000_000
    G = stats.invgauss.rvs(mu=1.0 / lam, scale=lam, size=n, random_state=RNG)  # mean=1, var=1/lam
    eps = np.sqrt(G) * RNG.standard_normal(n)
    eps *= sd / eps.std()                         # match width to z
    grid = np.linspace(-12 * sd, 12 * sd, 4001)
    hist, edges = np.histogram(eps, bins=grid, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    eps_abs = np.abs(eps)
    return {"name": "NIG / GIG (kurtosis)", "varG": float(varG), "scale": float(sd),
            "pdf": lambda x: np.interp(x, centers, hist, left=0, right=0),
            "sf": lambda k: float(np.mean(eps_abs > k))}


def main():
    print("collecting residuals ...")
    z = collect_residuals()
    z = z[np.isfinite(z)]
    sd = float(z.std()); m = float(z.mean())
    kappa = float(((z - m) ** 4).mean() / z.var() ** 2 - 3.0)        # excess kurtosis (raw)
    # robust (trimmed) kurtosis at 99.9% to show the raw one is outlier-driven
    cap = np.quantile(np.abs(z), 0.999)
    zt = z[np.abs(z) <= cap]
    kappa_tr = float(((zt - zt.mean()) ** 4).mean() / zt.var() ** 2 - 3.0)
    print(f"n={len(z):,}  std(z)={sd:.3f}  excess-kurtosis raw={kappa:.0f}  trimmed@99.9%={kappa_tr:.1f}")

    # fits — use the *trimmed* kurtosis for the kurtosis-match curves (raw is degenerate: nu->4)
    laws = []
    laws.append({"name": "best-fit Gaussian", "scale": sd,
                 "pdf": (lambda x, s=sd: stats.norm.pdf(x / s) / s),
                 "sf": (lambda k, s=sd: 2.0 * stats.norm.sf(k / s))})
    laws.append(fit_student_t_ML(z))
    laws.append(fit_student_t_kurt(z, kappa_tr, sd))
    try:
        laws.append(fit_gh_ML(z))
    except Exception as e:
        print("GH ML fit failed:", e)
    laws.append(fit_gh_kurt(z, kappa_tr, sd))

    # empirical tails and a per-law table
    ks = np.arange(1, 8)
    emp = np.array([np.mean(np.abs(z) > k) for k in ks])
    rows = []
    for L in laws:
        sf = np.array([L["sf"](k) for k in ks])
        rows.append({"law": L["name"],
                     **{f"P|>{k}": float(sf[i]) for i, k in enumerate(ks)},
                     "nu": L.get("nu", np.nan), "varG": L.get("varG", np.nan), "scale": L.get("scale", np.nan)})
    tab = pd.DataFrame(rows)
    out_t = HERE / "output" / "tables" / "sde_heavytail_summary.csv"
    out_t.parent.mkdir(parents=True, exist_ok=True)
    extra = pd.DataFrame([{"law": "DATA (empirical)", **{f"P|>{k}": float(emp[i]) for i, k in enumerate(ks)}}])
    pd.concat([extra, tab], ignore_index=True).to_csv(out_t, index=False)
    print("\n=== Tail probabilities P(|eps| > k), data vs fitted innovations ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.concat([extra, tab], ignore_index=True)[["law"] + [f"P|>{k}" for k in (2, 3, 4, 5, 6)]]
              .to_string(index=False, float_format=lambda v: f"{v:.2e}"))

    # ---- figure: A density (log-y), D tail exceedance (log-y) ----
    import sys
sys.path.insert(0, str(CODE))
from plot_style import ACCENT, ACCENT_DARK, WARN, MUTED, INK, POS, finish, setup_mpl, despine
    plt = setup_mpl()
    palette = {"best-fit Gaussian": WARN, "Student-$t$ (ML)": ACCENT_DARK,
               "Student-$t$ (kurtosis)": ACCENT, "Gen.\\ Hyperbolic (ML)": POS,
               "NIG / GIG (kurtosis)": INK}
    styles = {"best-fit Gaussian": "--", "Student-$t$ (ML)": "-", "Student-$t$ (kurtosis)": ":",
              "Gen.\\ Hyperbolic (ML)": "-", "NIG / GIG (kurtosis)": ":"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    a = ax[0]; lo, hi = -8 * sd, 8 * sd; gridp = np.linspace(lo, hi, 400)
    binp = np.linspace(lo, hi, 160)
    a.hist(z[(z > lo) & (z < hi)], bins=binp, density=True, histtype="step", color=MUTED, lw=1.5,
           label=r"data $z$")
    for L in laws:
        a.plot(gridp, L["pdf"](gridp), color=palette[L["name"]], ls=styles[L["name"]], lw=1.4, label=L["name"])
    a.set_yscale("log"); a.set_ylim(1e-5, 2)
    a.set_xlabel(r"standardised increment $z=(\Delta x-b_x)/\sqrt{a_{xx}}$")
    a.set_ylabel(r"density (log scale)")
    a.set_title(r"Panel A$'$: shape of the kick --- Gaussian vs heavy-tailed mixtures")
    a.legend(loc="lower center", ncol=2, fontsize=7); despine(a)

    d = ax[1]
    d.plot(ks, emp, "-o", color=MUTED, ms=4, lw=1.6, label=r"data")
    for L in laws:
        d.plot(ks, [L["sf"](k) for k in ks], ls=styles[L["name"]], color=palette[L["name"]], lw=1.4,
               marker="s", ms=3, label=L["name"])
    d.set_yscale("log"); d.set_ylim(1e-7, 1)
    d.set_xlabel(r"threshold $k$ (in units of $\sqrt{a_{xx}}$)")
    d.set_ylabel(r"$P(|\,z\,| > k)$")
    d.set_title(r"Panel D$'$: tail exceedance --- which kick reproduces the jumps?")
    d.legend(loc="upper right", fontsize=7); despine(d)

    fig.suptitle(r"Heavy-tailed innovation $\varepsilon=\sqrt{G}\,Z$ with $Z\sim N(0,1)$ and $G$ the random "
                 r"variance ($\mathbb{E}[G]{=}1$): Student-$t$ (Inv.-Gamma $G$) and Gen.-Hyperbolic "
                 r"(GIG $G$) vs.\ Gaussian", y=1.02)
    finish(fig, HERE / "output" / "figures" / "sde_heavytail.png")
    print("\n[heavytail] wrote figures/sde_heavytail.png and tables/sde_heavytail_summary.csv")


if __name__ == "__main__":
    main()
