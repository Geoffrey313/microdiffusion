#!/usr/bin/env python3
"""
conditional_sde_ghsim.py — (1) fold the Generalized-Hyperbolic kick into the SDE simulation and redraw the
tail panel with GH-driven paths; (2) test whether the tail knob Var(G) is itself a function of (I,S).

Model:  Delta x_t = b_x(c) + sqrt(a_xx(c)) * eps_t,   eps_t = sqrt(G) Z,  E[G]=1  =>  exkurt(eps)=3 Var(G).
We fit a symmetric GH to the pooled held-out standardised residuals z=(Delta x-b_x)/sqrt(a_xx), normalise it to
unit variance, then:
  (1) SIMULATE Delta x with the GH innovation per held-out point and compare P(|Delta x|/sigma_loc > k) to data
      and to the original Gaussian-driven simulation;
  (2) split the residuals by book state and estimate Var(G | state) = exkurt(z | state)/3 (trimmed, robust),
      plus P(|z|>3 | state), to see if the FAT-TAIL THICKNESS depends on (I,S) the way the variance does.

Outputs: figures/sde_ghsim_tail.png, figures/sde_tailknob_state.png ; tables/sde_ghsim_summary.csv
Run: python3 conditional_sde_ghsim.py
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


def collect():
    """Per held-out point: z, drift b_x, scale sqrt(a_xx), imbalance-bin, spread-tercile."""
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)
    Z, BX, SIG, IB, SB = [], [], [], [], []
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
        ck = cte[keep]; dk = dxte[keep]
        bxv = np.array([bx[c] for c in ck]); axxv = np.array([axx[c] for c in ck])
        Z.append((dk - bxv) / np.sqrt(axxv)); BX.append(bxv); SIG.append(np.sqrt(axxv))
        IB.append(ck // M_S); SB.append(ck % M_S)
    return (np.concatenate(Z), np.concatenate(BX), np.concatenate(SIG),
            np.concatenate(IB), np.concatenate(SB))


def gh_unit(z):
    """Fit symmetric GH (b=0, loc=0) by ML on a subsample; return a UNIT-VARIANCE frozen dist + sampler."""
    zz = z if len(z) <= 150_000 else RNG.choice(z, 150_000, replace=False)
    p, a, b, loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    fr = stats.genhyperbolic(p, a, 0.0, 0.0, s)
    sd = np.sqrt(fr.var())
    pdf = lambda x: sd * fr.pdf(sd * x)
    sf = lambda k: fr.sf(sd * k) + fr.cdf(-sd * k)
    rvs = lambda n: fr.rvs(size=n, random_state=RNG) / sd
    return {"p": float(p), "a": float(a), "sd": float(sd), "pdf": pdf, "sf": sf, "rvs": rvs}


def main():
    print("collecting held-out residuals ...")
    z, bxv, sig, ib, sb = collect()
    ok = np.isfinite(z); z, bxv, sig, ib, sb = z[ok], bxv[ok], sig[ok], ib[ok], sb[ok]
    n = len(z); print(f"n = {n:,}  std(z) = {z.std():.3f}")

    gh = gh_unit(z)
    print(f"GH fit: p={gh['p']:.3f} a={gh['a']:.3f}")

    # ---------- (1) GH-driven SDE simulation vs Gaussian, tail panel ----------
    nsim = min(n, 500_000)
    idx = RNG.choice(n, nsim, replace=False)
    eps_gh = gh["rvs"](nsim)
    eps_n = RNG.standard_normal(nsim)
    dx_gh = bxv[idx] + sig[idx] * eps_gh          # GH-driven simulated move
    dx_n = bxv[idx] + sig[idx] * eps_n            # Gaussian-driven (original prototype)
    u_gh = np.abs(dx_gh) / sig[idx]; u_n = np.abs(dx_n) / sig[idx]
    ks = np.arange(1, 8)
    emp = np.array([np.mean(np.abs(z) > k) for k in ks])
    t_gh = np.array([np.mean(u_gh > k) for k in ks])
    t_n = np.array([np.mean(u_n > k) for k in ks])
    t_norm = np.array([2 * stats.norm.sf(k) for k in ks])

    pd.DataFrame({"k": ks, "data": emp, "GH_sim": t_gh, "Gaussian_sim": t_n, "N01": t_norm}).to_csv(
        HERE / "output" / "tables" / "sde_ghsim_summary.csv", index=False)
    print("\nP(|.|>k):  k   data     GH-sim   Gauss-sim")
    for i, k in enumerate(ks):
        print(f"           {k}  {emp[i]:.2e} {t_gh[i]:.2e} {t_n[i]:.2e}")

    # ---------- (2) is Var(G) = exkurt/3 a function of (I,S)? ----------
    cap = np.quantile(np.abs(z), 0.999)              # common robust trim for all groups

    def varG(zz):
        zz = zz[np.abs(zz) <= cap]
        if len(zz) < 500:
            return np.nan, np.nan
        k = ((zz - zz.mean()) ** 4).mean() / zz.var() ** 2 - 3.0
        return k / 3.0, float(np.mean(np.abs(zz) > 3))    # Var(G), P(|z|>3) (untrimmed tail prob)

    by_S = [varG(z[sb == s]) for s in range(M_S)]                        # spread tercile
    by_I = [varG(z[ib == i]) for i in range(N_I)]                        # imbalance bin
    VG_S = [v[0] for v in by_S]; VG_I = [v[0] for v in by_I]
    P3_S = [float(np.mean(np.abs(z[sb == s]) > 3)) for s in range(M_S)]
    P3_I = [float(np.mean(np.abs(z[ib == i]) > 3)) for i in range(N_I)]
    imb_centers = (np.arange(N_I) + 0.5) / N_I * 2 - 1                   # bin midpoints in [-1,1]
    sp_I = stats.spearmanr(np.abs(imb_centers), VG_I).correlation
    print(f"\nVar(G) by spread tercile (tight->wide): {[f'{v:.1f}' for v in VG_S]}")
    print(f"Var(G) by imbalance bin: {[f'{v:.1f}' for v in VG_I]}")
    print(f"Spearman(|imbalance|, Var(G)) across bins = {sp_I:.3f}")

    # ---------- figures ----------
    import sys
sys.path.insert(0, str(CODE))
from plot_style import ACCENT, ACCENT_DARK, WARN, MUTED, INK, POS, finish, setup_mpl, despine
    plt = setup_mpl()

    # Fig 1: tail panel with GH simulation
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(ks, emp, "-o", color=MUTED, ms=5, lw=1.8, label=r"data (held-out)")
    ax.plot(ks, t_gh, "-s", color=POS, ms=4, lw=1.5, label=r"GH-driven SDE simulation")
    ax.plot(ks, t_n, "-^", color=ACCENT_DARK, ms=4, lw=1.5, label=r"Gaussian SDE simulation")
    ax.plot(ks, t_norm, "--", color=WARN, lw=1.2, label=r"$N(0,1)$ reference")
    ax.set_yscale("log"); ax.set_ylim(1e-7, 1)
    ax.set_xlabel(r"threshold $k$ (in units of $\sqrt{a_{xx}}$)")
    ax.set_ylabel(r"$P(|\Delta x|/\sqrt{a_{xx}} > k)$")
    ax.set_title(r"Folding the GH kick into the SDE: simulated paths now carry the jumps")
    ax.legend(loc="upper right"); despine(ax)
    finish(fig, HERE / "output" / "figures" / "sde_ghsim_tail.png")

    # Fig 2: is the tail knob Var(G) state-dependent?
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.6))
    axa.bar([0, 1, 2], VG_S, color=[MUTED, ACCENT, ACCENT_DARK], width=0.6)
    axa.set_xticks([0, 1, 2]); axa.set_xticklabels([r"tight $S$", r"mid $S$", r"wide $S$"])
    axa.set_ylabel(r"$\mathrm{Var}(G)=\mathrm{exkurt}(z)/3$ (trimmed)")
    axa.set_title(r"(A) Tail knob by spread tercile")
    for i, v in enumerate(VG_S):
        axa.text(i, v, f"{v:.1f}", ha="center", va="bottom"); despine(axa)
    axb.plot(imb_centers, VG_I, "-o", color=ACCENT_DARK, ms=5, lw=1.6)
    axb.set_xlabel(r"order-book imbalance $I$ (bin midpoint)")
    axb.set_ylabel(r"$\mathrm{Var}(G)$ (trimmed)")
    axb.set_title(rf"(B) Tail knob by imbalance (Spearman$(|I|,\mathrm{{Var}}\,G)={sp_I:+.2f}$)")
    despine(axb)
    fig2.suptitle(r"Is the fat-tail thickness $\mathrm{Var}(G)$ itself a function of the book state $(I,S)$?",
                  y=1.0)
    finish(fig2, HERE / "output" / "figures" / "sde_tailknob_state.png")
    print("\n[ghsim] wrote figures/sde_ghsim_tail.png, figures/sde_tailknob_state.png, tables/sde_ghsim_summary.csv")


if __name__ == "__main__":
    main()
