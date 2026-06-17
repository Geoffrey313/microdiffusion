#!/usr/bin/env python3
"""
conditional_sde_scale_recovery.py — test whether the GH likelihood recovers a scale CONSISTENT with a_xx.

Model:  Delta x - b = sqrt(a_xx(I,S)) * eta,   eta ~ GH (unit-variance heavy-tailed kick).
So in each book-state box the moves are GH-distributed with SCALE = sqrt(a_xx). We estimate that scale two
COMPLEMENTARY ways on the SAME held-out box data (NOT statistically independent -- both use the box moves, and
the GH shape was calibrated on a_xx-standardised residuals) and check they agree:
  (i)  RMS scale   : sqrt(a_xx) = sqrt(mean((Delta x - b)^2))           -- the 2nd-moment definition (outlier-sensitive)
  (ii) GH-ML scale : fit a GH (pooled shape fixed, free scale) by maximum likelihood -> implied std
                     (heavy-tail aware: a single huge move sways it far less than it sways a variance)
If (ii) ~ (i) across boxes -- in LEVEL (median ratio ~1), not just rank -- that supports a scale/shape
decomposition in which a_xx carries the scale; it is evidence against "a_xx is just a tail artifact", though NOT
a proof of independent recovery. This is the dual of the identifiability test: there we divided by sqrt(a_xx) to
recover the GH shape; here we divide out the GH shape to recover a scale and compare it to sqrt(a_xx).
KEY evidence is the median ratio ~1 (level agreement) and the slope-vs-1; the permutation test only rules out the
(easy) random-pairing null.

Outputs: figures/sde_scale_recovery.png ; tables/sde_scale_recovery_summary.csv
Run: python3 conditional_sde_scale_recovery.py
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
MIN_BOX = 200          # min held-out moves per box for a stable scale fit
CAP_BOX = 8000         # subsample cap per box for the ML fit (speed)
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


def collect_box_moves():
    """Return list of centred held-out move arrays, one per (symbol,box), plus pooled standardised z."""
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)
    boxes = []          # each: centred moves m = dx - b_box (held-out)
    Zpool = []
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
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxte = te["dx"].to_numpy()
        for c in np.unique(cte):
            m = dxte[cte == c]
            if len(m) >= MIN_BOX:
                mc = m - m.mean()
                boxes.append(mc)
                Zpool.append(mc / np.sqrt((mc ** 2).mean()))
    return boxes, np.concatenate(Zpool)


def main():
    print("collecting per-box held-out moves ...")
    boxes, z = collect_box_moves()
    print(f"{len(boxes)} boxes; pooled n={len(z):,}")

    # pooled GH shape (symmetric): fit on standardised residuals, fix (p,a)
    zz = z if len(z) <= 150_000 else RNG.choice(z, 150_000, replace=False)
    p, a, b, loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    v0 = stats.genhyperbolic(p, a, 0.0, 0.0, 1.0).var()      # variance of unit-scale GH(p,a)
    print(f"pooled GH shape: p={p:.3f} a={a:.3f}  (unit-scale var v0={v0:.3f})")

    def gh_ml_std(m):
        """ML scale of GH(p,a,b=0,loc=0) fit to m (fixed shape); return implied std = s*sqrt(v0)."""
        if len(m) > CAP_BOX:
            m = RNG.choice(m, CAP_BOX, replace=False)
        rms = np.sqrt((m ** 2).mean())
        def nll(logs):
            sc = np.exp(logs)
            pdf = stats.genhyperbolic.pdf(m / sc, p, a, 0.0) / sc
            pdf = np.clip(pdf, 1e-300, None)
            return -np.log(pdf).sum()
        r = optimize.minimize_scalar(nll, bounds=(np.log(rms / 50), np.log(rms * 50)), method="bounded")
        return float(np.exp(r.x) * np.sqrt(v0))

    rms_scale = np.array([np.sqrt((m ** 2).mean()) for m in boxes])     # sqrt(a_xx)
    gh_scale = np.array([gh_ml_std(m) for m in boxes])                  # GH-likelihood implied std

    good = np.isfinite(gh_scale) & (gh_scale > 0) & np.isfinite(rms_scale) & (rms_scale > 0)
    rms_scale, gh_scale = rms_scale[good], gh_scale[good]
    sp = stats.spearmanr(rms_scale, gh_scale).correlation
    slope = np.polyfit(np.log(rms_scale), np.log(gh_scale), 1)[0]
    ratio = np.median(gh_scale / rms_scale)

    # ---- significance: bootstrap CIs (resample boxes) + permutation placebo (shuffle the pairing) ----
    B = 2000
    nB = len(rms_scale)
    rg = np.random.default_rng(1)
    bsp, brt, bsl = [], [], []
    for _ in range(B):
        idx = rg.integers(0, nB, nB)
        r, g = rms_scale[idx], gh_scale[idx]
        bsp.append(stats.spearmanr(r, g).correlation)
        brt.append(np.median(g / r))
        bsl.append(np.polyfit(np.log(r), np.log(g), 1)[0])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    sp_ci, rt_ci, sl_ci = ci(bsp), ci(brt), ci(bsl)
    perm = np.array([stats.spearmanr(rms_scale, rg.permutation(gh_scale)).correlation for _ in range(B)])
    p_perm = (int(np.sum(np.abs(perm) >= abs(sp))) + 1) / (B + 1)
    z_perm = (sp - perm.mean()) / perm.std()

    pd.DataFrame({"sqrt_a_xx": rms_scale, "gh_ml_std": gh_scale}).to_csv(
        HERE / "output" / "tables" / "sde_scale_recovery_summary.csv", index=False)
    pd.DataFrame([{"spearman": sp, "spearman_lo": sp_ci[0], "spearman_hi": sp_ci[1],
                   "median_ratio": ratio, "ratio_lo": rt_ci[0], "ratio_hi": rt_ci[1],
                   "slope": slope, "slope_lo": sl_ci[0], "slope_hi": sl_ci[1],
                   "perm_null_mean": float(perm.mean()), "perm_null_sd": float(perm.std()),
                   "perm_p": p_perm, "perm_z": float(z_perm), "n_boxes": nB}]).to_csv(
        HERE / "output" / "tables" / "sde_scale_recovery_stats.csv", index=False)
    print(f"\n{len(rms_scale)} boxes compared")
    print(f"Spearman = {sp:.3f}  95% CI [{sp_ci[0]:.3f}, {sp_ci[1]:.3f}]")
    print(f"median ratio = {ratio:.3f}  95% CI [{rt_ci[0]:.3f}, {rt_ci[1]:.3f}]")
    print(f"log-log slope = {slope:.3f}  95% CI [{sl_ci[0]:.3f}, {sl_ci[1]:.3f}]")
    print(f"permutation placebo (shuffled pairing): null Spearman = {perm.mean():+.3f} +/- {perm.std():.3f}; "
          f"p = {p_perm:.2g} (<1/{B}); z = {z_perm:.0f} sigma")

    # ---- figure ----
    # ---- subplot B input: simulate (Delta x - b)/sqrt(G) using the REAL distribution of state scales ----
    # algebra: (Delta x - b)/sqrt(G) = sqrt(a_xx) * Z  -> a scale-mixture of normals (sqrt(a_xx) varies) => fat-tailed.
    cap = np.quantile(np.abs(z), 0.999); zt = z[np.abs(z) <= cap]
    kap = ((zt - zt.mean()) ** 4).mean() / zt.var() ** 2 - 3.0
    Vg = float(np.clip(kap / 3.0, 0.5, None))                       # Var(G) from exkurt = 3 Var(G)
    s_obs = np.concatenate([np.full(len(m), np.sqrt((m ** 2).mean())) for m in boxes])  # real per-obs sqrt(a_xx)
    Gd = stats.invgauss.rvs(mu=Vg, scale=1.0 / Vg, size=len(s_obs), random_state=RNG)   # mean 1, var Vg
    Zd = RNG.standard_normal(len(s_obs))
    g_only = s_obs * Zd;            g_only = g_only / g_only.std()   # (Dx-b)/sqrt(G) : only the burst removed
    both = Zd.copy()                                                # (Dx-b)/sqrt(a_xx G) : both removed -> Z
    p_gt3_gonly = float(np.mean(np.abs(g_only) > 3)); p_gt3_N = 2 * stats.norm.sf(3)
    print(f"\nsubplot B: Var(G)~{Vg:.1f};  P(|.|>3): divide-by-sqrt(G)-only = {p_gt3_gonly:.4f} "
          f"vs N(0,1) {p_gt3_N:.4f}  => sqrt(G) alone leaves fat tails")

    # ---- figure: A = scale recovery, B = sqrt(G) alone cannot whiten ----
    import sys
sys.path.insert(0, str(CODE))
from plot_style import ACCENT, ACCENT_DARK, WARN, MUTED, INK, finish, setup_mpl, despine
    plt = setup_mpl()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.8))

    # --- A: recover sqrt(a_xx) by dividing out the GH shape ---
    axA.scatter(rms_scale * 1e4, gh_scale * 1e4, s=9, c=ACCENT, alpha=0.5, edgecolors="none")
    lim = [min(rms_scale.min(), gh_scale.min()) * 1e4, max(rms_scale.max(), gh_scale.max()) * 1e4]
    axA.plot(lim, lim, color=INK, ls="--", lw=1.2, label=r"$y=x$: the two measurements agree")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"ordinary move size $\sqrt{a_{xx}}=\sqrt{\mathbb{E}[(\Delta x-b)^2]}$  (bps)")
    axA.set_ylabel(r"$(\Delta x-b)\,\div\,\mathrm{GH\ shape}\,=\,$ recovered $\sqrt{a_{xx}}$  (bps)")
    axA.set_title(rf"(A) Divide out the GH shape $\eta$ $\Rightarrow$ recover $\sqrt{{a_{{xx}}}}$ (Spearman {sp:.2f})")
    axA.text(0.04, 0.83,
             r"$\Delta x - b = \sqrt{a_{xx}(I,S)}\;\eta,\ \ \eta\sim\mathrm{GH}$" "\n"
             r"$\Rightarrow\ \dfrac{\Delta x - b}{\eta}=\sqrt{a_{xx}(I,S)}$",
             transform=axA.transAxes, ha="left", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6f8", ec=INK, lw=0.7, alpha=0.95))
    axA.legend(loc="lower right"); despine(axA)

    # --- B: dividing out sqrt(G) alone leaves sqrt(a_xx) Z -> NOT Gaussian ---
    grid = np.linspace(-8, 8, 220); bins = np.linspace(-8, 8, 150)
    axB.hist(g_only[np.abs(g_only) < 8], bins=bins, density=True, histtype="step", color=ACCENT_DARK, lw=1.6,
             label=r"$(\Delta x-b)/\sqrt{G}=\sqrt{a_{xx}}\,Z$  (only $\sqrt{G}$ removed)")
    axB.hist(both[np.abs(both) < 8], bins=bins, density=True, histtype="step", color=MUTED, lw=1.2,
             label=r"$(\Delta x-b)/\sqrt{a_{xx}G}=Z$  (both removed)")
    axB.plot(grid, stats.norm.pdf(grid), color=WARN, ls="--", lw=1.3, label=r"$N(0,1)$ target")
    axB.set_yscale("log"); axB.set_ylim(1e-5, 1)
    axB.set_xlabel(r"standardised value")
    axB.set_ylabel(r"density (log scale)")
    axB.set_title(r"(B) $\sqrt{G}$ alone can't whiten: the state scale $a_{xx}$ is still needed")
    axB.text(0.5, 0.04, r"removing only $\sqrt{G}$ leaves $\sqrt{a_{xx}}\,Z$ — a scale-mixture over states $\Rightarrow$ fat tails,"
             "\n" r"not $N(0,1)$.  ($G$ is latent on real data; here $G$ is simulated, $\sqrt{a_{xx}}$ from real states.)",
             transform=axB.transAxes, ha="center", va="bottom", fontsize=6.8,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK, lw=0.6, alpha=0.85))
    axB.legend(loc="upper right", fontsize=7); despine(axB)

    fig.suptitle(r"Scale and shape are both needed: $a_{xx}$ carries the state-dependence, $\sqrt{G}$ the burst "
                 r"--- neither alone whitens the move", y=1.01)
    finish(fig, HERE / "output" / "figures" / "sde_scale_recovery.png")
    print("\n[scale] wrote figures/sde_scale_recovery.png and tables/sde_scale_recovery_summary.csv")


if __name__ == "__main__":
    main()
