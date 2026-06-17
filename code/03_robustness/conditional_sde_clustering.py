#!/usr/bin/env python3
"""
conditional_sde_clustering.py — does book-state conditioning reduce VaR violation clustering?

Three VaR models compared on Christoffersen LR_ind (independence of violations):

  (1) Flat Gaussian   : constant per-instrument scale sigma_flat = sqrt(mean(dx^2)),
                        Gaussian quantile.  No book-state conditioning.
  (2) State Gaussian  : a_xx(I,S) state-dependent scale, Gaussian quantile.
                        Book-state scale, wrong tail shape.
  (3) State GH        : a_xx(I,S) state-dependent scale, GH quantile.
                        Book-state scale + correct tail shape.

Two-step decomposition:
  State Gaussian vs Flat Gaussian  =>  book-state scale effect on clustering
  State GH       vs State Gaussian =>  tail-law effect on clustering

For each model: Christoffersen LR_ind and p_ind per (instrument, alpha).
Figure: two scatter panels of p_ind (y-axis model vs x-axis baseline),
        points above diagonal = improvement (less evidence of clustering).
Decision: report State Gaussian vs Flat in manuscript if clear majority of
          instruments show improvement; otherwise frame as tail-law result only.

Output: paper/paper/figures/fig_clustering.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUTF = CODE / "paper" / "figures"
OUTF.mkdir(parents=True, exist_ok=True)

N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_CELL = 50
ALPHAS = [0.01, 0.05]
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# data helpers — identical to sibling scripts
# ---------------------------------------------------------------------------

def parse_date(s):
    mm, dd, yy = s.split("-")
    return (int(yy), int(mm), int(dd))


def session_increments(path):
    l1 = pd.read_parquet(path)
    mid = l1["mid"].to_numpy(float)
    bid = l1["bid"].to_numpy(float)
    ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float)
    asz = l1["ask_sz"].to_numpy(float)
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz)
    if good.sum() < 200:
        return None
    x = np.log(mid)
    dx = np.diff(x)
    I = (bsz - asz) / (bsz + asz)
    S = (ask - bid) / mid
    df = pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1]}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    return df if len(df) >= 50 else None


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


# ---------------------------------------------------------------------------
# Christoffersen LR_ind (session-reset, same as var_backtest)
# ---------------------------------------------------------------------------

def lr_ind(viol_sequences):
    n00 = n01 = n10 = n11 = 0
    for seq in viol_sequences:
        if len(seq) < 2:
            continue
        v0, v1 = seq[:-1], seq[1:]
        n00 += int(((v0 == 0) & (v1 == 0)).sum())
        n01 += int(((v0 == 0) & (v1 == 1)).sum())
        n10 += int(((v0 == 1) & (v1 == 0)).sum())
        n11 += int(((v0 == 1) & (v1 == 1)).sum())
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return np.nan, np.nan
    pi01 = n01 / n0
    pi11 = n11 / n1
    pi2  = (n01 + n11) / (n00 + n01 + n10 + n11)
    if any(v <= 0 or v >= 1 for v in [pi01, pi11, pi2]):
        return np.nan, np.nan
    log_a = (n00 * np.log(1 - pi2) + n01 * np.log(pi2)
             + n10 * np.log(1 - pi2) + n11 * np.log(pi2))
    log_b = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
             + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    lr = float(-2.0 * (log_a - log_b))
    return lr, float(stats.chi2.sf(lr, df=1))


# ---------------------------------------------------------------------------
# per-instrument backtest — all three models
# ---------------------------------------------------------------------------

def backtest_instrument(sym, paths, alpha):
    paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
    ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
    tr_paths, te_paths = paths[:ntr], paths[ntr:]
    if not te_paths:
        return None

    # train data
    tr_frames = [session_increments(p) for p in tr_paths]
    tr_frames = [f for f in tr_frames if f is not None]
    if not tr_frames:
        return None
    tr = pd.concat(tr_frames, ignore_index=True)
    if len(tr) < 2000:
        return None

    # (1) Flat: per-instrument scale from training data
    sigma_flat = float(np.sqrt((tr["dx"].to_numpy() ** 2).mean()))

    # (2)+(3) State surface
    s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
    ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
    dxtr = tr["dx"].to_numpy()
    bx, axx = {}, {}
    for c in range(N_I * M_S):
        m = ctr == c
        if m.sum() >= MIN_CELL:
            bx[c]  = float(dxtr[m].mean())
            axx[c] = float((dxtr[m] ** 2).mean())

    # GH fit on train standardised residuals
    z_tr = np.array(
        [(dxtr[i] - bx[c]) / np.sqrt(axx[c])
         for i, c in enumerate(ctr) if c in axx and axx[c] > 0],
        dtype=float)
    z_tr = z_tr[np.isfinite(z_tr)]
    if len(z_tr) < 500:
        return None
    zz = z_tr if len(z_tr) <= 150_000 else RNG.choice(z_tr, 150_000, replace=False)
    try:
        p_, a_, _b, _loc, sc_ = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
        gh = stats.genhyperbolic(p_, a_, 0.0, 0.0, sc_)
        sd_gh = float(np.sqrt(gh.var()))
        q_gh = float(gh.ppf(1 - alpha / 2) / sd_gh)
    except Exception:
        return None

    q_gauss = float(stats.norm.ppf(1 - alpha / 2))

    # test sessions
    flat_seqs, state_g_seqs, state_gh_seqs = [], [], []

    for p in te_paths:
        df = session_increments(p)
        if df is None or len(df) < 20:
            continue
        cte = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
        dx  = df["dx"].to_numpy()

        flat_s = state_g_s = state_gh_s = []
        flat_s, state_g_s, state_gh_s = [], [], []
        for i, c in enumerate(cte):
            # flat: use overall sigma_flat, no drift (or zero drift)
            resid_flat = abs(dx[i])
            flat_s.append(1 if resid_flat > sigma_flat * q_gauss else 0)

            # state models: use b_x(c) and sqrt(a_xx(c))
            if c not in axx or axx[c] <= 0:
                # fallback to flat for this observation
                state_g_s.append(1 if resid_flat > sigma_flat * q_gauss else 0)
                state_gh_s.append(1 if resid_flat > sigma_flat * q_gh   else 0)
            else:
                sigma_c = np.sqrt(axx[c])
                resid_c = abs(dx[i] - bx[c])
                state_g_s.append(1 if resid_c > sigma_c * q_gauss else 0)
                state_gh_s.append(1 if resid_c > sigma_c * q_gh   else 0)

        if flat_s:
            flat_seqs.append(np.array(flat_s, dtype=int))
            state_g_seqs.append(np.array(state_g_s, dtype=int))
            state_gh_seqs.append(np.array(state_gh_s, dtype=int))

    if not flat_seqs:
        return None

    def ind_result(seqs):
        lr, p = lr_ind(seqs)
        return {"LR_ind": lr, "p_ind": p}

    return {
        "sym": sym, "alpha": alpha,
        "flat_g":    ind_result(flat_seqs),
        "state_g":   ind_result(state_g_seqs),
        "state_gh":  ind_result(state_gh_seqs),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files: dict[str, list] = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    results = []
    for sym, paths in sorted(files.items()):
        for alpha in ALPHAS:
            r = backtest_instrument(sym, paths, alpha)
            if r is not None:
                results.append(r)

    if not results:
        print("No results — check data path.")
        return

    # --- summary by alpha ---
    print("\n=== Christoffersen LR_ind clustering comparison ===")
    print("Higher p_ind = less evidence of violation clustering\n")

    for alpha in ALPHAS:
        sub = [r for r in results if r["alpha"] == alpha]
        N = len(sub)

        def med_p(model):
            vals = [r[model]["p_ind"] for r in sub if np.isfinite(r[model].get("p_ind", np.nan))]
            return np.median(vals) if vals else np.nan

        def n_improve(model_new, model_base):
            cnt = sum(
                1 for r in sub
                if np.isfinite(r[model_new].get("p_ind", np.nan))
                and np.isfinite(r[model_base].get("p_ind", np.nan))
                and r[model_new]["p_ind"] > r[model_base]["p_ind"]
            )
            return cnt

        fg_med  = med_p("flat_g")
        sg_med  = med_p("state_g")
        sgh_med = med_p("state_gh")
        n_sg_beats_fg  = n_improve("state_g",  "flat_g")
        n_sgh_beats_sg = n_improve("state_gh", "state_g")

        print(f"alpha={alpha}  (N={N} instruments)")
        print(f"  Median p_ind:")
        print(f"    Flat Gaussian  : {fg_med:.3f}")
        print(f"    State Gaussian : {sg_med:.3f}  (book-state scale effect)")
        print(f"    State GH       : {sgh_med:.3f}  (tail-law effect on top)")
        print(f"  State Gaussian > Flat Gaussian  (improvement): {n_sg_beats_fg}/{N} instruments")
        print(f"  State GH > State Gaussian        (improvement): {n_sgh_beats_sg}/{N} instruments")

        majority = N * 2 // 3
        if n_sg_beats_fg >= majority:
            print(f"  => BOOK-STATE SCALE reduces clustering in clear majority ({n_sg_beats_fg}/{N})")
        elif n_sg_beats_fg > N // 2:
            print(f"  => Book-state scale reduces clustering in majority ({n_sg_beats_fg}/{N}), but not 2/3")
        else:
            print(f"  => Book-state scale does NOT clearly reduce clustering ({n_sg_beats_fg}/{N})")
        print()

    # --- per-instrument table ---
    print(f"{'sym':8s}  {'alpha':5s}  {'p_flat_g':>9s}  {'p_state_g':>10s}  {'p_state_gh':>11s}  "
          f"{'SG>FG':>6s}  {'SGH>SG':>7s}")
    for r in results:
        fg  = r["flat_g"]["p_ind"]
        sg  = r["state_g"]["p_ind"]
        sgh = r["state_gh"]["p_ind"]
        sg_beats  = "yes" if (np.isfinite(sg)  and np.isfinite(fg) and sg  > fg) else "no"
        sgh_beats = "yes" if (np.isfinite(sgh) and np.isfinite(sg) and sgh > sg) else "no"
        print(f"{r['sym']:8s}  {r['alpha']:.2f}  "
              f"{fg:>9.3f}  {sg:>10.3f}  {sgh:>11.3f}  "
              f"{sg_beats:>6s}  {sgh_beats:>7s}")

    # --- figure ---
    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, WARN, MUTED, POS
    plt = setup_mpl()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
    marker_by_alpha = {0.01: "o", 0.05: "s"}
    color_by_alpha  = {0.01: ACCENT, 0.05: ACCENT_DARK}

    for alpha in ALPHAS:
        sub = [r for r in results if r["alpha"] == alpha
               and np.isfinite(r["flat_g"]["p_ind"])
               and np.isfinite(r["state_g"]["p_ind"])
               and np.isfinite(r["state_gh"]["p_ind"])]
        x_fg  = [r["flat_g"]["p_ind"]   for r in sub]
        y_sg  = [r["state_g"]["p_ind"]  for r in sub]
        y_sgh = [r["state_gh"]["p_ind"] for r in sub]

        ax0.scatter(x_fg, y_sg, s=28, marker=marker_by_alpha[alpha],
                    color=color_by_alpha[alpha], alpha=0.75,
                    label=rf"$\alpha={alpha}$")
        ax1.scatter(y_sg, y_sgh, s=28, marker=marker_by_alpha[alpha],
                    color=color_by_alpha[alpha], alpha=0.75,
                    label=rf"$\alpha={alpha}$")

    for ax in (ax0, ax1):
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls="--", alpha=0.5)
        ax.axhline(0.05, color=MUTED, lw=0.7, ls=":")
        ax.axvline(0.05, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
        despine(ax)

    ax0.set_xlabel(r"Christoffersen $p_{\mathrm{ind}}$: Flat Gaussian (no conditioning)")
    ax0.set_ylabel(r"Christoffersen $p_{\mathrm{ind}}$: State Gaussian $a_{xx}(I,S)$")
    ax0.set_title(r"(a) Book-state scale effect on violation clustering")
    ax0.text(0.05, 0.92, "above diagonal = state scale\nreduces clustering",
             transform=ax0.transAxes, fontsize=7, color=INK,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, lw=0.5, alpha=0.85))

    ax1.set_xlabel(r"Christoffersen $p_{\mathrm{ind}}$: State Gaussian")
    ax1.set_ylabel(r"Christoffersen $p_{\mathrm{ind}}$: State GH $a_{xx}(I,S)$")
    ax1.set_title(r"(b) Tail-law effect on violation clustering")
    ax1.text(0.05, 0.92, "above diagonal = GH tail\nreduces clustering",
             transform=ax1.transAxes, fontsize=7, color=INK,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, lw=0.5, alpha=0.85))

    fig.suptitle(
        r"Violation clustering: Christoffersen $p_{\mathrm{ind}}$ across three VaR models "
        r"(higher = less clustering evidence). Points above diagonal = improvement.",
        y=1.02
    )
    finish(fig, OUTF / "fig_clustering.png")
    print("\n[clustering] wrote paper/figures/fig_clustering.png")


if __name__ == "__main__":
    main()
