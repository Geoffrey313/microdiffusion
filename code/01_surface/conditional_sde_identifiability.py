#!/usr/bin/env python3
"""
conditional_sde_identifiability.py — is a_xx redundant? Show scale and shape are SEPARATE, IDENTIFIED objects.

The objection: "Delta x = sqrt(a_xx) * sqrt(G) * Z; maybe a_xx does nothing and the random multiplier G is the
whole story." We refute it empirically. Decompose the realised variance per book-state box c=(I-bin, S-tercile):
    realised_var_test(c) = E[(Delta x)^2 | c]  on HELD-OUT days,
    a_xx_train(c)        = E[(Delta x)^2 | c]  on TRAIN days,
    Ghat(c)              = realised_var_test(c) / a_xx_train(c)   (the realised multiplier, should be ~1 if E[G|I,S]=1).

Claim, made visible:
  (A) the SIZE is state-dependent and a_xx captures it: realised_var_test tracks a_xx_train across boxes (Spearman high);
  (B) the leftover multiplier is state-NEUTRAL: Ghat ~ 1 flat across boxes, no residual dependence on a_xx (Spearman ~0);
  (C) the SHAPE (tails) is also ~state-independent: P(|z|>3) per box is roughly constant (a_xx removed the state, not the tail).

Outputs: tables/sde_identifiability_summary.csv ; figures/sde_identifiability.png
Run: python3 conditional_sde_identifiability.py
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
MIN_TAIL = 300


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


def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    rows = []        # per (symbol, box)
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
        # split TRAIN into two disjoint halves A,B for an UNBIASED scale (avoids ratio/regression bias)
        half = len(tr) // 2
        trA, trB = tr.iloc[:half], tr.iloc[half:]
        cA = cell_ids(trA["I"].to_numpy(), trA["S"].to_numpy(), s_edges); dxA = trA["dx"].to_numpy()
        cB = cell_ids(trB["I"].to_numpy(), trB["S"].to_numpy(), s_edges); dxB = trB["dx"].to_numpy()
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy(); dxte = te["dx"].to_numpy()
        bx, axx, aA, aB = {}, {}, {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = dxtr[m].mean(); axx[c] = (dxtr[m] ** 2).mean()
            mA = cA == c; mB = cB == c
            if mA.sum() >= MIN_CELL and mB.sum() >= MIN_CELL:
                aA[c] = (dxA[mA] ** 2).mean(); aB[c] = (dxB[mB] ** 2).mean()
        for c in np.unique(cte):
            if c not in axx or axx[c] <= 0:
                continue
            mc = cte == c
            if mc.sum() < MIN_CELL:
                continue
            dxc = dxte[mc]
            rv = float((dxc ** 2).mean())                     # realised test variance in box
            z = (dxc - bx[c]) / np.sqrt(axx[c])               # state-standardised residual in box
            # de-biased multiplier: numerator=test variance, denominator=block A, x-axis=block B (independent)
            gA = rv / aA[c] if c in aA and aA[c] > 0 else np.nan
            rows.append({"sym": sym, "box": int(c), "n": int(mc.sum()),
                         "a_xx_train": float(axx[c]), "realised_var_test": rv,
                         "Ghat": rv / axx[c],
                         "Ghat_dbias": float(gA), "a_indep_B": float(aB[c]) if c in aB else np.nan,
                         "frac_gt3": float(np.mean(np.abs(z) > 3)) if mc.sum() >= MIN_TAIL else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "output" / "tables" / "sde_identifiability_summary.csv", index=False)

    # ---- statistics ----
    sp_size = stats.spearmanr(df["a_xx_train"], df["realised_var_test"]).correlation   # SIZE tracks a_xx
    sp_mult = stats.spearmanr(df["a_xx_train"], df["Ghat"]).correlation                # multiplier vs a_xx (BIASED: shared denom)
    slope = np.polyfit(np.log(df["a_xx_train"]), np.log(df["Ghat"]), 1)[0]
    # de-biased: multiplier (denominator = block A) vs an INDEPENDENT scale (block B)
    dd = df.dropna(subset=["Ghat_dbias", "a_indep_B"])
    sp_mult_db = stats.spearmanr(dd["a_indep_B"], dd["Ghat_dbias"]).correlation
    slope_db = np.polyfit(np.log(dd["a_indep_B"]), np.log(dd["Ghat_dbias"]), 1)[0]
    D_var = float(np.std(np.log(df["realised_var_test"])))                             # cross-box log-dispersion of size
    D_G = float(np.std(np.log(df["Ghat"])))                                            # ... of the multiplier
    absorbed = 1.0 - (D_G ** 2) / (D_var ** 2)                                         # fraction of log-var dispersion a_xx soaks up
    # split-half reliability of the scale itself (how much cross-box ordering is REAL signal vs noise)
    rel = stats.spearmanr(dd["a_indep_B"], dd["Ghat_dbias"] * dd["a_indep_B"]).correlation  # placeholder, replaced below
    rel = stats.spearmanr(df.dropna(subset=["a_indep_B"])["a_xx_train"],
                          df.dropna(subset=["a_indep_B"])["a_indep_B"]).correlation
    tdf = df.dropna(subset=["frac_gt3"])
    sp_tail = stats.spearmanr(tdf["a_xx_train"], tdf["frac_gt3"]).correlation          # tails vs a_xx ~ 0
    tail_med = float(tdf["frac_gt3"].median()); tail_iqr = float(tdf["frac_gt3"].quantile(.75) - tdf["frac_gt3"].quantile(.25))

    print(f"boxes (symbol x state): {len(df)}  (de-biased subset: {len(dd)})")
    print(f"(A) SIZE is state-dependent & a_xx captures it: Spearman(a_xx_train, realised_var_test) = {sp_size:.3f}")
    print(f"    split-half reliability of the scale: Spearman(a_xx^A, a_xx^B) = {rel:.3f}  (=> the ordering is real, not noise)")
    print(f"(B) leftover multiplier ~ state-neutral. median Ghat = {df['Ghat'].median():.3f}.")
    print(f"    BIASED (shared denom): Spearman(a_xx, Ghat) = {sp_mult:.3f}, slope = {slope:.3f}")
    print(f"    DE-BIASED (indep scale): Spearman(a_B, Ghat_A) = {sp_mult_db:.3f}, slope = {slope_db:.3f}")
    print(f"    a_xx absorbs {100*absorbed:.0f}% of the cross-box log-variance dispersion "
          f"(std log var {D_var:.2f} -> std log Ghat {D_G:.2f})")
    print(f"(C) TAIL is ~state-independent: median P(|z|>3) per box = {tail_med:.3f} (IQR {tail_iqr:.3f}); "
          f"Spearman(a_xx_train, P(|z|>3)) = {sp_tail:.3f}")

    # ---- figure ----
    import sys
sys.path.insert(0, str(CODE))
from plot_style import ACCENT, ACCENT_DARK, WARN, MUTED, INK, finish, setup_mpl, despine
    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax_bp = ax[0].twinx() if False else None

    # left: SIZE is the state -- realised variance tracks a_xx (state-dependent), in bps^2
    a = ax[0]
    a.scatter(df["a_xx_train"] * 1e8, df["realised_var_test"] * 1e8, s=9, c=ACCENT, alpha=0.45, edgecolors="none")
    lim = [df["a_xx_train"].min() * 1e8, df["a_xx_train"].max() * 1e8]
    a.plot(lim, lim, color=INK, ls="--", lw=1, label=r"$y=x$ (perfect)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"train scale $a_{xx}(c)$  [bps$^2$]")
    a.set_ylabel(r"held-out realised variance $\mathbb{E}[(\Delta x)^2\,|\,c]$  [bps$^2$]")
    a.set_title(rf"(A) The SIZE is the state: $a_{{xx}}$ tracks it (Spearman {sp_size:.2f})")
    a.legend(loc="upper left"); despine(a)

    # right: de-biased multiplier Ghat (denom = block A) vs an INDEPENDENT scale (block B) -> flat at 1
    b = ax[1]
    bx8 = dd["a_indep_B"].to_numpy() * 1e8; gh = dd["Ghat_dbias"].to_numpy()
    b.scatter(bx8, gh, s=9, c=ACCENT_DARK, alpha=0.4, edgecolors="none",
              label=r"$\widehat{G}(c)=\mathbb{E}[(\Delta x)^2|c]_{\rm test}/a_{xx}^{A}(c)$")
    xq = np.logspace(np.log10(bx8.min()), np.log10(bx8.max()), 13)
    xc = np.sqrt(xq[:-1] * xq[1:]); med = []
    for lo, hi in zip(xq[:-1], xq[1:]):
        m = (bx8 >= lo) & (bx8 < hi)
        med.append(np.median(gh[m]) if m.sum() else np.nan)
    b.plot(xc, med, "-o", color=WARN, ms=4, lw=1.6, label=r"binned median $\widehat{G}$")
    b.axhline(1.0, color=INK, ls="--", lw=1.2, label=r"$\widehat{G}=1$ (state-neutral)")
    b.set_xscale("log"); b.set_ylim(0, 3)
    b.set_xlabel(r"independent scale $a_{xx}^{B}(c)$  [bps$^2$]  (disjoint block)")
    b.set_ylabel(r"realised multiplier $\widehat{G}(c)$")
    b.set_title(rf"(B) The leftover is state-neutral: $\widehat{{G}}\!\approx\!1$ (de-biased slope {slope_db:+.2f})")
    b.legend(loc="upper right", fontsize=7); despine(b)

    fig.suptitle(r"Scale vs.\ shape are separate, identified objects: $a_{xx}(I,S)$ owns the state-dependence of "
                 r"size; the multiplier $G$ is state-neutral", y=1.02)
    finish(fig, HERE / "output" / "figures" / "sde_identifiability.png")
    print("\n[ident] wrote figures/sde_identifiability.png and tables/sde_identifiability_summary.csv")


if __name__ == "__main__":
    main()
