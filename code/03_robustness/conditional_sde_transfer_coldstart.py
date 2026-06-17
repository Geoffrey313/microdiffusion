#!/usr/bin/env python3
"""
conditional_sde_transfer_coldstart.py — cross-instrument transfer of the diffusion
surface shape after a one-scalar target volatility calibration.

Leave-one-instrument-out. For each target j:
  - donor shape      : average of the OTHER instruments' a_xx surfaces, each normalised to unit mean over
                       cells (so only the SHAPE over (I,S) is borrowed, not the level);
  - transferred a_xx : donor_shape(I,S) * A_j, where A_j is j's overall variance (a single scalar) -- so
                       the entire state-conditional surface for j is borrowed from other instruments, knowing
                       only j's overall volatility;
  - own a_xx         : j's own trained surface (the within-instrument benchmark);
  - constant         : A_j for every observation -- the one-scalar variance floor.
We compare the out-of-sample Gaussian predictive log-likelihood of transferred / own / constant, and the
cross-instrument transfer rank correlation (donor surface vs target realised cell variance).
This is a minimal target-calibration test, not a zero-data listing-day forecast.

Outputs: tables/sde_transfer_coldstart.csv ; figures/fig_transfer_coldstart.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUTF = CODE / "paper" / "figures"; OUTF.mkdir(parents=True, exist_ok=True)
N_I, M_S = 10, 3
NCELL = N_I * M_S
TRAIN_FRAC = 0.60
MIN_CELL = 50
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


def gll(r2, s2):
    s2 = np.maximum(s2, 1e-12)
    return float(np.mean(-0.5 * (np.log(2 * np.pi) + np.log(s2) + r2 / s2)))


def main():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    inst = {}                  # per instrument: own surface, A_j, test (r2, cell), realised test cell var
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
        r2tr = (tr["dx"].to_numpy() * BPS) ** 2
        own = np.full(NCELL, np.nan)
        for c in range(NCELL):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                own[c] = r2tr[m].mean()
        if np.isfinite(own).sum() < 5:
            continue
        A = float(np.nanmean(r2tr))                      # overall variance scalar (bps^2)
        r2te = (te["dx"].to_numpy() * BPS) ** 2
        # realised test per-cell variance (for the transfer correlation)
        realc = np.full(NCELL, np.nan)
        for c in range(NCELL):
            m = cte == c
            if m.sum() >= MIN_CELL:
                realc[c] = r2te[m].mean()
        inst[sym] = {"own": own, "A": A, "cte": cte, "r2te": r2te, "realc": realc}
        print(f"  loaded {sym:6s} cells={np.isfinite(own).sum():2d} A={A:.3f}")

    syms = list(inst)
    # normalised shapes (unit mean over present cells)
    norm = {s: inst[s]["own"] / np.nanmean(inst[s]["own"]) for s in syms}

    rows, pair_t, pair_r = [], [], []
    for j in syms:
        donors = [s for s in syms if s != j]
        shape = np.nanmean(np.vstack([norm[s] for s in donors]), axis=0)     # donor shape over cells
        shape = np.where(np.isfinite(shape), shape, 1.0)
        A = inst[j]["A"]; cte = inst[j]["cte"]; r2 = inst[j]["r2te"]; own = inst[j]["own"]
        keep = np.isfinite(own[cte])                                          # obs whose cell has an own estimate
        c = cte[keep]; rr = r2[keep]
        s2_trans = shape[c] * A
        s2_own = own[c]
        s2_const = np.full(len(rr), A)
        rows.append({"sym": j,
                     "ll_const": gll(rr, s2_const),
                     "ll_transfer": gll(rr, s2_trans),
                     "ll_own": gll(rr, s2_own)})
        # transfer correlation: donor shape*A vs realised test cell variance
        rc = inst[j]["realc"]
        ok = np.isfinite(rc) & np.isfinite(own)
        for cc in np.where(ok)[0]:
            pair_t.append(shape[cc] * A); pair_r.append(rc[cc])
        print(f"  LOO {j:6s} LL const={rows[-1]['ll_const']:.3f} transfer={rows[-1]['ll_transfer']:.3f} "
              f"own={rows[-1]['ll_own']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "output" / "tables" / "sde_transfer_coldstart.csv", index=False)
    pair_t = np.array(pair_t); pair_r = np.array(pair_r)
    sp = stats.spearmanr(pair_t, pair_r).correlation

    def agg(x):
        x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); return m, 1.96 * se
    print(f"\n=== Leave-one-instrument-out transfer / cold start ({len(df)} instruments) ===")
    for k in ["ll_const", "ll_transfer", "ll_own"]:
        m, e = agg(df[k]); print(f"  mean OOS Gaussian LL {k:11s} = {m:.4f} +/- {e:.4f}")
    print(f"  transfer gain (transfer - const) = {(df.ll_transfer-df.ll_const).mean():+.4f} "
          f"(positive for {100*np.mean(df.ll_transfer>df.ll_const):.0f}% of names)")
    print(f"  loss vs own  (transfer - own)    = {(df.ll_transfer-df.ll_own).mean():+.4f}")
    print(f"  cross-instrument transfer Spearman(donor surface, target realised) = {sp:.3f}")

    # ---- figure ----
    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, MUTED
    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a = ax[0]
    a.scatter(pair_t, pair_r, s=10, c=ACCENT, alpha=0.5, edgecolors="none")
    lim = [np.nanmin(pair_t), np.nanmax(pair_t)]
    a.plot(lim, lim, color=INK, ls="--", lw=1, label=r"$y=x$")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"transferred $a_{xx}$ from other instruments  [bps$^2$]")
    a.set_ylabel(r"realised held-out variance of the target  [bps$^2$]")
    a.set_title(rf"(a) The surface transfers across instruments (Spearman {sp:.2f})")
    a.legend(loc="upper left"); despine(a)

    b = ax[1]
    gain_t = (df.ll_transfer - df.ll_const).to_numpy()
    gain_o = (df.ll_own - df.ll_const).to_numpy()
    means = [0.0, gain_t.mean(), gain_o.mean()]
    errs = [0.0, agg(gain_t)[1], agg(gain_o)[1]]
    labels = ["one-scalar floor\n(constant variance)", "transferred $a_{xx}$\n(other instruments)",
              "own $a_{xx}$\n(this instrument)"]
    cols = [MUTED, ACCENT, ACCENT_DARK]
    b.bar(range(3), means, yerr=errs, color=cols, width=0.6, capsize=3)
    b.axhline(0, color=INK, lw=0.8)
    b.set_xticks(range(3)); b.set_xticklabels(labels, fontsize=8)
    b.set_ylabel(r"OOS log-likelihood gain over the one-scalar floor")
    b.set_title(r"(b) A surface from other names captures the state structure")
    despine(b)
    fig.suptitle(r"The diffusion surface is cross-sectionally portable: a shape learned from other "
                 r"instruments forecasts the target's state-dependent variance", y=1.01)
    finish(fig, OUTF / "fig_transfer_coldstart.png")
    print("[transfer] wrote fig_transfer_coldstart.png and tables/sde_transfer_coldstart.csv")


if __name__ == "__main__":
    main()
