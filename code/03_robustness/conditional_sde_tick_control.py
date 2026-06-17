#!/usr/bin/env python3
"""
conditional_sde_tick_control.py — T1, the GATING test of the revision plan.

Referee concern #1: the diffusion surface a_xx(I,S) may be a restatement of the tick grid
rather than a new structural object, because in a coarse-tick market a wider quoted spread
mechanically permits larger mid-price moves. The spread axis carries most of the raw
transferable structure (spread-only transfer 0.77), so the surface could be "spread in
ticks" in disguise.

This script controls for that mechanically:
  1. recover the tick size per instrument from the exact decimal quote grid;
  2. express moves in ticks      : dx_tick = (M_{t+1} - M_t) / tick;
     express spread in ticks     : S_tick  = round((P^a - P^b) / tick);
  3. STRATIFY by spread-in-ticks (1, 2, 3, >=4) -- i.e. hold the tick mechanics fixed;
  4. WITHIN each spread-tick stratum ask whether IMBALANCE still moves the conditional
     variance a_xx, testing BOTH a signed monotone effect AND a symmetric (U-shaped)
     effect in |I| -- the diffusion surface, unlike the signed microprice drift, can rise
     for extreme imbalance of EITHER sign;
  5. test whether the imbalance-conditional surface TRANSFERS train->test WITHIN strata
     (the tick-controlled analogue of the headline 0.54 transfer).

Decision (printed at the end):
  Branch A  strong  : imbalance (signed or symmetric) explains within-stratum variance AND
                      transfers within strata -> full structural claim stands.
  Branch C  partial : spread dominates but a residual imbalance/interaction survives within
                      strata -> structural claim stands with softer wording.
  Branch B  none    : nothing survives the spread-tick control -> narrow to spread-scaled
                      risk bands.

Outputs: output/tables/sde_tick_control.csv, output/tables/sde_tick_control_summary.csv,
         ../paper/figures/fig_tick_control.png
Run: python3 conditional_sde_tick_control.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[1]
OUTT = HERE / "output" / "tables"; OUTT.mkdir(parents=True, exist_ok=True)
OUTF = CODE / "paper" / "figures"; OUTF.mkdir(parents=True, exist_ok=True)

N_I = 10                      # imbalance bins
TRAIN_FRAC = 0.60
MIN_CELL = 50                 # min obs per (stratum, imbalance-bin) to estimate a_xx
SPREAD_TICKS = [1, 2, 3, 4]   # 4 means ">=4 ticks"
I_EDGES = np.linspace(-1, 1, N_I + 1)
I_CENT = 0.5 * (I_EDGES[:-1] + I_EDGES[1:])     # bin centres in [-0.9, .. 0.9]


def parse_date(s):
    mm, dd, yy = s.split("-"); return (int(yy), int(mm), int(dd))


def recover_tick(price_strings):
    """Exact tick from the decimal quote grid: GCD of integer-scaled distinct prices."""
    s = pd.Series(price_strings).dropna().astype(str)
    s = s[s.str.match(r"^\d+\.?\d*$")]
    if len(s) < 10:
        return None
    dec = s.str.split(".").apply(lambda p: len(p[1]) if len(p) > 1 else 0).max()
    scale = 10 ** int(dec)
    ints = np.unique(np.round(s.astype(float).to_numpy() * scale).astype(np.int64))
    if len(ints) < 2:
        return None
    diffs = np.diff(ints)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    g = int(np.gcd.reduce(diffs))
    return g / scale if g > 0 else None


def tick_increments(l1, tick):
    """Forward one-step frame in TICK units: dx_tick paired with the pre-move state."""
    mid = l1["mid"].to_numpy(float); bid = l1["bid"].to_numpy(float); ask = l1["ask"].to_numpy(float)
    bsz = l1["bid_sz"].to_numpy(float); asz = l1["ask_sz"].to_numpy(float)
    good = np.isfinite(mid) & (mid > 0) & np.isfinite(bsz) & np.isfinite(asz) & np.isfinite(bid) & np.isfinite(ask)
    if good.sum() < 200:
        return None
    dxk = np.diff(mid) / tick                          # mid move in ticks (forward)
    I = (bsz - asz) / (bsz + asz)
    Sk = np.round((ask - bid) / tick).astype(float)    # spread in (integer) ticks
    df = pd.DataFrame({"dxk": dxk, "I": I[:-1], "Stick": Sk[:-1]})
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def spread_stratum(stick):
    """Map integer spread-in-ticks to a stratum index 0..3 (last is >=4)."""
    s = np.clip(stick.astype(int), 1, 4)
    return s - 1


def load_instruments():
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)
    out = {}
    for sym, paths in sorted(files.items()):
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr_raw = [pd.read_parquet(p) for p in paths[:ntr]]
        te_raw = [pd.read_parquet(p) for p in paths[ntr:]]
        if not tr_raw or not te_raw:
            continue
        # tick recovered on TRAINING quotes only
        allpx = pd.concat([d["bid"] for d in tr_raw] + [d["ask"] for d in tr_raw], ignore_index=True)
        tick = recover_tick(allpx)
        if tick is None or tick <= 0:
            continue
        tr = [tick_increments(d, tick) for d in tr_raw]
        te = [tick_increments(d, tick) for d in te_raw]
        tr = pd.concat([t for t in tr if t is not None], ignore_index=True) if any(t is not None for t in tr) else None
        te = pd.concat([t for t in te if t is not None], ignore_index=True) if any(t is not None for t in te) else None
        if tr is None or te is None or len(tr) < 3000 or len(te) < 1000:
            continue
        out[sym] = {"tick": tick, "tr": tr, "te": te}
    return out


def axx_by_ibin(df, strat):
    """Per imbalance-bin a_xx = mean(dx_tick^2) within one spread-tick stratum; NaN if < MIN_CELL."""
    sub = df[spread_stratum(df["Stick"].to_numpy()) == strat]
    if len(sub) < MIN_CELL:
        return None, None
    ib = np.clip(np.digitize(sub["I"].to_numpy(), I_EDGES) - 1, 0, N_I - 1)
    dxk2 = sub["dxk"].to_numpy() ** 2
    a = np.full(N_I, np.nan); n = np.zeros(N_I, int)
    for b in range(N_I):
        m = ib == b
        n[b] = m.sum()
        if m.sum() >= MIN_CELL:
            a[b] = dxk2[m].mean()
    return a, n


def main():
    print("loading instruments and recovering ticks ...")
    inst = load_instruments()
    print(f"  usable instruments: {len(inst)}")
    for s, d in list(inst.items())[:6]:
        print(f"    {s:6s} tick={d['tick']:.5g}  n_tr={len(d['tr']):>6d}  n_te={len(d['te']):>6d}")

    rows = []
    # pooled containers for the within-stratum transfer and the figure.
    # a_tr/a_te keep raw levels (figure panel a); d_tr/d_te are log-demeaned WITHIN each
    # instrument x stratum, so pooling them removes both the instrument level and the spread
    # level and leaves only the imbalance shape -- the honest tick-controlled transfer object.
    pooled = {s: {"a_tr": [], "a_te": [], "ic": [], "d_tr": [], "d_te": []}
              for s in range(len(SPREAD_TICKS))}
    for sym, d in inst.items():
        tr, te = d["tr"], d["te"]
        for strat in range(len(SPREAD_TICKS)):
            a_tr, n_tr = axx_by_ibin(tr, strat)
            a_te, n_te = axx_by_ibin(te, strat)
            if a_tr is None or a_te is None:
                continue
            ok = np.isfinite(a_tr) & np.isfinite(a_te)
            if ok.sum() < 4:                          # need a few imbalance bins to correlate
                continue
            ic = I_CENT[ok]; atr = a_tr[ok]; ate = a_te[ok]
            # --- within-stratum imbalance structure (on TRAIN a_xx) ---
            rho_signed = stats.spearmanr(ic, atr).correlation                 # monotone in I
            rho_abs = stats.spearmanr(np.abs(ic), atr).correlation            # U-shape in |I|
            # extreme vs central contrast (ratio of mean a_xx, |I|>0.6 vs |I|<0.2)
            ext = atr[np.abs(ic) > 0.6]; cen = atr[np.abs(ic) < 0.2]
            ext_ratio = float(np.mean(ext) / np.mean(cen)) if len(ext) and len(cen) else np.nan
            # shape-agnostic Kruskal across imbalance bins within stratum (raw dx^2)
            sub = tr[spread_stratum(tr["Stick"].to_numpy()) == strat]
            ib = np.clip(np.digitize(sub["I"].to_numpy(), I_EDGES) - 1, 0, N_I - 1)
            dxk2 = sub["dxk"].to_numpy() ** 2
            groups = [dxk2[ib == b] for b in range(N_I) if (ib == b).sum() >= MIN_CELL]
            kruskal_p = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan
            # --- within-stratum TRANSFER (train a_xx vs test a_xx across imbalance bins) ---
            transfer_rho = stats.spearmanr(atr, ate).correlation if ok.sum() >= 4 else np.nan
            rows.append({"sym": sym, "tick": d["tick"], "spread_ticks": SPREAD_TICKS[strat],
                         "n_ibins": int(ok.sum()),
                         "rho_signed_I": rho_signed, "rho_abs_I": rho_abs,
                         "extreme_central_ratio": ext_ratio, "kruskal_p": kruskal_p,
                         "transfer_rho_within": transfer_rho})
            pooled[strat]["a_tr"].append(atr); pooled[strat]["a_te"].append(ate)
            pooled[strat]["ic"].append(ic)
            # log-demeaned within this instrument x stratum (removes instrument & spread level)
            pooled[strat]["d_tr"].append(np.log(atr) - np.log(atr).mean())
            pooled[strat]["d_te"].append(np.log(ate) - np.log(ate).mean())

    df = pd.DataFrame(rows)
    df.to_csv(OUTT / "sde_tick_control.csv", index=False)

    # ---------- pooled summary ----------
    def agg(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if len(x) == 0:
            return (np.nan, np.nan)
        m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return m, 1.96 * se

    summ = []
    print("\n=== Within spread-tick strata: does imbalance still move a_xx? ===")
    print("  (rho_signed: monotone in I; rho_abs: U-shape in |I|; transfer: train->test within stratum)")
    for strat in range(len(SPREAD_TICKS)):
        sd = df[df["spread_ticks"] == SPREAD_TICKS[strat]]
        if len(sd) == 0:
            continue
        ms, _ = agg(sd["rho_signed_I"]); ma, _ = agg(sd["rho_abs_I"])
        mt, et = agg(sd["transfer_rho_within"]); mr, _ = agg(sd["extreme_central_ratio"])
        frac_kr = float(np.mean(sd["kruskal_p"] < 0.05))
        lab = f">={SPREAD_TICKS[strat]}" if strat == len(SPREAD_TICKS) - 1 else f"{SPREAD_TICKS[strat]}"
        print(f"  spread={lab:>3s} tick   n_inst={len(sd):2d}  rho_signed={ms:+.3f}  rho_|I|={ma:+.3f}  "
              f"ext/cen={mr:.2f}  Kruskal<.05={frac_kr:.0%}  transfer={mt:+.3f}[+/-{et:.3f}]")
        summ.append({"spread_ticks": lab, "n_inst": len(sd), "rho_signed_I": ms, "rho_abs_I": ma,
                     "extreme_central_ratio": mr, "frac_kruskal_sig": frac_kr,
                     "transfer_rho_within": mt, "transfer_ci": et})
    pd.DataFrame(summ).to_csv(OUTT / "sde_tick_control_summary.csv", index=False)

    # CONFOUNDED pooled transfer (kept only as a cautionary contrast): pooling raw a_xx across
    # instruments and strata reintroduces the instrument and spread LEVELS, so it is NOT
    # within-stratum evidence and is biased upward.
    alltr = np.concatenate([np.concatenate(pooled[s]["a_tr"]) for s in pooled if pooled[s]["a_tr"]])
    allte = np.concatenate([np.concatenate(pooled[s]["a_te"]) for s in pooled if pooled[s]["a_te"]])
    rho_confounded = stats.spearmanr(alltr, allte).correlation
    # HONEST tick-controlled transfer: pool the log-demeaned (within instrument x stratum)
    # residuals, so only the imbalance shape remains.
    dtr = np.concatenate([np.concatenate(pooled[s]["d_tr"]) for s in pooled if pooled[s]["d_tr"]])
    dte = np.concatenate([np.concatenate(pooled[s]["d_te"]) for s in pooled if pooled[s]["d_te"]])
    rho_demeaned = stats.spearmanr(dtr, dte).correlation
    per_stratum_mean = float(np.nanmean(df["transfer_rho_within"]))
    print(f"\n  CONFOUNDED pooled transfer (levels not removed) Spearman = {rho_confounded:.3f}  "
          f"<-- biased up, do NOT report as within-stratum")
    print(f"  HONEST within-stratum imbalance transfer (log-demeaned pooled) Spearman = {rho_demeaned:.3f}  "
          f"(n cells={len(dtr)})")
    print(f"  per-stratum within-instrument transfer, averaged = {per_stratum_mean:+.3f}")

    # ---------- decision ----------
    # The two questions are separate: (1) does imbalance MOVE the variance within fixed
    # spread-ticks (Kruskal, U-shape)?  (2) does that imbalance SHAPE TRANSFER train->test?
    wide = df[df["spread_ticks"] >= 2]
    sig_imb = (np.nanmean(np.abs(wide["rho_abs_I"])) > 0.2) or (np.nanmean(np.abs(wide["rho_signed_I"])) > 0.2)
    transfers_strong = rho_demeaned > 0.4          # honest, level-removed threshold
    transfers_weak = rho_demeaned > 0.1
    frac_kr_all = float(np.mean(df["kruskal_p"] < 0.05))
    if sig_imb and transfers_strong:
        branch = ("A (strong): imbalance explains within-stratum variance AND the shape transfers strongly "
                  "after removing levels; full structural claim.")
    elif (sig_imb or frac_kr_all > 0.5) and transfers_weak:
        branch = ("C (partial survival): imbalance robustly MOVES the variance within fixed spread-ticks "
                  f"(Kruskal sig in {frac_kr_all:.0%} of cells, U-shaped |I|), so the surface is NOT a tick "
                  f"artifact; but the within-stratum imbalance SHAPE transfers only weakly "
                  f"(level-removed Spearman {rho_demeaned:.2f}). Keep the structural claim with SOFTER wording: "
                  "spread carries the transferable level structure, imbalance carries a real but modestly "
                  "reproducible symmetric modulation.")
    elif sig_imb or frac_kr_all > 0.5:
        branch = ("C- (effect present, transfer near zero): imbalance moves the variance but its shape barely "
                  "reproduces out of sample; report as a contemporaneous, weakly-transferable modulation.")
    else:
        branch = "B (none): nothing survives the spread-tick control; narrow to spread-scaled risk bands."
    print(f"\n  Kruskal imbalance-effect significant in {frac_kr_all:.0%} of (instrument,stratum) cells")
    print(f"\n>>> DECISION: Branch {branch}")

    # ---------- figure ----------
    import sys
sys.path.insert(0, str(CODE))
from plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, MUTED, POS, WARN
    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
    cols = [MUTED, ACCENT, ACCENT_DARK, INK]

    # (a) normalised a_xx vs imbalance bin, one curve per spread-tick stratum (train), pooled across instruments
    a = ax[0]
    for strat in range(len(SPREAD_TICKS)):
        if not pooled[strat]["a_tr"]:
            continue
        # average the per-instrument normalised profiles over imbalance bins
        prof = np.full((len(pooled[strat]["a_tr"]), N_I), np.nan)
        for r, (atr, ic) in enumerate(zip(pooled[strat]["a_tr"], pooled[strat]["ic"])):
            idx = np.searchsorted(I_CENT, ic)
            v = atr / np.nanmean(atr)               # normalise out the instrument/stratum level
            for k, ii in enumerate(idx):
                if 0 <= ii < N_I:
                    prof[r, ii] = v[k]
        mprof = np.nanmean(prof, axis=0)
        lab = rf"$S={SPREAD_TICKS[strat]}$ tick" + (r"$+$" if strat == len(SPREAD_TICKS) - 1 else "")
        a.plot(I_CENT, mprof, "-o", color=cols[strat], ms=4, lw=1.6, label=lab)
    a.axvline(0, color=INK, lw=0.6, ls=":")
    a.set_xlabel(r"best-level imbalance $I$")
    a.set_ylabel(r"normalised diffusion $a_{xx}/\langle a_{xx}\rangle$ (within stratum)")
    a.set_title(r"(a) Imbalance still shapes $a_{xx}$ within fixed spread-in-ticks")
    a.legend(loc="upper center", fontsize=8, ncol=2); despine(a)

    # (b) HONEST within-stratum transfer of the imbalance SHAPE: log-demeaned train vs test
    #     (instrument and spread levels removed), coloured by stratum.
    b = ax[1]
    for strat in range(len(SPREAD_TICKS)):
        if not pooled[strat]["d_tr"]:
            continue
        xt = np.concatenate(pooled[strat]["d_tr"]); yt = np.concatenate(pooled[strat]["d_te"])
        b.scatter(xt, yt, s=12, c=cols[strat], alpha=0.5, edgecolors="none",
                  label=rf"$S={SPREAD_TICKS[strat]}$" + ("+" if strat == len(SPREAD_TICKS) - 1 else ""))
    lim = [np.nanmin(dtr), np.nanmax(dtr)]
    b.plot(lim, lim, color=INK, ls="--", lw=1, label=r"$y=x$")
    b.axhline(0, color=INK, lw=0.5, ls=":"); b.axvline(0, color=INK, lw=0.5, ls=":")
    b.set_xlabel(r"train $\log a_{xx}$, demeaned within instrument$\times$stratum")
    b.set_ylabel(r"held-out $\log a_{xx}$, demeaned")
    b.set_title(rf"(b) Within-stratum imbalance transfer (Spearman {rho_demeaned:.2f}, weak)")
    b.legend(loc="upper left", fontsize=8); despine(b)

    fig.suptitle(r"Tick-mechanics control: imbalance still moves $a_{xx}$ within fixed spread-in-ticks "
                 r"(panel a), but its shape transfers only weakly out of sample (panel b)", y=1.01)
    finish(fig, OUTF / "fig_tick_control.png")
    print("[tick-control] wrote fig_tick_control.png, sde_tick_control.csv, sde_tick_control_summary.csv")


if __name__ == "__main__":
    main()
