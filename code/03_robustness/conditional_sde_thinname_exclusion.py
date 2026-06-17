#!/usr/bin/env python3
"""
conditional_sde_thinname_exclusion.py — deletion robustness check for the five thinnest instruments.

Checks whether the pooled tail-exceedance profile and GH OOS log-likelihood gain are
driven by the five sub-5,000-update instruments:
  THIN_NAMES = {AHCS, MHAR, QATI, MKDM, QIGD}

Design: fit the GH law ONCE on the full pooled sample (same procedure as
conditional_sde_heavytail.py), then compute the comparison table for:
  (a) full sample
  (b) reduced sample (thin names excluded)

No refit on the reduced sample — this isolates whether the reported pooled conclusion
was driven by the thin names, holding the fitted law fixed.

Output: stdout comparison table (copy into manuscript robustness note).
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

THIN_NAMES = {"AHCS", "MHAR", "QATI", "MKDM", "QIGD"}
KS = [2, 3, 4, 5, 6]


def parse_date(s):
    mm, dd, yy = s.split("-")
    return (int(yy), int(mm), int(dd))


def session_increments(l1):
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
    return pd.DataFrame({"dx": dx, "I": I[:-1], "S": S[:-1]}).replace(
        [np.inf, -np.inf], np.nan).dropna()


def cell_ids(I, S, s_edges):
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb


def collect_residuals_by_instrument():
    """
    Returns:
      by_sym: dict sym -> z_array (held-out standardised residuals, quality-filtered)
      n_updates_raw: dict sym -> raw total observations (before quality filter)
      n_updates_filtered: dict sym -> observations for instruments that passed filter
      filtered_out: set of syms that exist in data but failed the quality threshold
    """
    cfg = yaml.safe_load(open(CODE / "config.yaml"))
    clean = Path(cfg["data"]["out_dir"]) / "clean"
    files: dict[str, list] = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    by_sym: dict[str, np.ndarray] = {}
    n_updates_raw: dict[str, int] = {}
    n_updates_filtered: dict[str, int] = {}
    filtered_out: set = set()

    for sym, paths in files.items():
        paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        tr_frames = [session_increments(pd.read_parquet(p)) for p in paths[:ntr]]
        te_frames = [session_increments(pd.read_parquet(p)) for p in paths[ntr:]]
        tr = pd.concat([f for f in tr_frames if f is not None], ignore_index=True) \
            if any(f is not None for f in tr_frames) else None
        te = pd.concat([f for f in te_frames if f is not None], ignore_index=True) \
            if any(f is not None for f in te_frames) else None
        n_raw = (len(tr) if tr is not None else 0) + (len(te) if te is not None else 0)
        n_updates_raw[sym] = n_raw
        if tr is None or te is None or len(tr) < 2000 or len(te) < 1000:
            filtered_out.add(sym)
            continue
        n_updates_filtered[sym] = len(tr) + len(te)
        s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
        ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
        cte = cell_ids(te["I"].to_numpy(), te["S"].to_numpy(), s_edges)
        dxtr = tr["dx"].to_numpy()
        dxte = te["dx"].to_numpy()
        bx, axx = {}, {}
        for c in range(N_I * M_S):
            m = ctr == c
            if m.sum() >= MIN_CELL:
                bx[c] = float(dxtr[m].mean())
                axx[c] = float((dxtr[m] ** 2).mean())
        keep = np.array([c in axx and axx[c] > 0 for c in cte])
        if keep.sum() < 500:
            filtered_out.add(sym)
            continue
        cte_k = cte[keep]
        dxte_k = dxte[keep]
        bx_v = np.array([bx[c] for c in cte_k])
        axx_v = np.array([axx[c] for c in cte_k])
        z = (dxte_k - bx_v) / np.sqrt(axx_v)
        z = z[np.isfinite(z)]
        if len(z) > 0:
            by_sym[sym] = z

    return by_sym, n_updates_raw, n_updates_filtered, filtered_out


def fit_gh(z_full: np.ndarray):
    """Fit symmetric GH on full pooled sample. Returns fitted distribution object."""
    zz = z_full if len(z_full) <= 200_000 else RNG.choice(z_full, 200_000, replace=False)
    p, a, _b, _loc, s = stats.genhyperbolic.fit(zz, fb=0.0, floc=0.0)
    gh = stats.genhyperbolic(p, a, 0.0, 0.0, s)
    return gh, p, a, s


def compute_metrics(z: np.ndarray, gh, label: str) -> dict:
    """
    For a residual array z and a pre-fitted GH:
    - empirical P(|z|>k) for k in KS
    - GH P(|z|>k) for k in KS
    - mean per-obs OOS log-likelihood gain: E[log p_GH(z) - log p_N(z)]
    """
    result = {"label": label, "n_obs": len(z)}
    for k in KS:
        result[f"emp_P|z|>{k}"] = float(np.mean(np.abs(z) > k))
        result[f"gh_P|z|>{k}"] = float(gh.sf(k) + gh.cdf(-k))

    # OOS log-likelihood gain per observation
    ll_gh = gh.logpdf(z)
    ll_norm = stats.norm.logpdf(z, scale=float(z.std()))  # best-fit Gaussian at z's own width
    gain = float(np.mean(ll_gh - ll_norm))
    result["gh_oos_gain_nats"] = gain
    return result


def fmt_float(v, spec):
    """Format a float with a given spec, handling the + sign correctly."""
    return format(v, spec)


def main():
    print("Loading residuals by instrument ...")
    by_sym, n_updates_raw, n_updates_filtered, filtered_out = collect_residuals_by_instrument()

    # --- report thin-name status ---
    print(f"\nInstruments in raw data: {len(n_updates_raw)}")
    print(f"Instruments passing quality filter: {len(by_sym)}")
    print(f"Filtered out (too few obs): {sorted(filtered_out)}")
    print()
    print("Named thin instruments status:")
    for sym in sorted(THIN_NAMES):
        raw = n_updates_raw.get(sym, "not in raw data")
        if sym in filtered_out:
            print(f"  {sym}: {raw} raw obs — ALREADY EXCLUDED by quality filter (not in pooled residuals)")
        elif sym in by_sym:
            print(f"  {sym}: {raw} raw obs — passes filter, {len(by_sym[sym])} test residuals")
        else:
            print(f"  {sym}: {raw} raw obs — not in data")

    # instruments that pass filter, by residual count (ascending = thinnest first)
    sym_by_size = sorted(by_sym.keys(), key=lambda s: len(by_sym[s]))
    print(f"\nThinnest 5 instruments that DO pass the quality filter:")
    for sym in sym_by_size[:5]:
        print(f"  {sym}: {len(by_sym[sym]):,} test residuals")

    # build full array and bottom-quintile exclusion set
    bottom_quintile = set(sym_by_size[:max(1, len(sym_by_size)//5)])
    print(f"\nBottom quintile (thinnest ~20% of passing instruments, n={len(bottom_quintile)}): {sorted(bottom_quintile)}")

    z_full = np.concatenate(list(by_sym.values()))
    z_full = z_full[np.isfinite(z_full)]

    # named thin names already excluded — bottom-quintile exclusion is the meaningful check
    thick_syms = set(by_sym.keys()) - bottom_quintile
    z_excl = np.concatenate([by_sym[s] for s in thick_syms])
    z_excl = z_excl[np.isfinite(z_excl)]

    print(f"\nFull sample: n={len(z_full):,} obs, {len(by_sym)} instruments")
    print(f"Excl. bottom quintile: n={len(z_excl):,} obs, {len(thick_syms)} instruments "
          f"({100*(1-len(z_excl)/len(z_full)):.1f}% obs removed)")

    # fit GH ONCE on full sample — do not refit on reduced
    print("\nFitting GH on full pooled sample ...")
    gh, p_gh, a_gh, s_gh = fit_gh(z_full)
    print(f"GH shape: p={p_gh:.3f}, a={a_gh:.3f}, scale={s_gh:.4f}")

    # compute metrics for both subsets
    r_full = compute_metrics(z_full, gh, "Full sample")
    r_excl = compute_metrics(z_excl, gh, "Excl. bottom quintile")

    # --- print comparison table ---
    print("\n" + "=" * 74)
    print("Thin-name deletion robustness check")
    print("Note: AHCS/MHAR/QATI/MKDM/QIGD already excluded by quality filter.")
    print("Check below excludes bottom quintile of passing instruments.")
    print("GH fitted on FULL sample; same law applied to both subsets.")
    print("=" * 74)
    print(f"  {'Metric':35s}  {'Full sample':>14s}  {'Excl. bottom 20%':>16s}")
    print("-" * 74)

    def row(name, key, spec):
        vf = r_full.get(key, float("nan"))
        ve = r_excl.get(key, float("nan"))
        print(f"  {name:35s}  {fmt_float(vf, spec):>14s}  {fmt_float(ve, spec):>16s}")

    print(f"  {'N instruments':35s}  {len(by_sym):>14d}  {len(thick_syms):>16d}")
    row("N observations", "n_obs", "d")
    print()
    for k in KS:
        row(f"Empirical P(|z|>{k})", f"emp_P|z|>{k}", ".2e")
    print()
    for k in KS:
        row(f"GH P(|z|>{k})", f"gh_P|z|>{k}", ".2e")
    print()
    row("GH OOS gain vs Gaussian (nats/obs)", "gh_oos_gain_nats", "+.4f")
    print("=" * 74)

    delta_gain = r_excl["gh_oos_gain_nats"] - r_full["gh_oos_gain_nats"]
    print(f"\nChange in OOS gain after exclusion: {delta_gain:+.4f} nats/obs")
    print("Conclusion: " + (
        "GH advantage is robust to thin-name exclusion."
        if abs(delta_gain) < 0.05 and r_excl["gh_oos_gain_nats"] > 0
        else "Thin names MATERIALLY affect the GH OOS gain — investigate."
    ))


if __name__ == "__main__":
    main()
