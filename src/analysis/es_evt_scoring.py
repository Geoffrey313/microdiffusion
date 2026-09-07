#!/usr/bin/env python3
"""
es_evt_scoring.py - Expected Shortfall + EVT tail assessment for the state-dependent SDE.

Companion to var_backtest.py. The VaR backtest tests a single tail quantile (coverage);
this script adds the two complementary tail diagnostics a referee typically asks for
alongside a Kupiec/Christoffersen backtest, and which the audit (Section 7) flags as a
likely R&R request:

  1. EXPECTED SHORTFALL (severity). For each innovation law (GH, Gaussian) and each
     coverage alpha, compare the model-implied ES - the mean magnitude of the standardised
     innovation beyond the model's own two-sided VaR threshold - with the realised mean
     exceedance on held-out sessions. The ratio

         R = realised_ES / model_ES

     is 1 when the model captures tail SEVERITY; R > 1 means realised tail shocks are more
     severe than the model predicts. This is a SEVERITY DIAGNOSTIC - a point-estimate ratio
     (the mean tail exceedance normalised by the model ES, the quantity behind the
     Acerbi-Szekely (2014) Z2 idea) - and NOT a formal ES backtest: it reports no test
     statistic and no p-value. It separates COVERAGE (does the tail fire at rate alpha? -
     the VaR backtest) from SEVERITY (when it fires, how big is it? - here). A formal
     proper-score ES contest (Fissler-Ziegel 2016) or an Acerbi-Szekely test with inference
     would require additional code and is deliberately not claimed here.

  2. EVT TAIL INDEX. Fit a Generalised Pareto Distribution (peaks-over-threshold) to the
     pooled standardised residual magnitudes above a high threshold and report the shape
     xi (tail index). This is a threshold-based (semiparametric peaks-over-threshold /
     GPD) fit: a nonparametric threshold with a parametric excess law, so it does not
     assume the GH shape. xi > 0 supports a heavy Frechet-domain tail, corroborating H3.
     A small sensitivity sweep over the threshold is reported.

Design: reuses var_backtest.py's EXACT residual construction (same surface, same per-symbol
GH fit, same 60/40 held-out split), so the ES numbers are directly comparable to the VaR
backtest and nothing new is assumed about the data.

Outputs (NOT part of the reproduce.py digest gate):
  results/diagnostics/tables/es_evt_summary.csv   per-instrument ES ratios (diagnostic)
  results/diagnostics/tables/es_evt_evt.csv       EVT threshold sweep (diagnostic)
  results/diagnostics/figures/fig_es_evt.png/.pdf diagnostic figure (not used in the paper)
  <SECTIONS_DIR>/tab_es_evt.tex                    the LaTeX table \\input by the Robustness
                                                   section (auto-generated, like the pooled
                                                   descriptive table; the manuscript prints
                                                   exactly the numbers this script computes)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_config, CLEAN_DIR, DIAG_DIR, SECTIONS_DIR
from common.grid import N_I, M_S, cell_ids
from common.plot_style import finish, setup_mpl, despine, INK, ACCENT, ACCENT_DARK, WARN, MUTED, POS, T
# Reuse the identical per-session reader and date parser from the VaR backtest (no duplication).
from analysis.var_backtest import session_increments, parse_date

TRAIN_FRAC = 0.60
MIN_CELL   = 50
ALPHAS     = [0.01, 0.05]
EVT_QUANTILES = [0.90, 0.95, 0.975]   # POT threshold sweep on |z|
RNG        = np.random.default_rng(0)
_GH_SIM    = 400_000                  # sample size for model-implied GH ES (deterministic seed)


# ---------------------------------------------------------------------------
# per-instrument: train surface + GH fit, then held-out standardised residuals
# ---------------------------------------------------------------------------

def fit_and_heldout_residuals(sym, paths):
    """Return (z_te, gh, sd_gh) where z_te are the held-out standardised residuals
    |Delta x - b_x|/sqrt(a_xx) (as signed z as well), gh the fitted unit-variance-rescaled
    innovation, sd_gh its raw sd. None if the symbol does not clear the gates.
    Mirrors var_backtest.backtest_instrument up to the residual construction."""
    paths = sorted(paths, key=lambda p: parse_date(p.parent.name))
    ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
    tr_paths, te_paths = paths[:ntr], paths[ntr:]
    if not te_paths:
        return None

    tr_frames = [session_increments(p) for p in tr_paths]
    tr_frames = [f for f in tr_frames if f is not None]
    if not tr_frames:
        return None
    tr = pd.concat(tr_frames, ignore_index=True)
    if len(tr) < 2000:
        return None

    s_edges = np.quantile(tr["S"], [1 / 3, 2 / 3])
    ctr = cell_ids(tr["I"].to_numpy(), tr["S"].to_numpy(), s_edges)
    dxtr = tr["dx"].to_numpy()
    bx, axx = {}, {}
    for c in range(N_I * M_S):
        m = ctr == c
        if m.sum() >= MIN_CELL:
            bx[c] = float(dxtr[m].mean())
            axx[c] = float((dxtr[m] ** 2).mean())

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
    except Exception:
        return None
    if not np.isfinite(sd_gh) or sd_gh <= 0:
        return None

    # held-out standardised residuals (signed z); magnitude is what the VaR interval tests
    z_list = []
    for p in te_paths:
        df = session_increments(p)
        if df is None or len(df) < 20:
            continue
        cte = cell_ids(df["I"].to_numpy(), df["S"].to_numpy(), s_edges)
        dx = df["dx"].to_numpy()
        for i, c in enumerate(cte):
            if c not in axx or axx[c] <= 0:
                continue
            z_list.append((dx[i] - bx[c]) / np.sqrt(axx[c]))
    z_te = np.asarray(z_list, dtype=float)
    z_te = z_te[np.isfinite(z_te)]
    if len(z_te) < 500:
        return None
    return z_te, gh, sd_gh


# ---------------------------------------------------------------------------
# ES helpers (two-sided magnitude convention)
# ---------------------------------------------------------------------------

def gaussian_var_es(alpha):
    """(VaR threshold t, ES) for a standard-normal innovation, two-sided coverage alpha.
    P(|Z|>t)=alpha => t = Phi^{-1}(1-alpha/2); ES = E[|Z| : |Z|>t] = phi(t)/(alpha/2)."""
    t = float(stats.norm.ppf(1 - alpha / 2))
    es = float(stats.norm.pdf(t) / (alpha / 2))
    return t, es


def gh_var_es(gh, sd_gh, alpha):
    """(VaR threshold t, ES) for the fitted GH rescaled to unit variance, two-sided.
    t = q_{1-alpha/2}(X)/sd; ES = E[|Z| : |Z|>t] by deterministic-seed simulation."""
    t = float(gh.ppf(1 - alpha / 2) / sd_gh)
    sim = gh.rvs(size=_GH_SIM, random_state=np.random.default_rng(12345)) / sd_gh
    az = np.abs(sim)
    tail = az[az > t]
    es = float(tail.mean()) if tail.size else np.nan
    return t, es


def realised_es(z_te, t):
    """Realised mean exceedance magnitude and exceedance count at model threshold t."""
    az = np.abs(z_te)
    tail = az[az > t]
    return (float(tail.mean()) if tail.size else np.nan), int(tail.size)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    load_config()
    files: dict[str, list] = {}
    for p in CLEAN_DIR.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    rows = []
    pooled_absz = []   # for the EVT fit and the pooled tail figure
    for sym, paths in sorted(files.items()):
        out = fit_and_heldout_residuals(sym, paths)
        if out is None:
            continue
        z_te, gh, sd_gh = out
        pooled_absz.append(np.abs(z_te))
        for alpha in ALPHAS:
            tg, es_g = gaussian_var_es(alpha)
            th, es_h = gh_var_es(gh, sd_gh, alpha)
            r_es_g, n_g = realised_es(z_te, tg)
            r_es_h, n_h = realised_es(z_te, th)
            rows.append(dict(
                sym=sym, alpha=alpha, T=len(z_te),
                gh_t=th, gh_es_model=es_h, gh_es_real=r_es_h, gh_R=r_es_h / es_h if es_h else np.nan, gh_nexc=n_h,
                gauss_t=tg, gauss_es_model=es_g, gauss_es_real=r_es_g, gauss_R=r_es_g / es_g if es_g else np.nan, gauss_nexc=n_g,
            ))
        print(f"  {sym:8s} T={len(z_te):>7d}  "
              f"GH R(0.01)={rows[-2]['gh_R']:.2f}/Gauss {rows[-2]['gauss_R']:.2f}  |  "
              f"GH R(0.05)={rows[-1]['gh_R']:.2f}/Gauss {rows[-1]['gauss_R']:.2f}")

    if not rows:
        print("No results - check data path.")
        return
    df = pd.DataFrame(rows)

    # ---- ES summary (pooled medians of R across instruments) ----
    print("\n=== Expected Shortfall severity: R = realised/model ES (1 = calibrated, >1 = model understates) ===")
    for alpha in ALPHAS:
        s = df[df["alpha"] == alpha]
        print(f"alpha={alpha}: N={len(s)}  "
              f"GH median R={s['gh_R'].median():.3f} (IQR {s['gh_R'].quantile(.25):.3f}-{s['gh_R'].quantile(.75):.3f})  |  "
              f"Gauss median R={s['gauss_R'].median():.3f} (IQR {s['gauss_R'].quantile(.25):.3f}-{s['gauss_R'].quantile(.75):.3f})")

    # ---- EVT: pooled GPD (peaks-over-threshold) on |z|, threshold sweep ----
    absz = np.concatenate(pooled_absz)
    evt_rows = []
    for q in EVT_QUANTILES:
        u = float(np.quantile(absz, q))
        exc = absz[absz > u] - u
        if len(exc) < 200:
            continue
        xi, _loc, beta = stats.genpareto.fit(exc, floc=0.0)   # xi = shape (tail index)
        # approx SE of xi via nonparametric bootstrap
        bs = []
        for _ in range(200):
            b = RNG.choice(exc, len(exc), replace=True)
            try:
                xib, _l, _s = stats.genpareto.fit(b, floc=0.0)
                bs.append(xib)
            except Exception:
                pass
        se = float(np.std(bs)) if bs else np.nan
        evt_rows.append(dict(threshold_q=q, u=u, n_exc=len(exc), xi=float(xi), xi_se=se, beta=float(beta)))
        print(f"[EVT] u=q{q:.3f}={u:.2f}  n_exc={len(exc):>6d}  xi(tail index)={xi:+.3f} +- {se:.3f}  beta={beta:.3f}")
    evt = pd.DataFrame(evt_rows)

    # ---- write diagnostic tables ----
    (DIAG_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(DIAG_DIR / "tables" / "es_evt_summary.csv", index=False)
    evt.to_csv(DIAG_DIR / "tables" / "es_evt_evt.csv", index=False)

    # ---- LaTeX table fragment for the Robustness section (auto-generated, bilingual) ----
    # Mirrors descriptive_stats.py: the manuscript \input's this file, so the printed numbers
    # are exactly the ones computed above (nothing entered by hand). The French build
    # (MICRODIFFUSION_LANG=fr, MICRODIFFUSION_SECTIONS_DIR=.../fr/.../sections) writes the
    # French fragment into the French tree.
    import os
    fr = os.environ.get("MICRODIFFUSION_LANG", "en") == "fr"
    venue = os.environ.get("MICRODIFFUSION_VENUE", "QSE")  # relabelled by reproduce.py --us
    a01, a05 = df[df["alpha"] == 0.01], df[df["alpha"] == 0.05]
    n_inst = int(a01.shape[0])

    def _cell(s):
        return "$%.2f\\ [%.2f,\\,%.2f]$" % (s.median(), s.quantile(.25), s.quantile(.75))

    def _nexc(n):
        return "$" + "{:,}".format(int(n)).replace(",", "\\,") + "$"

    def _qs(q):                      # 0.90 / 0.95 / 0.975 with a consistent 2+ decimals
        s = ("%.3f" % q).rstrip("0")
        if len(s.split(".")[1]) < 2:
            s += "0"
        return s

    evt_body = "\n".join(
        "$q_{%s}$ & $%.3f\\ (%.3f)$ & %s \\\\" % (_qs(r["threshold_q"]), r["xi"], r["xi_se"], _nexc(r["n_exc"]))
        for r in evt_rows)

    if fr:
        caption = (
            "Diagnostic de sévérité de queue et indice de queue en valeurs extrêmes, sessions %s "
            "de test. Panneau~A : ratio de sévérité d'expected shortfall $R$ (expected shortfall "
            "réalisé divisé par l'expected shortfall implicite du modèle) --- le dépassement "
            "standardisé moyen au-delà du seuil VaR bilatéral propre au modèle divisé par "
            "l'expected shortfall implicite de ce modèle (médiane sur les $%d$ instruments, écart "
            "interquartile entre crochets) ; $R=1$ = sévérité calibrée, $R>1$ = pertes de queue "
            "réalisées supérieures au modèle. Panneau~B : paramètre de forme $\\xi$ (indice de queue) "
            "d'une loi de Pareto généralisée ajustée par dépassements de seuil aux amplitudes des "
            "résidus standardisés poolés, à trois quantiles de seuil (erreur-type bootstrap entre "
            "parenthèses). Diagnostic de sévérité et vérification d'indice de queue par seuil, non "
            "un backtest formel d'expected shortfall." % (venue, n_inst))
        panelA  = "\\emph{Panneau A : ratio de sévérité ES $R$ (médiane [IIQ])}"
        headerA = "Couverture $\\alpha_{\\mathrm{VaR}}$ & Hyperbolique gén. & Gaussienne"
        panelB  = "\\emph{Panneau B : indice de queue EVT $\\xi$ (GPD, dépassements de seuil)}"
        headerB = "Seuil $u$ & $\\xi$ (é.-t.) & Nb.\\ dépassements"
    else:
        caption = (
            "Tail-severity diagnostic and extreme-value tail index on held-out %s sessions. "
            "Panel~A: expected-shortfall severity ratio "
            "$R=\\mathrm{ES}_{\\mathrm{realised}}/\\mathrm{ES}_{\\mathrm{model}}$, the mean "
            "standardised exceedance beyond each model's own two-sided VaR threshold divided by "
            "that model's implied expected shortfall (median across the $%d$ instruments, "
            "interquartile range in brackets); $R=1$ is calibrated severity, $R>1$ means realised "
            "tail losses exceed the model. Panel~B: shape parameter $\\xi$ (tail index) of a "
            "generalised Pareto distribution fitted by peaks-over-threshold to the pooled "
            "standardised-residual magnitudes, at three threshold quantiles (bootstrap standard "
            "error in parentheses). This is a severity diagnostic and a threshold-based tail-index "
            "check, not a formal expected-shortfall backtest." % (venue, n_inst))
        panelA  = "\\emph{Panel A: ES severity ratio $R$ (median [IQR])}"
        headerA = "Coverage $\\alpha_{\\mathrm{VaR}}$ & Gen.\\ Hyperbolic & Gaussian"
        panelB  = "\\emph{Panel B: EVT tail index $\\xi$ (GPD, peaks-over-threshold)}"
        headerB = "Threshold $u$ & $\\xi$ (s.e.) & No.\\ exceedances"

    tex = (
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{" + caption + "}\n"
        "\\label{tab:es-evt}\n\\small\n"
        "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
        "\\multicolumn{3}{@{}l}{" + panelA + "} \\\\\n\\midrule\n"
        + headerA + " \\\\\n\\midrule\n"
        + "$0.01$ & " + _cell(a01["gh_R"]) + " & " + _cell(a01["gauss_R"]) + " \\\\\n"
        + "$0.05$ & " + _cell(a05["gh_R"]) + " & " + _cell(a05["gauss_R"]) + " \\\\\n\\midrule\n"
        "\\multicolumn{3}{@{}l}{" + panelB + "} \\\\\n\\midrule\n"
        + headerB + " \\\\\n\\midrule\n"
        + evt_body + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (SECTIONS_DIR / "tab_es_evt.tex").write_text(tex)
    print("wrote %s" % (SECTIONS_DIR / "tab_es_evt.tex"))

    # ---- figure: (a) ES ratio R by model x alpha; (b) pooled tail vs GH/Gauss/GPD ----
    plt = setup_mpl()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) violin of R across instruments
    positions = [1, 2, 4, 5]
    colors = [ACCENT, WARN, ACCENT, WARN]
    labels = [r"GH, $\alpha{=}0.01$", r"Gauss, $\alpha{=}0.01$",
              r"GH, $\alpha{=}0.05$", r"Gauss, $\alpha{=}0.05$"]
    data_R = []
    for alpha in ALPHAS:
        s = df[df["alpha"] == alpha]
        data_R.append(s["gh_R"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        data_R.append(s["gauss_R"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    parts = ax0.violinplot(data_R, positions=positions, showmedians=True, widths=0.7)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col); body.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(INK); parts[key].set_linewidth(1.0)
    ax0.axhline(1.0, color=INK, ls="--", lw=1.0, label=T(r"calibrated ($R=1$)"))
    ax0.set_xticks(positions); ax0.set_xticklabels(labels, fontsize=7)
    ax0.set_ylabel(T(r"ES ratio $R=\mathrm{realised}/\mathrm{model}$"))
    ax0.set_title("(a) " + T(r"Expected-shortfall severity: GH vs Gaussian by $\alpha$"))
    ax0.legend(fontsize=7); despine(ax0)

    # (b) pooled empirical tail vs GH, Gaussian, GPD
    ks = np.linspace(1.5, 6.0, 60)
    emp = np.array([(absz > k).mean() for k in ks])
    gauss_tail = 2 * (1 - stats.norm.cdf(ks))
    # GH pooled tail: use a large GH sample rescaled to unit variance (median sd across fits ~1)
    ax1.plot(ks, emp, "-o", color=ACCENT_DARK, ms=3, label=T(r"data (pooled $|z|$)"))
    ax1.plot(ks, gauss_tail, "--", color=WARN, lw=1.2, label=r"$N(0,1)$")
    if len(evt):
        row = evt[evt["threshold_q"] == 0.95].iloc[0] if (evt["threshold_q"] == 0.95).any() else evt.iloc[0]
        u, xi, beta = row["u"], row["xi"], row["beta"]
        pu = (absz > u).mean()
        gpd_tail = np.where(ks > u, pu * np.power(np.maximum(1 + xi * (ks - u) / beta, 1e-12), -1.0 / xi), np.nan)
        ax1.plot(ks, gpd_tail, "-", color=POS, lw=1.4, label=T(r"GPD (EVT) fit"))
    ax1.set_yscale("log")
    ax1.set_xlabel(T(r"threshold $k$ (in units of $\sqrt{a_{xx}}$)"))
    ax1.set_ylabel(T(r"$P(|z|>k)$"))
    ax1.set_title("(b) " + T(r"Pooled tail: data vs Gaussian vs EVT"))
    ax1.legend(fontsize=7); despine(ax1)

    fig.suptitle(T("Expected shortfall and EVT tail assessment, %s held-out sessions" % venue), y=1.02)
    (DIAG_DIR / "figures").mkdir(parents=True, exist_ok=True)
    finish(fig, DIAG_DIR / "figures" / "fig_es_evt.png")
    print(f"\n[es_evt] wrote {DIAG_DIR}/tables/es_evt_summary.csv, es_evt_evt.csv and figures/fig_es_evt.png")


if __name__ == "__main__":
    main()
