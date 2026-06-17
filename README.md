# Microdiffusion — Replication Package

Code and paper source for:

> **From Microprice to Microdiffusion: Heavy-Tailed Price Diffusion in Limit Order Books**
> Geoffrey Ducournau, Yibo Wang, Jinliang Li

## Repository layout

```
microdiffusion/
  code/
    config.yaml          shared config (data path, session list, model params)
    plot_style.py        shared figure style
    01_surface/          §3  conditional SDE surface estimation
    02_innovation/       §4  GH heavy-tailed innovation
    03_robustness/       §5  robustness and validation
    04_benchmarks/       §5/App  GARCH benchmarks
  paper/
    main.tex
    references.bib
    cover_letter_revision.tex
    figures/             committed PNG outputs
  data/
    clean/               processed QSE L1 parquet files (see data/README.md)
  environment.yml
```

## Requirements

```bash
conda env create -f environment.yml
conda activate market-maker
```

## Run order

Scripts are run from their own subfolder. Each reads `config.yaml` and `plot_style.py`
from the `code/` root via `CODE = Path(__file__).resolve().parents[1]`.

### §3 — Surface estimation (`code/01_surface/`)

| Script | Output | Paper |
|--------|--------|-------|
| `conditional_sde_scale_recovery.py` | `fig_axx_surface.png`, `fig_drift_surface.png` | §3 |
| `conditional_sde_cellcounts.py` | `fig_axx_heatmap_counts.png` | §3 |
| `conditional_sde_identifiability.py` | identifiability diagnostics (stdout) | §3 App |
| `conditional_sde_prototype.py` | prototype surface (exploratory) | — |

### §4 — GH innovation (`code/02_innovation/`)

| Script | Output | Paper |
|--------|--------|-------|
| `conditional_sde_heavytail.py` | tail exceedance table (stdout) | §4 |
| `conditional_sde_gh_oos.py` | OOS log-likelihood gain (stdout) | §4 |
| `conditional_sde_gh_loio.py` | `fig_gh_loio.png` | §4 |
| `conditional_sde_ghsim.py` | GH simulation diagnostics | App |
| `conditional_sde_price_band.py` | `fig_price_band.png`, `fig_model_vs_realized.png` | §4 |

### §5 — Robustness (`code/03_robustness/`)

| Script | Output | Paper |
|--------|--------|-------|
| `conditional_sde_var_backtest.py` | `fig_var_backtest.png` | §5 |
| `conditional_sde_regime_transfer.py` | `fig_regime_transfer.png` | §5 |
| `conditional_sde_thinname_exclusion.py` | thin-name robustness table (stdout) | §5 |
| `conditional_sde_tick_control.py` | `fig_tick_control.png` | §5 |
| `conditional_sde_zeromove.py` | `fig_zeromove.png` | §5 |
| `conditional_sde_transfer_coldstart.py` | `fig_transfer_coldstart.png` | §5 |
| `conditional_sde_clustering.py` | violation clustering diagnostics (stdout) | diagnostic — not in manuscript |

### §5/App — GARCH benchmarks (`code/04_benchmarks/`)

| Script | Output | Paper |
|--------|--------|-------|
| `conditional_sde_garch_compare.py` | `fig_garch_compare.png` | §5 |
| `conditional_sde_garch_grid.py` | `fig_garch_calibration.png` | App |
| `conditional_sde_garch_fixedclock.py` | fixed-clock GARCH diagnostics | App |
| `conditional_sde_benchmarks.py` | benchmark comparison table | App |
| `generate_garch_figure.py` | GARCH figure entry point | App |
| `generate_paper1_figures.py` | gap/scale figures | §3 |

## Data

See `data/README.md`. Processed L1 parquet files available on request:
geoffrey.ducournau@111dimtech.com

Raw QSE tick data is proprietary and cannot be redistributed.
