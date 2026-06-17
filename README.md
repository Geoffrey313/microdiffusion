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
  data/
    clean/               processed QSE L1 parquet files (see data/README.md)
  paper/
    main.tex             current unblinded manuscript source
    main.pdf             current unblinded manuscript PDF
    figures/             manuscript figures
    sections/            included descriptive-statistics tables
  journal/
    JEF/                 Journal of Empirical Finance submission copy
  environment.yml
```

## Requirements

```bash
conda env create -f environment.yml
conda activate market-maker
```

## Run full pipeline

To reproduce all results in order:

```bash
cd code

python 01_surface/conditional_sde_scale_recovery.py
python 01_surface/conditional_sde_cellcounts.py

python 02_innovation/conditional_sde_heavytail.py
python 02_innovation/conditional_sde_gh_oos.py
python 02_innovation/conditional_sde_gh_loio.py
python 02_innovation/conditional_sde_price_band.py

python 03_robustness/conditional_sde_var_backtest.py
python 03_robustness/conditional_sde_regime_transfer.py
python 03_robustness/conditional_sde_thinname_exclusion.py
python 03_robustness/conditional_sde_tick_control.py
python 03_robustness/conditional_sde_zeromove.py
python 03_robustness/conditional_sde_transfer_coldstart.py

python 04_benchmarks/conditional_sde_garch_compare.py
python 04_benchmarks/conditional_sde_garch_grid.py
python 04_benchmarks/generate_paper1_figures.py
```

## Run individual scripts

### §3 — Surface estimation

```bash
python code/01_surface/conditional_sde_scale_recovery.py   # fig_axx_surface.png, fig_drift_surface.png
python code/01_surface/conditional_sde_cellcounts.py       # fig_axx_heatmap_counts.png
python code/01_surface/conditional_sde_identifiability.py  # identifiability diagnostics (stdout)
```

### §4 — GH innovation

```bash
python code/02_innovation/conditional_sde_heavytail.py     # tail exceedance table (stdout)
python code/02_innovation/conditional_sde_gh_oos.py        # OOS log-likelihood gain (stdout)
python code/02_innovation/conditional_sde_gh_loio.py       # fig_gh_loio.png
python code/02_innovation/conditional_sde_price_band.py    # fig_price_band.png, fig_model_vs_realized.png
python code/02_innovation/conditional_sde_ghsim.py         # GH simulation diagnostics (appendix)
```

### §5 — Robustness

```bash
python code/03_robustness/conditional_sde_var_backtest.py       # fig_var_backtest.png
python code/03_robustness/conditional_sde_regime_transfer.py    # fig_regime_transfer.png
python code/03_robustness/conditional_sde_thinname_exclusion.py # thin-name table (stdout)
python code/03_robustness/conditional_sde_tick_control.py       # fig_tick_control.png
python code/03_robustness/conditional_sde_zeromove.py           # fig_zeromove.png
python code/03_robustness/conditional_sde_transfer_coldstart.py # fig_transfer_coldstart.png
# diagnostic only — not in manuscript:
python code/03_robustness/conditional_sde_clustering.py         # violation clustering (stdout)
```

### §5/App — GARCH benchmarks

```bash
python code/04_benchmarks/conditional_sde_garch_compare.py      # fig_garch_compare.png
python code/04_benchmarks/conditional_sde_garch_grid.py         # fig_garch_calibration.png
python code/04_benchmarks/conditional_sde_garch_fixedclock.py   # fixed-clock diagnostics
python code/04_benchmarks/conditional_sde_benchmarks.py         # benchmark table
python code/04_benchmarks/generate_garch_figure.py
python code/04_benchmarks/generate_paper1_figures.py
```

All scripts can be run from the repo root or from inside `code/`. Figures are written to
`paper/figures/` when they are manuscript figures; diagnostic-only outputs may be written
to a local `output/` subfolder within each section.

## Paper and submission files

- `paper/main.tex` and `paper/main.pdf`: current unblinded paper.
- `paper/figures/`: complete 20-figure set used by the manuscript.
- `journal/JEF/manuscript_anonymized.pdf`: double-anonymized manuscript for review.
- `journal/JEF/title_page.pdf`: separate title page with authors, declarations, funding,
  code availability, and corresponding-author details.
- `journal/JEF/cover_letter_JEF.pdf`: JEF cover letter.
- `journal/JEF/highlights_JEF.txt`: Elsevier highlights.
- `journal/JEF/paper1_JEF_submission_package.zip`: complete JEF upload/archive package.

## Data

See `data/README.md`. Processed L1 parquet files available on request:
geoffrey.ducournau@111dimtech.com

Raw QSE tick data is proprietary and cannot be redistributed.
