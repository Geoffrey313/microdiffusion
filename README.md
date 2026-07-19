# Microdiffusion — Replication Package

Replication code for:

> **The Microdiffusion: Book-State Price Risk and Heavy Tails in Event Time**
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
    05_forecast/         §5.3  nested forecast comparison (which part of the state predicts)
  data/
    clean/               processed QSE L1 parquet files (see data/README.md)
    data_export/         event panel built by 05_forecast (regenerated, not committed)
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

# forecast comparison: build the event panel once, then run the four analyses
python 05_forecast/export_event_panel.py     # writes data/data_export/event_panel.parquet
python 05_forecast/a4_reproduce.py           # panel-fidelity check vs the published transfer
python 05_forecast/a3_nested.py              # nested variance ladder (Table 3)
python 05_forecast/a2_twopart.py             # occurrence x magnitude decomposition (Table 5)
python 05_forecast/a5_losses_garch.py        # forecast losses vs GARCH / joint GARCH-t (Table 8)
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

### §5.3 — Forecast comparison (which part of the state predicts)

These four scripts read a single event panel rather than the per-session parquet files. Build
the panel once with the export step, then run the analyses; each writes its reference tables to
`code/05_forecast/output/`.

```bash
python code/05_forecast/export_event_panel.py   # data/data_export/event_panel.parquet (+ tick_sizes, sbin_edges, garch_spec)
python code/05_forecast/a4_reproduce.py         # a4_reproduction.csv — panel fidelity vs the published transfer
python code/05_forecast/a3_nested.py            # a3_ladder_{levels,pairs}.csv — nested variance ladder
python code/05_forecast/a2_twopart.py           # a2_twopart_{levels,pairs}.csv — occurrence x magnitude
python code/05_forecast/a5_losses_garch.py      # a5_losses_{levels,pairs}.csv + predictions/residuals fast-path files
```

The verdict these produce: the quoted spread carries the transferable one-step forecast content
(it predicts both whether the mid moves and how large the move is); the best-level imbalance is a
real contemporaneous second-moment modulation but adds no one-step point-forecast increment and
does not transfer as a day-to-day shape.

All scripts can be run from the repo root or from inside `code/`. Manuscript figures are
written to `paper/figures/` if a local paper folder is present; diagnostic-only outputs may
be written to a local `output/` subfolder within each section.

## Data

See `data/README.md`. Processed L1 parquet files available on request:
geoffrey.ducournau@111dimtech.com

Raw QSE tick data is proprietary and cannot be redistributed.
