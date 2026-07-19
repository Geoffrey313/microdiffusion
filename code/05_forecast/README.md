# 05_forecast — nested forecast comparison

These scripts answer a single question that the surface estimation in `01_surface` does not:
of the two book-state axes, which one actually carries out-of-sample forecast value for the
next mid-price move? They back Section 5.3 of the manuscript.

Unlike the `01`–`04` scripts, which read the per-session L1 parquet files directly, everything
here reads one flat **event panel**. Build it once:

```bash
python export_event_panel.py
```

This writes, into `data/data_export/` at the repo root (regenerated, not committed):

- `event_panel.parquet` — one row per retained book update, all instruments, full period,
  with the log-mid increment, the state `(I, S)`, the per-instrument bins, and the
  chronological 60/40 train/test split. Built to match `generate_paper1_figures.py` exactly.
- `tick_sizes.csv`, `sbin_edges.csv`, `garch_spec.txt` — the tick rule, per-instrument spread
  tercile edges (train-only), and the GARCH estimation protocol used by the benchmarks.

Then run the four analyses (each writes its reference tables to `output/`):

| script | what it does | table |
|---|---|---|
| `a4_reproduce.py` | walk-forward transfer from the panel alone, checked against the published numbers (panel-fidelity check) | — |
| `a3_nested.py` | nested variance ladder (constant → spread → imbalance → additive → full surface) under out-of-sample winsorised squared-error loss, instrument-clustered bootstrap | Table 3 |
| `a2_twopart.py` | two-part decomposition `E[dx^2] = P(move) x E[dx^2 | move]`: occurrence (log-loss, Brier) and magnitude (QLIKE on nonzero moves) scored separately | Table 5 |
| `a5_losses_garch.py` | predictive log score and variance losses against GARCH and jointly estimated GARCH-t, same protocol; also emits `predictions_test.parquet` and `residuals_var.parquet` | Table 8 |

The consistent finding across all three non-degenerate losses: the **spread** predicts both
whether the mid moves and how large the move is, and this transfers out of sample; the
**imbalance** moves variance contemporaneously (it survives the tick-mechanics controls and
shows the U-shape) but adds no one-step point-forecast increment and does not transfer as a
day-to-day shape. The 30-cell full surface pays more in estimation noise than it earns in
signal at one step, so a spread-only forecaster is not beaten by the full surface.
