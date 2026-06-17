# Data

The `clean/` subdirectory contains per-session L1 order-book snapshots for 36 QSE
instruments derived from the Qatar Stock Exchange (QSE) MITch feed, covering
January 2025 – April 2026 (216 trading sessions).

Each file is a Parquet table with columns:
  mid, bid, ask, bid_sz, ask_sz

organised as `clean/<MM-DD-YY>/<SYM>_l1.parquet`.

## Availability

The processed data are available to referees and replicators on request.
Contact: geoffrey.ducournau@111dimtech.com

Raw tick-by-tick feed data cannot be redistributed; it is proprietary to QSE/Dimtech.
