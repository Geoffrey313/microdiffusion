# Data

The `clean/` subdirectory contains per-session L1 order-book snapshots for 41 QSE
instruments derived from the Qatar Stock Exchange (QSE) MITch feed, covering
January 2025 to April 2026 (216 trading sessions, 4,032,350 retained events). Of
the 41 retained names, 36 have enough train and test observations to receive
per-symbol spread-tercile edges and enter the forecast and benchmark analyses;
the remaining five are exported but carry S_bin = -1.

Each file is a Parquet table with columns:
  mid, bid, ask, bid_sz, ask_sz

organised as `clean/<MM-DD-YY>/<SYM>_l1.parquet`.

## Availability

The processed data are available to referees and replicators on request.
Contact: geoffrey.ducournau@111dimtech.com

Raw tick-by-tick feed data cannot be redistributed; it is proprietary to QSE/Dimtech.
