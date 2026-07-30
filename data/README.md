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

## External cryptocurrency sample (Appendix E)

The external-sample robustness check uses a one-second aggregated level-one feed
for three cryptocurrencies (Bitcoin, Ether, Cardano) over 2021-04-07 to
2021-04-19. It is read directly from three comma-separated files:

  crypto/BTC_1sec.csv
  crypto/ETH_1sec.csv
  crypto/ADA_1sec.csv

Each file carries at least the columns `system_time`, `midpoint`, `spread`,
`buys`, `sells`, `bids_notional_0`, and `asks_notional_0`. Place them under
`data/crypto/` or point the environment variable `MICRODIFFUSION_CRYPTO_DIR` at
their directory. The cleaning, state construction, and transfer test are then run
by `src/analysis/crypto_transfer.py`, wired into `reproduce.py` as the external
stage. When the feed is absent the stage self-skips and the rest of the chain
completes on the QSE sample alone. This feed is not redistributed with the
package and is not part of the digest gate.

## Availability

The processed data are available to referees and replicators on request.
Contact: geoffrey.ducournau@111dimtech.com

Raw tick-by-tick feed data cannot be redistributed; it is proprietary to QSE/Dimtech.
