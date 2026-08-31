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

The external-sample robustness check uses an event-time best-quote feed for five
cryptocurrencies (Bitcoin, Ether, Zcash, Ondo, Stellar; symbols `BTC-USD`,
`ETH-USD`, `ZEC-USD`, `ONDO-USD`, `XLM-USD`) traded on Coinbase. The feed is the
Coinbase "quotes" archive of the Tardis.dev public datasets service: one row per
change of the top of book, price or size, so the sampling clock is the book event
itself rather than a fixed wall-clock bin. Each day-block is one calendar day; the
panel spans the first calendar day of each month from 2025-10-01 to 2026-06-01
(nine day-blocks), which the service serves at no cost and without a key.

Each source file is a gzip-compressed comma-separated archive named

  crypto/<SYM>_<YYYY-MM-DD>.csv.gz

with the columns `exchange`, `symbol`, `timestamp`, `local_timestamp`,
`ask_amount`, `ask_price`, `bid_price`, and `bid_amount`; the state builder reads
`timestamp` (microseconds) and the best `ask_price`, `bid_price`, `ask_amount`,
and `bid_amount`. Place the files under `data/crypto/` or point the environment
variable `MICRODIFFUSION_CRYPTO_DIR` at their directory. The panel is fetched by
`src/data/crypto_fetch.py`; the instrument selection rule (a pair is retained only
if its spread varies materially and its imbalance is dispersed on every screened
date) is recorded and reproducible in `src/data/crypto_screen.py`. Cleaning, state
construction, and the walk-forward transfer test are run by
`src/analysis/crypto_transfer.py`, wired into `reproduce.py` as the external stage.
When the feed is absent the stage self-skips and the rest of the chain completes on
the QSE sample alone. This feed is external to the package: its redistribution is
governed by the Tardis.dev terms, so it is not included in the repository and is
not part of the digest gate.

## External United States large-cap sample (Appendix F)

The second external check uses a reconstructed event-time top-of-book feed for a
fixed ex-ante panel of large, actively traded capitalisations listed on the
Nasdaq Stock Market over the sample window (`AAPL`, `MSFT`, `AMZN`, `GOOGL`,
`FB`, `INTC`, `CSCO`, `PEP`, `NVDA`, `CMCSA`). `FB` is the canonical key for
the Facebook/Meta instrument and is matched to `META` from 2022-06-09 onward.
The source is the free Nasdaq TotalView-ITCH 5.0 sample, an
order-by-order message feed served without a key or a fee from
`https://emi.nasdaq.com/ITCH/Nasdaq ITCH/`. Seventeen dated sample days are
currently configured and each is one walk-forward day-block:

  01302019, 03272019, 07302019, 08302019, S101819-v50,
  10302019, 12302019, 01302020, S071321-v50, S081321-v50,
  S112825-v50, S120825-v50, S120925-v50, S121025-v50,
  S121125-v50, S121225-v50, S061226-v50

`src/data/us_itch_fetch.py` streams each day through `curl` and decompresses it in
flight, reconstructs the top of book for the target instruments, and writes one
compact level-one panel per instrument and day. The raw feed (about five gigabytes
per day once decompressed) is never written to disk, and only the collapsed panel
is kept: one row per change of the top of book, restricted to the regular session
and capped per session, so the whole reconstructed sample is a few hundred
megabytes rather than tens of gigabytes. Each output is a Parquet table named

  us_itch/<SYM>_<YYYY-MM-DD>.parquet

with the columns `ts`, `bid`, `bid_sz`, `ask`, `ask_sz`. Place the files under
`data/us_itch/` or point the environment variable `MICRODIFFUSION_US_DIR` at their
directory. Cleaning, state construction, and the walk-forward transfer test are run
by `src/analysis/us_transfer.py`, wired into `reproduce.py` as an external stage.
When the panel is absent the stage self-skips and the rest of the chain completes
on the QSE sample alone. United States capitalisations whose primary listing is
another venue are not carried by this single-venue feed and are out of scope;
this is stated with the result.

## Availability

The processed data are available to referees and replicators on request through the
journal's editorial system (author contact withheld for double-anonymized review).

Raw tick-by-tick feed data cannot be redistributed; it is proprietary to the exchange
and its data provider.
