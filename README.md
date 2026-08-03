# The Microdiffusion: Book-State Price Risk and Heavy Tails in Event Time

Anonymized replication package for:

> **The Microdiffusion: Book-State Price Risk and Heavy Tails in Event Time**
> (author identities withheld for double-anonymized peer review)

## What the paper shows

In event time, the one-step conditional variance of the log mid-price is a
function of the visible book state, the diffusion surface a_xx(I, S) over
best-level imbalance I and relative spread S. The innovation that remains after
scaling by that surface is heavy-tailed and is well described by a Generalized
Hyperbolic law. A nested forecast comparison then asks which part of the state
carries out-of-sample predictive value: the quoted spread carries the
transferable one-step forecast content (it predicts both whether the mid moves
and how large the move is), while the imbalance is a real contemporaneous
second-moment modulation that adds no one-step point-forecast increment and does
not transfer as a day-to-day shape.

## Core hypotheses and results

The paper rests on one maintained assumption (H1) and four substantive
hypotheses (H2-H5):

- **H1 (maintained):** the conditional drift is effectively two-dimensional in
  the book state (I, S); the price level adds no further information about the
  expected next move, and is the coordinate left free to diffuse.
- **H2:** the microdiffusion a_xx(I, S) is a stable function of the book state,
  reappearing on disjoint, unseen days.
- **H3:** a Gaussian innovation is insufficient; the scaled residual is
  heavy-tailed and requires a Generalized Hyperbolic law.
- **H4:** scale (the surface) and shape (the heavy tail) are separately
  identified and each is necessary.
- **H5:** the one-step forecast content of the state is concentrated in the
  spread; the imbalance moves the contemporaneous variance but adds no further
  one-step accuracy.

**Headline results:** the state-to-variance ranking transfers out of sample, the
spread-driven level transfers across days, instruments, and volatility regimes,
the imbalance transfers weakly as a shape; the heavy-tailed innovation reduces
extreme-tail under-coverage in intraday VaR backtests and ties a jointly
estimated GARCH-t on predictive log score while beating its Gaussian version.

## Repository layout

```
microdiffusion/
  src/
    common/      shared configuration, figure style, and portable paths
    data/        event-panel assembly from the processed L1 files
    engine/      the model object: diffusion surface and GH innovation law
    analysis/    the tests and decompositions that produce the results
    figures/     manuscript figure and table scripts
  results/       article tables written by the analyses (with digests.sha256)
  data/          raw and processed inputs, git-ignored (see data/README.md)
  reproduce.py   single deterministic entry point with digest checks
  requirements.txt
  .env.example
```

The `src/` subpackages are split by role, not by chapter. Every path resolves
relative to the repository root through `src/common/paths.py`; nothing depends
on the current working directory or a machine-specific path.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11. No credentials are needed; see `.env.example`.

## Reproduce

```bash
python reproduce.py            # full chain, then verify every tracked digest
python reproduce.py --fast     # event panel + gated tables + verify
python reproduce.py --check    # verify digests of existing results only
```

`reproduce.py` runs the stages in dependency order (data, engine, analysis,
figures) and checks each tracked result in `results/` against
`results/digests.sha256`. The digest-gated outputs are the forecast tables plus
the pooled descriptive table `descriptive_stats_qse.csv`; these are exactly what
`--fast` regenerates before verifying. The remaining engine, robustness,
benchmark and figure stages are run for completeness and write diagnostic tables
and manuscript figures that are not tracked.

Note the difference between the modes: `--check` only re-hashes the tracked
`results/*.csv` already on disk against the manifest (a fast integrity check, it
recomputes nothing), whereas `--fast` and the full run **regenerate** the gated
tables from the data and then verify. A genuine reproduction therefore requires a
`--fast` or full run, not `--check` alone.

## Data

See `data/README.md`. The processed L1 parquet files are available to referees and
replicators on request through the journal's editorial system (author contact
withheld for double-anonymized review). Raw exchange data is proprietary and cannot
be redistributed.
