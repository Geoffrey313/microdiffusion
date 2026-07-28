# The Microdiffusion: Book-State Price Risk and Heavy Tails in Event Time

Replication package for:

> **The Microdiffusion: Book-State Price Risk and Heavy Tails in Event Time**
> Geoffrey Ducournau, Yibo Wang, Jinliang Li

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
  environment.yml
  .env.example
```

The `src/` subpackages are split by role, not by chapter. Every path resolves
relative to the repository root through `src/common/paths.py`; nothing depends
on the current working directory or a machine-specific path.

## Requirements

```bash
conda env create -f environment.yml
conda activate market-maker
```

No credentials are needed; see `.env.example`.

## Reproduce

```bash
python reproduce.py            # full chain, then verify every tracked digest
python reproduce.py --fast     # event panel + forecast tables + verify
python reproduce.py --check    # verify digests of existing results only
```

`reproduce.py` runs the stages in dependency order (data, engine, analysis,
figures) and checks each tracked result in `results/` against
`results/digests.sha256`. The forecast tables are the digest-gated outputs; the
engine, robustness, benchmark and figure stages are run for completeness and
write diagnostic tables and manuscript figures that are not tracked.

## Data

See `data/README.md`. The processed L1 parquet files are available on request:
geoffrey.ducournau@111dimtech.com. Raw exchange data is proprietary and cannot
be redistributed.
