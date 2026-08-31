"""
paths.py - portable path resolution shared by every module in the package.

All locations are resolved relative to the repository root (found from this
file), never from the current working directory or a hard-coded machine path,
so a script behaves identically whether it is run from the repo root, from
inside src/, or through reproduce.py. Import the anchors you need:

    from common.paths import CLEAN_DIR, PANEL, RESULTS_DIR, FIG_DIR, load_config

Anchors
    REPO_ROOT    repository root
    DATA_DIR     data/                 (raw and processed inputs, git-ignored)
    CLEAN_DIR    data/clean/           processed per-session L1 parquet
    DATA_EXPORT  data/data_export/     flat event panel and precomputed files
    PANEL        the event_panel.parquet used by the forecast analyses
    CRYPTO_DIR   external event-time crypto quotes feed (Appendix E, git-ignored)
    US_DIR       external United States level-one panel (Appendix F, git-ignored)
    RESULTS_DIR  results/              article tables written by the analyses
    DIAG_DIR     results/diagnostics/  diagnostic-only tables and figures
    FIG_DIR      paper/figures/        manuscript figures (local paper build)
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
COMMON = SRC / "common"

# Data lives at repo/data by default; an optional environment override lets the
# data sit outside the repository without any code change.
DATA_DIR = Path(os.environ.get("MICRODIFFUSION_DATA_DIR", REPO_ROOT / "data"))
CLEAN_DIR = DATA_DIR / "clean"
DATA_EXPORT = DATA_DIR / "data_export"
PANEL = DATA_EXPORT / "event_panel.parquet"

# External event-time cryptocurrency quote feed for the Appendix E robustness check.
# It is not redistributed with the package; place the Tardis/Coinbase
# <SYM>_<YYYY-MM-DD>.csv.gz files under data/crypto or point
# MICRODIFFUSION_CRYPTO_DIR at them.
CRYPTO_DIR = Path(os.environ.get("MICRODIFFUSION_CRYPTO_DIR", DATA_DIR / "crypto"))

# External United States large-cap level-one panel for the Appendix F robustness
# check, rebuilt from the free Nasdaq TotalView-ITCH 5.0 sample by
# data/us_itch_fetch.py. Neither the raw feed nor the reconstructed panel is
# redistributed; the <SYM>_<YYYY-MM-DD>.parquet files live under data/us_itch or
# wherever MICRODIFFUSION_US_DIR points.
US_DIR = Path(os.environ.get("MICRODIFFUSION_US_DIR", DATA_DIR / "us_itch"))

# Output roots. Defaults sit under the repository so the primary QSE reproduction
# is unchanged; optional environment overrides let a parallel run (e.g. the
# external United States battery) write its tables and figures to a separate tree
# without editing any stage. Every stage imports these anchors, so overriding the
# environment redirects the whole battery at once.
RESULTS_DIR = Path(os.environ.get("MICRODIFFUSION_RESULTS_DIR", REPO_ROOT / "results"))
DIAG_DIR = RESULTS_DIR / "diagnostics"
FIG_DIR = Path(os.environ.get("MICRODIFFUSION_FIG_DIR", REPO_ROOT / "paper" / "figures"))
# LaTeX table fragments \input by the manuscript. Overridable so a parallel run
# writes its tables outside the manuscript tree instead of overwriting the primary
# QSE fragments.
SECTIONS_DIR = Path(os.environ.get("MICRODIFFUSION_SECTIONS_DIR", REPO_ROOT / "paper" / "sections"))


def load_config() -> dict:
    """Return the shared protocol configuration (src/common/config.yaml)."""
    with open(COMMON / "config.yaml") as fh:
        return yaml.safe_load(fh)


def ensure(path: Path) -> Path:
    """Create a directory (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
