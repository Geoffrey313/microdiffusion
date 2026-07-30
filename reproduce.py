#!/usr/bin/env python3
"""
reproduce.py - single deterministic entry point for the replication package.

It runs the pipeline in dependency order - data assembly, then the model
(engine), then the analyses that produce the article tables, then the figures -
and verifies that every tracked result keeps the same SHA-256 digest it had when
the manuscript was written. The digest manifest is results/digests.sha256.

Usage
    python reproduce.py                 run the full chain, then verify digests
    python reproduce.py --fast          panel (if missing) + forecast tables + verify
    python reproduce.py --check         only verify digests of existing results
    python reproduce.py --rebuild-panel force a rebuild of the event panel

The forecast tables in results/ are the digest-checked outputs. The engine,
robustness, benchmark, external and figure stages produce diagnostic and
manuscript figures that are not tracked; they are run for completeness and to
confirm the whole chain executes, but they are not part of the digest gate. The
external stage (Appendix E) needs the one-second cryptocurrency feed, which is
not redistributed with the package; it self-skips when the feed is absent.
"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
PANEL = ROOT / "data" / "data_export" / "event_panel.parquet"
MANIFEST = RESULTS / "digests.sha256"

# Stages, in dependency order. Each entry is a module path under src/.
DATA = ["data/event_panel.py"]
ENGINE = [
    "engine/surface_scale_recovery.py",
    "engine/surface_prototype.py",
    "engine/innovation_heavytail.py",
    "engine/innovation_oos.py",
    "engine/innovation_loio.py",
    "engine/innovation_simulation.py",
]
# The forecast analyses write the tracked tables in results/.
FORECAST = [
    "analysis/forecast_panel_repro.py",
    "analysis/forecast_nested_ladder.py",
    "analysis/forecast_two_part.py",
    "analysis/forecast_losses_garch.py",
    "analysis/microprice_oos.py",
]
ROBUSTNESS = [
    "analysis/surface_identifiability.py",
    "analysis/var_backtest.py",
    "analysis/regime_transfer.py",
    "analysis/thinname_exclusion.py",
    "analysis/tick_control.py",
    "analysis/transfer_coldstart.py",
    "analysis/zeromove.py",
    "analysis/violation_clustering.py",
]
BENCHMARKS = [
    "analysis/garch_compare.py",
    "analysis/garch_grid.py",
    "analysis/garch_fixedclock.py",
    "analysis/return_history_benchmarks.py",
]
# External-sample robustness (Appendix E). The one-second cryptocurrency feed is
# not redistributed with the package, so this stage self-skips when the feed is
# absent and its diagnostic outputs are not part of the digest gate.
EXTERNAL = [
    "analysis/crypto_transfer.py",
]
FIGURES = [
    "figures/surface_heatmap_counts.py",
    "figures/price_band_series.py",
    "figures/gap_figures.py",
    "figures/garch_figure.py",
    "figures/descriptive_stats.py",
]


def run(module: str) -> None:
    """Run one pipeline module as a subprocess, aborting on failure."""
    path = SRC / module
    print(f"\n=== run {module} ===", flush=True)
    r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"[reproduce] FAILED: {module} (exit {r.returncode})")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify() -> None:
    """Compare results/*.csv against the recorded digest manifest."""
    if not MANIFEST.exists():
        sys.exit(f"[reproduce] no manifest at {MANIFEST}")
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    ok = True
    for name, want in sorted(expected.items()):
        f = RESULTS / name
        if not f.exists():
            print(f"[MISS] {name}"); ok = False; continue
        got = sha256(f)
        if got == want:
            print(f"[ OK ] {name}")
        else:
            print(f"[FAIL] {name}\n       expected {want}\n       got      {got}"); ok = False
    if not ok:
        sys.exit("[reproduce] digest verification FAILED")
    print("\n[reproduce] all tracked results reproduce byte-for-byte.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the replication package.")
    ap.add_argument("--fast", action="store_true", help="panel + forecast tables + verify only")
    ap.add_argument("--check", action="store_true", help="verify digests only, run nothing")
    ap.add_argument("--rebuild-panel", action="store_true", help="rebuild the event panel even if present")
    args = ap.parse_args()

    if args.check:
        verify()
        return

    # Some diagnostic scripts write into results/diagnostics without creating it.
    for d in (RESULTS / "diagnostics" / "tables", RESULTS / "diagnostics" / "figures"):
        d.mkdir(parents=True, exist_ok=True)

    if args.rebuild_panel or not PANEL.exists():
        for m in DATA:
            run(m)
    else:
        print(f"[reproduce] event panel present, skipping build ({PANEL.relative_to(ROOT)})")

    if args.fast:
        for m in FORECAST:
            run(m)
        verify()
        return

    for m in ENGINE + FORECAST + ROBUSTNESS + BENCHMARKS + EXTERNAL + FIGURES:
        run(m)
    verify()


if __name__ == "__main__":
    main()
