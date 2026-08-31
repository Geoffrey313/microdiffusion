#!/usr/bin/env python3
"""
reproduce.py - single deterministic entry point for the replication package.

It runs the pipeline in dependency order - data assembly, then the model
(engine), then the analyses that produce the article tables, then the figures -
and verifies that every tracked result keeps the same SHA-256 digest it had when
the manuscript was written. The digest manifest is results/digests.sha256.

Usage
    python reproduce.py                 run the full chain, then verify digests
    python reproduce.py --fast          panel (if missing) + gated tables + verify
    python reproduce.py --check         only verify digests of existing results
    python reproduce.py --rebuild-panel force a rebuild of the event panel
    python reproduce.py --us            stage the external US sample and run the
                                        full battery into results_us/

The digest-checked outputs are the forecast tables plus the pooled descriptive
table descriptive_stats_qse.csv (written by figures/descriptive_stats.py). These
gated producers are exactly the modules --fast runs before verify(). The
remaining engine, robustness, benchmark, external and figure stages produce
diagnostic and manuscript figures that are not tracked; they are run for
completeness and to confirm the whole chain executes, but they are not part of
the digest gate. The
external stages (Appendix E, event-time cryptocurrency quotes; Appendix F, a
United States large-cap level-one panel) need feeds that are not redistributed
with the package; each self-skips when its feed is absent.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
PANEL = ROOT / "data" / "data_export" / "event_panel.parquet"
MANIFEST = RESULTS / "digests.sha256"

# Parallel output tree for the external United States battery (--us): the same
# stages run against the US sample, written here so the primary QSE results are
# untouched. Populated by staging the US panel into the QSE clean layout and
# pointing the path-resolution environment overrides at this root.
US_ROOT = ROOT / "data_us"
US_RESULTS = ROOT / "results_us"

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
# Figure-stage producer of a digest-gated table (the pooled descriptive table
# descriptive_stats_qse.csv). It is gated, so it must run under --fast too.
GATED_FIGURES = ["figures/descriptive_stats.py"]
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
# External-sample robustness (Appendices E and F). Neither the event-time
# cryptocurrency quotes feed nor the United States level-one panel is
# redistributed with the package, so each stage self-skips when its feed is
# absent and its diagnostic outputs are not part of the digest gate.
EXTERNAL = [
    "analysis/crypto_transfer.py",
    "figures/crypto_appendix_figures.py",
    "analysis/us_transfer.py",
    "figures/us_appendix_figures.py",
]
FIGURES = [
    "figures/surface_heatmap_counts.py",
    "figures/price_band_series.py",
    "figures/gap_figures.py",
    "figures/garch_figure.py",
]


def run(module: str, env: dict | None = None, abort: bool = True,
        extra_args: list[str] | None = None) -> int:
    """Run one pipeline module as a subprocess.

    Aborts the whole run on failure when abort is True (the primary QSE chain);
    when abort is False the return code is reported and returned so the caller can
    continue past a stage that does not port to the external sample.
    """
    path = SRC / module
    print(f"\n=== run {module} ===", flush=True)
    r = subprocess.run([sys.executable, str(path), *(extra_args or [])],
                       cwd=str(ROOT), env=env)
    if r.returncode != 0 and abort:
        sys.exit(f"[reproduce] FAILED: {module} (exit {r.returncode})")
    return r.returncode


def us_battery() -> None:
    """Run the full battery on the external United States sample (Appendix F).

    Stages the flat US panel into the QSE clean layout, rebuilds the US event
    panel from that staged snapshot, then runs the same data,
    engine, forecast, robustness, benchmark and figure stages against it via the
    path-resolution environment overrides, writing every table and figure under
    results_us/. The digest gate does not apply (the US sample is external and not
    tracked); stage failures are collected and reported rather than aborting, so a
    stage that does not port to this venue does not stop the rest.
    """
    clean = US_ROOT / "clean"
    fig = US_RESULTS / "figures"
    sections = US_RESULTS / "sections"
    for d in (clean, US_ROOT / "data_export", US_RESULTS / "tables",
              US_RESULTS / "diagnostics" / "tables",
              US_RESULTS / "diagnostics" / "figures", fig, sections):
        d.mkdir(parents=True, exist_ok=True)

    # Stage US_DIR -> US clean layout. Run without the data-dir override so the
    # adapter reads the real US panel, and write into the US root's clean tree.
    print("\n=== stage US panel into clean layout ===", flush=True)
    if run("data/us_clean_adapter.py", abort=False,
           extra_args=["--out", str(clean)]) != 0:
        sys.exit("[reproduce --us] staging failed; is the US panel present?")

    env = {**os.environ,
           "MICRODIFFUSION_DATA_DIR": str(US_ROOT),
           "MICRODIFFUSION_RESULTS_DIR": str(US_RESULTS),
           "MICRODIFFUSION_FIG_DIR": str(fig),
           "MICRODIFFUSION_SECTIONS_DIR": str(sections)}

    for m in DATA:
        if run(m, env=env, abort=False) != 0:
            sys.exit(f"[reproduce --us] failed to build the US event panel ({m})")

    stages = ENGINE + FORECAST + ROBUSTNESS + BENCHMARKS + GATED_FIGURES + FIGURES
    failed = []
    for m in stages:
        if run(m, env=env, abort=False) != 0:
            failed.append(m)

    print("\n[reproduce --us] US battery complete. "
          f"Results under {US_RESULTS.relative_to(ROOT)}/.")
    if failed:
        print("[reproduce --us] stages that did not port (review case by case):")
        for m in failed:
            print(f"    - {m}")
    else:
        print("[reproduce --us] every stage ran.")


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
    ap.add_argument("--fast", action="store_true", help="panel + gated tables + verify only")
    ap.add_argument("--check", action="store_true", help="verify digests only, run nothing")
    ap.add_argument("--rebuild-panel", action="store_true", help="rebuild the event panel even if present")
    ap.add_argument("--us", action="store_true",
                    help="run the full battery on the external US sample (Appendix F) into results_us/")
    args = ap.parse_args()

    if args.check:
        verify()
        return

    if args.us:
        us_battery()
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
        for m in FORECAST + GATED_FIGURES:
            run(m)
        verify()
        return

    for m in ENGINE + FORECAST + ROBUSTNESS + BENCHMARKS + EXTERNAL + GATED_FIGURES + FIGURES:
        run(m)
    verify()


if __name__ == "__main__":
    main()
