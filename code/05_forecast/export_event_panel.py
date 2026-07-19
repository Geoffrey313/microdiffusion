#!/usr/bin/env python3
"""
export_event_panel.py — build the co-author data export for Article B.

Produces, in data/data_export/ (repo root):
  event_panel.parquet   one row per retained book update, all instruments, full period
  tick_sizes.csv        instrument -> tick size (price units) + typical price level
  garch_spec.txt        the GARCH estimation protocol actually used by the benchmarks

Conventions reproduce the paper's headline scripts EXACTLY (generate_paper1_figures.py):
  - increments on the log mid:      dx_log_n = log(mid_{n+1}) - log(mid_n)   (primary target)
  - state sampled at event n (before the move), NaN/inf rows dropped
  - imbalance  I = (bsz - asz) / (bsz + asz)                     (continuous, [-1, 1])
  - spread     S_mid = (ask - bid) / mid                          (the surface scripts' S)
               S_rel = (ask - bid) / (ask + bid) = S_mid / 2      (state-builder convention)
  - I_bin: N_I = 10 uniform bins on [-1, 1]
  - S_bin: M_S = 3 per-instrument spread terciles, edges estimated on the TRAIN days only
  - split_original: sessions ordered chronologically per instrument, first 60% train
    (TRAIN_FRAC = 0.60), final 40% test — the paper's rule
  - tick_size: 0.001 if bid < 10 else 0.01 (QSE rule, as in 03_construct_state.py)

The last event of each session has no next update: dx_* are NaN there and zero = <NA>.
Filters (same as the paper): a session-instrument file needs >= 200 finite mids; an
instrument needs >= 2000 train and >= 1000 test retained rows to contribute S_bin edges
(instruments failing that keep S_bin = -1 but are still exported).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent          # code/05_forecast
OUT = HERE.parents[1] / "data" / "data_export"   # repo/data/data_export
OUT.mkdir(parents=True, exist_ok=True)

N_I, M_S = 10, 3
TRAIN_FRAC = 0.60
MIN_MIDS = 200          # per session file, as in session_increments()
MIN_TRAIN, MIN_TEST = 2000, 1000   # per instrument, as in collect()


def parse_date(s: str):
    mm, dd, yy = s.split("-")
    return (int(yy), int(mm), int(dd))


def session_frame(path: Path, instrument: str, session: str) -> pd.DataFrame | None:
    l1 = pd.read_parquet(path)
    mid = l1["mid"].astype(float)
    if (np.isfinite(mid) & (mid > 0)).sum() < MIN_MIDS:
        return None
    bid = l1["bid"].astype(float); ask = l1["ask"].astype(float)
    bsz = l1["bid_sz"].astype(float); asz = l1["ask_sz"].astype(float)
    ns = pd.to_numeric(l1.get("ns", 0), errors="coerce").fillna(0).astype("int64")
    ts_event = pd.to_datetime(l1["ts"]) + pd.to_timedelta(ns, unit="ns")

    df = pd.DataFrame({
        "instrument": instrument,
        "date": pd.Timestamp(f"20{session[6:8]}-{session[0:2]}-{session[3:5]}"),
        "ts_event": ts_event,
        "mid": mid,
        "I": (bsz - asz) / (bsz + asz).replace(0, np.nan),
        "S_mid": (ask - bid) / mid,
        "S_rel": (ask - bid) / (ask + bid),
        "tick_size": np.where(bid < 10, 0.001, 0.01),
    })
    df["S_tick"] = np.round((ask - bid) / df["tick_size"]).astype("Int64")
    # event order within the session, then forward increments (last row: no next update)
    df = df.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
    df["dx_log"] = np.log(df["mid"]).shift(-1) - np.log(df["mid"])
    df["dx_price"] = df["mid"].shift(-1) - df["mid"]
    df["dx_ticks"] = np.round(df["dx_price"] / df["tick_size"]).astype("Int64")
    df["zero"] = (df["dx_ticks"] == 0).astype("Int64")
    df.loc[df["dx_ticks"].isna(), "zero"] = pd.NA
    # drop rows with non-finite state (same effect as the paper's dropna on I, S)
    df = df[np.isfinite(df["I"]) & np.isfinite(df["S_mid"]) & np.isfinite(df["mid"])]
    return df if len(df) else None


def main():
    cfg_path = HERE.parent / "config.yaml"       # code/config.yaml
    cfg = yaml.safe_load(open(cfg_path))
    clean = (cfg_path.parent / cfg["data"]["out_dir"] / "clean").resolve()
    files: dict[str, list[Path]] = {}
    for p in clean.glob("*/*_l1.parquet"):
        files.setdefault(p.name.replace("_l1.parquet", ""), []).append(p)

    frames, spec_rows = [], []
    for sym in sorted(files):
        paths = sorted(files[sym], key=lambda p: parse_date(p.parent.name))
        ntr = max(1, int(round(TRAIN_FRAC * len(paths))))
        train_sessions = {p.parent.name for p in paths[:ntr]}
        parts = []
        for p in paths:
            f = session_frame(p, sym, p.parent.name)
            if f is None:
                continue
            f["split_original"] = "train" if p.parent.name in train_sessions else "test"
            parts.append(f)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)

        # bins: I uniform on [-1,1]; S per-instrument train terciles (paper rule)
        df["I_bin"] = np.clip(((df["I"] + 1) / 2 * N_I).astype(int), 0, N_I - 1)
        tr = df[df["split_original"] == "train"]
        te = df[df["split_original"] == "test"]
        if len(tr) >= MIN_TRAIN and len(te) >= MIN_TEST:
            s_edges = np.quantile(tr["S_mid"], [1 / 3, 2 / 3])
            df["S_bin"] = np.clip(
                np.searchsorted(s_edges, df["S_mid"].to_numpy(), side="right"), 0, M_S - 1)
            spec_rows.append({"instrument": sym, "s_edge_1": s_edges[0], "s_edge_2": s_edges[1],
                              "n_train": len(tr), "n_test": len(te),
                              "n_sessions": len(paths), "n_train_sessions": ntr})
        else:
            df["S_bin"] = -1   # instrument too thin for the paper's per-symbol edges
            spec_rows.append({"instrument": sym, "s_edge_1": np.nan, "s_edge_2": np.nan,
                              "n_train": len(tr), "n_test": len(te),
                              "n_sessions": len(paths), "n_train_sessions": ntr})
        frames.append(df)
        print(f"{sym}: {len(df):>9,} rows  ({len(paths)} sessions, {ntr} train)")

    panel = pd.concat(frames, ignore_index=True)
    cols = ["instrument", "date", "ts_event", "mid",
            "dx_log", "dx_price", "dx_ticks", "zero",
            "I", "S_mid", "S_rel", "S_tick", "I_bin", "S_bin",
            "tick_size", "split_original"]
    panel = panel[cols]
    panel.to_parquet(OUT / "event_panel.parquet", index=False)
    pd.DataFrame(spec_rows).to_csv(OUT / "sbin_edges.csv", index=False)

    # tick_sizes.csv: tick rule + typical price level per instrument
    ts = (panel.groupby("instrument")
                .agg(tick_size=("tick_size", lambda s: s.mode().iloc[0]),
                     median_mid=("mid", "median"),
                     n_updates=("mid", "size"))
                .reset_index())
    ts.to_csv(OUT / "tick_sizes.csv", index=False)

    (OUT / "garch_spec.txt").write_text(
        "GARCH(1,1) benchmarks (conditional_sde_garch_compare.py / _benchmarks.py / _garch_grid.py):\n"
        "fit once on the TRAINING partition per instrument, with the series passed to the\n"
        "optimiser capped at the most recent 12,000 observations (FIT_CAP = 12000); the fitted\n"
        "filter is then run over the full ordered series and evaluated on the test partition.\n"
        "No rolling re-estimation. Split: chronological 60/40 by session (TRAIN_FRAC = 0.60).\n")

    n_zero = int(panel["zero"].sum())
    n_nonnull = int(panel["zero"].notna().sum())
    print(f"\nTOTAL: {len(panel):,} rows, {panel['instrument'].nunique()} instruments, "
          f"{panel['date'].nunique()} sessions")
    print(f"zero-move share (of rows with a next update): {n_zero / n_nonnull:.3f}")
    print(f"train share: {(panel['split_original'] == 'train').mean():.3f}")
    print(f"wrote: {OUT}/event_panel.parquet, tick_sizes.csv, garch_spec.txt, sbin_edges.csv")


if __name__ == "__main__":
    main()
