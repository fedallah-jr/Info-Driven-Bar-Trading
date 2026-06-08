#!/usr/bin/env python3
"""exp04 GOLD layer — sampled bars (silver) -> model-ready feature table (gold).

For each (coin, bartype, threshold): read data/bars/<bartype>/<COIN>_thr<θ>.parquet, compute
the indicator features PER SEGMENT (so no rolling window crosses a data gap), drop per-segment warm-up
rows (leading NaNs), and write data/datasets/<COIN>_<bartype>_thr<θ>.parquet.

The gold file is RAW (unscaled) features + close + segment_id + t_open. Scaling, windowing, the
expanding-window split, and labeling are done in the loader at train time (the scaler must be fit on
train-only per quarter, so scaled/windowed tensors are deliberately NOT materialized here).

    python3 model/build_dataset.py                       # full grid present on disk
    python3 model/build_dataset.py --coins ETHUSDT --bartypes cusum --thresholds 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.features import compute_features, feature_columns

KEEP = ["t_open", "t_close", "segment_id", "open", "high", "low", "close", "volume"]


def build_one(bars_path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(bars_path).sort_values("t_open").reset_index(drop=True)
    feats = feature_columns()
    parts = []
    for _, seg in bars.groupby("segment_id", sort=False):
        out = compute_features(seg)
        out = out.dropna(subset=feats)              # drop per-segment warm-up
        if len(out):
            parts.append(out)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[KEEP + feats]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build exp04 gold feature datasets from silver bars.")
    ap.add_argument("--coins", default="")           # blank = all coins found
    ap.add_argument("--bartypes", default="cusum,range")
    ap.add_argument("--thresholds", default="0.01,0.02,0.03")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--out-dir", default="data/datasets")
    args = ap.parse_args()

    bartypes = [b.strip() for b in args.bartypes.split(",") if b.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    nfeat = len(feature_columns())

    print(f"{'coin':<9}{'bartype':<7}{'thr':>6}{'bars_in':>10}{'rows_out':>10}{'n_feat':>8}", flush=True)
    for bartype in bartypes:
        bdir = Path(args.bars_dir) / bartype
        for thr in thresholds:
            for path in sorted(bdir.glob("*_thr%.3f.parquet" % thr)):
                coin = path.stem.split("_thr")[0]
                if coins and coin not in coins:
                    continue
                n_in = pd.read_parquet(path, columns=["t_open"]).shape[0]
                df = build_one(path)
                stem = f"{coin}_{bartype}_thr{thr:.3f}"
                df.to_parquet(out / f"{stem}.parquet")
                print(f"{coin:<9}{bartype:<7}{thr:>6.2f}{n_in:>10,}{len(df):>10,}{nfeat:>8}", flush=True)
    print(f"\nDONE -> {out}/  ({nfeat} features/bar)", flush=True)


if __name__ == "__main__":
    main()
