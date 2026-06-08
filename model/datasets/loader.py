"""Loader: gold features -> (train/val/test) windowed arrays for one expanding-window quarter.

Leakage discipline, centralized here:
  * StandardScaler is fit on the TRAIN feature rows only, then applied to val/test.
  * A 96-bar window never crosses a data gap (segment_id) OR the train/val/test boundary.
  * The label bar is the window's last bar.

`load_quarter`        — one coin.
`load_quarter_pooled` — many coins, each scaled by its OWN train stats (so price-level features are
                        comparable), then train/val pooled; test kept per-coin. This is the
                        cross-sectional ("predict coins together") path.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # model on path
from config import parse_quarter, quarter_ord
from datasets.features import feature_columns
from datasets.labeling import make_labels


def _windows(Xs, y, close, topen, seg, lo, hi, W):
    """Windows ending in [lo+W-1, hi): label bar y[i]>=0 and window stays in one segment.
    Returns X, y, close, t_open, fwd_ret (next-bar return at the label bar)."""
    F = Xs.shape[1]
    empty = (np.empty((0, W, F), np.float32), np.empty(0, np.int64),
             np.empty(0), np.empty(0, np.int64), np.empty(0))
    if hi - lo < W:
        return empty
    ends = np.arange(lo + W - 1, hi)
    ends = ends[(y[ends] >= 0) & (seg[ends] == seg[ends - W + 1])]
    if len(ends) == 0:
        return empty
    win = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]
    n = len(close)
    nxt = np.clip(ends + 1, 0, n - 1)
    ok = (ends + 1 < n) & (seg[nxt] == seg[ends])
    fwd = np.where(ok, close[nxt] / close[ends] - 1.0, np.nan)
    return (Xs[win].astype(np.float32), y[ends].astype(np.int64),
            close[ends].astype(np.float64), topen[ends].astype(np.int64), fwd)


def _coin_splits(df: pd.DataFrame, cfg, test_quarter: str) -> dict:
    """One coin's scaled, windowed train/val/test arrays for the given test quarter.
    Scaler is fit on this coin's TRAIN rows only."""
    df = df.sort_values("t_open").reset_index(drop=True)
    feats = feature_columns()
    W = cfg.window
    close = df["close"].to_numpy(np.float64)
    topen = df["t_open"].to_numpy(np.int64)
    seg = df["segment_id"].to_numpy()
    y = make_labels(close, seg, cfg.label_scheme, cfg.barrier, cfg.vertical)
    qord = quarter_ord(topen)

    test_ord = parse_quarter(test_quarter)
    val_ord = test_ord - 1
    tr_hi = int(np.searchsorted(qord, val_ord, "left"))
    va_lo, va_hi = tr_hi, int(np.searchsorted(qord, val_ord, "right"))
    te_lo, te_hi = (int(np.searchsorted(qord, test_ord, "left")),
                    int(np.searchsorted(qord, test_ord, "right")))

    F = len(feats)
    if tr_hi < W:                       # not enough train history (e.g. late-listing coin, early quarter)
        e = np.empty
        return {"Xtr": e((0, W, F), np.float32), "ytr": e(0, np.int64),
                "Xva": e((0, W, F), np.float32), "yva": e(0, np.int64),
                "Xte": e((0, W, F), np.float32), "yte": e(0, np.int64),
                "test_close": e(0), "test_topen": e(0, np.int64), "test_fwd_ret": e(0)}

    Xraw = df[feats].to_numpy(np.float64)
    Xs = StandardScaler().fit(Xraw[:tr_hi]).transform(Xraw)        # FIT ON TRAIN ONLY
    Xtr, ytr, *_ = _windows(Xs, y, close, topen, seg, 0, tr_hi, W)
    Xva, yva, *_ = _windows(Xs, y, close, topen, seg, va_lo, va_hi, W)
    Xte, yte, cte, tte, rte = _windows(Xs, y, close, topen, seg, te_lo, te_hi, W)
    return {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte,
            "test_close": cte, "test_topen": tte, "test_fwd_ret": rte}


def load_quarter(cfg, test_quarter: str, root: str = "data/datasets") -> dict:
    df = pd.read_parquet(cfg.dataset_path(root))
    d = _coin_splits(df, cfg, test_quarter)
    d["n_features"] = len(feature_columns())
    d["test_quarter"] = test_quarter
    d["sizes"] = {"windows": {"train": len(d["ytr"]), "val": len(d["yva"]), "test": len(d["yte"])}}
    return d


def load_quarter_pooled(coins, cfg, test_quarter: str, root: str = "data/datasets") -> dict:
    """Pool TRAIN/VAL windows across coins (each scaled by its own train stats); keep TEST per-coin."""
    parts, test = [], {}
    for c in coins:
        df = pd.read_parquet(replace(cfg, coin=c).dataset_path(root))
        s = _coin_splits(df, cfg, test_quarter)
        parts.append(s)
        if len(s["yte"]) > 0:
            test[c] = {k: s[k] for k in ("Xte", "yte", "test_close", "test_topen", "test_fwd_ret")}

    def cat(key):
        arrs = [p[key] for p in parts if len(p[key]) > 0]
        return np.concatenate(arrs) if arrs else parts[0][key]

    return {"Xtr": cat("Xtr"), "ytr": cat("ytr"), "Xva": cat("Xva"), "yva": cat("yva"),
            "test": test, "n_features": len(feature_columns()), "test_quarter": test_quarter}
