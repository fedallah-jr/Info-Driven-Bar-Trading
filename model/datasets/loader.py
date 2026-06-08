"""exp04 loader: gold features -> (train/val/test) windowed arrays for one expanding-window quarter.

Leakage discipline, all centralized here:
  * StandardScaler is fit on the TRAIN feature rows only, then applied to val/test.
  * A 96-bar window never crosses a data gap (segment_id) OR the train/val/test boundary.
  * The label bar is the window's last bar.

Returns numpy arrays (framework-agnostic); the GPU train.py wraps them in tensors. Each split also
returns the label-bar close + t_open so the backtest can compute next-bar returns aligned to predictions.
"""
from __future__ import annotations

import sys
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
    Returns X, y, close, t_open, fwd_ret — fwd_ret = next-bar return at the label bar
    (close[i+1]/close[i]-1, same-segment, else NaN) for the backtest."""
    F = Xs.shape[1]
    empty = (np.empty((0, W, F), np.float32), np.empty(0, np.int64),
             np.empty(0), np.empty(0, np.int64), np.empty(0))
    if hi - lo < W:
        return empty
    ends = np.arange(lo + W - 1, hi)
    ends = ends[(y[ends] >= 0) & (seg[ends] == seg[ends - W + 1])]
    if len(ends) == 0:
        return empty
    win = ends[:, None] - np.arange(W - 1, -1, -1)[None, :]      # (n, W) row indices
    n = len(close)
    nxt = np.clip(ends + 1, 0, n - 1)
    ok = (ends + 1 < n) & (seg[nxt] == seg[ends])
    fwd = np.where(ok, close[nxt] / close[ends] - 1.0, np.nan)
    return (Xs[win].astype(np.float32), y[ends].astype(np.int64),
            close[ends].astype(np.float64), topen[ends].astype(np.int64), fwd)


def load_quarter(cfg, test_quarter: str, root: str = "data/datasets") -> dict:
    df = pd.read_parquet(cfg.dataset_path(root)).sort_values("t_open").reset_index(drop=True)
    feats = feature_columns()
    W = cfg.window
    close = df["close"].to_numpy(np.float64)
    topen = df["t_open"].to_numpy(np.int64)
    seg = df["segment_id"].to_numpy()
    y = make_labels(close, seg, cfg.label_scheme, cfg.barrier, cfg.vertical)
    qord = quarter_ord(topen)

    test_ord = parse_quarter(test_quarter)
    val_ord = test_ord - 1
    tr_hi = int(np.searchsorted(qord, val_ord, "left"))                       # train: q < val
    va_lo, va_hi = tr_hi, int(np.searchsorted(qord, val_ord, "right"))        # val:   q == val
    te_lo, te_hi = (int(np.searchsorted(qord, test_ord, "left")),
                    int(np.searchsorted(qord, test_ord, "right")))            # test:  q == test

    Xraw = df[feats].to_numpy(np.float64)
    scaler = StandardScaler().fit(Xraw[:tr_hi])          # FIT ON TRAIN ONLY
    Xs = scaler.transform(Xraw)

    Xtr, ytr, *_ = _windows(Xs, y, close, topen, seg, 0, tr_hi, W)
    Xva, yva, *_ = _windows(Xs, y, close, topen, seg, va_lo, va_hi, W)
    Xte, yte, cte, tte, rte = _windows(Xs, y, close, topen, seg, te_lo, te_hi, W)
    return {
        "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte,
        "test_close": cte, "test_topen": tte, "test_fwd_ret": rte,
        "n_features": len(feats), "test_quarter": test_quarter,
        "sizes": {"train_rows": tr_hi, "windows": {"train": len(ytr), "val": len(yva), "test": len(yte)}},
    }
