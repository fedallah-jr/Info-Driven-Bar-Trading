"""Technical-indicator features for exp04 info-driven bars (paper's ~33-variable set).

Reimplemented in pandas (no pandas_ta dependency) so the feature set is deterministic and
dependency-free. Computed on the SAMPLED bars (after CUSUM/range sampling), and applied PER
SEGMENT by the builder so no rolling window crosses a data gap (segment_id boundary).

The paper lists: EMA & rolling-std of close {5,10,15,20,50}; MACD(12,26,9); RSI{6,10,14};
Stochastic %K/%D(14,3); Williams %R(14); Bollinger(len 5, 2σ); per-bar return; CMF(21);
MFI(14); and sin/cos of hour & weekday. Exact column count depends on the indicator library
(the paper ends at 33 via pandas_ta's multi-column outputs); here it is reported by
`feature_columns()` and the model is built with whatever that count is (not hard-coded to 33).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMA_STD_PERIODS = (5, 10, 15, 20, 50)
RSI_PERIODS = (6, 10, 14)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:  # Wilder's smoothing (used by RSI/MFI)
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    gain = _rma(d.clip(lower=0.0), n)
    loss = _rma(-d.clip(upper=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to one contiguous segment of bars (must be time-sorted).

    `bars` needs: t_open (int64 ms), open, high, low, close, volume. Returns a copy with the
    feature columns added; warm-up rows (leading NaNs from rolling/ewm) are left as NaN for the
    builder to drop per-segment.
    """
    df = bars.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f: dict[str, pd.Series] = {}

    for n in EMA_STD_PERIODS:
        f[f"ema_{n}"] = _ema(c, n)
        f[f"std_{n}"] = c.rolling(n).std()

    macd = _ema(c, 12) - _ema(c, 26)               # MACD line / signal / histogram
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd, sig, macd - sig

    for n in RSI_PERIODS:
        f[f"rsi_{n}"] = _rsi(c, n)

    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    rng14 = (hh14 - ll14).replace(0.0, np.nan)
    k = 100.0 * (c - ll14) / rng14                 # Stochastic %K / %D
    f["stoch_k"], f["stoch_d"] = k, k.rolling(3).mean()
    f["willr_14"] = -100.0 * (hh14 - c) / rng14    # Williams %R

    mid = c.rolling(5).mean()                       # Bollinger (len 5, 2σ)
    sd = c.rolling(5).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    f["bb_mid"], f["bb_upper"], f["bb_lower"] = mid, upper, lower
    f["bb_bandwidth"] = (upper - lower) / mid
    f["bb_pctb"] = (c - lower) / (upper - lower).replace(0.0, np.nan)

    f["ret"] = c.pct_change()                        # per-bar return

    hl = (h - l).replace(0.0, np.nan)                # Chaikin Money Flow (21)
    mfv = ((c - l) - (h - c)) / hl * v
    f["cmf_21"] = mfv.rolling(21).sum() / v.rolling(21).sum().replace(0.0, np.nan)

    tp = (h + l + c) / 3.0                            # Money Flow Index (14)
    rmf = tp * v
    up = rmf.where(tp > tp.shift(1), 0.0)
    dn = rmf.where(tp < tp.shift(1), 0.0)
    mfr = up.rolling(14).sum() / dn.rolling(14).sum().replace(0.0, np.nan)
    f["mfi_14"] = 100.0 - 100.0 / (1.0 + mfr)

    ts = pd.to_datetime(df["t_open"], unit="ms", utc=True)   # cyclical time encodings
    hour, wd = ts.dt.hour.to_numpy(), ts.dt.dayofweek.to_numpy()
    f["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    f["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    f["wd_sin"] = pd.Series(np.sin(2 * np.pi * wd / 7), index=df.index)
    f["wd_cos"] = pd.Series(np.cos(2 * np.pi * wd / 7), index=df.index)

    out = df.assign(**f)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns() -> list[str]:
    cols = []
    for n in EMA_STD_PERIODS:
        cols += [f"ema_{n}", f"std_{n}"]
    cols += ["macd", "macd_signal", "macd_hist"]
    cols += [f"rsi_{n}" for n in RSI_PERIODS]
    cols += ["stoch_k", "stoch_d", "willr_14",
             "bb_mid", "bb_upper", "bb_lower", "bb_bandwidth", "bb_pctb",
             "ret", "cmf_21", "mfi_14",
             "hour_sin", "hour_cos", "wd_sin", "wd_cos"]
    return cols
