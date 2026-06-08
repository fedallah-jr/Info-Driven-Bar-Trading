"""Labeling schemes for exp04: triple-barrier and next-bar.

Both are SEGMENT-AWARE: the forward look never crosses a data gap (segment_id boundary), matching
the bar layer's contract. Labels: 1 = up/long, 0 = down/short, -1 = unlabelable (no forward bar in
the segment) — the loader drops -1.

Triple barrier (close-based, per the paper): from each bar's close (hypothetical entry), set
upper = entry*(1+b), lower = entry*(1-b), and a vertical timeout `vertical` bars ahead (clipped to
the segment end). Whichever is touched first sets the label; if the vertical is hit first, the label
is the sign of the return at the timeout.
"""
from __future__ import annotations

import numpy as np


def _segment_ends(seg_id: np.ndarray) -> np.ndarray:
    """For each index, the last index belonging to the same contiguous segment."""
    n = len(seg_id)
    end = np.empty(n, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and seg_id[i] != seg_id[i + 1]:
            last = i
        end[i] = last
    return end


def triple_barrier(close: np.ndarray, seg_id: np.ndarray, barrier: float, vertical: int) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    seg_id = np.asarray(seg_id)
    seg_end = _segment_ends(seg_id)
    n = len(close)
    y = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        entry = close[i]
        up, dn = entry * (1.0 + barrier), entry * (1.0 - barrier)
        end = min(i + vertical, int(seg_end[i]))
        if end <= i:
            continue                                  # no forward bar in this segment
        lab = -1
        for j in range(i + 1, end + 1):
            if close[j] >= up:
                lab = 1; break                        # take-profit -> long
            if close[j] <= dn:
                lab = 0; break                        # stop-loss   -> short
        if lab == -1:                                 # vertical timeout -> sign at expiry
            lab = 1 if close[end] >= entry else 0
        y[i] = lab
    return y


def next_bar(close: np.ndarray, seg_id: np.ndarray) -> np.ndarray:
    """Direction of the next bar's close within the same segment."""
    close = np.asarray(close, dtype=np.float64)
    seg_id = np.asarray(seg_id)
    n = len(close)
    y = np.full(n, -1, dtype=np.int8)
    same = np.empty(n, dtype=bool)
    same[:-1] = seg_id[:-1] == seg_id[1:]
    same[-1] = False
    idx = np.where(same)[0]
    y[idx] = (close[idx + 1] >= close[idx]).astype(np.int8)
    return y


def make_labels(close: np.ndarray, seg_id: np.ndarray, scheme: str,
                barrier: float = 0.05, vertical: int = 24) -> np.ndarray:
    if scheme == "triple_barrier":
        return triple_barrier(close, seg_id, barrier, vertical)
    if scheme == "next_bar":
        return next_bar(close, seg_id)
    raise ValueError(f"unknown label scheme: {scheme}")
