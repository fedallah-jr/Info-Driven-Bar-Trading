#!/usr/bin/env python3
"""exp04 backtest — model probabilities -> trades -> equity (paper's rules).

Long if P(up) > 0.60, short if P(up) < 0.40, else flat. Realized PnL per predicted bar = position ×
next-bar return (fwd_ret). Transaction cost charged per side on every position change (0.1% default;
a position change of |Δpos| units = that many sides). Slippage omitted (BTC/ETH liquidity).

    python3 model/backtest.py results/model/ETHUSDT_cusum_thr0.020_triple_barrier_resnet_lstm_preds.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import COST, LONG_TH, SHORT_TH


def run_backtest(df: pd.DataFrame, cost: float = COST,
                 long_th: float = LONG_TH, short_th: float = SHORT_TH) -> dict:
    df = df.sort_values("t_open").reset_index(drop=True)
    p = df["p_up"].to_numpy()
    ret = np.nan_to_num(df["fwd_ret"].to_numpy(), nan=0.0)
    pos = np.where(p > long_th, 1, np.where(p < short_th, -1, 0)).astype(np.int64)
    pos = np.where(np.isnan(df["fwd_ret"].to_numpy()), 0, pos)        # can't hold across a gap
    sides = np.abs(np.diff(np.concatenate([[0], pos])))               # transactions (entry/exit) per bar
    gross = pos * ret
    net = gross - sides * cost
    equity = np.cumprod(1.0 + net)
    return {"pos": pos, "ret": ret, "gross": gross, "net": net, "equity": equity,
            "sides": int(sides.sum()), "round_trips": int(np.ceil(sides.sum() / 2)),
            "t_open": df["t_open"].to_numpy(), "y_true": df["y_true"].to_numpy(), "p_up": p}


def main():
    ap = argparse.ArgumentParser(description="exp04 backtest of a predictions parquet.")
    ap.add_argument("preds")
    ap.add_argument("--cost", type=float, default=COST)
    args = ap.parse_args()
    df = pd.read_parquet(args.preds)
    r = run_backtest(df, cost=args.cost)
    active = (r["pos"] != 0)
    print(f"preds: {len(df):,} | active {active.mean()*100:.0f}% | round-trips ~{r['round_trips']:,} "
          f"(sides {r['sides']:,})")
    print(f"total net return: {(r['equity'][-1]-1)*100:+.1f}%   final equity {r['equity'][-1]:.3f}")


if __name__ == "__main__":
    main()
