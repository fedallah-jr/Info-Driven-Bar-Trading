#!/usr/bin/env python3
"""exp04 evaluate — predictions -> backtest -> the paper's metric set + results table.

Metrics: Annual Net P/L %, annualized Sharpe (rf=3.14%), max drawdown %, % profitable trades,
classification accuracy %, % time active, round-trips, and buy-&-hold over the same span (context).

    python3 model/evaluate.py                          # all preds in results/model
    python3 model/evaluate.py results/model/ETHUSDT_*_preds.parquet
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import run_backtest
from config import RISK_FREE

MS_YEAR = 365.25 * 86400 * 1000


def metrics(df: pd.DataFrame) -> dict:
    df = df.sort_values("t_open").reset_index(drop=True)
    r = run_backtest(df)
    net, eq, pos, t = r["net"], r["equity"], r["pos"], r["t_open"]
    span = (t.max() - t.min()) / MS_YEAR
    total = eq[-1] - 1.0
    ann = (1.0 + total) ** (1.0 / span) - 1.0 if span > 0 and (1.0 + total) > 0 else float("nan")
    bpy = len(net) / span if span > 0 else float("nan")
    ann_vol = net.std(ddof=1) * math.sqrt(bpy) if bpy > 0 else float("nan")
    sharpe = (net.mean() * bpy - RISK_FREE) / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    dd = (eq / np.maximum.accumulate(eq) - 1.0).min()
    active = pos != 0
    acc = ((df["p_up"].to_numpy() > 0.5).astype(int) == df["y_true"].to_numpy()).mean()
    prof = (net[active] > 0).mean() if active.any() else float("nan")
    bh = np.prod(1.0 + np.nan_to_num(df["fwd_ret"].to_numpy())) - 1.0
    return {"n": len(df), "ann_pct": ann * 100, "total_pct": total * 100, "sharpe": sharpe,
            "maxdd_pct": dd * 100, "pct_profitable": prof * 100, "accuracy": acc * 100,
            "pct_active": active.mean() * 100, "round_trips": r["round_trips"],
            "buyhold_pct": bh * 100, "span_years": span}


def main():
    ap = argparse.ArgumentParser(description="exp04 evaluate predictions.")
    ap.add_argument("preds", nargs="*", help="preds parquet(s); default: all in results/model")
    ap.add_argument("--out", default="results/model/RESULTS.md")
    args = ap.parse_args()
    paths = args.preds or sorted(glob.glob("results/model/*_preds.parquet"))
    if not paths:
        raise SystemExit("no predictions found — run train.py first")

    f = lambda x, n=1: ("n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}")
    rows, lines = [], [
        "# Experiment 04 — ResNet-LSTM / info-driven bars: results", "",
        "Backtest: long if P(up)>0.60, short if <0.40, else flat; 0.1%/side cost; "
        "Sharpe annualized at rf=3.14%. Paper headline (ETH 2% CUSUM, triple-barrier): Net P/L 91.6%, Sharpe 1.42.",
        "",
        "| Config | OOS preds | Ann Net % | Sharpe | MaxDD % | %Profit | Acc % | %Active | Round-trips | Buy&Hold % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in paths:
        m = metrics(pd.read_parquet(p))
        tag = Path(p).stem.replace("_preds", "")
        lines.append(f"| {tag} | {m['n']:,} | {f(m['ann_pct'])} | {f(m['sharpe'],2)} | {f(m['maxdd_pct'])} | "
                     f"{f(m['pct_profitable'])} | {f(m['accuracy'])} | {f(m['pct_active'])} | "
                     f"{m['round_trips']:,} | {f(m['buyhold_pct'])} |")
        rows.append((tag, m))
        print(f"[{tag}] ann {f(m['ann_pct'])}%  Sharpe {f(m['sharpe'],2)}  acc {f(m['accuracy'])}%  "
              f"active {f(m['pct_active'])}%  RT {m['round_trips']:,}", flush=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
