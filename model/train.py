#!/usr/bin/env python3
"""Training — expanding-window protocol, GPU (CUDA).

Per-coin (default):  for each test quarter, train `--seeds` models, ensemble P(up), save OOS preds.
Pooled (--pooled):   train ONE model across all --coins/--basket each quarter, predict each coin's test.

Saves results/model/<tag>_preds.parquet (t_open, close, fwd_ret, y_true, p_up, quarter) for
backtest.py / evaluate.py. Sampling/labeling params are FIXED across quarters.

    python3 model/train.py --grid                                   # paper v1 grid (ETH/BTC, paper window)
    python3 model/train.py --coin ETHUSDT --quarters extended       # ETH, full 2022->2026 OOS (robustness)
    python3 model/train.py --basket --quarters extended --seeds 0   # every coin, per-coin, extended OOS
    python3 model/train.py --basket --pooled --quarters extended    # ONE cross-sectional model over all coins
    python3 model/train.py --coin ETHUSDT --model transformer --quarters extended
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASKET, EXTENDED_QUARTERS, TEST_QUARTERS, ExperimentConfig, v1_grid
from datasets.loader import load_quarter, load_quarter_pooled
from models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _seed(s: int):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _eval_loss(model, X, y, lossfn, bs=1024):
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs].to(DEVICE), y[i:i + bs].to(DEVICE)
            tot += lossfn(model(xb), yb).item() * len(xb); n += len(xb)
    return tot / max(n, 1)


def _predict(model, X, bs=1024) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            p = torch.softmax(model(X[i:i + bs].to(DEVICE)), dim=1)[:, 1]
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def train_model(cfg, Xtr, ytr, Xva, yva, n_features, seed):
    """Train one model (early-stop on val loss, AMP on CUDA); return the trained model."""
    _seed(seed)
    model = build_model(cfg.model, n_features, cfg.window).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossfn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.batch_size, shuffle=True)
    have_val = len(yva) > 0
    best, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ep in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                loss = lossfn(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if have_val:
            vl = _eval_loss(model, Xva_t, yva_t, lossfn)
            if vl < best - 1e-5:
                best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
    if have_val:
        model.load_state_dict(best_state)
    return model


def _save(rows, path: Path):
    preds = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(path)
    acc = float(((preds["p_up"] > 0.5).astype(int) == preds["y_true"]).mean())
    print(f"  wrote {path}  ({len(preds):,} preds, acc {acc:.3f})", flush=True)


def run_config(cfg: ExperimentConfig, out_dir: Path):
    print(f"[{cfg.tag()}] device={DEVICE} quarters={len(cfg.test_quarters)} seeds={list(cfg.seeds)}", flush=True)
    rows = []
    for q in cfg.test_quarters:
        d = load_quarter(cfg, q)
        if len(d["yte"]) == 0:
            continue
        probs = np.mean([_predict(train_model(cfg, d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_features"], s),
                                  torch.from_numpy(d["Xte"])) for s in cfg.seeds], axis=0)
        print(f"  {q}: train={len(d['ytr']):,} test={len(d['yte']):,} "
              f"acc={float(((probs>0.5).astype(int)==d['yte']).mean()):.3f}", flush=True)
        rows.append(pd.DataFrame({"t_open": d["test_topen"], "close": d["test_close"],
                                  "fwd_ret": d["test_fwd_ret"], "y_true": d["yte"], "p_up": probs, "quarter": q}))
    if rows:
        _save(rows, out_dir / f"{cfg.tag()}_preds.parquet")


def run_pooled(coins, cfg: ExperimentConfig, out_dir: Path):
    tag = f"pooled{len(coins)}_{cfg.bartype}_thr{cfg.threshold:.3f}_{cfg.model}"
    print(f"[{tag}] device={DEVICE} coins={len(coins)} quarters={len(cfg.test_quarters)} seeds={list(cfg.seeds)}",
          flush=True)
    rows_by_coin = {c: [] for c in coins}
    for q in cfg.test_quarters:
        d = load_quarter_pooled(coins, cfg, q)
        if len(d["ytr"]) == 0 or not d["test"]:
            continue
        models = [train_model(cfg, d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_features"], s) for s in cfg.seeds]
        for c, td in d["test"].items():
            probs = np.mean([_predict(m, torch.from_numpy(td["Xte"])) for m in models], axis=0)
            rows_by_coin[c].append(pd.DataFrame({"t_open": td["test_topen"], "close": td["test_close"],
                                                 "fwd_ret": td["test_fwd_ret"], "y_true": td["yte"],
                                                 "p_up": probs, "quarter": q}))
        print(f"  {q}: pooled train={len(d['ytr']):,} coins_tested={len(d['test'])}", flush=True)
    for c, rows in rows_by_coin.items():
        if rows:
            _save(rows, out_dir / f"{c}_{tag}_preds.parquet")


def main():
    ap = argparse.ArgumentParser(description="Expanding-window training (CUDA).")
    ap.add_argument("--grid", action="store_true", help="run the paper v1 grid")
    ap.add_argument("--basket", action="store_true", help="use the full 10-coin basket")
    ap.add_argument("--pooled", action="store_true", help="train ONE model across all coins (cross-sectional)")
    ap.add_argument("--coins", default="", help="comma list of coins (overrides --coin)")
    ap.add_argument("--coin", default="ETHUSDT")
    ap.add_argument("--bartype", default="cusum"); ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--model", default="resnet_lstm"); ap.add_argument("--label-scheme", default="triple_barrier")
    ap.add_argument("--barrier", type=float, default=0.05); ap.add_argument("--vertical", type=int, default=24)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--quarters", default="", help="comma list, or 'extended' (2022->2026) or 'paper'")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default="results/model")
    args = ap.parse_args()
    out = Path(args.out_dir)

    if args.grid:
        for cfg in v1_grid():
            run_config(cfg, out)
        return

    qk = args.quarters.strip().lower()
    if qk in ("extended", "all"):
        qs = tuple(EXTENDED_QUARTERS)
    elif qk in ("", "paper"):
        qs = tuple(TEST_QUARTERS)
    else:
        qs = tuple(q.strip() for q in args.quarters.split(",") if q.strip())

    coins = (list(BASKET) if args.basket
             else [c.strip() for c in args.coins.split(",") if c.strip()] or [args.coin])

    base = dict(bartype=args.bartype, threshold=args.threshold, model=args.model,
                label_scheme=args.label_scheme, barrier=args.barrier, vertical=args.vertical,
                seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
                epochs=args.epochs, batch_size=args.batch_size, test_quarters=qs)

    if args.pooled:
        run_pooled(coins, ExperimentConfig(coin=coins[0], **base), out)
    else:
        for c in coins:
            run_config(ExperimentConfig(coin=c, **base), out)


if __name__ == "__main__":
    main()
