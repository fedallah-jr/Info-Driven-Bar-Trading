#!/usr/bin/env python3
"""exp04 training — expanding-window protocol, GPU (CUDA). Training runs on the GPU box.

For one config: for each test quarter, train `--seeds` independent models (Adam + CrossEntropyLoss,
early-stopping on val loss, AMP on CUDA), ensemble their P(up) by averaging, and collect the
out-of-sample test predictions. Saves results/model/<tag>_preds.parquet with one row per predicted
test bar: t_open, close, fwd_ret, y_true, p_up, quarter — consumed by backtest.py / evaluate.py.

Params (bar/threshold/barrier/window) are FIXED across quarters (no per-quarter re-tuning).

    python3 model/train.py --coin ETHUSDT --bartype cusum --threshold 0.02 --barrier 0.05
    python3 model/train.py --grid               # the v1 grid
    python3 model/train.py --coin ETHUSDT --epochs 2 --quarters 2022Q4 --seeds 0   # smoke
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
from config import ExperimentConfig, v1_grid
from datasets.loader import load_quarter
from models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _seed(s: int):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _eval_loss(model, X, y, lossfn, bs=1024):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(DEVICE); yb = y[i:i + bs].to(DEVICE)
            tot += lossfn(model(xb), yb).item() * len(xb); n += len(xb)
    return tot / max(n, 1)


def _predict(model, X, bs=1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            p = torch.softmax(model(X[i:i + bs].to(DEVICE)), dim=1)[:, 1]
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def train_seed(cfg, data, seed: int) -> np.ndarray:
    _seed(seed)
    model = build_model(cfg.model, data["n_features"], cfg.window).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossfn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    Xtr = torch.from_numpy(data["Xtr"]); ytr = torch.from_numpy(data["ytr"])
    Xva = torch.from_numpy(data["Xva"]); yva = torch.from_numpy(data["yva"])
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
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
            vl = _eval_loss(model, Xva, yva, lossfn)
            if vl < best - 1e-5:
                best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
    if have_val:
        model.load_state_dict(best_state)
    return _predict(model, torch.from_numpy(data["Xte"]))


def run_config(cfg: ExperimentConfig, out_dir: Path) -> Path | None:
    print(f"[{cfg.tag()}] device={DEVICE} quarters={list(cfg.test_quarters)} seeds={list(cfg.seeds)}", flush=True)
    rows = []
    for q in cfg.test_quarters:
        data = load_quarter(cfg, q)
        if len(data["yte"]) == 0:
            print(f"  {q}: no test windows, skip", flush=True); continue
        probs = np.mean([train_seed(cfg, data, s) for s in cfg.seeds], axis=0)   # seed ensemble
        acc = float(((probs > 0.5).astype(int) == data["yte"]).mean())
        print(f"  {q}: train={len(data['ytr']):,} test={len(data['yte']):,} acc={acc:.3f}", flush=True)
        rows.append(pd.DataFrame({"t_open": data["test_topen"], "close": data["test_close"],
                                  "fwd_ret": data["test_fwd_ret"], "y_true": data["yte"],
                                  "p_up": probs, "quarter": q}))
    if not rows:
        print("  no predictions produced", flush=True); return None
    preds = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.tag()}_preds.parquet"
    preds.to_parquet(path)
    print(f"  wrote {path}  ({len(preds):,} OOS predictions, overall acc "
          f"{float(((preds['p_up']>0.5).astype(int)==preds['y_true']).mean()):.3f})", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="exp04 expanding-window training (CUDA).")
    ap.add_argument("--grid", action="store_true", help="run the v1 grid")
    ap.add_argument("--coin", default="ETHUSDT"); ap.add_argument("--bartype", default="cusum")
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--model", default="resnet_lstm"); ap.add_argument("--label-scheme", default="triple_barrier")
    ap.add_argument("--barrier", type=float, default=0.05); ap.add_argument("--vertical", type=int, default=24)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--quarters", default="")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default="results/model")
    args = ap.parse_args()
    out = Path(args.out_dir)

    if args.grid:
        for cfg in v1_grid():
            run_config(cfg, out)
        return
    kw = dict(coin=args.coin, bartype=args.bartype, threshold=args.threshold, model=args.model,
              label_scheme=args.label_scheme, barrier=args.barrier, vertical=args.vertical,
              seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
              epochs=args.epochs, batch_size=args.batch_size)
    if args.quarters.strip():
        kw["test_quarters"] = tuple(q.strip() for q in args.quarters.split(",") if q.strip())
    run_config(ExperimentConfig(**kw), out)


if __name__ == "__main__":
    main()
