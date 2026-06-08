# Experiment 04 — Model Layer: Build Record

Model-side codebase (features → labeling → loader → models → train → backtest → evaluate). Built &
verified end-to-end on CPU 2026-06-08; **full training runs on a GPU box** (code is CUDA-ready, the
local machine is CPU-only). Reads the silver bars produced by `data_pipeline/build_bars.py`.

Reproduces the paper: Y. info-driven bars (CUSUM/range) + Triple-Barrier labels + ResNet-LSTM,
quarterly expanding window. Headline target: **ETH 2% CUSUM, triple-barrier → Net P/L 91.6%, Sharpe 1.42**.

## 1. What was built (`exp04/model/`)

| File | Role | Verified |
|---|---|---|
| `datasets/features.py` | ~33-var indicator set (EMA/std, MACD, RSI, stoch, Williams %R, Bollinger, CMF, MFI, returns, time sin/cos), pandas, no `pandas_ta` | 31 cols, 0 NaN after warm-up |
| `datasets/labeling.py` | triple-barrier + next-bar, **segment-aware** (no forward look across a gap) | TB labels balanced 51/49 |
| `datasets/loader.py` | expanding-window split → **scaler fit on train only** → 96-bar windows (never cross gap/split) → numpy arrays + fwd-return for backtest | shapes (N,96,31), train-only scaling confirmed |
| `models/resnet_lstm.py` | paper headline (Conv×3 + residual + LSTM + head), PyTorch | forward (B,2), 64.5K params |
| `models/tsmixer.py` | all-MLP time mixer (paper's MLP-Mixer) | forward (B,2), 107K params |
| `models/__init__.py` | `build_model(name, …)` registry | resnet_lstm, tsmixer ✓; transformer stubbed |
| `build_dataset.py` | silver bars → gold feature parquet (`data/datasets/`) | ETH 26,098→24,687; BTC 17,403→16,257 |
| `config.py` | `ExperimentConfig` + v1 grid + quarter split (test Q2-2022…Q2-2023) | — |
| `train.py` | expanding-window training (Adam, CE, early-stop, AMP), 3-seed ensemble → OOS preds parquet | runs E2E |
| `backtest.py` | 60/40 band → 0.1%/side cost → equity; trade count = position changes | runs E2E |
| `evaluate.py` | Ann Net %, Sharpe@rf 3.14%, MaxDD, %profit, accuracy, %active, buy&hold → `RESULTS.md` | runs E2E |

## 2. Pipeline

```
silver bars (data/bars)  →  build_dataset.py  →  gold features (data/datasets)
   →  loader (split + train-only scaling + 96-windows + TB labels)
   →  train.py (per quarter × seed → ensemble) → OOS predictions parquet
   →  backtest.py (60/40, costs) → evaluate.py (metrics → RESULTS.md)
```

Design constraints honored: gold is **raw** features (scaler must re-fit per expanding quarter, so
scaled/windowed tensors are never materialized); `n_features` is dynamic (31 here, not hard-coded 33);
windows never straddle a data gap or the train/test boundary; sampling/labeling params fixed across quarters.

## 3. Run on the GPU box

```bash
VPY=python3   # an env with torch (CUDA), pandas, polars, sklearn, pyarrow
# (1) gold datasets — ship data/bars + data/datasets, or rebuild:
$VPY exp04/model/build_dataset.py
# (2) train the v1 grid {ETH,BTC}×{cusum,range}×2%×TB×resnet_lstm×3 seeds×5 quarters:
$VPY exp04/model/train.py --grid
#     or one cell (the 91.6% headline check):
$VPY exp04/model/train.py --coin ETHUSDT --bartype cusum --threshold 0.02 --barrier 0.05
# (3) metrics table vs the paper:
$VPY exp04/model/evaluate.py
```

## 4. Status / deferred

- **Verified:** every stage runs end-to-end; a 1-epoch CPU smoke produced 307 OOS preds → backtest →
  `RESULTS.md`. Numbers from that smoke are meaningless (undertrained, ~0% active); real metrics need
  full-epoch GPU training.
- **Deferred:** `models/transformer.py` (vanilla encoder) + Autoformer/FEDformer (external PyTorch
  repos) and XGBoost (not installed); Hyperband HP search (v1 uses fixed sensible hparams); the full
  paper grid (5 bars × 3 thr × 2 labels × 6 models). All reachable by extending `config.py` + `models/`.
- **First reproduction check:** ETH 2% CUSUM / triple-barrier should approach the paper's 91.6% Net P/L
  / 1.42 Sharpe once trained full-epoch with the 3-seed ensemble across the 5 quarters.
