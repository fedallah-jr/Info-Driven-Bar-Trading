# exp04 — DATA SIDE (information-driven bar construction)

This directory is the **data pipeline**, deliberately separate from the future **model
codebase** (features → triple-barrier labels → ResNet-LSTM training → backtest). The two
sides share **nothing but bar files on disk** — the contract in §2 of
`experiment-04-infodriven-bars-plan.md`. That is what keeps the model bar-agnostic and lets
us re-cut bars at new thresholds without touching ML code.

```
RAW (bronze, immutable)        BARS (silver, regenerable)            ── contract ──>   MODEL SIDE
ingest.py                      build_bars.py                                           (separate;
spot 1-min klines  ──────────> CUSUM / range bars (1-min-approx)                        consumes bars)
exp04/data/raw/klines_1m/      exp04/data/bars/<bartype>/<SYM>_thr<θ>.parquet (+ .manifest.json)
```

- **`ingest.py`** — download + checksum-verify monthly spot 1-min klines → `data/raw/klines_1m/` (verbatim, immutable).
- **`build_bars.py`** — raw 1-min klines → CUSUM/range bars at {1,2,3%}. OHLCV only; **no features/labels/model code.**

**Run everything with the project venv** (the non-interactive shell does not auto-activate it):

```bash
/home/caner/.venv/bin/python3 exp04/data_pipeline/ingest.py        # raw layer (~1.4 GB, 10 coins)
/home/caner/.venv/bin/python3 exp04/data_pipeline/build_bars.py    # bars layer (CUSUM+range × 1/2/3%)
```

The bars are an **approximation**: at 1-minute resolution the CUSUM/range *trigger* is evaluated on
minute closes (not tick), so bars are minute-grid rather than tick-exact. Each bar's OHLC still uses its
constituent minutes' extremes. This is the agreed size/fidelity tradeoff (tick-exact would need the
~100× larger aggTrades tape). The approximation is recorded in every bar's manifest.
