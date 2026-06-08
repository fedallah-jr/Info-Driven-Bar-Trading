# Experiment 04 — Data Layer: Build Record & Verification

Objective record of the **data side** (ingest + information-driven bars). The model side
(features → triple-barrier → ResNet-LSTM → backtest) is **not started**. Built & verified 2026-06-08.

Plan: `experiment-04-infodriven-bars-plan.md`. Data side lives in `exp04/data_pipeline/` and shares
nothing with the (future) model codebase but the bar-parquet contract.

---

## 1. What was built

| File | Role |
|---|---|
| `exp04/data_pipeline/ingest.py` | Download + SHA256-verify monthly **spot 1-min klines** → `data/raw/klines_1m/` (verbatim, immutable, resumable, 404-tolerant). |
| `exp04/data_pipeline/build_bars.py` | Raw 1-min klines → **CUSUM + range** bars at {1,2,3%}; OHLCV contract + `segment_id` + per-dataset `manifest.json`. **No feature/label/model code.** |
| `exp04/data_pipeline/README.md` | Data-side ↔ model-side boundary; venv requirement. |

**Decisions taken:** raw resolution = **1-min klines** (not aggTrades); coin set = **10-coin basket**;
period = **full history** (each coin's listing → 2026-05). aggTrades (tick) deferred as an optional
later add for BTC/ETH only.

## 2. Raw layer (bronze) — `exp04/data/raw/klines_1m/` — 1.4 GB

Spot 1-min klines, downloaded & checksum-verified from `data.binance.vision`.

| Coin | Months | Range | 1-min bars | Segments* |
|---|---:|---|---:|---:|
| BTC | 106 | 2017-08 → 2026-05 | 4.61 M | 22 |
| ETH | 106 | 2017-08 → 2026-05 | 4.61 M | 22 |
| BNB | 103 | 2017-11 → 2026-05 | 4.50 M | 21 |
| LTC | 102 | 2017-12 → 2026-05 | 4.44 M | 21 |
| ADA | 98 | 2018-04 → 2026-05 | 4.27 M | 19 |
| XRP | 97 | 2018-05 → 2026-05 | 4.24 M | 19 |
| LINK | 89 | 2019-01 → 2026-05 | 3.87 M | 15 |
| DOGE | 83 | 2019-07 → 2026-05 | 3.63 M | 13 |
| SOL | 70 | 2020-08 → 2026-05 | 3.05 M | 6 |
| AVAX | 69 | 2020-09 → 2026-05 | 2.99 M | 6 |
| **Total** | **923 files** | | **40.2 M** | |

*Segments = contiguous runs split by gaps >120 min (Binance maintenance/archive holes); older coins
have more. Verified: actual 1.4 GB vs ~1.5 GB predicted; each coin **contiguous from listing to
2026-05** (month counts match exactly, no interior gaps); BTC Jan-2023 = 44,640 rows (31×1440); all
923 parquet read cleanly. 2 transient download errors (SOL 2021-08, LTC 2026-01) were retried OK.

## 3. Bars layer (silver) — `exp04/data/bars/{cusum,range}/` — 58 MB, 60 datasets, 2,153,224 bars

Bar counts (10 coins × {cusum C, range R} × {1,2,3%}):

| Coin | C-1% | C-2% | C-3% | R-1% | R-2% | R-3% |
|---|---:|---:|---:|---:|---:|---:|
| BTC | 57,885 | 17,403 | 8,440 | 34,904 | 9,899 | 4,667 |
| ETH | 84,481 | 26,098 | 12,785 | 52,101 | 15,095 | 7,151 |
| SOL | 91,876 | 29,956 | 14,959 | 58,311 | 17,724 | 8,524 |
| BNB | 91,392 | 29,439 | 14,538 | 58,188 | 17,381 | 8,262 |
| XRP | 87,684 | 27,743 | 13,728 | 54,291 | 16,304 | 7,750 |
| DOGE | 94,003 | 31,603 | 16,194 | 60,668 | 19,271 | 9,641 |
| ADA | 102,334 | 31,857 | 15,441 | 62,673 | 18,182 | 8,627 |
| AVAX | 90,531 | 29,104 | 14,337 | 57,128 | 17,044 | 8,107 |
| LINK | 110,715 | 34,518 | 16,805 | 68,225 | 19,895 | 9,294 |
| LTC | 97,286 | 30,360 | 14,726 | 60,216 | 17,353 | 8,127 |

Contract (each parquet): `t_open, t_close` (int64 ms, exact) · `open, high, low, close, volume` (f64) ·
`n_trades` (int64) · `segment_id` (int32). Sidecar `*.manifest.json` records provenance + the
approximation flag.

Bars/day at 2% CUSUM: BTC 5.4, ETH 8.2, alts ~9–14 (SOL/AVAX/LINK/DOGE highest). 1% gives ~18–44/day.

## 4. Verification performed (all passed)

Integrity sweep over **all 60** datasets (`scan` of every parquet + manifest):
- **Schema:** 60/60 match the contract exactly (0 mismatches).
- **Invariants:** 0 violations of `high≥low`, `t_close>t_open`, OHLC-range (`low ≤ open,close ≤ high`),
  `close>0`, and `t_open` strictly increasing within each `segment_id`.
- **Manifests:** 60/60 parse and `n_bars` agrees with the parquet.
- **Sampler correctness:** CUSUM > range at every threshold; monotone decreasing in threshold;
  **30/30** range datasets show median per-bar |close/open−1| ≥ 0.9·threshold (range fires on point
  move); CUSUM median net move < threshold (fires on *accumulated* move) — both as expected.

## 5. The 1-minute approximation (recorded in every manifest)

Trigger evaluated on **1-minute closes**, not the tick tape — the agreed size/fidelity tradeoff
(tick-exact would need the ~100× larger aggTrades feed). Each bar's OHLC still uses its constituent
minutes' true extremes; only the trigger grid is minute-level. **Why it's safe here:** at 2% bars span
hours (BTC ~267 min/bar, alts ~120), and even at 1% ~80 min/bar — so the worst-case ±1-min trigger
error is <1% of a bar's span.

## 6. Caveats

- **Spot, not futures** (matches the paper). **Full history**, a superset of the paper's 2018–2023 window.
- **Indicators decision still open** (model side): `pip install pandas-ta` vs reimplement the 33.
- **Run with `/home/caner/.venv/bin/python3`** — the non-interactive shell does not auto-activate the venv.
- aggTrades tick layer not downloaded (optional later add for BTC/ETH).

## 7. Reproduce

```bash
VPY=/home/caner/.venv/bin/python3
$VPY exp04/data_pipeline/ingest.py        # raw 1-min klines (10 coins, full history) -> data/raw/klines_1m
$VPY exp04/data_pipeline/build_bars.py    # 60 CUSUM/range datasets -> data/bars/{cusum,range}
```

**Next:** model codebase (separate) reading `exp04/data/bars/<bartype>/<SYM>_thr<θ>.parquet`.
