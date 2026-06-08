# Experiment 04 — Information-Driven Bars (CUSUM + Range) → Triple Barrier → ResNet-LSTM

**Goal:** implement the information-driven-bar trading study (CUSUM filter + range bars, the paper's
two best samplers) on Binance data: sample price-movement bars, label them with the Triple Barrier
method, train a ResNet-LSTM classifier under an expanding-window walk-forward, and backtest net of
fees. Reference results: 2% CUSUM + Triple Barrier was profitable in all four paper configurations
(ETH/BTC × next-bar/triple-barrier); range bars second; dollar bars worst.

> **The one design decision that shapes this experiment.** Data *creation* is separated from
> *labeling/training* by a frozen **OHLCV bar contract**. The bar layer (download + sample) and the
> model layer (features + label + train + backtest) share nothing but bar files on disk. This is what
> makes the model bar-agnostic (one pipeline over CUSUM/range/volume/dollar/time bars, fairly compared)
> and lets us re-cut bars at new thresholds without touching ML code — *provided we keep the raw tape*.

---

> **STATUS (2026-06-08): the data side is BUILT & VERIFIED.** Decided — 1-min spot klines, full 10-coin
> basket, full history. Raw layer (1.4 GB, 40.2 M 1-min bars) and all **60** CUSUM/range bar datasets
> (2.15 M bars) are on disk and integrity-checked (schema/invariants/manifests all clean). Objective
> record: **`exp04/results/DATA_LAYER_STATUS.md`**. The model side (§4) is **not started**.

## 1. Architecture — three layers, one contract

```
RAW  (bronze, immutable)        BARS (silver, regenerable)         MODEL (gold, regenerable)
download once, finest res   →   CUSUM / range sampler          →   features → triple-barrier
klines_1m or aggTrades          (parameterized by type, θ)         → 96-bar windows → ResNet-LSTM
                                 OHLCV only, + manifest             → walk-forward → backtest
        [DATA SIDE]                   [DATA SIDE]                          [MODEL SIDE]
```

Principles (non-negotiable, they are what make the split pay off):

1. **Raw is immutable and the finest resolution we can afford** — because bar-building is lossy and
   parameterized; **the raw resolution permanently bounds the fidelity of every future re-cut.**
   (aggTrades → tick-exact re-cuts forever; 1-min klines → minute-grid approximation forever. See §6.)
2. **The bar layer emits OHLCV only — no features.** Features are cheap to recompute and iterated on
   far more than bar definitions; keeping them out of the bar layer means new indicators cost zero bars.
3. **The two sides meet only at the bar-parquet contract** (§2) — not shared code. Either can be
   rebuilt/tested independently; the model side can be developed today against any OHLCV (even the
   on-disk 1-h klines, or a synthetic sample) while the data side is built in parallel.

## 2. The bar↔model contract (the linchpin)

```
exp04/data/bars/<bartype>/<SYMBOL>_thr<θ>.parquet
  t_open    int64 (ms)    # exact bar open time
  t_close   int64 (ms)    # exact bar close time  ← preserves the exact-second info only the tape has
  open high low close  f64
  volume    f64           # summed to the trigger trade (exact iff raw=aggTrades)
  n_trades  int64         # enables MFI / CMF / dollar-volume features
  segment_id int32        # contiguous-coverage id — see caveat
+ sidecar manifest.json:  {symbol, source, source_resolution, bartype, threshold,
                           date_range, n_bars, code_version, built_utc}
```

**The one caveat to a fully blind split:** the model side can be agnostic about bar *content* but not
about **contiguity**. The 96-bar window and the 24-bar vertical barrier assume bars are consecutive in
sampling; if bars come from gappy data (our archived aggTrades are ~7-day islands — see §7) the
windower/labeler must **never cross a gap**. Encode it once as `segment_id` (one contiguous run = one
id); windowing and triple-barrier forward-looks stay within a single `segment_id`. The manifest makes
"change the criteria later" safe — you never silently mix a 1% and a 2% run.

## 3. Bar definitions (the two samplers)

**CUSUM** — clamped symmetric cumulative-sum filter on returns `rₜ` (close-to-close of the base series):
```
S⁺ₜ = max(0, S⁺ₜ₋₁ + rₜ) ;  S⁻ₜ = min(0, S⁻ₜ₋₁ + rₜ) ;  fire when max(S⁺ₜ, |S⁻ₜ|) ≥ h ; then reset 0,0
```
The clamp is the point: a positive run can't be dragged negative by one counter-move → fires on
*sustained directional drift*. Thresholds h ∈ {1%, 2%, 3%} (best: 2%).

**Range** — fire when price moves R away from the last sampled price: `|Pₜ − P_last|/P_last ≥ R`,
R ∈ {1%, 2%, 3%}. Fires on raw point-to-point distance → samples less often than CUSUM at equal θ.

Both are **path-dependent**, so they must run on the finest base series available (§6). Each emits an
ordinary OHLC bar; **the bar only sets *when* a period closes — it does not open/close trades** (that
is the Triple Barrier layer, §4).

## 4. Model side — spec (from the paper, with our stack deviations flagged)

| Stage | Spec | Our build |
|---|---|---|
| Features | 33 vars: EMA/std {5,10,15,20,50}, MACD(12,26), RSI{6,10,14}, Stoch%K/%D(14), Williams%R(14), Bollinger(5,2σ), bar returns, CMF(21), MFI(14), sin/cos hour+weekday | `pandas-ta` **not installed** → `pip install pandas-ta` or reimplement (codebase already computes TA-style features in exp01). **Scale on train only** (no leakage). |
| Label | Triple Barrier: ±b barriers (b=5% best for ETH-CUSUM; 2.5% for frequent intervals) + vertical 24 bars; first touch → long/short; vertical → sign at expiry | Reuse `exp01/src/triple_barrier_eval.py` (`walk_forward_triple_barrier`). Segment-aware (§2). |
| Windows | 96 timesteps × 33 vars; sample i = 96 bars ending at i, label of bar i | Never cross `segment_id`. |
| Split | Quarterly **expanding** walk-forward; train grows, val/test roll 1 quarter; fixed params across quarters (no per-quarter re-tuning) | Quarters defined over whatever period we download (paper: test Q2'22→Q2'23; recent: 2024-2026). |
| Model | ResNet-LSTM: 3× Conv1D(+BN+ReLU), residual skip around last conv, dropout, LSTM, dense+softmax, BCE | **PyTorch** (TensorFlow/Keras **not installed; no GPU**). Architecturally identical. |
| Ensemble | top-3 configs by val, majority vote; 3 seeds averaged | Same; the full 2,700-model grid is impractical on CPU → run the **headline subset** (CUSUM+range × {1,2,3%} × Triple Barrier × BTC/ETH × few seeds × available quarters). |
| Backtest | long if P(up)>0.60, short if <0.40, else flat; **0.1%** fee each side; Sharpe (rf 3.14%), PnL, DD, % active | Pure numpy. |

**Stack reality (verified):** Python 3.12; have `polars 1.40, pandas 2.3, numpy 1.26, scipy, sklearn,
torch 2.9 (CPU), lightgbm, pyarrow`; **missing** `tensorflow/keras` (→ use torch), `pandas-ta`
(→ pip or reimpl), `xgboost` (→ pip if a GBM baseline is wanted). 16 cores / 15 GB RAM, **no GPU**.

## 5. Reproducibility & leakage discipline

- Per-bar-dataset **manifest** (§2); raw cache is **skip-if-exists + checksum-verified** (reuse the
  exp01/exp02 ingest pattern).
- Scaler fit on **train segment only**; labels strictly forward; window never straddles train/test or a
  `segment_id`; bar θ and barrier widths **fixed across quarters** (the paper's honesty rule).

## 6. Data plan + download-size prediction (predicted 2026-06-07; **realized** 2026-06-08)

> **Realized:** chose **1-min klines, full 10-coin basket, full history** → **1.4 GB actual** (vs ~1.5 GB
> predicted), 923 monthly files, 40.2 M 1-min bars, then 60 CUSUM/range datasets. The prediction below held.
> Build record: `exp04/results/DATA_LAYER_STATUS.md`.

**Raw-resolution choice is the binding decision** (Principle 1). The two options differ ~100× in size:

| Raw option | Fidelity of bars | Continuity | Download (10 popular coins) |
|---|---|---|---|
| **1-min klines** (paper's own choice) | minute-grid approximation; misses intra-minute path/round-trips; volume not splittable at trigger | **continuous** | **~0.9 GB** (2018–2023) / **~1.5 GB** (full history) |
| **aggTrades** (tick tape) | **tick-exact** close-second, price, volume; any future re-cut at full fidelity | continuous if downloaded fresh | **~87 GB** (2018–2023) / **~142 GB** (full history) |

Per-coin prediction (GB), basket = BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC (all have spot history
back to 2017–2020; SOL 2020-08, AVAX 2020-09 the latest starts):

| Coin | aggTrades full | aggTrades 2018–23 | klines_1m full | klines_1m 2018–23 |
|---|---:|---:|---:|---:|
| BTC | 46.5 | 28.7 | 0.23 | 0.14 |
| ETH | 33.7 | 20.8 | 0.20 | 0.12 |
| SOL | 8.3 | 4.1 | 0.11 | 0.05 |
| BNB | 11.0 | 7.0 | 0.15 | 0.10 |
| XRP | 13.0 | 8.2 | 0.15 | 0.09 |
| DOGE | 8.7 | 5.0 | 0.14 | 0.08 |
| ADA | 6.2 | 3.9 | 0.15 | 0.10 |
| AVAX | 2.6 | 1.2 | 0.10 | 0.05 |
| LINK | 5.7 | 3.4 | 0.14 | 0.08 |
| LTC | 6.9 | 4.4 | 0.17 | 0.11 |
| **TOTAL** | **~142** | **~87** | **~1.5** | **~0.9** |

Notes: aggTrades months are **highly variable** (BTC: 89 MB in 2019-01 vs **2.1 GB** in 2023-01) so the
aggTrades figures are mid-year-sampled estimates (±~30%); klines are ~constant (~1.5–2.2 MB/month/coin)
so those are tight. Disk free: **873 GB** — even the 142 GB pull fits; process zips **streaming**
(never extract the whole corpus; the minute/bar output is tiny).

**Recommendation (sequenced):**
1. **Download 1-min klines, full basket, full history (~1.5 GB) first.** Continuous, matches the paper's
   own input, lets the entire pipeline be built, validated, and the paper's *approach* reproduced
   immediately. Spot, to match the paper.
2. **Add aggTrades selectively, later** — only BTC/ETH (the paper's coins), recent ~2–3 yr window
   (~20–50 GB) — *if/when* you want to prove tick-exact bars differ materially from the 1-min
   approximation, or to re-cut at very low thresholds. Storing the tape once is the only way to keep
   full re-cut fidelity (Principle 1), so do it for the coins you care most about.

## 7. What's already on disk vs what must be downloaded

- **On disk, continuous:** 1-h klines, 2023-01→2026-05, BTC/ETH/SOL (futures) + ~40 coins (spot,
  incl. BTC/ETH). Too coarse for faithful 1–3% bars (a 1-h candle hides full round-trips) — usable only
  for plumbing/dev, not the real run.
- **On disk, NOT continuous:** aggTrades for BTC/ETH/SOL but only ~129/126/129 **scattered days**
  (~7-day islands every 1–2 months, 2023-06→2026-03), futures, plus 5 alts × 3 days. Tick-exact but
  gappy → supports per-island bars only, **not** a multi-year walk-forward. → must download fresh.
- **Not on disk:** any 1-min data; any continuous tape; any pre-2023 data; any spot tape.

## 8. Known caveats

- **Spot vs futures:** paper uses **spot**; download spot to match. (On-disk fine data is futures perp.)
- **Period vs paper:** to reproduce the headline tables (91.6% etc.) the window must be **2018-01→2023-06**;
  a recent window (2023–2026) is a faithful *method* run but not comparable to the paper's numbers.
- **Decay prior:** exp02 showed edges in this venue decay (carry +10%/yr 2024 → −13%/yr 2026); treat any
  positive result as period-specific until walk-forward-confirmed.
- **CPU only:** no GPU → headline subset, not the 2,700-model grid.
- **Bar count vs depth model:** at high thresholds (2–3%) bar counts are modest; 96-bar windows need long
  contiguous segments — another reason continuity (klines) matters for the first pass.

## 9. Proposed repo structure

```
exp04/
  data_pipeline/                 # DATA SIDE  ✅ built & verified
    ingest.py                    # ✅ spot 1-min klines -> raw cache (checksum, resumable, 404-tolerant)
    build_bars.py                # ✅ raw -> CUSUM/range bars + manifest ; OHLCV only
    README.md                    # ✅ data-side ↔ model-side boundary + venv note
  model/                         # MODEL SIDE — ⬜ NOT STARTED (bar-agnostic; reads the §2 contract only)
    features.py                  # 33 features, train-only scaling
    label.py                     # triple barrier (reuse exp01/src/triple_barrier_eval.py)
    windows.py                   # 96-bar sequences, segment-aware
    model_resnet_lstm.py         # PyTorch
    train.py                     # expanding quarterly walk-forward, seeds, top-3 ensemble
    backtest.py                  # 60/40 band, 0.1% cost, metrics
  data/
    raw/klines_1m/<SYM>/<YYYY-MM>.parquet         # ✅ bronze, immutable (1.4 GB, 40.2 M 1-min bars)
    bars/<bartype>/<SYM>_thr<θ>.parquet + .manifest.json   # ✅ silver (60 datasets, 2.15 M bars)
  results/DATA_LAYER_STATUS.md   # ✅ build record + verification (this run)
  src/build_info_bars.py         # draft aggTrades→1-min probe (superseded by build_bars.py for klines)
```

## 10. Decisions (resolved)

1. **Raw resolution:** ✅ **1-min klines** (1.4 GB, full basket + history) — DONE. aggTrades deferred (optional BTC/ETH ~20–50 GB later).
2. **Period:** ✅ **full history** (each coin's listing → 2026-05) — a superset of both the paper window and a recent run.
3. **Coin set:** ✅ **10-coin basket** (BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC) — DONE.
4. **Indicators:** ⬜ still open — `pip install pandas-ta` vs reimplement the 33 (decide when building the model side).

*Sizes/availability verified live against `data.binance.vision` 2026-06-07; data side built & verified 2026-06-08 (`exp04/results/DATA_LAYER_STATUS.md`).*
