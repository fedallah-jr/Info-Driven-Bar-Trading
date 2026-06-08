#!/usr/bin/env python3
"""
exp04 DATA SIDE — build information-driven bars (CUSUM + range) from 1-min klines.

APPROXIMATE bars: the CUSUM/range *trigger* is evaluated on 1-minute closes (not on the
tick tape) — the agreed size/fidelity tradeoff (tick-exact would need the ~100x larger
aggTrades feed). Each emitted bar's OHLC still uses its constituent minutes' true extremes;
only the *trigger grid* is minute-level. The approximation is recorded in every manifest.

This is the DATA side. It produces bars and knows nothing about features / labels / training;
the model codebase consumes these parquet files and the two meet only at the bar contract:

    data/bars/<bartype>/<SYMBOL>_thr<θ>.parquet   columns:
        t_open  t_close (int64 ms, exact)   open high low close (f64)
        volume (f64)   n_trades (int64)     segment_id (int32)
    + sidecar <SYMBOL>_thr<θ>.manifest.json  (provenance: how the bars were cut)

`segment_id` marks contiguous-coverage runs; a gap larger than --max-gap-min breaks a segment,
and bars are never sampled across a break (so a return is never taken over a data hole). The
model side must keep its 96-bar windows / triple-barrier look-forward within one segment_id.

CUSUM:  S⁺=max(0,S⁺+r), S⁻=min(0,S⁻+r); fire when max(S⁺,|S⁻|) ≥ h; reset 0,0.   (continuous;
         the cross-bar return is accumulated into the next bar, per López de Prado.)
Range:  fire when |close − last_sampled_close| / last_sampled_close ≥ R.

    /home/caner/.venv/bin/python3 data_pipeline/build_bars.py            # full grid
    /home/caner/.venv/bin/python3 data_pipeline/build_bars.py --symbols BTCUSDT --bartypes cusum --thresholds 0.02
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def _to_ms(a: np.ndarray) -> np.ndarray:
    """Magnitude-based epoch-unit normalization (Binance switched ms→µs in 2025)."""
    a = a.astype(np.float64)
    return np.where(a > 1e17, a / 1e6, np.where(a > 1e14, a / 1e3, a)).astype(np.int64)


def load_minute(sym: str, raw_dir: str) -> dict | None:
    files = sorted(glob.glob(f"{raw_dir}/{sym}/*.parquet"))
    if not files:
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "count"]
    df = (pl.concat([pl.read_parquet(f, columns=cols) for f in files])
            .unique(subset="open_time", keep="first")
            .sort("open_time"))
    a = {
        "topen": _to_ms(df["open_time"].to_numpy()),
        "tclose": _to_ms(df["close_time"].to_numpy()),
        "o": df["open"].to_numpy().astype(np.float64),
        "h": df["high"].to_numpy().astype(np.float64),
        "l": df["low"].to_numpy().astype(np.float64),
        "c": df["close"].to_numpy().astype(np.float64),
        "v": df["volume"].to_numpy().astype(np.float64),
        "n": df["count"].to_numpy().astype(np.int64),
    }
    good = np.isfinite(a["c"]) & (a["c"] > 0)          # drop any bad/halted minutes
    if not good.all():
        a = {k: v[good] for k, v in a.items()}
    return a


def segments(topen: np.ndarray, max_gap_ms: int) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges of contiguous coverage."""
    if len(topen) == 0:
        return []
    brk = np.where(np.diff(topen) > max_gap_ms)[0]
    b = [0, *(brk + 1).tolist(), len(topen)]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def cuts_cusum(c: np.ndarray, s: int, e: int, h: float) -> list[int]:
    cuts, sp, sn = [], 0.0, 0.0
    for i in range(s + 1, e):
        r = c[i] / c[i - 1] - 1.0
        sp += r
        if sp < 0.0:
            sp = 0.0
        sn += r
        if sn > 0.0:
            sn = 0.0
        if (sp if sp >= -sn else -sn) >= h:           # max(S⁺, |S⁻|) ≥ h
            cuts.append(i)
            sp = sn = 0.0
    return cuts


def cuts_range(c: np.ndarray, s: int, e: int, r: float) -> list[int]:
    cuts, anchor = [], c[s]
    for i in range(s + 1, e):
        if abs(c[i] - anchor) / anchor >= r:
            cuts.append(i)
            anchor = c[i]
    return cuts


def assemble(a: dict, s: int, cuts: list[int], seg_id: int) -> dict | None:
    """Build bar columns for one segment. Bar k spans minutes [start_k .. cut_k] (inclusive);
    the trailing sub-threshold remainder after the last cut is dropped."""
    if not cuts:
        return None
    starts = np.array([s] + [cut + 1 for cut in cuts[:-1]], dtype=np.int64)
    ends = np.array(cuts, dtype=np.int64)
    last = int(ends[-1])
    rel = starts - s                                  # reduceat boundaries into the [s..last] slice
    sl = slice(s, last + 1)
    return {
        "t_open": a["topen"][starts],
        "t_close": a["tclose"][ends],
        "open": a["o"][starts],
        "high": np.maximum.reduceat(a["h"][sl], rel),
        "low": np.minimum.reduceat(a["l"][sl], rel),
        "close": a["c"][ends],
        "volume": np.add.reduceat(a["v"][sl], rel),
        "n_trades": np.add.reduceat(a["n"][sl], rel),
        "segment_id": np.full(len(starts), seg_id, dtype=np.int32),
    }


def build(a: dict, segs: list[tuple[int, int]], bartype: str, thr: float) -> pl.DataFrame:
    finder = cuts_cusum if bartype == "cusum" else cuts_range
    parts = []
    for seg_id, (s, e) in enumerate(segs):
        if e - s < 2:
            continue
        part = assemble(a, s, finder(a["c"], s, e, thr), seg_id)
        if part is not None:
            parts.append(part)
    if not parts:
        return pl.DataFrame()
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return pl.DataFrame(merged).with_columns(
        pl.col("t_open").cast(pl.Int64), pl.col("t_close").cast(pl.Int64),
        pl.col("n_trades").cast(pl.Int64), pl.col("segment_id").cast(pl.Int32),
    )


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CUSUM/range information-driven bars (exp04 silver layer).")
    ap.add_argument("--symbols", default=",".join(BASKET))
    ap.add_argument("--bartypes", default="cusum,range")
    ap.add_argument("--thresholds", default="0.01,0.02,0.03")
    ap.add_argument("--raw", default="data/raw/klines_1m")
    ap.add_argument("--out", default="data/bars")
    ap.add_argument("--max-gap-min", type=int, default=120, help="gap (minutes) that breaks a segment")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bartypes = [b.strip() for b in args.bartypes.split(",") if b.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    out = Path(args.out)
    max_gap_ms = args.max_gap_min * 60_000

    print(f"{'symbol':<9}{'bartype':<7}{'thr':>5}{'segs':>6}{'bars':>9}{'bars/day':>10}", flush=True)
    summary = []
    for sym in symbols:
        a = load_minute(sym, args.raw)
        if a is None:
            print(f"{sym:<9} no raw data", flush=True)
            continue
        segs = segments(a["topen"], max_gap_ms)
        active_days = len(a["c"]) / 1440.0
        for bartype in bartypes:
            for thr in thresholds:
                bars = build(a, segs, bartype, thr)
                d = out / bartype
                d.mkdir(parents=True, exist_ok=True)
                stem = f"{sym}_thr{thr:.3f}"
                bars.write_parquet(d / f"{stem}.parquet")
                manifest = {
                    "symbol": sym, "source": "data.binance.vision spot klines",
                    "source_resolution": "1m",
                    "approximation": "1-minute close-based trigger (not tick-exact)",
                    "bartype": bartype, "threshold": thr,
                    "n_bars": bars.height, "n_segments": len(segs),
                    "base_minute_rows": len(a["c"]),
                    "bars_per_active_day": round(bars.height / active_days, 2) if active_days else None,
                    "date_range": [_iso(int(a["topen"][0])), _iso(int(a["tclose"][-1]))],
                    "max_gap_min": args.max_gap_min,
                    "code": "data_pipeline/build_bars.py",
                    "built_utc": datetime.now(timezone.utc).isoformat(),
                }
                (d / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
                bpd = manifest["bars_per_active_day"]
                print(f"{sym:<9}{bartype:<7}{thr:>5.2f}{len(segs):>6}{bars.height:>9,}{bpd:>10.1f}", flush=True)
                summary.append((sym, bartype, thr, bars.height, bpd))
    tot = sum(s[3] for s in summary)
    print(f"\nDONE — {len(summary)} bar datasets, {tot:,} bars total -> {out}/", flush=True)


if __name__ == "__main__":
    main()
