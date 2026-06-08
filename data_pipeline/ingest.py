#!/usr/bin/env python3
"""
exp04 DATA SIDE — ingest spot 1-minute klines (raw / bronze layer).

Downloads monthly 1-min spot klines for the 10-coin popular basket from
data.binance.vision, SHA256-verifies each file against its .CHECKSUM sibling, and
writes it *verbatim* to parquet. Raw is immutable: epoch-unit and feature
normalization happen later in build_bars (per the exp04 layer separation), so the
bronze layer stays a faithful mirror of the source.

Resumable (skips files already converted). Full per-coin history is captured by
starting early and letting pre-listing months 404 -> counted as 'missing'.

    python3 data_pipeline/ingest.py                       # full basket, full history (~1.5 GB)
    python3 data_pipeline/ingest.py --symbols BTCUSDT --start 2023-01-01 --end 2023-03-31  # smoke

Logic (checksum, headerless detection, skip-if-exists, threaded 404-tolerant fetch)
mirrors exp02/src/ingest_lowfreq.py, which the handoff calls schema/unit-robust.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import polars as pl

SPOT_KLINES = "https://data.binance.vision/data/spot/monthly/klines"

# 10 popular coins; all have spot history back to 2017-2020 (verified 2026-06-07).
BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

# Spot kline CSVs in the archive are headerless -> applied positionally.
KLINES_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def month_starts(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fetch(url: str, timeout: int = 60) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_and_write(url: str, out_path: Path, verify: bool) -> str:
    if out_path.exists():
        return "skip"
    data = fetch(url)
    if data is None:
        return "missing"  # month before listing, or not yet published
    if verify:
        chk = fetch(url + ".CHECKSUM")
        if chk is not None:
            want = chk.decode().split()[0].strip()
            if want != hashlib.sha256(data).hexdigest():
                raise ValueError(f"checksum mismatch {url}")
    zf = zipfile.ZipFile(io.BytesIO(data))
    raw = zf.read(zf.namelist()[0])
    # Older archive CSVs ship without a header -> detect numeric first field, apply schema positionally.
    first = raw.split(b"\n", 1)[0].decode(errors="ignore").split(",")[0].strip()
    if _is_number(first):
        df = pl.read_csv(io.BytesIO(raw), has_header=False, new_columns=KLINES_COLS, infer_schema_length=10000)
    else:
        df = pl.read_csv(io.BytesIO(raw), infer_schema_length=10000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest spot 1-min klines (exp04 bronze layer).")
    ap.add_argument("--symbols", default=",".join(BASKET))
    ap.add_argument("--start", default="2017-08-01")          # Binance launch; per-coin pre-listing months 404
    ap.add_argument("--end", default="2026-05-31")            # last complete published month
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    out, iv = Path(args.out), args.interval

    jobs = []  # (url, out_path)
    for sym in symbols:
        for ms in month_starts(start, end):
            ym = ms.strftime("%Y-%m")
            jobs.append((f"{SPOT_KLINES}/{sym}/{iv}/{sym}-{iv}-{ym}.zip",
                         out / f"klines_{iv}" / sym / f"{ym}.parquet"))

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    print(f"[ingest] {len(jobs)} files | {len(symbols)} symbols {iv} spot klines {start}..{end}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_and_write, u, p, not args.no_verify): u for u, p in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:  # noqa: BLE001 - log and continue; ingest is resumable
                counts["error"] += 1
                print(f"[ingest] ERROR {futs[fut]}: {e}", file=sys.stderr, flush=True)
            if i % 100 == 0 or i == len(jobs):
                print(f"[ingest] {i}/{len(jobs)} {counts}", flush=True)
    print(f"[ingest] DONE {counts}", flush=True)


if __name__ == "__main__":
    main()
