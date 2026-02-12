#!/usr/bin/env python3
"""
Batch download LunarCrush time-series data for all coins and buckets.

Downloads in chunks with idempotent resume support.
Chunk cache: data/cache/chunks/lunarcrush_{coin}_{bucket}_{YYYYMMDD}_{YYYYMMDD}.csv
Final output: data/cache/lunarcrush_{coin}_{bucket}_full.csv

Usage:
    python scripts/download_lunarcrush.py              # all coins, all buckets
    python scripts/download_lunarcrush.py --coin btc   # single coin
    python scripts/download_lunarcrush.py --bucket day  # single bucket
    python scripts/download_lunarcrush.py --force       # re-download all chunks
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
from src.data.lunarcrush_api import (
    LUNARCRUSH_ALL_COLUMNS,
    fetch_lunarcrush_timeseries,
    to_unix_utc,
)

COINS = ["btc", "eth", "sol", "xrp", "bnb"]
BUCKETS = ["day", "hour"]

# Chunk sizes (days)
CHUNK_DAYS = {
    "hour": 30,   # ~720 rows per chunk
    "day": 365,   # ~365 rows per chunk
}

CACHE_DIR = Path("data/cache")
CHUNK_DIR = CACHE_DIR / "chunks"
AVAILABILITY_FILE = CACHE_DIR / "lunarcrush_availability.json"


def load_availability() -> dict:
    """Load probed availability dates."""
    if not AVAILABILITY_FILE.exists():
        print(
            f"Warning: {AVAILABILITY_FILE} not found. "
            "Run scripts/probe_lunarcrush.py first. Using 2020-01-01 as default start."
        )
        return {}
    with open(AVAILABILITY_FILE) as f:
        return json.load(f)


def get_start_date(coin: str, bucket: str, availability: dict) -> pd.Timestamp:
    """Get the start date for a coin/bucket pair from availability data."""
    key = f"{coin}_{bucket}"
    if key in availability and availability[key]:
        return pd.Timestamp(availability[key]).floor("D")
    # Conservative default
    return pd.Timestamp("2020-01-01", tz="UTC")


def chunk_filename(coin: str, bucket: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Build chunk cache file path."""
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    return CHUNK_DIR / f"lunarcrush_{coin}_{bucket}_{s}_{e}.csv"


def download_coin_bucket(coin: str, bucket: str, availability: dict, force: bool = False):
    """Download all chunks for a coin/bucket pair."""
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    start = get_start_date(coin, bucket, availability)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    chunk_days = CHUNK_DAYS[bucket]

    print(f"\n{'='*60}")
    print(f"Downloading {coin}/{bucket}: {start.date()} → {now.date()}")
    print(f"  Chunk size: {chunk_days} days")

    # Generate chunk boundaries
    chunks = []
    cursor = start
    while cursor < now:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), now)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end

    print(f"  Total chunks: {len(chunks)}")

    all_dfs = []
    for i, (c_start, c_end) in enumerate(chunks, 1):
        cache_file = chunk_filename(coin, bucket, c_start, c_end)

        if cache_file.exists() and not force:
            # Load cached chunk
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"  [{i}/{len(chunks)}] Cached: {cache_file.name} ({len(df)} rows)")
            all_dfs.append(df)
            continue

        # Download chunk
        start_ts = int(c_start.timestamp())
        end_ts = int(c_end.timestamp())
        print(f"  [{i}/{len(chunks)}] Downloading {c_start.date()} → {c_end.date()} ... ", end="", flush=True)

        try:
            df = fetch_lunarcrush_timeseries(
                coin=coin, bucket=bucket, start_ts=start_ts, end_ts=end_ts
            )
        except Exception as e:
            print(f"error: {e}")
            continue

        if len(df) == 0:
            print("no data")
            continue

        # Save chunk
        df.to_csv(cache_file)
        print(f"{len(df)} rows")
        all_dfs.append(df)

    if not all_dfs:
        print(f"  No data downloaded for {coin}/{bucket}")
        return

    # Concatenate, deduplicate, sort
    full = pd.concat(all_dfs)
    full = full[~full.index.duplicated(keep="last")]
    full = full.sort_index()

    # Ensure all 18 columns present
    for col in LUNARCRUSH_ALL_COLUMNS:
        if col not in full.columns:
            full[col] = float("nan")
    full = full[LUNARCRUSH_ALL_COLUMNS]

    # Save final file
    output_file = CACHE_DIR / f"lunarcrush_{coin}_{bucket}_full.csv"
    full.to_csv(output_file)
    print(f"  Final: {output_file} ({len(full)} rows, {full.index.min()} → {full.index.max()})")


def main():
    parser = argparse.ArgumentParser(description="Download LunarCrush data")
    parser.add_argument("--coin", type=str, help="Single coin to download (e.g. btc)")
    parser.add_argument("--bucket", type=str, help="Single bucket to download (day or hour)")
    parser.add_argument("--force", action="store_true", help="Re-download existing chunks")
    args = parser.parse_args()

    coins = [args.coin] if args.coin else COINS
    buckets = [args.bucket] if args.bucket else BUCKETS

    availability = load_availability()

    for coin in coins:
        for bucket in buckets:
            download_coin_bucket(coin, bucket, availability, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
