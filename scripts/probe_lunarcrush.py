#!/usr/bin/env python3
"""
Probe LunarCrush data availability for each coin/bucket combination.

Uses binary search to find the earliest timestamp with data.
Output: data/cache/lunarcrush_availability.json

Usage:
    python scripts/probe_lunarcrush.py
"""

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
from src.data.lunarcrush_api import fetch_lunarcrush_timeseries, to_unix_utc

COINS = ["btc", "eth", "sol", "xrp", "bnb"]
BUCKETS = ["day", "hour"]

# Search range
EARLIEST = "2019-01-01"
LATEST_OFFSET = pd.Timedelta(days=7)  # probe up to 7 days ago


def probe_earliest(coin: str, bucket: str) -> str | None:
    """
    Binary search for the earliest date with data for a coin/bucket pair.

    Returns ISO date string or None if no data found.
    """
    low = pd.Timestamp(EARLIEST, tz="UTC")
    high = pd.Timestamp.now(tz="UTC") - LATEST_OFFSET

    earliest_found = None

    while (high - low).days > 1:
        mid = low + (high - low) / 2
        # Probe a 7-day window starting at mid
        start_ts = int(mid.timestamp())
        end_ts = int((mid + pd.Timedelta(days=7)).timestamp())

        print(f"  [{coin}/{bucket}] Probing {mid.date()} ... ", end="", flush=True)
        try:
            df = fetch_lunarcrush_timeseries(
                coin=coin, bucket=bucket, start_ts=start_ts, end_ts=end_ts
            )
        except Exception as e:
            print(f"error: {e}")
            # Assume no data, move forward
            low = mid
            continue

        if len(df) > 0:
            earliest_found = df.index.min().isoformat()
            print(f"found {len(df)} rows (earliest: {df.index.min().date()})")
            high = mid
        else:
            print("no data")
            low = mid

    # Final check at 'high' boundary
    start_ts = int(high.timestamp())
    end_ts = int((high + pd.Timedelta(days=7)).timestamp())
    try:
        df = fetch_lunarcrush_timeseries(
            coin=coin, bucket=bucket, start_ts=start_ts, end_ts=end_ts
        )
        if len(df) > 0:
            candidate = df.index.min().isoformat()
            if earliest_found is None or candidate < earliest_found:
                earliest_found = candidate
    except Exception:
        pass

    return earliest_found


def main():
    output_path = Path("data/cache/lunarcrush_availability.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume support
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming from {output_path} ({len(results)} existing entries)")
    else:
        results = {}

    for coin in COINS:
        for bucket in BUCKETS:
            key = f"{coin}_{bucket}"
            if key in results:
                print(f"Skipping {key} (already probed: {results[key]})")
                continue

            print(f"\nProbing {coin}/{bucket} ...")
            earliest = probe_earliest(coin, bucket)
            results[key] = earliest
            print(f"  → {key}: {earliest}")

            # Save after each combo for idempotency
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
