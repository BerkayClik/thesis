#!/usr/bin/env python3
"""
Resample hourly LunarCrush data to 4-hourly with semantically correct aggregation,
then validate all 15 datasets (5 coins × 3 intervals).

Aggregation rules for 18 columns:
  OHLC          → first/max/min/last   : open, high, low, close
  Snapshots     → last                 : volume_24h, market_cap, circulating_supply,
                                          market_dominance, alt_rank
  Sentiment     → mean                 : sentiment, galaxy_score, social_dominance
  Event counts  → sum                  : interactions, contributors_created,
                                          posts_created, spam
  Unique active → max                  : contributors_active, posts_active

Output: data/cache/lunarcrush_{coin}_4hour_full.csv

Usage:
    python scripts/resample_4hourly.py
    python scripts/resample_4hourly.py --validate-only
"""

import argparse
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data.lunarcrush_api import LUNARCRUSH_ALL_COLUMNS

COINS = ["btc", "eth", "sol", "xrp", "bnb"]
CACHE_DIR = Path("data/cache")

# Aggregation rules per column
AGG_RULES = {
    # OHLC
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    # Snapshot values (point-in-time)
    "volume_24h": "last",
    "market_cap": "last",
    "circulating_supply": "last",
    "market_dominance": "last",
    "alt_rank": "last",
    # Sentiment / scores (average over window)
    "sentiment": "mean",
    "galaxy_score": "mean",
    "social_dominance": "mean",
    # Event counts (sum over window)
    "interactions": "sum",
    "contributors_created": "sum",
    "posts_created": "sum",
    "spam": "sum",
    # Unique active (max captures peak activity)
    "contributors_active": "max",
    "posts_active": "max",
}


def resample_hourly_to_4h(coin: str) -> pd.DataFrame | None:
    """Resample hourly data to 4-hourly for a single coin."""
    hourly_file = CACHE_DIR / f"lunarcrush_{coin}_hour_full.csv"
    if not hourly_file.exists():
        print(f"  Skipping {coin}: {hourly_file} not found")
        return None

    df = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Only aggregate columns that exist
    agg = {col: rule for col, rule in AGG_RULES.items() if col in df.columns}
    resampled = df.resample("4h").agg(agg).dropna(how="all")

    # Ensure canonical column order
    for col in LUNARCRUSH_ALL_COLUMNS:
        if col not in resampled.columns:
            resampled[col] = float("nan")
    resampled = resampled[LUNARCRUSH_ALL_COLUMNS]

    # Save
    output_file = CACHE_DIR / f"lunarcrush_{coin}_4hour_full.csv"
    resampled.to_csv(output_file)
    print(f"  {coin}: {len(df)} hourly → {len(resampled)} 4-hourly ({output_file})")

    return resampled


def validate_dataset(filepath: Path) -> list[str]:
    """Validate a single dataset file. Returns list of issues found."""
    issues = []

    if not filepath.exists():
        return [f"File not found: {filepath}"]

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Check 18 columns present
    missing_cols = [c for c in LUNARCRUSH_ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check no all-NaN columns
    all_nan_cols = [c for c in LUNARCRUSH_ALL_COLUMNS if c in df.columns and df[c].isna().all()]
    if all_nan_cols:
        issues.append(f"All-NaN columns: {all_nan_cols}")

    # Check monotonic index
    if not df.index.is_monotonic_increasing:
        issues.append("Index is not monotonically increasing")

    # Check no duplicate timestamps
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate timestamps")

    # Check minimum row count
    if len(df) < 100:
        issues.append(f"Very few rows: {len(df)}")

    return issues


def validate_all():
    """Validate all 15 datasets."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    all_ok = True
    datasets = []
    for coin in COINS:
        for suffix in ["day", "hour", "4hour"]:
            datasets.append(CACHE_DIR / f"lunarcrush_{coin}_{suffix}_full.csv")

    for filepath in datasets:
        issues = validate_dataset(filepath)
        status = "OK" if not issues else "FAIL"
        if issues:
            all_ok = False

        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            rows = len(df)
            date_range = f"{df.index.min()} → {df.index.max()}"
        else:
            rows = 0
            date_range = "N/A"

        print(f"  [{status}] {filepath.name}: {rows} rows, {date_range}")
        for issue in issues:
            print(f"        ⚠ {issue}")

    if all_ok:
        print("\nAll datasets passed validation.")
    else:
        print("\nSome datasets have issues (see above).")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Resample and validate LunarCrush data")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, skip resampling")
    args = parser.parse_args()

    if not args.validate_only:
        print("Resampling hourly → 4-hourly")
        for coin in COINS:
            resample_hourly_to_4h(coin)

    validate_all()


if __name__ == "__main__":
    main()
