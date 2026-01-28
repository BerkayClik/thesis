"""
S&P 500 data loader module.

Provides functions to download and cache S&P 500 OHLC data from Yahoo Finance.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
import yfinance as yf


def resample_ohlc(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample OHLC data to a larger interval.

    Args:
        df: DataFrame with OHLC columns and DatetimeIndex.
        target_interval: Target interval (e.g., "4h", "2h", "1d").

    Returns:
        Resampled DataFrame with proper OHLC aggregation.
    """
    resampled = df.resample(target_interval).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    return resampled


def download_sp500_data(
    ticker: str = "^GSPC",
    start_date: Optional[str] = "2000-01-01",
    end_date: Optional[str] = "2024-12-31",
    cache_dir: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    resample_interval: Optional[str] = None
) -> pd.DataFrame:
    """
    Download S&P 500 OHLC data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (default: ^GSPC for S&P 500).
        start_date: Start date in YYYY-MM-DD format (ignored if period is set).
        end_date: End date in YYYY-MM-DD format (ignored if period is set).
        cache_dir: Optional directory to cache downloaded data.
        interval: Data interval (default: "1d" for daily).
                  Options: "1h" (hourly), "1d" (daily), etc.
                  Note: Hourly data is limited to last ~730 days on Yahoo Finance.
        period: Data period (e.g., "730d", "2y", "max"). If set, overrides start/end dates.
                Recommended for hourly data to get maximum available data.
        resample_interval: Optional target interval for resampling (e.g., "4h").
                          Downloads at `interval` and resamples to this.

    Returns:
        DataFrame with columns [Open, High, Low, Close] and DatetimeIndex.
    """
    # Download data from Yahoo Finance
    if period is not None:
        # Use period for rolling window (recommended for hourly data)
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
    else:
        # Use fixed date range
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

    # Handle multi-level columns if present (yfinance returns MultiIndex for single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLC columns in correct order
    df = df[["Open", "High", "Low", "Close"]].copy()

    # Drop any rows with NaN values
    df = df.dropna()

    # Resample if requested
    if resample_interval is not None:
        df = resample_ohlc(df, resample_interval)

    # Cache if directory specified
    if cache_dir is not None and len(df) > 0:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        # Include interval in cache filename to differentiate daily vs hourly data
        interval_suffix = f"_{interval}" if interval != "1d" else ""
        resample_suffix = f"_resampled_{resample_interval}" if resample_interval else ""
        if period is not None:
            cache_file = cache_path / f"{ticker.replace('^', '').replace('=', '')}_{period}{interval_suffix}{resample_suffix}.csv"
        else:
            cache_file = cache_path / f"{ticker.replace('^', '').replace('=', '')}_{start_date}_{end_date}{interval_suffix}{resample_suffix}.csv"
        df.to_csv(cache_file)

    return df


def load_sp500_data(
    data_path: Optional[str] = None,
    ticker: str = "^GSPC",
    start_date: Optional[str] = "2000-01-01",
    end_date: Optional[str] = "2024-12-31",
    cache_dir: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    resample_interval: Optional[str] = None
) -> pd.DataFrame:
    """
    Load S&P 500 data from cache or download if not available.

    Args:
        data_path: Path to cached CSV file. If None, checks cache_dir or downloads.
        ticker: Yahoo Finance ticker symbol.
        start_date: Start date in YYYY-MM-DD format (ignored if period is set).
        end_date: End date in YYYY-MM-DD format (ignored if period is set).
        cache_dir: Directory to look for/store cached data.
        interval: Data interval (default: "1d" for daily).
                  Options: "1h" (hourly), "1d" (daily), etc.
        period: Data period (e.g., "730d", "2y", "max"). If set, overrides start/end dates.
        resample_interval: Optional target interval for resampling (e.g., "4h").

    Returns:
        DataFrame with columns [Open, High, Low, Close] and DatetimeIndex.
    """
    # If explicit data path provided, load from there
    if data_path is not None:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df = df[["Open", "High", "Low", "Close"]]
        return df

    # Check cache directory for existing file
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        interval_suffix = f"_{interval}" if interval != "1d" else ""
        resample_suffix = f"_resampled_{resample_interval}" if resample_interval else ""
        if period is not None:
            cache_file = cache_path / f"{ticker.replace('^', '').replace('=', '')}_{period}{interval_suffix}{resample_suffix}.csv"
        else:
            cache_file = cache_path / f"{ticker.replace('^', '').replace('=', '')}_{start_date}_{end_date}{interval_suffix}{resample_suffix}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

    # Download fresh data
    return download_sp500_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        interval=interval,
        period=period,
        resample_interval=resample_interval
    )


def dataframe_to_tensor(df: pd.DataFrame) -> Tuple[torch.Tensor, pd.DatetimeIndex]:
    """
    Convert pandas DataFrame to PyTorch tensor.

    Args:
        df: DataFrame with OHLC columns and DatetimeIndex.

    Returns:
        Tuple of (tensor of shape (N, 4), DatetimeIndex).
    """
    # Convert to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32)

    # Return tensor and dates
    return tensor, df.index
