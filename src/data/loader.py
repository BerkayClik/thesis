"""
Market data loader module.

Provides functions to download and cache market data from Yahoo Finance
or LunarCrush API.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import torch
import yfinance as yf


from .lunarcrush_api import LUNARCRUSH_ALL_COLUMNS, QUATERNION_FEATURE_COLS

YAHOO_COLUMNS = ["Open", "High", "Low", "Close"]
LUNARCRUSH_COLUMNS = LUNARCRUSH_ALL_COLUMNS


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


def resample_market_social(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample Close/Volume/Sentiment/Social data to a larger interval.

    Args:
        df: DataFrame with [Close, Volume, Sentiment, Social] columns.
        target_interval: Target interval (e.g., "4h", "1d").

    Returns:
        Resampled DataFrame.
    """
    resampled = df.resample(target_interval).agg({
        'Close': 'last',
        'Volume': 'last',
        'Sentiment': 'mean',
        'Social': 'mean'
    }).dropna()
    return resampled


def resample_lunarcrush_full(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample a full 18-column LunarCrush DataFrame to a larger interval.

    Uses semantically correct aggregation per column type:
      OHLC          → first/max/min/last
      Snapshots     → last
      Sentiment     → mean
      Event counts  → sum
      Unique active → max

    Args:
        df: DataFrame with 18 LunarCrush columns and DatetimeIndex.
        target_interval: Target interval (e.g., "4h", "1d").

    Returns:
        Resampled DataFrame.
    """
    agg_rules = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume_24h": "last", "market_cap": "last", "circulating_supply": "last",
        "market_dominance": "last", "alt_rank": "last",
        "sentiment": "mean", "galaxy_score": "mean", "social_dominance": "mean",
        "interactions": "sum", "contributors_created": "sum",
        "posts_created": "sum", "spam": "sum",
        "contributors_active": "max", "posts_active": "max",
    }
    agg = {col: rule for col, rule in agg_rules.items() if col in df.columns}
    resampled = df.resample(target_interval).agg(agg).dropna(how="all")
    return resampled


def _parse_period_to_timedelta(period: str) -> pd.Timedelta:
    """
    Parse compact period strings (e.g. 730d, 2w, 3m, 1y, 12h).
    """
    match = re.fullmatch(r"(\d+)([hdwmy])", period.lower().strip())
    if not match:
        raise ValueError(
            f"Invalid period '{period}'. Expected format like '730d', '2w', '3m', '1y', '12h'."
        )

    value = int(match.group(1))
    unit = match.group(2)

    if unit == 'h':
        return pd.Timedelta(hours=value)
    if unit == 'd':
        return pd.Timedelta(days=value)
    if unit == 'w':
        return pd.Timedelta(weeks=value)
    if unit == 'm':
        return pd.Timedelta(days=30 * value)
    if unit == 'y':
        return pd.Timedelta(days=365 * value)

    raise ValueError(f"Unsupported period unit '{unit}'")


def _to_unix_utc(timestamp_like: str) -> int:
    """
    Convert timestamp/date string to unix seconds in UTC.
    """
    from .lunarcrush_api import to_unix_utc
    return to_unix_utc(timestamp_like)


def _lunarcrush_cache_file(
    cache_dir: str,
    coin: str,
    interval: str,
    period: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    resample_interval: Optional[str]
) -> Path:
    """
    Build cache file path for LunarCrush datasets.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    coin_part = coin.lower().strip()
    interval_suffix = f"_{interval}" if interval else ""
    resample_suffix = f"_resampled_{resample_interval}" if resample_interval else ""

    if period is not None:
        filename = f"lunarcrush_{coin_part}_{period}{interval_suffix}{resample_suffix}.csv"
    else:
        filename = f"lunarcrush_{coin_part}_{start_date}_{end_date}{interval_suffix}{resample_suffix}.csv"

    return cache_path / filename


def download_lunarcrush_data(
    coin: str = "btc",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: Optional[str] = None,
    interval: str = "1h",
    period: Optional[str] = "730d",
    resample_interval: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: str = "LUNARCRUSH_API_KEY"
) -> pd.DataFrame:
    """
    Download market + social time series data from LunarCrush.

    Output columns are standardized to:
    [Close, Volume, Sentiment, Social]

    Social uses `social_dominance` when available and falls back to
    `interactions` if needed.

    Args:
        coin: Coin id/symbol accepted by LunarCrush (e.g., "btc").
        start_date: Start date/time (used when period is None).
        end_date: End date/time (used when period is None).
        cache_dir: Optional directory to cache downloaded data.
        interval: Data interval label for cache naming (typically "1h").
        period: Rolling period string (e.g., "730d"). If set, overrides start/end.
        resample_interval: Optional resample interval (e.g., "4h").
        api_key: Optional API key string; if None reads from api_key_env.
        api_key_env: Environment variable name for LunarCrush API key.

    Returns:
        DataFrame with columns [Close, Volume, Sentiment, Social] and UTC DatetimeIndex.
    """
    token = api_key or os.getenv(api_key_env)
    if not token:
        raise ValueError(
            f"Missing LunarCrush API key. Set {api_key_env} environment variable or pass api_key."
        )

    interval_normalized = interval.lower().strip()
    if interval_normalized not in {'1h', 'hour', '60m'}:
        raise ValueError(
            f"LunarCrush integration currently supports hourly interval only, got '{interval}'."
        )

    if period is not None:
        period_delta = _parse_period_to_timedelta(period)
        end_ts = int(pd.Timestamp.now(tz='UTC').floor('h').timestamp())
        start_ts = int((pd.Timestamp(end_ts, unit='s', tz='UTC') - period_delta).timestamp())
    else:
        if start_date is None or end_date is None:
            raise ValueError("Provide either period or both start_date and end_date for LunarCrush data.")
        start_ts = _to_unix_utc(start_date)
        end_ts = _to_unix_utc(end_date)

    params = {
        'bucket': 'hour',
        'start': start_ts,
        'end': end_ts,
    }
    url = f"https://lunarcrush.com/api4/public/coins/{coin}/time-series/v2?{urlencode(params)}"

    def fetch_with_curl() -> dict:
        """
        Fallback transport for environments where urllib user-agent is blocked.
        """
        curl_command = [
            'curl',
            '--silent',
            '--show-error',
            '--location',
            '--max-time',
            '30',
            '--header',
            f'Authorization: Bearer {token}',
            '--header',
            'Accept: application/json',
            '--header',
            'User-Agent: curl/8.7.1',
            url,
        ]

        result = subprocess.run(curl_command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"LunarCrush curl request failed: {result.stderr.strip()}")

        body = result.stdout
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            body_lower = body.lower()
            if 'cloudflare' in body_lower or '<html' in body_lower:
                raise RuntimeError(
                    "LunarCrush request blocked by Cloudflare (curl fallback also failed). "
                    "Try a different network/VPN or contact LunarCrush support for API allowlisting."
                ) from exc
            raise RuntimeError("LunarCrush returned non-JSON response via curl fallback.") from exc

    request = Request(
        url,
        headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
        },
        method='GET'
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')
        details_lower = details.lower()
        if exc.code == 403 and ('cloudflare' in details_lower or 'error 1010' in details_lower):
            payload = fetch_with_curl()
        else:
            raise RuntimeError(f"LunarCrush API error ({exc.code}): {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach LunarCrush API: {exc}") from exc

    if not isinstance(payload, dict) or 'data' not in payload:
        raise ValueError("Unexpected LunarCrush response format: missing 'data' field.")

    rows = payload.get('data')
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("LunarCrush response contains no time-series data.")

    raw = pd.DataFrame(rows)
    required_columns = {'time', 'close', 'volume_24h', 'sentiment'}
    missing = [col for col in required_columns if col not in raw.columns]
    if missing:
        raise ValueError(f"LunarCrush response missing required columns: {missing}")

    social_source_col = 'social_dominance' if 'social_dominance' in raw.columns else 'interactions'
    if social_source_col not in raw.columns:
        raise ValueError(
            "LunarCrush response missing social field. Expected 'social_dominance' or 'interactions'."
        )

    index = pd.to_datetime(raw['time'], unit='s', utc=True, errors='coerce')
    df = pd.DataFrame(
        {
            'Close': pd.to_numeric(raw['close'], errors='coerce'),
            'Volume': pd.to_numeric(raw['volume_24h'], errors='coerce'),
            'Sentiment': pd.to_numeric(raw['sentiment'], errors='coerce'),
            'Social': pd.to_numeric(raw[social_source_col], errors='coerce'),
        },
        index=index,
    )
    df = df[LUNARCRUSH_COLUMNS]
    df.index.name = 'Datetime'

    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    df = df.dropna()

    if resample_interval is not None:
        df = resample_market_social(df, resample_interval)

    if cache_dir is not None and len(df) > 0:
        cache_file = _lunarcrush_cache_file(
            cache_dir=cache_dir,
            coin=coin,
            interval=interval,
            period=period,
            start_date=start_date,
            end_date=end_date,
            resample_interval=resample_interval,
        )
        df.to_csv(cache_file)

    return df


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
    df = df[YAHOO_COLUMNS].copy()

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
    resample_interval: Optional[str] = None,
    source: str = "yahoo",
    coin: str = "btc",
    api_key: Optional[str] = None,
    api_key_env: str = "LUNARCRUSH_API_KEY"
) -> pd.DataFrame:
    """
    Load market data from cache or download if not available.

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
        source: Data source identifier ("yahoo" or "lunarcrush").
        coin: LunarCrush coin id/symbol (used when source="lunarcrush").
        api_key: Optional LunarCrush API key.
        api_key_env: Environment variable name for LunarCrush API key.

    Returns:
        DataFrame with source-specific feature columns and DatetimeIndex.
    """
    source = source.lower().strip()
    if source not in {'yahoo', 'lunarcrush'}:
        raise ValueError(f"Unsupported data source '{source}'. Use 'yahoo' or 'lunarcrush'.")

    # If explicit data path provided, load from there
    if data_path is not None:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        return df

    # Check cache directory for existing file
    if cache_dir is not None:
        if source == 'lunarcrush':
            cache_file = _lunarcrush_cache_file(
                cache_dir=cache_dir,
                coin=coin,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date,
                resample_interval=resample_interval,
            )
        else:
            cache_path = Path(cache_dir)
            interval_suffix = f"_{interval}" if interval != "1d" else ""
            resample_suffix = f"_resampled_{resample_interval}" if resample_interval else ""
            if period is not None:
                cache_file = cache_path / f"{ticker.replace('^', '').replace('=', '')}_{period}{interval_suffix}{resample_suffix}.csv"
            else:
                cache_file = cache_path / f"{ticker.replace('^', '').replace('=', '')}_{start_date}_{end_date}{interval_suffix}{resample_suffix}.csv"

        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df = df.sort_index()
            return df

    # Download fresh data
    if source == 'lunarcrush':
        return download_lunarcrush_data(
            coin=coin,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            interval=interval,
            period=period,
            resample_interval=resample_interval,
            api_key=api_key,
            api_key_env=api_key_env,
        )

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
        df: DataFrame with feature columns and DatetimeIndex.

    Returns:
        Tuple of (tensor of shape (N, num_features), DatetimeIndex).
    """
    tensor = torch.tensor(df.values, dtype=torch.float32)
    return tensor, df.index
