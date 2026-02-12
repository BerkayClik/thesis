"""
Shared LunarCrush API client.

Provides reusable functions for fetching time-series data from LunarCrush v2,
with retry logic, Cloudflare bypass via curl fallback, and rate limiting.
"""

import json
import os
import subprocess
import time
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# All 18 columns from LunarCrush Time Series v2, in canonical order.
# Index 9 (close) is the default forecasting target.
LUNARCRUSH_ALL_COLUMNS: List[str] = [
    "contributors_active",    # 0
    "contributors_created",   # 1
    "interactions",           # 2
    "posts_active",           # 3
    "posts_created",          # 4
    "sentiment",              # 5
    "spam",                   # 6
    "alt_rank",               # 7
    "circulating_supply",     # 8
    "close",                  # 9  ← target
    "galaxy_score",           # 10
    "high",                   # 11
    "low",                    # 12
    "market_cap",             # 13
    "market_dominance",       # 14
    "open",                   # 15
    "social_dominance",       # 16
    "volume_24h",             # 17
]

# Default feature columns for 4-feature quaternion models: close, high, low, open
QUATERNION_FEATURE_COLS: List[int] = [9, 11, 12, 15]

# Rate-limit delay between API requests (seconds)
_REQUEST_DELAY = 2.0

# Retry settings
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def to_unix_utc(date_str: str) -> int:
    """Convert a date/datetime string to Unix timestamp (UTC seconds)."""
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def _fetch_with_curl(url: str, token: str) -> dict:
    """Fallback transport using curl to bypass Cloudflare."""
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--max-time", "30",
        "--header", f"Authorization: Bearer {token}",
        "--header", "Accept: application/json",
        "--header", "User-Agent: curl/8.7.1",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr.strip()}")
    body = result.stdout
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        if "cloudflare" in body.lower() or "<html" in body.lower():
            raise RuntimeError(
                "Blocked by Cloudflare (curl fallback also failed)."
            ) from exc
        raise RuntimeError("Non-JSON response from curl fallback.") from exc


def _fetch_with_urllib(url: str, token: str) -> dict:
    """Primary transport using urllib."""
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
        },
        method="GET",
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_lunarcrush_timeseries(
    coin: str,
    bucket: str,
    start_ts: int,
    end_ts: int,
    api_key: Optional[str] = None,
    api_key_env: str = "LUNARCRUSH_API_KEY",
) -> pd.DataFrame:
    """
    Fetch time-series data from LunarCrush v2 for a single chunk.

    Returns a DataFrame with all 18 canonical columns, indexed by UTC datetime.
    Retries up to 3 times with exponential backoff on 429/5xx errors,
    and falls back to curl on Cloudflare 403.

    Args:
        coin: Coin symbol (e.g. "btc", "eth").
        bucket: Time bucket ("hour" or "day").
        start_ts: Start Unix timestamp (UTC).
        end_ts: End Unix timestamp (UTC).
        api_key: API key string. If None, reads from api_key_env.
        api_key_env: Environment variable name for the API key.

    Returns:
        DataFrame with LUNARCRUSH_ALL_COLUMNS and UTC DatetimeIndex.
        May be empty if the API returns no data for the range.
    """
    token = api_key or os.getenv(api_key_env)
    if not token:
        raise ValueError(
            f"Missing LunarCrush API key. Set {api_key_env} or pass api_key."
        )

    params = {"bucket": bucket, "start": start_ts, "end": end_ts}
    url = (
        f"https://lunarcrush.com/api4/public/coins/{coin}/time-series/v2"
        f"?{urlencode(params)}"
    )

    payload = None
    last_exc = None

    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            wait = _BACKOFF_BASE ** attempt
            time.sleep(wait)

        try:
            payload = _fetch_with_urllib(url, token)
            break
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            details_lower = details.lower()

            # Cloudflare block → curl fallback (no retry needed)
            if exc.code == 403 and (
                "cloudflare" in details_lower or "error 1010" in details_lower
            ):
                payload = _fetch_with_curl(url, token)
                break

            # Rate-limited or server error → retry
            if exc.code in (429, 500, 502, 503, 504):
                last_exc = exc
                continue

            raise RuntimeError(
                f"LunarCrush API error ({exc.code}): {details}"
            ) from exc
        except URLError as exc:
            last_exc = exc
            continue

    if payload is None:
        raise RuntimeError(
            f"LunarCrush request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(
            f"Unexpected response format (keys: "
            f"{list(payload.keys()) if isinstance(payload, dict) else type(payload)})"
        )

    rows = payload["data"]
    if not isinstance(rows, list) or len(rows) == 0:
        return pd.DataFrame(columns=LUNARCRUSH_ALL_COLUMNS)

    raw = pd.DataFrame(rows)

    # Build output with canonical columns
    if "time" not in raw.columns:
        raise ValueError("Response missing 'time' column.")

    index = pd.to_datetime(raw["time"], unit="s", utc=True, errors="coerce")

    out = pd.DataFrame(index=index)
    for col in LUNARCRUSH_ALL_COLUMNS:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce").values
        else:
            out[col] = float("nan")

    out.index.name = "Datetime"
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    # Rate-limit
    time.sleep(_REQUEST_DELAY)

    return out
