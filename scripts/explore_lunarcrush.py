"""
Fetch raw hourly BTC data from LunarCrush with ALL columns.

Standalone exploration script — does not modify the existing pipeline.
Run from project root:
  python scripts/explore_lunarcrush.py          # default 1d
  python scripts/explore_lunarcrush.py 30d      # last 30 days
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on environment variables directly

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY_ENV = "LUNARCRUSH_API_KEY"
COIN = "btc"
DEFAULT_PERIOD = "1d"
BUCKET = "hour"


def fetch_lunarcrush_raw(period: str) -> pd.DataFrame:
    """Fetch hourly BTC data from LunarCrush, keeping all columns."""

    token = os.getenv(API_KEY_ENV)
    if not token:
        print(f"ERROR: Set the {API_KEY_ENV} environment variable.", file=sys.stderr)
        sys.exit(1)

    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(period)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    params = {"bucket": BUCKET, "start": start_ts, "end": end_ts}
    url = f"https://lunarcrush.com/api4/public/coins/{COIN}/time-series/v2?{urlencode(params)}"

    print(f"Requesting: {url}")
    print(f"Time range: {start} → {end}  (UTC)\n")

    # --- primary: urllib ------------------------------------------------
    def _fetch_urllib() -> dict:
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

    # --- fallback: curl (Cloudflare bypass) -----------------------------
    def _fetch_curl() -> dict:
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

    # --- execute --------------------------------------------------------
    try:
        payload = _fetch_urllib()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and (
            "cloudflare" in details.lower() or "error 1010" in details.lower()
        ):
            print("urllib blocked by Cloudflare — falling back to curl …")
            payload = _fetch_curl()
        else:
            raise RuntimeError(f"LunarCrush API error ({exc.code}): {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach LunarCrush API: {exc}") from exc

    # --- parse ----------------------------------------------------------
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Unexpected response format (keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)})")

    rows = payload["data"]
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("Response contains no time-series data.")

    df = pd.DataFrame(rows)

    # Convert 'time' column to datetime index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise exploration summary."""
    print("=" * 72)
    print("LUNARCRUSH RAW DATA — ALL COLUMNS")
    print("=" * 72)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Time range: {df.index.min()} → {df.index.max()}")
    print(f"\nColumns ({df.shape[1]}):")
    for col in df.columns:
        print(f"  {col:<30s}  dtype={df[col].dtype}  nulls={df[col].isna().sum()}")

    print(f"\n{'— describe().T ' + '—' * 56}")
    with pd.option_context("display.max_columns", None, "display.width", 140, "display.float_format", "{:.4f}".format):
        print(df.describe().T)

    print(f"\n{'— head(3) ' + '—' * 61}")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.head(3))

    print(f"\n{'— tail(3) ' + '—' * 61}")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.tail(3))


def main() -> None:
    period = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    df = fetch_lunarcrush_raw(period)

    # Save CSV
    out = Path(f"data/cache/lunarcrush_btc_exploration_{period}_all_columns.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved → {out}  ({out.stat().st_size:,} bytes)\n")

    print_summary(df)


if __name__ == "__main__":
    main()
