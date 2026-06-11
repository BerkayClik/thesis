"""vectorbt single-coin backtest adapter (runs in the isolated backtest env).

Consumes a predictions CSV (decision_time/target_time/.../pred_return) plus the
source OHLC, builds leakage-free next-open signals (Oracle O2), runs a vectorbt
Portfolio, and writes stats JSON + trades CSV + a serialized Portfolio.

Run with the backtest venv:
    .venv-backtest/bin/python -m src.backtesting.vbt_adapter \
        --predictions <preds.csv> --ohlc <ohlc.csv> --outdir <dir>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.backtesting.signals import build_long_only_signals, build_position_signals


def load_ohlc(path):
    df = pd.read_csv(path)
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    df.columns = [c.lower() for c in df.columns]
    return df


def run_single_coin_backtest(
    predictions,
    ohlc,
    threshold=0.0,
    init_cash=10_000.0,
    fees=0.001,
    slippage=0.0005,
    freq="4h",
):
    """Run a long-only next-open backtest and return (portfolio, signals)."""
    sig = build_long_only_signals(predictions, ohlc, threshold=threshold)
    if len(sig) == 0:
        raise ValueError("No executable signals (no predictions matched OHLC opens).")

    price = pd.Series(sig["exec_price"].to_numpy(), index=sig.index)
    pf = vbt.Portfolio.from_signals(
        price,
        entries=sig["entries"].to_numpy(),
        exits=sig["exits"].to_numpy(),
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
    return pf, sig


def run_position_backtest(
    predictions,
    ohlc,
    threshold=0.0,
    allow_short=True,
    exit_mode="hold",
    init_cash=10_000.0,
    fees=0.001,
    slippage=0.0005,
    freq="4h",
):
    """Run a target-position backtest (long/short/flat) at next-bar opens.

    Positions come from ``build_position_signals`` (+1/-1/0 with a dead band).
    Orders are submitted only on position *changes* (size is NaN elsewhere), so
    holding a position across bars never pays fees or rebalances.

    Returns (portfolio, signals).
    """
    sig = build_position_signals(
        predictions, ohlc,
        threshold=threshold, allow_short=allow_short, exit_mode=exit_mode,
    )
    if len(sig) == 0:
        raise ValueError("No executable signals (no predictions matched OHLC opens).")

    price = pd.Series(sig["exec_price"].to_numpy(), index=sig.index)
    position = sig["position"].to_numpy()

    # Order only at transitions: target percent at change bars, NaN = no order.
    size = np.full(len(position), np.nan)
    prev = 0.0
    for i, p in enumerate(position):
        if p != prev:
            size[i] = p
            prev = p

    pf = vbt.Portfolio.from_orders(
        price,
        size=size,
        size_type="targetpercent",
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
    return pf, sig


def _stats_dict(pf):
    raw = pf.stats().to_dict()
    out = {}
    for k, v in raw.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            fv = float(v)
            out[k] = fv if np.isfinite(fv) else None
        elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
            out[k] = str(v)
        else:
            try:
                fv = float(v)
                out[k] = fv if np.isfinite(fv) else None
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--ohlc", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--init-cash", type=float, default=10_000.0)
    p.add_argument("--fees", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--freq", default="4h")
    p.add_argument("--label", default="single")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    ohlc = load_ohlc(args.ohlc)

    pf, sig = run_single_coin_backtest(
        predictions, ohlc,
        threshold=args.threshold, init_cash=args.init_cash,
        fees=args.fees, slippage=args.slippage, freq=args.freq,
    )

    stats = _stats_dict(pf)
    stats_path = os.path.join(args.outdir, f"{args.label}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    trades = pf.trades.records_readable
    trades_path = os.path.join(args.outdir, f"{args.label}_trades.csv")
    trades.to_csv(trades_path, index=False)

    equity_path = os.path.join(args.outdir, f"{args.label}_equity.csv")
    pf.value().to_csv(equity_path)

    print(f"Backtest: {args.label}")
    print(f"  signals: {len(sig)}  entries: {int(sig['entries'].sum())}")
    print(f"  Total Return [%]: {stats.get('Total Return [%]')}")
    print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio')}")
    print(f"  Max Drawdown [%]: {stats.get('Max Drawdown [%]')}")
    print(f"  stats  -> {stats_path}")
    print(f"  trades -> {trades_path}")
    print(f"  equity -> {equity_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
