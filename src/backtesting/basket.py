"""Oracle O4: multi-coin basket via equal-weight active longs + cash_sharing.

``compute_target_weights`` is pure-pandas (main env testable). ``run_basket``
requires vectorbt and runs in the isolated backtest env.

Each bar, every coin whose ``pred_return > threshold`` gets an equal target
weight ``1/|active|`` (long-only); inactive coins get 0; if no coin is active
the bar is all-cash. Weights feed ``Portfolio.from_orders`` with
``size_type="targetpercent"``, ``cash_sharing=True``, ``group_by=True`` so the
basket is one pooled NAV stream rather than independent per-coin runs.
"""

import pandas as pd


def compute_target_weights(per_coin_predictions, threshold=0.0):
    """Build a (target_time x coin) target-weight DataFrame.

    Args:
        per_coin_predictions: dict ``{coin: DataFrame}`` where each frame has
            ``target_time`` and ``pred_return`` columns.
        threshold: a coin is an active long when ``pred_return > threshold``.

    Returns:
        DataFrame indexed by the union of target_times, columns = coins, values
        in [0, 1]. Each row sums to 1 (active longs) or 0 (all-cash).
    """
    series = {}
    for coin, df in per_coin_predictions.items():
        s = df.copy()
        s["target_time"] = pd.to_datetime(s["target_time"])
        series[coin] = s.set_index("target_time")["pred_return"]

    pred = pd.DataFrame(series).sort_index()
    active = (pred > threshold) & pred.notna()
    counts = active.sum(axis=1)

    weights = active.astype(float)
    nonzero = counts > 0
    weights.loc[nonzero] = weights.loc[nonzero].div(counts[nonzero], axis=0)
    weights.loc[~nonzero] = 0.0
    return weights.fillna(0.0)


def run_basket(
    per_coin_predictions,
    per_coin_ohlc,
    threshold=0.0,
    init_cash=100_000.0,
    fees=0.001,
    slippage=0.0005,
    freq="4h",
):
    """Run a pooled multi-asset basket backtest (cash_sharing) and return the
    vectorbt Portfolio. Execution price is each coin's open at target_time."""
    import vectorbt as vbt

    weights = compute_target_weights(per_coin_predictions, threshold=threshold)

    open_cols = {}
    for coin, ohlc in per_coin_ohlc.items():
        o = ohlc["open"].copy()
        o.index = pd.to_datetime(o.index)
        open_cols[coin] = o

    open_df = pd.DataFrame(open_cols).reindex(weights.index)

    # Restrict to bars where every coin has an execution price (no NaN fills).
    valid = open_df.notna().all(axis=1)
    weights = weights.loc[valid]
    open_df = open_df.loc[valid]

    if len(weights) == 0:
        raise ValueError("No bars with complete OHLC opens across all coins.")

    pf = vbt.Portfolio.from_orders(
        close=open_df,
        size=weights,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
    return pf, weights


def _main() -> int:
    import argparse
    import json
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.backtesting.vbt_adapter import load_ohlc, _stats_dict

    p = argparse.ArgumentParser()
    p.add_argument("--coin", action="append", required=True,
                   help="coin:predictions.csv:ohlc.csv  (repeat per coin)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--init-cash", type=float, default=100_000.0)
    p.add_argument("--fees", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--freq", default="4h")
    p.add_argument("--label", default="basket")
    args = p.parse_args()

    per_pred, per_ohlc = {}, {}
    for spec in args.coin:
        coin, preds_path, ohlc_path = spec.split(":")
        per_pred[coin] = pd.read_csv(preds_path)
        per_ohlc[coin] = load_ohlc(ohlc_path)

    pf, weights = run_basket(
        per_pred, per_ohlc, threshold=args.threshold,
        init_cash=args.init_cash, fees=args.fees,
        slippage=args.slippage, freq=args.freq,
    )

    os.makedirs(args.outdir, exist_ok=True)
    stats = _stats_dict(pf)
    stats_path = os.path.join(args.outdir, f"{args.label}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    pf.value().to_csv(os.path.join(args.outdir, f"{args.label}_equity.csv"))

    print(f"Basket: {args.label}  coins: {list(per_pred)}")
    print(f"  bars: {len(weights)}")
    print(f"  Total Return [%]: {stats.get('Total Return [%]')}")
    print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio')}")
    print(f"  Max Drawdown [%]: {stats.get('Max Drawdown [%]')}")
    print(f"  stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
