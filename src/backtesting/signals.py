"""Pure-pandas, leakage-free signal construction (no vectorbt dependency).

Oracle O2: a prediction is decided at ``decision_time`` (bar t) but executed at
the next bar's open (``open`` at ``target_time`` = bar t+1). The output frame is
indexed by ``target_time`` (the execution bar) and carries the execution price
joined from the source OHLC, so the backtest never fills on the same bar the
model observed.
"""

import pandas as pd


def build_long_only_signals(predictions, ohlc, threshold=0.0):
    """Build a long-only entry/exit frame executed at the next bar open.

    Args:
        predictions: DataFrame with at least ``target_time`` and ``pred_return``.
        ohlc: DataFrame indexed by datetime with an ``open`` column.
        threshold: enter long only when ``pred_return > threshold``.

    Returns:
        DataFrame indexed by ``target_time`` with columns ``entries`` (bool),
        ``exits`` (bool), ``exec_price`` (open at target_time). Rows whose
        target_time has no matching open are dropped (no NaN execution).
    """
    preds = predictions.copy()
    preds["target_time"] = pd.to_datetime(preds["target_time"])

    open_series = ohlc["open"]
    open_series.index = pd.to_datetime(open_series.index)

    exec_price = preds["target_time"].map(open_series)
    keep = exec_price.notna()

    preds = preds.loc[keep].reset_index(drop=True)
    exec_price = exec_price.loc[keep].reset_index(drop=True)

    entries = preds["pred_return"] > threshold
    exits = ~entries

    out = pd.DataFrame({
        "entries": entries.to_numpy(),
        "exits": exits.to_numpy(),
        "exec_price": exec_price.to_numpy(),
    }, index=pd.DatetimeIndex(preds["target_time"], name="target_time"))
    return out
