"""Pure-pandas, leakage-free signal construction (no vectorbt dependency).

Oracle O2: a prediction is decided at ``decision_time`` (bar t) but executed at
the next bar's open (``open`` at ``target_time`` = bar t+1). The output frame is
indexed by ``target_time`` (the execution bar) and carries the execution price
joined from the source OHLC, so the backtest never fills on the same bar the
model observed.
"""

import pandas as pd


def build_position_signals(
    predictions,
    ohlc,
    threshold=0.0,
    allow_short=True,
    exit_mode="hold",
):
    """Build a target-position frame (+1 long / -1 short / 0 flat) executed at
    the next bar open, with a symmetric dead band around zero.

    Position rule per bar (c = threshold):
        pred_return >  +c  ->  +1
        pred_return <  -c  ->  -1 (or 0 when allow_short=False)
        |pred_return| <= c ->  exit_mode "hold": keep previous position
                               exit_mode "flat": go to 0

    The dead band plus "hold" gives hysteresis: a position is only changed when
    the prediction crosses the opposite band, which cuts churn (and fees) from
    near-zero predictions flickering around 0.

    Args:
        predictions: DataFrame with at least ``target_time`` and ``pred_return``.
        ohlc: DataFrame indexed by datetime with an ``open`` column.
        threshold: half-width of the no-trade dead band (>= 0).
        allow_short: if False, sell signals go flat instead of short.
        exit_mode: "hold" (keep position inside the band) or "flat".

    Returns:
        DataFrame indexed by ``target_time`` with columns ``position`` (float)
        and ``exec_price`` (open at target_time). Rows whose target_time has no
        matching open are dropped (no NaN execution).
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if exit_mode not in ("hold", "flat"):
        raise ValueError(f"exit_mode must be 'hold' or 'flat', got {exit_mode!r}")

    preds = predictions.copy()
    preds["target_time"] = pd.to_datetime(preds["target_time"])

    open_series = ohlc["open"]
    open_series.index = pd.to_datetime(open_series.index)

    exec_price = preds["target_time"].map(open_series)
    keep = exec_price.notna()
    preds = preds.loc[keep].reset_index(drop=True)
    exec_price = exec_price.loc[keep].reset_index(drop=True)

    pred_return = preds["pred_return"].to_numpy()
    short_pos = -1.0 if allow_short else 0.0

    positions = []
    pos = 0.0
    for r in pred_return:
        if r > threshold:
            pos = 1.0
        elif r < -threshold:
            pos = short_pos
        elif exit_mode == "flat":
            pos = 0.0
        # exit_mode == "hold": keep previous pos inside the band
        positions.append(pos)

    out = pd.DataFrame({
        "position": positions,
        "exec_price": exec_price.to_numpy(),
    }, index=pd.DatetimeIndex(preds["target_time"], name="target_time"))
    return out


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
