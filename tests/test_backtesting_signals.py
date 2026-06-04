"""Pure-pandas signal construction for the vectorbt adapter (runs in main env).

Oracle O2: signals are decided at decision_time t but EXECUTED at the next bar
open (open at target_time t+1). The signal builder joins each prediction to the
source OHLC open at target_time and produces a long-only entry/exit frame.
"""

import numpy as np
import pandas as pd

from src.backtesting.signals import build_long_only_signals


def _preds():
    return pd.DataFrame({
        "decision_time": pd.to_datetime(
            ["2024-01-01T00", "2024-01-01T04", "2024-01-01T08", "2024-01-01T12"]
        ),
        "target_time": pd.to_datetime(
            ["2024-01-01T04", "2024-01-01T08", "2024-01-01T12", "2024-01-01T16"]
        ),
        "prev_close": [100.0, 101.0, 100.0, 102.0],
        "pred_close": [101.5, 100.5, 101.0, 103.0],
        "true_close": [101.0, 100.0, 102.0, 103.5],
        "pred_return": [0.015, -0.005, 0.010, 0.0098],
        "true_return": [0.010, -0.0099, 0.020, 0.0147],
    })


def _ohlc():
    idx = pd.to_datetime([
        "2024-01-01T04", "2024-01-01T08", "2024-01-01T12", "2024-01-01T16",
    ])
    return pd.DataFrame({"open": [101.0, 100.5, 101.0, 103.0]}, index=idx)


def test_entries_from_positive_pred_return_above_threshold():
    sig = build_long_only_signals(_preds(), _ohlc(), threshold=0.0)
    # pred_return > 0 on rows 0,2,3 -> entry True; row1 (-0.005) -> exit
    assert list(sig["entries"]) == [True, False, True, True]
    assert list(sig["exits"]) == [False, True, False, False]


def test_threshold_filters_small_signals():
    sig = build_long_only_signals(_preds(), _ohlc(), threshold=0.012)
    # only row0 (0.015) clears 0.012; rows 2 (0.010), 3 (0.0098) do not
    assert list(sig["entries"]) == [True, False, False, False]


def test_execution_price_is_open_at_target_time():
    sig = build_long_only_signals(_preds(), _ohlc(), threshold=0.0)
    # exec_price column == open joined on target_time
    assert list(sig["exec_price"]) == [101.0, 100.5, 101.0, 103.0]


def test_index_is_target_time():
    sig = build_long_only_signals(_preds(), _ohlc(), threshold=0.0)
    assert list(sig.index) == list(pd.to_datetime(
        ["2024-01-01T04", "2024-01-01T08", "2024-01-01T12", "2024-01-01T16"]
    ))


def test_missing_open_drops_row_no_leakage():
    ohlc = _ohlc().iloc[:3]  # drop last bar's open
    sig = build_long_only_signals(_preds(), ohlc, threshold=0.0)
    # last prediction has no exec open -> excluded, no NaN exec price
    assert len(sig) == 3
    assert not sig["exec_price"].isna().any()
