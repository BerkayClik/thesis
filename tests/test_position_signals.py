"""Long/short dead-band position signals (pure pandas, runs in main env).

build_position_signals maps pred_return to a target position (+1/-1/0) with a
symmetric dead band: predictions inside [-c, +c] either hold the previous
position (exit_mode="hold", hysteresis) or go flat (exit_mode="flat"). Like the
long-only builder, execution is the next bar open joined on target_time.
"""

import pandas as pd
import pytest

from src.backtesting.signals import build_position_signals


def _frame(pred_returns):
    n = len(pred_returns)
    times = pd.date_range("2024-01-01", periods=n + 1, freq="4h")
    return pd.DataFrame({
        "decision_time": times[:n],
        "target_time": times[1:],
        "pred_return": pred_returns,
    })


def _ohlc(n):
    idx = pd.date_range("2024-01-01 04:00", periods=n, freq="4h")
    return pd.DataFrame({"open": [100.0 + i for i in range(n)]}, index=idx)


def test_long_short_signs_no_band():
    preds = _frame([0.01, -0.01, 0.02, -0.005])
    sig = build_position_signals(preds, _ohlc(4), threshold=0.0)
    assert list(sig["position"]) == [1.0, -1.0, 1.0, -1.0]


def test_dead_band_hold_keeps_position():
    # band 0.005: 0.01 -> long, 0.001 inside band -> hold long,
    # -0.01 -> short, -0.002 inside band -> hold short
    preds = _frame([0.01, 0.001, -0.01, -0.002])
    sig = build_position_signals(preds, _ohlc(4), threshold=0.005, exit_mode="hold")
    assert list(sig["position"]) == [1.0, 1.0, -1.0, -1.0]


def test_dead_band_flat_exits_position():
    preds = _frame([0.01, 0.001, -0.01, -0.002])
    sig = build_position_signals(preds, _ohlc(4), threshold=0.005, exit_mode="flat")
    assert list(sig["position"]) == [1.0, 0.0, -1.0, 0.0]


def test_starts_flat_inside_band():
    preds = _frame([0.001, -0.001, 0.01])
    sig = build_position_signals(preds, _ohlc(3), threshold=0.005, exit_mode="hold")
    assert list(sig["position"]) == [0.0, 0.0, 1.0]


def test_no_short_goes_flat_on_sell_signal():
    preds = _frame([0.01, -0.02, 0.015])
    sig = build_position_signals(preds, _ohlc(3), threshold=0.0, allow_short=False)
    assert list(sig["position"]) == [1.0, 0.0, 1.0]


def test_band_boundary_is_inside_band():
    # pred exactly == threshold does not trigger entry (strict inequality)
    preds = _frame([0.005, -0.005])
    sig = build_position_signals(preds, _ohlc(2), threshold=0.005, exit_mode="flat")
    assert list(sig["position"]) == [0.0, 0.0]


def test_execution_price_and_index_from_target_time():
    preds = _frame([0.01, -0.01, 0.02])
    ohlc = _ohlc(3)
    sig = build_position_signals(preds, ohlc, threshold=0.0)
    assert list(sig["exec_price"]) == list(ohlc["open"])
    assert list(sig.index) == list(ohlc.index)


def test_missing_open_drops_row():
    preds = _frame([0.01, -0.01, 0.02])
    ohlc = _ohlc(3).iloc[:2]
    sig = build_position_signals(preds, ohlc, threshold=0.0)
    assert len(sig) == 2
    assert not sig["exec_price"].isna().any()


def test_invalid_args_raise():
    preds = _frame([0.01])
    with pytest.raises(ValueError):
        build_position_signals(preds, _ohlc(1), threshold=-0.1)
    with pytest.raises(ValueError):
        build_position_signals(preds, _ohlc(1), exit_mode="bogus")
