"""Oracle O4: multi-coin basket via equal-weight active longs.

compute_target_weights joins per-coin pred_return on target_time and assigns
weight 1/|active| to each coin with pred_return > threshold, 0 otherwise. When
no coin is active the row is all-cash (weights sum to 0).
"""

import numpy as np
import pandas as pd

from src.backtesting.basket import compute_target_weights


def _preds(target_times, pred_returns):
    return pd.DataFrame({
        "target_time": pd.to_datetime(target_times),
        "pred_return": pred_returns,
    })


def test_equal_weight_among_active_longs():
    t = ["2024-01-01T04", "2024-01-01T08"]
    per_coin = {
        "btc": _preds(t, [0.02, -0.01]),
        "eth": _preds(t, [0.01, 0.03]),
        "sol": _preds(t, [-0.02, 0.04]),
    }
    w = compute_target_weights(per_coin, threshold=0.0)
    # row0: btc,eth active -> 0.5 each, sol 0
    assert w.loc[w.index[0], "btc"] == 0.5
    assert w.loc[w.index[0], "eth"] == 0.5
    assert w.loc[w.index[0], "sol"] == 0.0
    # row1: eth,sol active -> 0.5 each, btc 0
    assert w.loc[w.index[1], "btc"] == 0.0
    assert w.loc[w.index[1], "eth"] == 0.5
    assert w.loc[w.index[1], "sol"] == 0.5


def test_all_cash_when_none_active():
    t = ["2024-01-01T04"]
    per_coin = {
        "btc": _preds(t, [-0.02]),
        "eth": _preds(t, [-0.01]),
    }
    w = compute_target_weights(per_coin, threshold=0.0)
    assert w.iloc[0].sum() == 0.0


def test_weights_sum_to_one_or_zero():
    t = ["2024-01-01T04", "2024-01-01T08", "2024-01-01T12"]
    per_coin = {
        "btc": _preds(t, [0.02, -0.01, 0.05]),
        "eth": _preds(t, [0.01, -0.03, 0.04]),
    }
    w = compute_target_weights(per_coin, threshold=0.0)
    sums = w.sum(axis=1).to_numpy()
    for s in sums:
        assert abs(s - 1.0) < 1e-9 or abs(s) < 1e-9


def test_no_negative_weights_long_only():
    t = ["2024-01-01T04"]
    per_coin = {"btc": _preds(t, [0.02]), "eth": _preds(t, [-0.5])}
    w = compute_target_weights(per_coin, threshold=0.0)
    assert (w.to_numpy() >= 0).all()


def test_join_on_target_time_handles_offset_index():
    per_coin = {
        "btc": _preds(["2024-01-01T04", "2024-01-01T08"], [0.02, 0.03]),
        "eth": _preds(["2024-01-01T08", "2024-01-01T12"], [0.01, 0.04]),
    }
    w = compute_target_weights(per_coin, threshold=0.0)
    # union of times = 3 rows; missing pred -> treated as inactive (no crash)
    assert len(w) == 3
    assert not np.isnan(w.to_numpy()).any()
