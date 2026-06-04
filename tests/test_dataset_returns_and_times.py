"""Tests for return-mode targets and timestamp propagation in SP500Dataset.

Oracle O1 alignment for a window of length W starting at idx:
    window rows = data[idx : idx+W]
    decision_time = index[idx + W - 1]   (last observed bar, "t")
    target_time   = index[idx + W]       (predicted bar, "t+1")
    prev_close    = data[idx + W - 1, target_col]
    y_price       = data[idx + W, target_col]
    y_return      = y_price / prev_close - 1
    y_log_return  = log(y_price) - log(prev_close)
"""

import math

import numpy as np
import pandas as pd
import torch

from src.data.dataset import SP500Dataset


def _close_tensor(closes):
    """Build a (N, 1) tensor with the single column acting as close (target_col=0)."""
    return torch.tensor([[c] for c in closes], dtype=torch.float32)


def test_price_mode_unchanged():
    """Default price mode must yield the absolute next close (regression lock)."""
    data = _close_tensor([10, 11, 12, 13, 14])
    ds = SP500Dataset(data, window_size=2, target_col=0)  # default target_mode="price"
    # idx=0 -> window [10,11], target data[2]=12
    _, y0 = ds[0]
    assert float(y0) == 12.0
    # idx=2 -> window [12,13], target data[4]=14
    _, y2 = ds[2]
    assert float(y2) == 14.0


def test_simple_return_target():
    data = _close_tensor([10, 11, 12, 13, 14])
    ds = SP500Dataset(data, window_size=2, target_col=0, target_mode="return")
    # idx=0 -> prev_close=data[1]=11, target=data[2]=12 -> 12/11 - 1
    _, y0 = ds[0]
    assert float(y0) == pytest_approx(12 / 11 - 1)
    # idx=2 -> prev_close=data[3]=13, target=data[4]=14 -> 14/13 - 1
    _, y2 = ds[2]
    assert float(y2) == pytest_approx(14 / 13 - 1)


def test_log_return_target():
    data = _close_tensor([10, 11, 12, 13, 14])
    ds = SP500Dataset(data, window_size=2, target_col=0, target_mode="log_return")
    _, y0 = ds[0]
    assert float(y0) == pytest_approx(math.log(12) - math.log(11))


def test_decision_and_target_times():
    closes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    data = _close_tensor(closes)
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    ds = SP500Dataset(data, window_size=3, target_col=0, index=index)
    # idx=2 -> decision=index[2+3-1]=index[4]=2020-01-05, target=index[5]=2020-01-06
    assert ds.decision_times[2] == np.datetime64("2020-01-05")
    assert ds.target_times[2] == np.datetime64("2020-01-06")
    # arrays aligned to len(dataset)
    assert len(ds.decision_times) == len(ds)
    assert len(ds.target_times) == len(ds)


def test_prev_closes_exposed():
    """prev_close array aligned to dataset length for downstream reconstruction."""
    closes = [10, 11, 12, 13, 14]
    data = _close_tensor(closes)
    ds = SP500Dataset(data, window_size=2, target_col=0, target_mode="return")
    # idx=0 -> prev_close=data[1]=11 ; idx=2 -> prev_close=data[3]=13
    assert float(ds.prev_closes[0]) == 11.0
    assert float(ds.prev_closes[2]) == 13.0
    assert len(ds.prev_closes) == len(ds)


# --- tiny local approx helper (avoid extra import noise) ---
def pytest_approx(value, rel=1e-6):
    import pytest

    return pytest.approx(value, rel=rel)
