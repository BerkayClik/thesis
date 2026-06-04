"""Oracle Design B: in return-mode the dataset can carry a separate raw_data
companion so the model input may be z-scored while the return target and
prev_close are computed from RAW closes.

    x      = data[i : i+W]                         (model input, possibly z-scored)
    prev   = raw_data[i+W-1, target_col]           (raw close at decision bar t)
    next   = raw_data[i+W,   target_col]           (raw close at target bar t+1)
    y      = next / prev - 1   (return)            (computed from RAW, not input)
"""

import math

import torch

from src.data.dataset import SP500Dataset


def _col(vals):
    return torch.tensor([[v] for v in vals], dtype=torch.float32)


def test_raw_companion_drives_return_target():
    # Input data is deliberately a DIFFERENT scale than raw (simulating z-score).
    input_data = _col([0.0, 0.1, 0.2, 0.3, 0.4])
    raw_data = _col([10.0, 11.0, 12.0, 13.0, 14.0])
    ds = SP500Dataset(
        input_data, window_size=2, target_col=0,
        target_mode="return", raw_data=raw_data,
    )
    x0, y0 = ds[0]
    # x comes from input_data (z-scored), y from raw_data: 12/11 - 1
    assert abs(float(x0[0, 0]) - 0.0) < 1e-6 and abs(float(x0[1, 0]) - 0.1) < 1e-6
    assert abs(float(y0) - (12 / 11 - 1)) < 1e-6


def test_raw_companion_prev_closes_are_raw():
    input_data = _col([0.0, 0.1, 0.2, 0.3, 0.4])
    raw_data = _col([10.0, 11.0, 12.0, 13.0, 14.0])
    ds = SP500Dataset(
        input_data, window_size=2, target_col=0,
        target_mode="return", raw_data=raw_data,
    )
    # prev_close for idx=0 = raw_data[1] = 11 ; idx=2 = raw_data[3] = 13
    assert float(ds.prev_closes[0]) == 11.0
    assert float(ds.prev_closes[2]) == 13.0


def test_raw_companion_log_return():
    input_data = _col([0.0, 0.1, 0.2, 0.3, 0.4])
    raw_data = _col([10.0, 11.0, 12.0, 13.0, 14.0])
    ds = SP500Dataset(
        input_data, window_size=2, target_col=0,
        target_mode="log_return", raw_data=raw_data,
    )
    _, y0 = ds[0]
    assert abs(float(y0) - (math.log(12) - math.log(11))) < 1e-6


def test_raw_companion_length_must_match():
    input_data = _col([0.0, 0.1, 0.2, 0.3, 0.4])
    raw_data = _col([10.0, 11.0, 12.0])  # wrong length
    try:
        SP500Dataset(
            input_data, window_size=2, target_col=0,
            target_mode="return", raw_data=raw_data,
        )
        assert False, "expected an assertion on mismatched raw_data length"
    except AssertionError:
        pass


def test_no_raw_companion_uses_own_data():
    # Backward compatible: without raw_data, target computed from input data.
    data = _col([10.0, 11.0, 12.0, 13.0, 14.0])
    ds = SP500Dataset(data, window_size=2, target_col=0, target_mode="return")
    _, y0 = ds[0]
    assert abs(float(y0) - (12 / 11 - 1)) < 1e-6
