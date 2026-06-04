"""
S&P 500 Dataset module.

Provides sliding window dataset class for time-series forecasting.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SP500Dataset(Dataset):
    """
    Sliding window dataset for S&P 500 OHLC data.

    Args:
        data: Tensor of shape (num_samples, num_features) containing feature values.
        window_size: Number of time steps to include in each sample.
        target_col: Index of the target column (default: 3 for Close).
        target_mode: One of ``"price"`` (default), ``"return"``, ``"log_return"``.
            ``"price"`` yields the absolute next close (unchanged legacy behavior).
            ``"return"`` yields ``close[t+1] / close[t] - 1``.
            ``"log_return"`` yields ``log(close[t+1]) - log(close[t])``.
        index: Optional pandas DatetimeIndex aligned to ``data`` rows. When given,
            ``decision_times`` and ``target_times`` arrays are exposed for
            leakage-free backtest alignment (Oracle O1).
        raw_data: Optional raw (unnormalized) tensor aligned to ``data`` rows.
            When given, the return/log_return target and ``prev_closes`` are
            computed from ``raw_data`` instead of ``data``. This lets the model
            input window be z-scored while targets stay on the raw close scale
            (Oracle Design B), so predicted returns reconstruct to real prices.

    Alignment (window length W starting at idx):
        decision_time = index[idx + W - 1]   (last observed bar, "t")
        target_time   = index[idx + W]       (predicted bar, "t+1")
        prev_close    = target_source[idx + W - 1, target_col]
    """

    VALID_TARGET_MODES = ("price", "return", "log_return")

    def __init__(
        self,
        data: torch.Tensor,
        window_size: int,
        target_col: int = 3,
        target_mode: str = "price",
        index: Optional["object"] = None,
        raw_data: Optional[torch.Tensor] = None,
    ):
        assert data.ndim == 2, f"Expected 2D data, got {data.shape}"
        assert 0 < window_size < len(data), f"window_size must be in (0, {len(data)}), got {window_size}"
        assert 0 <= target_col < data.shape[1], f"target_col must be in [0, {data.shape[1]}), got {target_col}"
        if target_mode not in self.VALID_TARGET_MODES:
            raise ValueError(
                f"Invalid target_mode {target_mode!r}; expected one of {self.VALID_TARGET_MODES}"
            )
        if raw_data is not None:
            assert len(raw_data) == len(data), (
                f"raw_data length {len(raw_data)} != data length {len(data)}"
            )

        self.data = data
        self.window_size = window_size
        self.target_col = target_col
        self.target_mode = target_mode
        self.target_source = raw_data if raw_data is not None else data

        n = len(self.data) - self.window_size

        prev_close_rows = self.target_source[
            self.window_size - 1 : self.window_size - 1 + n, self.target_col
        ]
        self.prev_closes = prev_close_rows.detach().cpu().numpy().astype(np.float64)

        self.decision_times = None
        self.target_times = None
        if index is not None:
            idx_values = np.asarray(index)
            assert len(idx_values) == len(self.data), (
                f"index length {len(idx_values)} != data length {len(self.data)}"
            )
            # Oracle O1: decision=index[idx+W-1] (bar t), target=index[idx+W] (bar t+1)
            self.decision_times = idx_values[self.window_size - 1 : self.window_size - 1 + n]
            self.target_times = idx_values[self.window_size : self.window_size + n]

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def _compute_target(self, idx: int) -> torch.Tensor:
        prev_close = self.target_source[idx + self.window_size - 1, self.target_col]
        next_close = self.target_source[idx + self.window_size, self.target_col]

        if self.target_mode == "price":
            return next_close
        if self.target_mode == "return":
            return next_close / prev_close - 1.0
        return torch.log(next_close) - torch.log(prev_close)

    def __getitem__(self, idx: int):
        x = self.data[idx:idx + self.window_size]
        y = self._compute_target(idx)
        return x, y
