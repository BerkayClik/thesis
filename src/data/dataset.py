"""
S&P 500 Dataset module.

Provides sliding window dataset class for time-series forecasting.
"""

import torch
from torch.utils.data import Dataset


class SP500Dataset(Dataset):
    """
    Sliding window dataset for S&P 500 OHLC data.

    Args:
        data: Tensor of shape (num_samples, 4) containing OHLC values.
        window_size: Number of time steps to include in each sample.
        target_col: Index of the target column (default: 3 for Close).
        predict_returns: If True, target is percentage return instead of price.
    """

    def __init__(
        self,
        data: torch.Tensor,
        window_size: int,
        target_col: int = 3,
        predict_returns: bool = True
    ):
        assert data.ndim == 2, f"Expected 2D data, got {data.shape}"
        assert 0 < window_size < len(data), f"window_size must be in (0, {len(data)}), got {window_size}"
        assert 0 <= target_col < data.shape[1], f"target_col must be in [0, {data.shape[1]}), got {target_col}"

        self.data = data
        self.window_size = window_size
        self.target_col = target_col
        self.predict_returns = predict_returns

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int):
        # X: window of OHLC data
        x = self.data[idx:idx + self.window_size]
        # Next close price
        next_close = self.data[idx + self.window_size, self.target_col]

        if self.predict_returns:
            # y: percentage return from last close to next close
            prev_close = self.data[idx + self.window_size - 1, self.target_col]
            y = (next_close - prev_close) / (prev_close.abs() + 1e-8)
        else:
            # y: next-step close price (legacy behavior)
            y = next_close

        return x, y
