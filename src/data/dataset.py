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
        returns: Pre-computed returns from raw (unnormalized) prices.
                 If provided and predict_returns=True, these are used as targets
                 instead of computing returns from the (normalized) data.
                 This is critical for correct training - see preprocessing.py.
    """

    def __init__(
        self,
        data: torch.Tensor,
        window_size: int,
        target_col: int = 3,
        predict_returns: bool = True,
        returns: torch.Tensor = None
    ):
        assert data.ndim == 2, f"Expected 2D data, got {data.shape}"
        assert 0 < window_size < len(data), f"window_size must be in (0, {len(data)}), got {window_size}"
        assert 0 <= target_col < data.shape[1], f"target_col must be in [0, {data.shape[1]}), got {target_col}"

        self.data = data
        self.window_size = window_size
        self.target_col = target_col
        self.predict_returns = predict_returns
        self.returns = returns  # Pre-computed returns from raw prices

        # Validate returns tensor if provided
        if returns is not None:
            # Returns tensor should have len(data) - 1 elements
            # (one return per consecutive price pair)
            expected_len = len(data) - 1
            assert len(returns) == expected_len, (
                f"Returns tensor length ({len(returns)}) should be data length - 1 ({expected_len})"
            )

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int):
        # X: window of OHLC data (normalized)
        x = self.data[idx:idx + self.window_size]

        if self.predict_returns and self.returns is not None:
            # Use pre-computed return from raw prices
            # returns[t] = return from time t to time t+1
            # For window ending at idx+window_size-1, we want return to idx+window_size
            # That's returns[idx + window_size - 1]
            y = self.returns[idx + self.window_size - 1]
        elif self.predict_returns:
            # Fallback: compute from normalized data (NOT RECOMMENDED - causes variance issues)
            # This path is kept for backward compatibility only
            prev_close = self.data[idx + self.window_size - 1, self.target_col]
            next_close = self.data[idx + self.window_size, self.target_col]
            y = (next_close - prev_close) / (prev_close.abs() + 1e-8)
        else:
            # y: next-step close price (legacy behavior)
            y = self.data[idx + self.window_size, self.target_col]

        return x, y
