"""
Data preprocessing module.

Provides normalization, temporal splitting, and quaternion encoding functions.
"""

import torch
import pandas as pd
from typing import Tuple, Dict, Union


def normalize_data(
    data: torch.Tensor,
    stats: Dict[str, torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply Z-score normalization to data.

    Args:
        data: Tensor of shape (num_samples, num_features).
        stats: Optional dict with 'mean' and 'std' tensors. If None, computes from data.

    Returns:
        Tuple of (normalized_data, stats_dict).
    """
    if stats is None:
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        stats = {'mean': mean, 'std': std}

    normalized = (data - stats['mean']) / torch.clamp(stats['std'], min=1e-6)
    return normalized, stats


def temporal_split(
    data: torch.Tensor,
    dates: pd.DatetimeIndex,
    train_end_year: int = 2018,
    val_end_year: int = 2021
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Split data temporally by year boundaries without shuffling.

    Args:
        data: Full dataset tensor of shape (num_samples, num_features).
        dates: DatetimeIndex corresponding to data rows.
        train_end_year: Last year of training data (inclusive).
        val_end_year: Last year of validation data (inclusive).

    Returns:
        Tuple of (train_data, val_data, test_data, split_info).
        split_info contains indices and dates for each split.

    Example:
        train_end_year=2018, val_end_year=2021 gives:
        - Train: 2000-2018 (up to and including Dec 31, 2018)
        - Val: 2019-2021 (Jan 1, 2019 to Dec 31, 2021)
        - Test: 2022+ (Jan 1, 2022 onwards)
    """
    # Create boolean masks based on year boundaries
    years = dates.year  # Returns numpy array
    train_mask = years <= train_end_year
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = years > val_end_year

    # Split data using boolean masks
    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Return split info for debugging/verification
    split_info = {
        'train_dates': dates[train_mask],
        'val_dates': dates[val_mask],
        'test_dates': dates[test_mask],
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }

    return train_data, val_data, test_data, split_info


def temporal_split_ratio(
    data: torch.Tensor,
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Split data temporally by ratio without shuffling.

    Args:
        data: Full dataset tensor of shape (num_samples, num_features).
        dates: DatetimeIndex corresponding to data rows.
        train_ratio: Fraction of data for training (default: 0.70).
        val_ratio: Fraction of data for validation (default: 0.10).
        test_ratio: Fraction of data for testing (default: 0.20).

    Returns:
        Tuple of (train_data, val_data, test_data, split_info).
        split_info contains indices and dates for each split.

    Example:
        train_ratio=0.70, val_ratio=0.10, test_ratio=0.20 gives:
        - Train: First 70% of data
        - Val: Next 10% of data
        - Test: Final 20% of data
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split data using indices
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Return split info for debugging/verification
    split_info = {
        'train_dates': dates[:train_end],
        'val_dates': dates[train_end:val_end],
        'test_dates': dates[val_end:],
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'train_ratio_actual': len(train_data) / n,
        'val_ratio_actual': len(val_data) / n,
        'test_ratio_actual': len(test_data) / n
    }

    return train_data, val_data, test_data, split_info


def encode_quaternion(ohlc: torch.Tensor) -> torch.Tensor:
    """
    Encode OHLC data as quaternions.

    Mapping: q_t = O_t + H_t * i + L_t * j + C_t * k

    The quaternion components are stored as:
    [real, i, j, k] = [Open, High, Low, Close]

    Args:
        ohlc: Tensor of shape (..., 4) with [Open, High, Low, Close].

    Returns:
        Tensor of same shape, interpreted as quaternion components [r, i, j, k].

    Note:
        While the output tensor has the same shape as input, this function
        serves as a semantic transformation marking the data as quaternion-encoded.
        The actual quaternion operations are performed in quaternion_ops.py.
    """
    if ohlc.shape[-1] != 4:
        raise ValueError(f"Expected 4 features (OHLC), got {ohlc.shape[-1]}")

    # OHLC naturally maps to quaternion components:
    # q = O + H*i + L*j + C*k
    # This is a semantic transformation - the data layout is already correct
    # [Open, High, Low, Close] -> [real, i, j, k]
    return ohlc.clone()


def preprocess_data(
    data: torch.Tensor,
    dates: pd.DatetimeIndex,
    train_end_year: int = 2018,
    val_end_year: int = 2021
) -> Dict:
    """
    Complete preprocessing pipeline: split, normalize, encode.

    CRITICAL: Normalization statistics are computed from training data ONLY
    to prevent look-ahead bias.

    Args:
        data: Raw OHLC tensor of shape (num_samples, 4).
        dates: DatetimeIndex corresponding to data rows.
        train_end_year: Last year of training data.
        val_end_year: Last year of validation data.

    Returns:
        Dictionary containing:
        - train_data: Normalized quaternion-encoded training data
        - val_data: Normalized quaternion-encoded validation data
        - test_data: Normalized quaternion-encoded test data
        - norm_stats: Normalization statistics (from train only)
        - split_info: Split boundary information
    """
    # Step 1: Temporal split (on RAW data, before normalization!)
    train_raw, val_raw, test_raw, split_info = temporal_split(
        data, dates, train_end_year, val_end_year
    )

    # Step 2: Compute normalization stats from TRAIN ONLY
    _, norm_stats = normalize_data(train_raw)

    # Compute training-set return std for 3-class directional threshold
    train_close = train_raw[:, 3]
    train_returns = (train_close[1:] - train_close[:-1]) / (train_close[:-1].abs() + 1e-8)
    norm_stats['return_std'] = train_returns.std().item()

    # Step 3: Apply normalization using train stats to all splits
    train_norm, _ = normalize_data(train_raw, stats=norm_stats)
    val_norm, _ = normalize_data(val_raw, stats=norm_stats)
    test_norm, _ = normalize_data(test_raw, stats=norm_stats)

    # Step 4: Quaternion encoding (semantic transformation)
    train_quat = encode_quaternion(train_norm)
    val_quat = encode_quaternion(val_norm)
    test_quat = encode_quaternion(test_norm)

    return {
        'train_data': train_quat,
        'val_data': val_quat,
        'test_data': test_quat,
        'norm_stats': norm_stats,
        'split_info': split_info
    }


def preprocess_data_ratio(
    data: torch.Tensor,
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20
) -> Dict:
    """
    Complete preprocessing pipeline with ratio-based splitting: split, normalize, encode.

    CRITICAL: Normalization statistics are computed from training data ONLY
    to prevent look-ahead bias.

    Args:
        data: Raw OHLC tensor of shape (num_samples, 4).
        dates: DatetimeIndex corresponding to data rows.
        train_ratio: Fraction of data for training (default: 0.70).
        val_ratio: Fraction of data for validation (default: 0.10).
        test_ratio: Fraction of data for testing (default: 0.20).

    Returns:
        Dictionary containing:
        - train_data: Normalized quaternion-encoded training data
        - val_data: Normalized quaternion-encoded validation data
        - test_data: Normalized quaternion-encoded test data
        - norm_stats: Normalization statistics (from train only)
        - split_info: Split boundary information
    """
    # Step 1: Temporal split by ratio (on RAW data, before normalization!)
    train_raw, val_raw, test_raw, split_info = temporal_split_ratio(
        data, dates, train_ratio, val_ratio, test_ratio
    )

    # Step 2: Compute normalization stats from TRAIN ONLY
    _, norm_stats = normalize_data(train_raw)

    # Compute training-set return std for 3-class directional threshold
    train_close = train_raw[:, 3]
    train_returns = (train_close[1:] - train_close[:-1]) / (train_close[:-1].abs() + 1e-8)
    norm_stats['return_std'] = train_returns.std().item()

    # Step 3: Apply normalization using train stats to all splits
    train_norm, _ = normalize_data(train_raw, stats=norm_stats)
    val_norm, _ = normalize_data(val_raw, stats=norm_stats)
    test_norm, _ = normalize_data(test_raw, stats=norm_stats)

    # Step 4: Quaternion encoding (semantic transformation)
    train_quat = encode_quaternion(train_norm)
    val_quat = encode_quaternion(val_norm)
    test_quat = encode_quaternion(test_norm)

    return {
        'train_data': train_quat,
        'val_data': val_quat,
        'test_data': test_quat,
        'norm_stats': norm_stats,
        'split_info': split_info
    }
