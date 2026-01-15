"""
Phase 1 Data Pipeline Tests.

Tests data loading, preprocessing, and dataset functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from src.data import (
    SP500Dataset,
    normalize_data,
    temporal_split,
    encode_quaternion,
    preprocess_data,
    download_sp500_data,
    load_sp500_data,
    dataframe_to_tensor
)


class TestDataLoader:
    """Tests for data loading functionality."""

    def test_download_sp500_data_returns_dataframe(self):
        """Verify download returns proper DataFrame structure."""
        # Download a small date range to speed up test
        df = download_sp500_data(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_download_sp500_data_columns(self):
        """Verify OHLC columns are present."""
        df = download_sp500_data(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        expected_cols = ["Open", "High", "Low", "Close"]
        assert list(df.columns) == expected_cols

    def test_download_sp500_data_date_range(self):
        """Verify data covers requested date range."""
        df = download_sp500_data(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        # Check that data starts in January 2023
        assert df.index.min().year == 2023
        assert df.index.min().month == 1
        # Check that data ends in December 2023
        assert df.index.max().year == 2023
        assert df.index.max().month == 12

    def test_dataframe_to_tensor_shape(self):
        """Verify tensor shape matches DataFrame."""
        df = download_sp500_data(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        tensor, dates = dataframe_to_tensor(df)
        assert tensor.shape[0] == len(df)
        assert tensor.shape[1] == 4

    def test_dataframe_to_tensor_values(self):
        """Verify tensor values match DataFrame values."""
        df = download_sp500_data(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        tensor, dates = dataframe_to_tensor(df)
        # Check first row values
        np.testing.assert_allclose(
            tensor[0].numpy(),
            df.iloc[0].values,
            rtol=1e-5
        )

    def test_dataframe_to_tensor_returns_dates(self):
        """Verify dates are returned alongside tensor."""
        df = download_sp500_data(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        tensor, dates = dataframe_to_tensor(df)
        assert isinstance(dates, pd.DatetimeIndex)
        assert len(dates) == len(tensor)


class TestTemporalSplit:
    """Tests for temporal splitting functionality."""

    @pytest.fixture
    def sample_data_with_dates(self):
        """Create sample data spanning multiple years."""
        # Create dates from 2017 to 2023
        dates = pd.date_range("2017-01-01", "2023-12-31", freq="B")
        data = torch.randn(len(dates), 4)
        return data, dates

    def test_temporal_split_by_year_boundaries(self, sample_data_with_dates):
        """Verify splits occur at correct year boundaries."""
        data, dates = sample_data_with_dates
        train, val, test, info = temporal_split(
            data, dates, train_end_year=2018, val_end_year=2021
        )

        # Train should be <= 2018
        assert all(d.year <= 2018 for d in info['train_dates'])
        # Val should be 2019-2021
        assert all(2019 <= d.year <= 2021 for d in info['val_dates'])
        # Test should be > 2021
        assert all(d.year > 2021 for d in info['test_dates'])

    def test_temporal_split_no_overlap(self, sample_data_with_dates):
        """Verify no data leakage between splits."""
        data, dates = sample_data_with_dates
        train, val, test, info = temporal_split(
            data, dates, train_end_year=2018, val_end_year=2021
        )

        train_set = set(info['train_dates'])
        val_set = set(info['val_dates'])
        test_set = set(info['test_dates'])

        assert len(train_set & val_set) == 0
        assert len(val_set & test_set) == 0
        assert len(train_set & test_set) == 0

    def test_temporal_split_preserves_order(self, sample_data_with_dates):
        """Verify chronological order is maintained."""
        data, dates = sample_data_with_dates
        train, val, test, info = temporal_split(
            data, dates, train_end_year=2018, val_end_year=2021
        )

        # Check that dates are sorted within each split
        assert info['train_dates'].is_monotonic_increasing
        assert info['val_dates'].is_monotonic_increasing
        assert info['test_dates'].is_monotonic_increasing

        # Check that train max < val min < test min
        assert info['train_dates'].max() < info['val_dates'].min()
        assert info['val_dates'].max() < info['test_dates'].min()

    def test_temporal_split_coverage(self, sample_data_with_dates):
        """Verify all data is included in exactly one split."""
        data, dates = sample_data_with_dates
        train, val, test, info = temporal_split(
            data, dates, train_end_year=2018, val_end_year=2021
        )

        total = info['train_size'] + info['val_size'] + info['test_size']
        assert total == len(data)


class TestNormalization:
    """Tests for normalization functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known statistics."""
        return torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
        ])

    def test_normalize_data_zero_mean(self, sample_data):
        """Verify normalized data has approximately zero mean."""
        normalized, stats = normalize_data(sample_data)
        mean = normalized.mean(dim=0)
        np.testing.assert_allclose(mean.numpy(), np.zeros(4), atol=1e-6)

    def test_normalize_data_unit_std(self, sample_data):
        """Verify normalized data has approximately unit std."""
        normalized, stats = normalize_data(sample_data)
        std = normalized.std(dim=0)
        np.testing.assert_allclose(std.numpy(), np.ones(4), atol=1e-2)

    def test_normalize_with_stats_uses_provided_stats(self):
        """Verify external stats are used when provided."""
        data = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        stats = {
            'mean': torch.tensor([5.0, 10.0, 15.0, 20.0]),
            'std': torch.tensor([1.0, 2.0, 3.0, 4.0])
        }
        normalized, _ = normalize_data(data, stats=stats)

        expected = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        torch.testing.assert_close(normalized, expected)

    def test_no_look_ahead_bias(self):
        """Critical: Verify val/test normalization uses only train stats."""
        # Create data where train/val/test have very different distributions
        train_data = torch.tensor([
            [100.0, 200.0, 300.0, 400.0],
            [101.0, 201.0, 301.0, 401.0],
            [102.0, 202.0, 302.0, 402.0],
        ])
        val_data = torch.tensor([
            [1000.0, 2000.0, 3000.0, 4000.0],
            [1001.0, 2001.0, 3001.0, 4001.0],
        ])
        test_data = torch.tensor([
            [10000.0, 20000.0, 30000.0, 40000.0],
        ])

        # Normalize train and compute stats
        _, train_stats = normalize_data(train_data)

        # Apply train stats to val and test
        val_norm, _ = normalize_data(val_data, stats=train_stats)
        test_norm, _ = normalize_data(test_data, stats=train_stats)

        # Val and test should NOT have zero mean when using train stats
        # because they come from different distributions
        assert val_norm.mean().abs() > 1.0
        assert test_norm.mean().abs() > 1.0


class TestQuaternionEncoding:
    """Tests for quaternion encoding functionality."""

    def test_encode_quaternion_shape(self):
        """Verify output shape matches input shape."""
        ohlc = torch.randn(100, 4)
        encoded = encode_quaternion(ohlc)
        assert encoded.shape == ohlc.shape

    def test_encode_quaternion_mapping(self):
        """Verify OHLC maps correctly to quaternion components."""
        ohlc = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        encoded = encode_quaternion(ohlc)
        # [Open, High, Low, Close] -> [r, i, j, k]
        torch.testing.assert_close(encoded, ohlc)

    def test_encode_quaternion_batch_support(self):
        """Verify batch processing works correctly."""
        # Test with 3D tensor (batch, seq, features)
        ohlc = torch.randn(32, 20, 4)
        encoded = encode_quaternion(ohlc)
        assert encoded.shape == (32, 20, 4)

    def test_encode_quaternion_invalid_features(self):
        """Verify error for invalid number of features."""
        ohlc = torch.randn(100, 3)  # Wrong number of features
        with pytest.raises(ValueError, match="Expected 4 features"):
            encode_quaternion(ohlc)

    def test_encode_quaternion_returns_copy(self):
        """Verify encoding returns a copy, not a view."""
        ohlc = torch.randn(10, 4)
        encoded = encode_quaternion(ohlc)
        # Modify encoded
        encoded[0, 0] = 999.0
        # Original should be unchanged
        assert ohlc[0, 0] != 999.0


class TestDataset:
    """Tests for SP500Dataset functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        data = torch.randn(100, 4)
        window_size = 20
        return SP500Dataset(data, window_size), data, window_size

    def test_dataset_length(self, sample_dataset):
        """Verify dataset length accounts for window size."""
        dataset, data, window_size = sample_dataset
        assert len(dataset) == len(data) - window_size

    def test_dataset_getitem_shapes(self, sample_dataset):
        """Verify X and y shapes are correct."""
        dataset, data, window_size = sample_dataset
        x, y = dataset[0]
        assert x.shape == (window_size, 4)
        assert y.shape == ()  # Scalar

    def test_dataset_window_contents(self):
        """Verify window contains correct time steps."""
        # Create predictable data
        data = torch.arange(100).float().unsqueeze(1).expand(-1, 4)
        dataset = SP500Dataset(data, window_size=5)

        x, y = dataset[10]
        # Window should contain steps 10-14
        expected_x = torch.arange(10, 15).float().unsqueeze(1).expand(-1, 4)
        torch.testing.assert_close(x, expected_x)

    def test_dataset_target_is_next_close(self):
        """Verify target is the close price at t+window_size."""
        data = torch.arange(100).float().unsqueeze(1).expand(-1, 4)
        dataset = SP500Dataset(data, window_size=5, target_col=3)

        x, y = dataset[10]
        # Target should be the close price at step 15
        assert y == 15.0

    def test_dataset_no_look_ahead(self):
        """Verify target is not included in input window."""
        data = torch.arange(100).float().unsqueeze(1).expand(-1, 4)
        dataset = SP500Dataset(data, window_size=5)

        x, y = dataset[10]
        # Window should be [10, 11, 12, 13, 14], target should be 15
        # Check that target value is not in the window
        assert y not in x[:, 0]


class TestDataPipelineIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def full_data(self):
        """Load real S&P 500 data for integration tests."""
        df = download_sp500_data(
            start_date="2015-01-01",
            end_date="2023-12-31"
        )
        return dataframe_to_tensor(df)

    def test_full_pipeline_no_look_ahead_bias(self, full_data):
        """Verify entire pipeline maintains temporal integrity."""
        data, dates = full_data
        result = preprocess_data(
            data, dates, train_end_year=2018, val_end_year=2021
        )

        # Check year boundaries
        train_years = result['split_info']['train_dates'].year
        val_years = result['split_info']['val_dates'].year
        test_years = result['split_info']['test_dates'].year

        assert train_years.max() <= 2018
        assert val_years.min() >= 2019
        assert val_years.max() <= 2021
        assert test_years.min() >= 2022

    def test_pipeline_with_dataloader(self, full_data):
        """Verify compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        data, dates = full_data
        result = preprocess_data(data, dates)

        train_dataset = SP500Dataset(result['train_data'], window_size=20)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Verify we can iterate through batches
        for x, y in train_loader:
            assert x.shape == (32, 20, 4) or x.shape[0] < 32  # Last batch may be smaller
            assert len(y.shape) == 1
            break

    def test_pipeline_split_sizes(self, full_data):
        """Verify split sizes are reasonable."""
        data, dates = full_data
        result = preprocess_data(data, dates)

        # All splits should have data
        assert result['split_info']['train_size'] > 0
        assert result['split_info']['val_size'] > 0
        assert result['split_info']['test_size'] > 0

        # Total should equal input
        total = (
            result['split_info']['train_size'] +
            result['split_info']['val_size'] +
            result['split_info']['test_size']
        )
        assert total == len(data)

    def test_norm_stats_from_train_only(self, full_data):
        """Verify normalization stats come only from training data."""
        data, dates = full_data
        result = preprocess_data(data, dates)

        # Recompute train stats manually
        train_mask = dates.year <= 2018
        train_raw = data[train_mask]
        expected_mean = train_raw.mean(dim=0)
        expected_std = train_raw.std(dim=0)

        torch.testing.assert_close(
            result['norm_stats']['mean'],
            expected_mean,
            rtol=1e-5,
            atol=1e-5
        )
        torch.testing.assert_close(
            result['norm_stats']['std'],
            expected_std,
            rtol=1e-5,
            atol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
