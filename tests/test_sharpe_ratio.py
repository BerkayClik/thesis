"""
Tests for Sharpe Ratio computation.
"""

import pytest
import torch
from src.evaluation.sharpe_ratio import compute_sharpe_ratio, compute_sharpe_ratio_returns
from src.evaluation.directional_accuracy import compute_directional_accuracy_returns


class TestSharpeRatio:
    """Tests for Sharpe Ratio computation."""

    def test_perfect_prediction_positive_sharpe(self):
        """Perfect predictions should yield positive Sharpe Ratio."""
        # Perfect directional predictions in upward market
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        prev = torch.tensor([1.0, 2.0, 3.0])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert sharpe > 0

    def test_inverse_prediction_negative_sharpe(self):
        """Opposite predictions should yield negative Sharpe Ratio."""
        # Predict down when market goes up
        pred = torch.tensor([0.5, 1.5, 2.5])  # Predict down
        target = torch.tensor([2.0, 3.0, 4.0])  # Actually goes up
        prev = torch.tensor([1.0, 2.0, 3.0])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert sharpe < 0

    def test_random_prediction_bounded(self):
        """Random predictions should yield Sharpe within reasonable bounds."""
        torch.manual_seed(42)
        pred = torch.randn(1000) + 100
        target = torch.randn(1000) + 100
        prev = torch.ones(1000) * 100

        sharpe = compute_sharpe_ratio(pred, target, prev)
        # Should be within reasonable bounds (not wildly large)
        assert abs(sharpe) < 5.0

    def test_zero_std_returns_zero(self):
        """Zero std of returns should return 0 (no division by zero)."""
        # All same returns
        pred = torch.tensor([2.0, 2.0, 2.0])
        target = torch.tensor([2.0, 2.0, 2.0])
        prev = torch.tensor([1.0, 1.0, 1.0])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert sharpe == 0.0

    def test_annualization_factor(self):
        """Annualization factor should scale the Sharpe Ratio."""
        pred = torch.tensor([2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([2.0, 3.0, 4.0, 5.0])
        prev = torch.tensor([1.0, 2.0, 3.0, 4.0])

        sharpe_daily = compute_sharpe_ratio(pred, target, prev, annualization_factor=1.0)
        sharpe_annual = compute_sharpe_ratio(pred, target, prev, annualization_factor=15.87)

        # Allow for small floating point differences
        assert pytest.approx(sharpe_annual / sharpe_daily, rel=0.01) == 15.87

    def test_risk_free_rate(self):
        """Risk-free rate should reduce Sharpe Ratio."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        prev = torch.tensor([1.0, 2.0, 3.0])

        sharpe_no_rf = compute_sharpe_ratio(pred, target, prev, risk_free_rate=0.0)
        sharpe_with_rf = compute_sharpe_ratio(pred, target, prev, risk_free_rate=0.001)

        assert sharpe_with_rf < sharpe_no_rf

    def test_handles_near_zero_prev(self):
        """Should handle near-zero previous values without error."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0])
        prev = torch.tensor([0.001, 0.001])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert not torch.isnan(torch.tensor(sharpe))
        assert not torch.isinf(torch.tensor(sharpe))

    def test_handles_negative_prev(self):
        """Should handle negative previous values (short positions) without error."""
        pred = torch.tensor([-2.0, -3.0])
        target = torch.tensor([-2.0, -3.0])
        prev = torch.tensor([-1.0, -2.0])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert not torch.isnan(torch.tensor(sharpe))
        assert not torch.isinf(torch.tensor(sharpe))

    def test_single_sample(self):
        """Should handle single sample (std will be 0)."""
        pred = torch.tensor([2.0])
        target = torch.tensor([2.0])
        prev = torch.tensor([1.0])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert sharpe == 0.0  # Single sample has no std

    def test_output_is_float(self):
        """Output should be a Python float."""
        pred = torch.tensor([2.0, 3.0])
        target = torch.tensor([2.0, 3.0])
        prev = torch.tensor([1.0, 2.0])

        sharpe = compute_sharpe_ratio(pred, target, prev)
        assert isinstance(sharpe, float)


class TestSharpeRatioReturns:
    """Tests for return-based Sharpe Ratio computation."""

    def test_perfect_prediction_positive_sharpe(self):
        """Perfect predictions should yield positive Sharpe Ratio."""
        # Predicted returns match actual returns
        pred_return = torch.tensor([0.01, 0.02, -0.01, 0.015])
        target_return = torch.tensor([0.01, 0.02, -0.01, 0.015])

        sharpe = compute_sharpe_ratio_returns(pred_return, target_return)
        assert sharpe > 0

    def test_opposite_prediction_negative_sharpe(self):
        """Opposite predictions should yield negative Sharpe Ratio."""
        # Predicted direction is always wrong
        pred_return = torch.tensor([0.01, 0.02, -0.01, 0.015])
        target_return = torch.tensor([-0.01, -0.02, 0.01, -0.015])

        sharpe = compute_sharpe_ratio_returns(pred_return, target_return)
        assert sharpe < 0

    def test_zero_std_returns_zero(self):
        """Zero std of returns should return 0."""
        # All same returns
        pred_return = torch.tensor([0.01, 0.01, 0.01])
        target_return = torch.tensor([0.01, 0.01, 0.01])

        sharpe = compute_sharpe_ratio_returns(pred_return, target_return)
        assert sharpe == 0.0

    def test_output_is_float(self):
        """Output should be a Python float."""
        pred_return = torch.tensor([0.01, -0.02])
        target_return = torch.tensor([0.01, -0.02])

        sharpe = compute_sharpe_ratio_returns(pred_return, target_return)
        assert isinstance(sharpe, float)

    def test_annualization_factor(self):
        """Annualization factor should scale the Sharpe Ratio."""
        pred_return = torch.tensor([0.01, 0.02, -0.005, 0.015])
        target_return = torch.tensor([0.01, 0.02, -0.005, 0.015])

        sharpe_daily = compute_sharpe_ratio_returns(pred_return, target_return, annualization_factor=1.0)
        sharpe_annual = compute_sharpe_ratio_returns(pred_return, target_return, annualization_factor=15.87)

        assert pytest.approx(sharpe_annual / sharpe_daily, rel=0.01) == 15.87


class TestDirectionalAccuracyReturns:
    """Tests for return-based directional accuracy computation."""

    def test_perfect_prediction_100_percent(self):
        """Perfect predictions should yield 100% accuracy."""
        pred_return = torch.tensor([0.01, 0.02, -0.01, -0.02])
        target_return = torch.tensor([0.01, 0.02, -0.01, -0.02])

        accuracy = compute_directional_accuracy_returns(pred_return, target_return)
        assert accuracy == 100.0

    def test_opposite_prediction_0_percent(self):
        """Always-wrong predictions should yield 0% accuracy."""
        pred_return = torch.tensor([0.01, 0.02, -0.01, -0.02])
        target_return = torch.tensor([-0.01, -0.02, 0.01, 0.02])

        accuracy = compute_directional_accuracy_returns(pred_return, target_return)
        assert accuracy == 0.0

    def test_half_correct_50_percent(self):
        """Half correct predictions should yield 50% accuracy."""
        pred_return = torch.tensor([0.01, 0.02, 0.01, 0.02])  # All positive
        target_return = torch.tensor([0.01, 0.02, -0.01, -0.02])  # Half positive

        accuracy = compute_directional_accuracy_returns(pred_return, target_return)
        assert accuracy == 50.0

    def test_returns_percentage(self):
        """Output should be in percentage (0-100)."""
        pred_return = torch.tensor([0.01, 0.02])
        target_return = torch.tensor([0.01, 0.02])

        accuracy = compute_directional_accuracy_returns(pred_return, target_return)
        assert 0.0 <= accuracy <= 100.0

    def test_output_is_float(self):
        """Output should be a Python float."""
        pred_return = torch.tensor([0.01, -0.02])
        target_return = torch.tensor([0.01, -0.02])

        accuracy = compute_directional_accuracy_returns(pred_return, target_return)
        assert isinstance(accuracy, float)
