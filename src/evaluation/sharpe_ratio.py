"""
Sharpe Ratio module.

Computes Sharpe Ratio for risk-adjusted returns in simulated trading scenarios.
"""

import torch


def compute_sharpe_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0
) -> float:
    """
    Compute Sharpe Ratio for a simple directional trading strategy.

    Strategy: Go long when predicted direction is up, short when down.
    Returns are computed as: sign(pred - prev) * (target - prev) / prev

    Args:
        pred: Predictions tensor of shape (n,).
        target: Target tensor of shape (n,).
        prev: Previous values tensor of shape (n,).
        risk_free_rate: Risk-free rate (default: 0.0).
        annualization_factor: Factor to annualize Sharpe Ratio.
            For daily returns: sqrt(252) ≈ 15.87
            For hourly returns: sqrt(252 * 6.5) ≈ 40.45
            Default: 1.0 (no annualization)

    Returns:
        Sharpe Ratio as float. Returns 0.0 if std of returns is 0.
    """
    # Compute actual returns (percentage change)
    actual_returns = (target - prev) / (prev.abs() + 1e-8)

    # Compute predicted direction
    pred_direction = torch.sign(pred - prev)

    # Strategy returns: actual returns when direction is correct, negative otherwise
    strategy_returns = pred_direction * actual_returns

    # Compute mean and std of strategy returns
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std(unbiased=False)

    # Avoid division by zero or NaN (single sample case)
    if std_return < 1e-8 or torch.isnan(std_return):
        return 0.0

    # Compute Sharpe Ratio
    sharpe = (mean_return - risk_free_rate) / std_return

    # Annualize
    sharpe = sharpe * annualization_factor

    return sharpe.item()
