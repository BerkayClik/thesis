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

    # Compute predicted direction (default to long if no change predicted)
    pred_direction = torch.sign(pred - prev)
    pred_direction = torch.where(pred_direction == 0, torch.ones_like(pred_direction), pred_direction)

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


def compute_sharpe_ratio_3class(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    flat_threshold: float = 0.0,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0
) -> float:
    """
    Compute Sharpe Ratio with 3-class positioning (long / no position / short).

    Strategy: Go long when predicted return > +threshold,
              short when < -threshold, no position when FLAT.
    Sharpe is computed only over active (non-FLAT) periods.

    Args:
        pred: Predictions tensor of shape (n,).
        target: Target tensor of shape (n,).
        prev: Previous values tensor of shape (n,).
        flat_threshold: Threshold in return space for the FLAT zone.
        risk_free_rate: Risk-free rate (default: 0.0).
        annualization_factor: Factor to annualize Sharpe Ratio (default: 1.0).

    Returns:
        Sharpe Ratio as float. Returns 0.0 if no positions taken or std is 0.
    """
    # Compute actual returns (percentage change)
    actual_returns = (target - prev) / (prev.abs() + 1e-8)

    # Compute predicted returns for classification
    pred_return = (pred - prev) / (prev.abs() + 1e-8)

    # Classify predicted direction: +1 (long), 0 (flat/no position), -1 (short)
    position = torch.zeros_like(pred_return)
    position[pred_return > flat_threshold] = 1.0
    position[pred_return < -flat_threshold] = -1.0

    # Only keep active (non-FLAT) periods
    active_mask = position != 0
    if active_mask.sum() == 0:
        return 0.0

    active_returns = position[active_mask] * actual_returns[active_mask]

    # Compute mean and std of strategy returns
    mean_return = active_returns.mean()
    std_return = active_returns.std(unbiased=False)

    # Avoid division by zero or NaN
    if std_return < 1e-8 or torch.isnan(std_return):
        return 0.0

    # Compute Sharpe Ratio
    sharpe = (mean_return - risk_free_rate) / std_return

    # Annualize
    sharpe = sharpe * annualization_factor

    return sharpe.item()
