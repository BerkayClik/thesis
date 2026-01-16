"""
Evaluation metrics module.

Provides MAE, MSE, and MAPE computation functions.
"""

import torch


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        pred: Predictions tensor.
        target: Target tensor.

    Returns:
        MAE value.
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.

    Args:
        pred: Predictions tensor.
        target: Target tensor.

    Returns:
        MSE value.
    """
    return torch.mean((pred - target) ** 2).item()


def compute_mape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        pred: Predictions tensor.
        target: Target tensor.
        epsilon: Small value to prevent division by zero.

    Returns:
        MAPE value as a percentage.
    """
    return (torch.abs((target - pred) / (torch.abs(target) + epsilon))).mean().item() * 100
