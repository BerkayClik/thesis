"""
Evaluation metrics module.

Provides MAE and MSE computation functions.
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
