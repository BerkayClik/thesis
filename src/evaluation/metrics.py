"""
Evaluation metrics module.

Provides MAPE computation function.
"""

import torch


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
