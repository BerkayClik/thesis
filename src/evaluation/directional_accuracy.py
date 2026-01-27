"""
Directional accuracy module.

Computes directional correctness for time-series forecasting.
"""

import torch


def compute_directional_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor
) -> float:
    """
    Compute directional accuracy.

    Direction is correct when:
    sign(pred - prev) == sign(target - prev)

    Args:
        pred: Predictions tensor of shape (n,).
        target: Target tensor of shape (n,).
        prev: Previous values tensor of shape (n,).

    Returns:
        Directional accuracy as percentage (0-100).
    """
    pred_direction = torch.sign(pred - prev)
    target_direction = torch.sign(target - prev)

    # Handle zero change case: default to "up" direction (consistent with sharpe_ratio.py)
    # This ensures sign(0) is treated as +1, not 0
    pred_direction = torch.where(pred_direction == 0, torch.ones_like(pred_direction), pred_direction)
    target_direction = torch.where(target_direction == 0, torch.ones_like(target_direction), target_direction)

    correct = (pred_direction == target_direction).float()
    accuracy = correct.mean().item() * 100

    return accuracy
