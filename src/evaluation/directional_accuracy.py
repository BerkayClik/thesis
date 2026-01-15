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

    correct = (pred_direction == target_direction).float()
    accuracy = correct.mean().item() * 100

    return accuracy


def compute_directional_accuracy_returns(
    pred_return: torch.Tensor,
    target_return: torch.Tensor
) -> float:
    """
    Compute directional accuracy when predictions are returns.

    Direction is correct when:
    sign(pred_return) == sign(target_return)

    Args:
        pred_return: Predicted returns tensor of shape (n,).
        target_return: Actual returns tensor of shape (n,).

    Returns:
        Directional accuracy as percentage (0-100).
    """
    pred_direction = torch.sign(pred_return)
    target_direction = torch.sign(target_return)

    correct = (pred_direction == target_direction).float()
    accuracy = correct.mean().item() * 100

    return accuracy
