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


def compute_directional_accuracy_3class(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    flat_threshold: float = 0.0
) -> float:
    """
    Compute 3-class directional accuracy (UP / FLAT / DOWN).

    Returns are classified into three classes based on flat_threshold:
    - UP (+1): return > +flat_threshold
    - FLAT (0): |return| <= flat_threshold
    - DOWN (-1): return < -flat_threshold

    Args:
        pred: Predictions tensor of shape (n,).
        target: Target tensor of shape (n,).
        prev: Previous values tensor of shape (n,).
        flat_threshold: Threshold in return space for the FLAT zone.
            Typically computed as fraction * training_return_std.

    Returns:
        Directional accuracy as percentage (0-100).
    """
    # Compute returns
    pred_return = (pred - prev) / (prev.abs() + 1e-8)
    target_return = (target - prev) / (prev.abs() + 1e-8)

    # Classify into 3 classes: UP (+1), FLAT (0), DOWN (-1)
    def classify(returns, threshold):
        classes = torch.zeros_like(returns)
        classes[returns > threshold] = 1.0
        classes[returns < -threshold] = -1.0
        return classes

    pred_class = classify(pred_return, flat_threshold)
    target_class = classify(target_return, flat_threshold)

    correct = (pred_class == target_class).float()
    accuracy = correct.mean().item() * 100

    return accuracy
