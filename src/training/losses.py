"""
Loss functions module.

Provides loss functions for training.
"""

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss.

    Args:
        pred: Predictions tensor.
        target: Target tensor.

    Returns:
        MSE loss value.
    """
    # Flatten both tensors to ensure matching shapes
    return F.mse_loss(pred.view(-1), target.view(-1))
