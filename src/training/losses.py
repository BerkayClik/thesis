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


def directional_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_dir: float = 1.0,
    k: float = 1000.0,
) -> torch.Tensor:
    """
    MSE plus a sign-agreement penalty for return-mode training.

    loss = MSE(pred, target) + lambda_dir * mean(softplus(-k * pred * target))

    The product ``pred * target`` is positive when both share a sign (correct
    direction) and negative when they disagree. ``softplus(-k * x)`` is small for
    correct-direction pairs and grows for wrong-direction pairs, so the model is
    pushed off the persistence (predict ~0) optimum toward committing to a
    direction. ``k`` scales the small return products (O(1e-2)^2) into the
    softplus's sensitive range. With ``lambda_dir=0`` this is exactly ``mse_loss``.
    """
    p = pred.view(-1)
    t = target.view(-1)
    mse = F.mse_loss(p, t)
    if lambda_dir == 0.0:
        return mse
    directional = F.softplus(-k * p * t).mean()
    return mse + lambda_dir * directional
