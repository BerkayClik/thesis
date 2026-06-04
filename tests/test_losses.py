"""Directional MSE loss for return-mode training.

directional_mse_loss = MSE(pred, target) + lambda_dir * mean(softplus(-k*pred*target))

The penalty term is small when pred and target share sign (correct direction)
and large when they disagree, pushing the model off the persistence (~0) optimum.
With lambda_dir=0 it reduces to plain MSE (backward compatible).
"""

import torch

from src.training.losses import mse_loss, directional_mse_loss


def test_reduces_to_mse_when_lambda_zero():
    pred = torch.tensor([0.01, -0.02, 0.03])
    target = torch.tensor([0.015, -0.01, 0.02])
    base = mse_loss(pred, target)
    got = directional_mse_loss(pred, target, lambda_dir=0.0)
    assert torch.allclose(got, base, atol=1e-7)


def test_correct_direction_cheaper_than_wrong_direction():
    target = torch.tensor([0.02, -0.03, 0.01, -0.04])
    # same magnitude error in both cases, but one set agrees in sign, one disagrees
    pred_right = torch.tensor([0.03, -0.02, 0.02, -0.03])   # all signs match target
    pred_wrong = torch.tensor([-0.03, 0.02, -0.02, 0.03])   # all signs oppose target
    loss_right = directional_mse_loss(pred_right, target, lambda_dir=1.0)
    loss_wrong = directional_mse_loss(pred_wrong, target, lambda_dir=1.0)
    assert loss_right < loss_wrong


def test_penalty_increases_with_lambda():
    pred = torch.tensor([-0.03, 0.02, -0.02, 0.03])   # wrong direction
    target = torch.tensor([0.02, -0.03, 0.01, -0.04])
    low = directional_mse_loss(pred, target, lambda_dir=0.5)
    high = directional_mse_loss(pred, target, lambda_dir=5.0)
    assert high > low


def test_returns_scalar_and_finite():
    pred = torch.randn(16, requires_grad=True)
    target = torch.randn(16)
    loss = directional_mse_loss(pred, target, lambda_dir=1.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_handles_2d_shapes_like_mse():
    pred = torch.tensor([[0.01], [-0.02], [0.03]])
    target = torch.tensor([0.015, -0.01, 0.02])
    loss = directional_mse_loss(pred, target, lambda_dir=1.0)
    assert torch.isfinite(loss)
