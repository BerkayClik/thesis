"""
Quaternion operations module.

Provides fundamental quaternion operations for neural networks.
"""

import math
import torch
import torch.nn as nn


def hamilton_product(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compile-friendly Hamilton product of two quaternions.

    For quaternions p = (a, b, c, d) and q = (e, f, g, h):
    p * q = (ae - bf - cg - dh,
             af + be + ch - dg,
             ag - bh + ce + df,
             ah + bg - cf + de)

    Uses slice indexing (0:1) instead of direct indexing to prevent graph breaks
    during torch.compile. Uses cat instead of stack for better kernel fusion.

    Args:
        p: Quaternion tensor of shape (..., 4).
        q: Quaternion tensor of shape (..., 4).

    Returns:
        Hamilton product of shape (..., 4).
    """
    # Slice-based component extraction (compile-friendly, avoids graph breaks)
    a = p[..., 0:1]
    b = p[..., 1:2]
    c = p[..., 2:3]
    d = p[..., 3:4]

    e = q[..., 0:1]
    f = q[..., 1:2]
    g = q[..., 2:3]
    h = q[..., 3:4]

    # Hamilton product formulas
    r = a * e - b * f - c * g - d * h
    i = a * f + b * e + c * h - d * g
    j = a * g - b * h + c * e + d * f
    k = a * h + b * g - c * f + d * e

    # cat instead of stack for better compile fusion
    return torch.cat([r, i, j, k], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate.

    For q = (a, b, c, d), conjugate is (a, -b, -c, -d).

    Args:
        q: Quaternion tensor of shape (..., 4).

    Returns:
        Conjugate quaternion of shape (..., 4).
    """
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion norm.

    ||q|| = sqrt(a^2 + b^2 + c^2 + d^2)

    Args:
        q: Quaternion tensor of shape (..., 4).

    Returns:
        Norm tensor of shape (...).
    """
    return torch.norm(q, dim=-1)


class QuaternionLinear(nn.Module):
    """
    Quaternion linear layer using Hamilton product.

    Args:
        in_features: Number of input quaternion features.
        out_features: Number of output quaternion features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight quaternion for each input-output pair
        # Shape: (out_features, in_features, 4)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 4))
        self.bias = nn.Parameter(torch.zeros(out_features, 4))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using Xavier-like initialization for quaternions.

        Uses fan_in + fan_out scaling similar to Xavier initialization, but adjusted
        for quaternion dimension (4 components). The factor of 4 accounts for the
        Hamilton product summing over all 4 quaternion components.

        Bias is initialized using the same uniform distribution as weights (matching
        PyTorch's nn.Linear behavior) rather than zeros, which helps with symmetry
        breaking and gradient flow.
        """
        # Xavier uniform: stdv = sqrt(6 / (fan_in + fan_out))
        # For quaternions, we scale by the quaternion dimension
        # Use symmetric fan calculation for consistent initialization across layers
        fan_in = self.in_features * 4  # Each quaternion has 4 components
        fan_out = self.out_features * 4
        stdv = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.weight, -stdv, stdv)
        # Initialize bias with same distribution as weights (matches nn.Linear)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Hamilton product.

        Args:
            x: Input of shape (batch, in_features, 4).

        Returns:
            Output of shape (batch, out_features, 4).
        """
        assert x.ndim == 3, f"Expected 3D input (batch, features, 4), got {x.shape}"
        assert x.shape[1] == self.in_features, f"Expected {self.in_features} input features, got {x.shape[1]}"
        assert x.shape[2] == 4, f"Expected quaternion dim 4, got {x.shape[2]}"

        # x: (batch, in_features, 4), weight: (out_features, in_features, 4)
        # Use broadcasting to compute all hamilton products at once
        x_expanded = x.unsqueeze(1)  # (batch, 1, in_features, 4)
        w_expanded = self.weight.unsqueeze(0)  # (1, out_features, in_features, 4)
        products = hamilton_product(x_expanded, w_expanded)  # (batch, out_features, in_features, 4)
        output = products.sum(dim=2) + self.bias  # (batch, out_features, 4)
        return output
