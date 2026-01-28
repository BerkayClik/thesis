"""
Quaternion operations module.

Provides fundamental quaternion operations for neural networks.
Optimized for GPU performance using matmul-based implementations.
"""

import math
import torch
import torch.nn as nn


def hamilton_product(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternions using matrix formulation.

    For quaternions p = (a, b, c, d) and q = (e, f, g, h):
    p * q = (ae - bf - cg - dh,
             af + be + ch - dg,
             ag - bh + ce + df,
             ah + bg - cf + de)

    Uses batched matrix multiplication for better GPU performance.
    The Hamilton product can be expressed as a matrix-vector multiplication:

    [r]   [a  -b  -c  -d] [e]
    [i] = [b   a  -d   c] [f]
    [j]   [c   d   a  -b] [g]
    [k]   [d  -c   b   a] [h]

    Args:
        p: Quaternion tensor of shape (..., 4).
        q: Quaternion tensor of shape (..., 4).

    Returns:
        Hamilton product of shape (..., 4).
    """
    a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

    # Build rotation matrix from p: (..., 4, 4)
    row0 = torch.stack([a, -b, -c, -d], dim=-1)
    row1 = torch.stack([b, a, -d, c], dim=-1)
    row2 = torch.stack([c, d, a, -b], dim=-1)
    row3 = torch.stack([d, -c, b, a], dim=-1)
    mat = torch.stack([row0, row1, row2, row3], dim=-2)

    # Matrix-vector multiplication
    q_col = q.unsqueeze(-1)
    return torch.matmul(mat, q_col).squeeze(-1)


def hamilton_product_broadcast(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product using broadcast operations.

    This version is used for QuaternionLinear where we need to compute
    products between tensors of different shapes and sum the results.
    Uses slice indexing for torch.compile compatibility.

    Args:
        p: Quaternion tensor of shape (..., 4).
        q: Quaternion tensor of shape (..., 4).

    Returns:
        Hamilton product of shape (..., 4).
    """
    a = p[..., 0:1]
    b = p[..., 1:2]
    c = p[..., 2:3]
    d = p[..., 3:4]

    e = q[..., 0:1]
    f = q[..., 1:2]
    g = q[..., 2:3]
    h = q[..., 3:4]

    r = a * e - b * f - c * g - d * h
    i = a * f + b * e + c * h - d * g
    j = a * g - b * h + c * e + d * f
    k = a * h + b * g - c * f + d * e

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

    Optimized implementation using matrix multiplications (cuBLAS) instead of
    element-wise broadcast operations. The Hamilton product + sum is reformulated as:

    y_r = x_r @ W_r.T - x_i @ W_i.T - x_j @ W_j.T - x_k @ W_k.T
    y_i = x_r @ W_i.T + x_i @ W_r.T + x_j @ W_k.T - x_k @ W_j.T
    y_j = x_r @ W_j.T - x_i @ W_k.T + x_j @ W_r.T + x_k @ W_i.T
    y_k = x_r @ W_k.T + x_i @ W_j.T - x_j @ W_i.T + x_k @ W_r.T

    This uses 4 matrix multiplications with concatenated weights, which is
    significantly faster than the broadcast approach on GPU.

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
        Initialize quaternion weights using Glorot normal initialization.

        The Hamilton product means each output scalar sums over 4 * in_features
        terms (all 4 quaternion components of each input feature), and each input
        scalar contributes to 4 * out_features output scalars. Fan counts are
        scaled by 4 to account for this.
        """
        fan_in = 4 * self.in_features
        fan_out = 4 * self.out_features
        stdv = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.weight, 0, stdv)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using optimized matmul-based Hamilton product.

        Args:
            x: Input of shape (batch, in_features, 4).

        Returns:
            Output of shape (batch, out_features, 4).
        """
        # Extract quaternion components
        x_r, x_i, x_j, x_k = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        W_r, W_i, W_j, W_k = self.weight[..., 0], self.weight[..., 1], self.weight[..., 2], self.weight[..., 3]

        # Concatenate input components: (batch, 4*in_features)
        x_flat = torch.cat([x_r, x_i, x_j, x_k], dim=-1)

        # Build combined weight matrices for each output component
        # Each: (out_features, 4*in_features)
        W_for_r = torch.cat([W_r, -W_i, -W_j, -W_k], dim=1)
        W_for_i = torch.cat([W_i, W_r, W_k, -W_j], dim=1)
        W_for_j = torch.cat([W_j, -W_k, W_r, W_i], dim=1)
        W_for_k = torch.cat([W_k, W_j, -W_i, W_r], dim=1)

        # 4 matmuls instead of element-wise broadcast + sum
        y_r = x_flat @ W_for_r.T
        y_i = x_flat @ W_for_i.T
        y_j = x_flat @ W_for_j.T
        y_k = x_flat @ W_for_k.T

        return torch.stack([y_r, y_i, y_j, y_k], dim=-1) + self.bias
