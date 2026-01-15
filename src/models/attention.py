"""
Temporal Attention mechanism.

Provides attention over time steps for sequence models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for sequence data.

    Computes attention weights over time steps and returns weighted sum.

    Args:
        hidden_size: Size of input features.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).
            return_weights: If True, also return attention weights.

        Returns:
            Context vector of shape (batch, hidden_size).
            Optionally attention weights of shape (batch, seq_len).
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Apply softmax to get weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Compute weighted sum
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_size)

        if return_weights:
            return context, weights
        return context
