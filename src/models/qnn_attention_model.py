"""
Full Quaternion Neural Network with Temporal Attention model.

Architecture:
Input (OHLC) -> Quaternion Encoding -> Quaternion LSTM ->
Projection -> Temporal Attention -> Regression Head -> Output

Design Principle:
- Feature correlation -> Quaternion space (Hamilton product preserves OHLC relationships)
- Temporal importance -> Real-valued space (attention over time steps)
"""

import torch
import torch.nn as nn
from .quaternion_lstm import QuaternionLSTM
from .attention import TemporalAttention


class QuaternionLSTMBase(nn.Module):
    """
    Base class for Quaternion LSTM models.

    Contains shared components: QLSTM backbone, projection layer, output head.
    Subclasses implement specific forward() logic.

    Args:
        hidden_size: Size of quaternion hidden features.
        num_layers: Number of Quaternion LSTM layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = 1

        # Quaternion LSTM backbone
        self.qlstm = QuaternionLSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Projection from quaternion to real space
        self.projection_size = hidden_size * 4
        self.projection = nn.Linear(self.projection_size, hidden_size)

        # Regression head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def encode_quaternion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode OHLC data as quaternions.

        q_t = O_t + H_t * i + L_t * j + C_t * k

        Args:
            x: OHLC input of shape (batch, seq_len, 4).

        Returns:
            Quaternion tensor of shape (batch, seq_len, 1, 4).
        """
        return x.unsqueeze(2)


class QNNAttentionModel(QuaternionLSTMBase):
    """
    Quaternion LSTM with Temporal Attention for S&P 500 forecasting.

    Args:
        hidden_size: Size of quaternion hidden features.
        num_layers: Number of Quaternion LSTM layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__(hidden_size, num_layers, dropout)
        self.attention = TemporalAttention(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass.

        Args:
            x: OHLC input of shape (batch, seq_len, 4).
            return_attention: If True, also return attention weights.

        Returns:
            Prediction of shape (batch, 1).
            Optionally attention weights of shape (batch, seq_len).
        """
        batch_size, seq_len, _ = x.size()

        # Quaternion encoding and LSTM
        q_input = self.encode_quaternion(x)
        qlstm_out, _ = self.qlstm(q_input)

        # Flatten and project to real space
        qlstm_flat = qlstm_out.view(batch_size, seq_len, -1)
        projected = self.projection(qlstm_flat)

        # Temporal attention
        if return_attention:
            context, attention_weights = self.attention(projected, return_weights=True)
        else:
            context = self.attention(projected)

        # Regression head
        output = self.output_head(context)

        if return_attention:
            return output, attention_weights
        return output


class QuaternionLSTMNoAttention(QuaternionLSTMBase):
    """
    Quaternion LSTM without Temporal Attention (for ablation studies).

    Uses the last hidden state instead of attention-weighted context.

    Args:
        hidden_size: Size of quaternion hidden features.
        num_layers: Number of Quaternion LSTM layers.
        dropout: Dropout rate.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: OHLC input of shape (batch, seq_len, 4).

        Returns:
            Prediction of shape (batch, 1).
        """
        batch_size = x.size(0)

        # Quaternion encoding and LSTM
        q_input = self.encode_quaternion(x)
        qlstm_out, _ = self.qlstm(q_input)

        # Use last hidden state (no attention)
        last_hidden = qlstm_out[:, -1]
        last_hidden_flat = last_hidden.view(batch_size, -1)

        # Project and output
        projected = self.projection(last_hidden_flat)
        output = self.output_head(projected)

        return output
