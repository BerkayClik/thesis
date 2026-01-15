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


class QNNAttentionModel(nn.Module):
    """
    Quaternion LSTM with Temporal Attention for S&P 500 forecasting.

    The model encodes OHLC data as quaternions, processes through Quaternion LSTM
    to capture feature correlations, then projects to real space for temporal
    attention and regression.

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

        # Input size is 1 quaternion feature (OHLC encoded as single quaternion)
        self.input_size = 1

        # Quaternion LSTM backbone
        # Input: (batch, seq_len, 1, 4) -> Output: (batch, seq_len, hidden_size, 4)
        self.qlstm = QuaternionLSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Projection from quaternion to real space
        # Flatten quaternion output: hidden_size * 4 -> projection_size
        self.projection_size = hidden_size * 4
        self.projection = nn.Linear(self.projection_size, hidden_size)

        # Temporal attention on real-valued features
        self.attention = TemporalAttention(hidden_size)

        # Regression head: predict next close price
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
               Last dimension is [Open, High, Low, Close].

        Returns:
            Quaternion tensor of shape (batch, seq_len, 1, 4).
        """
        # x is already in quaternion format (OHLC -> 4 components)
        # Just add the "num_features" dimension
        return x.unsqueeze(2)  # (batch, seq_len, 1, 4)

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

        # Step 1: Quaternion encoding
        # (batch, seq_len, 4) -> (batch, seq_len, 1, 4)
        q_input = self.encode_quaternion(x)

        # Step 2: Quaternion LSTM
        # (batch, seq_len, 1, 4) -> (batch, seq_len, hidden_size, 4)
        qlstm_out, _ = self.qlstm(q_input)

        # Step 3: Flatten quaternion features
        # (batch, seq_len, hidden_size, 4) -> (batch, seq_len, hidden_size * 4)
        qlstm_flat = qlstm_out.view(batch_size, seq_len, -1)

        # Step 4: Project to real space
        # (batch, seq_len, hidden_size * 4) -> (batch, seq_len, hidden_size)
        projected = self.projection(qlstm_flat)

        # Step 5: Temporal attention
        # (batch, seq_len, hidden_size) -> (batch, hidden_size)
        if return_attention:
            context, attention_weights = self.attention(projected, return_weights=True)
        else:
            context = self.attention(projected)

        # Step 6: Regression head
        # (batch, hidden_size) -> (batch, 1)
        output = self.output_head(context)

        if return_attention:
            return output, attention_weights
        return output


class QuaternionLSTMNoAttention(nn.Module):
    """
    Quaternion LSTM without Temporal Attention (for ablation studies).

    Uses the last hidden state instead of attention-weighted context.

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
        """Encode OHLC data as quaternions."""
        return x.unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: OHLC input of shape (batch, seq_len, 4).

        Returns:
            Prediction of shape (batch, 1).
        """
        batch_size, seq_len, _ = x.size()

        # Quaternion encoding
        q_input = self.encode_quaternion(x)

        # Quaternion LSTM
        qlstm_out, _ = self.qlstm(q_input)

        # Use last hidden state (no attention)
        last_hidden = qlstm_out[:, -1]  # (batch, hidden_size, 4)
        last_hidden_flat = last_hidden.view(batch_size, -1)  # (batch, hidden_size * 4)

        # Project to real space
        projected = self.projection(last_hidden_flat)

        # Regression output
        output = self.output_head(projected)

        return output
