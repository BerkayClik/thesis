"""
Real-valued LSTM with Temporal Attention baseline model.

Combines LSTM encoder with temporal attention mechanism.
"""

import torch
import torch.nn as nn
from .attention import TemporalAttention


class RealLSTMAttention(nn.Module):
    """
    Real-valued LSTM with Temporal Attention for time-series regression.

    Architecture:
        1. LSTM encodes full sequence
        2. Temporal attention weights all time steps
        3. Linear head produces regression output

    Args:
        input_size: Number of input features (default: 4 for OHLC).
        hidden_size: Size of LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM encoder - processes full sequence
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Temporal attention over LSTM outputs
        self.attention = TemporalAttention(hidden_size)

        # Regression output head
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            return_attention_weights: If True, also return attention weights.

        Returns:
            Predictions of shape (batch, 1).
            Optionally attention weights of shape (batch, seq_len).
        """
        # LSTM encoding - get all time step outputs
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Apply temporal attention
        if return_attention_weights:
            context, attn_weights = self.attention(lstm_out, return_weights=True)
        else:
            context = self.attention(lstm_out)

        # Regression output
        output = self.output_head(context)

        if return_attention_weights:
            return output, attn_weights
        return output
