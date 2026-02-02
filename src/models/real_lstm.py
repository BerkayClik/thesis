"""
Real-valued LSTM baseline model.

Standard LSTM encoder with regression output head.
"""

import torch
import torch.nn as nn


class RealLSTM(nn.Module):
    """
    Real-valued LSTM for time-series regression.

    Args:
        input_size: Number of input features.
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_head = nn.Linear(hidden_size, 1)
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self):
        """Initialize forget gate bias to +1.0 (Jozefowicz et al. 2015)."""
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    n = self.hidden_size
                    param.data[n:2*n].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, 1).
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Use last time step output
        last_out = lstm_out[:, -1, :]

        # Regression output
        output = self.output_head(last_out)
        return output
