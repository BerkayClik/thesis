"""Real-valued LSTM baselines with internal RevIN / Dish-TS normalization.

Ablation models that separate the architecture effect from the normalization
effect. The standard real LSTM baselines receive statically Z-scored inputs
(training-set mean/std), which puts test-period prices far outside the train
distribution; quaternion models instead use per-window RevIN. Any performance
gap between the two therefore confounds architecture with normalization.

These wrappers give the real LSTM the EXACT same normalization protocol as the
quaternion models (see qnn_attention_model.py): normalize each input window
with RevIN (or Dish-TS), run the backbone on the normalized window, and in
price-mode denormalize the scalar output back to the window's price scale via
``denorm_scalar``. In return-mode the output is a return, so denormalization
is bypassed (Oracle O3), identical to the quaternion models.
"""

import torch
import torch.nn as nn

from .real_lstm import RealLSTM
from .real_lstm_attention import RealLSTMAttention
from .revin import RevIN
from .dish_ts import DishTS


class _RealRevINBase(nn.Module):
    """Shared normalization wrapper around a real-valued backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        num_features: int,
        target_col: int = 3,
        norm_type: str = "revin",
        seq_len: int = 20,
        dish_init: str = "standard",
        target_mode: str = "price",
    ):
        super().__init__()
        self.target_col = target_col
        self.target_mode = target_mode

        # Normalization layer (swappable), mirroring the quaternion models
        if norm_type == "dish_ts":
            self.norm_layer = DishTS(
                num_features=num_features,
                seq_len=seq_len,
                dish_init=dish_init,
            )
        else:
            self.norm_layer = RevIN(num_features)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RAW input windows of shape (batch, seq_len, features).

        Returns:
            Predictions of shape (batch, 1) — original price scale in
            price-mode, raw return scale in return/log_return mode.
        """
        x = self.norm_layer(x, "norm")
        output = self.backbone(x)
        if self.target_mode == "price":
            output = self.norm_layer.denorm_scalar(output, self.target_col)
        return output


class RealLSTMRevIN(_RealRevINBase):
    """RealLSTM with per-window RevIN (or Dish-TS) normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        target_col: int = 3,
        norm_type: str = "revin",
        seq_len: int = 20,
        dish_init: str = "standard",
        target_mode: str = "price",
    ):
        backbone = RealLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        super().__init__(
            backbone,
            num_features=input_size,
            target_col=target_col,
            norm_type=norm_type,
            seq_len=seq_len,
            dish_init=dish_init,
            target_mode=target_mode,
        )


class RealLSTMAttentionRevIN(_RealRevINBase):
    """RealLSTMAttention with per-window RevIN (or Dish-TS) normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        target_col: int = 3,
        norm_type: str = "revin",
        seq_len: int = 20,
        dish_init: str = "standard",
        target_mode: str = "price",
    ):
        backbone = RealLSTMAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        super().__init__(
            backbone,
            num_features=input_size,
            target_col=target_col,
            norm_type=norm_type,
            seq_len=seq_len,
            dish_init=dish_init,
            target_mode=target_mode,
        )
