"""
Hierarchical quaternion neural networks for multiquaternion feature fusion.

Architecture:
Input (16 features) -> 4 semantic quaternions -> Quaternion LSTM (4 -> 4)
-> Quaternion LSTM (4 -> 1) -> Projection -> Temporal Attention -> Output
"""

import torch
import torch.nn as nn

from .attention import TemporalAttention
from .dish_ts import DishTS
from .quaternion_lstm import QuaternionLSTM
from .revin import RevIN


class HierarchicalQuaternionLSTMBase(nn.Module):
    """Base class for hierarchical quaternion forecasting models."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_size: int = 16,
        num_features: int = 16,
        target_col: int = 3,
        norm_type: str = "revin",
        seq_len: int = 20,
        dish_init: str = "standard",
    ):
        super().__init__()
        if input_size != 16:
            raise ValueError(
                f"Hierarchical quaternion models require 16 input features, got {input_size}."
            )
        if num_features != 16:
            raise ValueError(
                f"Hierarchical quaternion models require 16 normalization features, got {num_features}."
            )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.target_col = target_col
        self.num_features = num_features
        self.num_quaternions = 4
        self.quaternion_dim = 4

        if norm_type == "dish_ts":
            self.norm_layer = DishTS(
                num_features=num_features,
                seq_len=seq_len,
                dish_init=dish_init,
            )
        else:
            self.norm_layer = RevIN(num_features)

        self.level1_qlstm = QuaternionLSTM(
            input_size=self.num_quaternions,
            hidden_size=self.num_quaternions,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.level2_qlstm = QuaternionLSTM(
            input_size=self.num_quaternions,
            hidden_size=1,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.projection = nn.Linear(self.quaternion_dim, hidden_size)
        self.output_head = nn.Linear(hidden_size, 1)

    def encode_hierarchical_quaternions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape 16 ordered features into 4 semantic quaternions.

        Expected order:
        [open, high, low, close,
         circulating_supply, market_cap, market_dominance, volume_24h,
         contributors_active, contributors_created, posts_active, posts_created,
         interactions, sentiment, galaxy_score, social_dominance]
        """
        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.num_features:
            raise ValueError(
                f"Expected input shape (batch, seq_len, {self.num_features}), got {tuple(x.shape)}."
            )
        return x.view(batch_size, seq_len, self.num_quaternions, self.quaternion_dim)

    def forward_hierarchy(self, x: torch.Tensor) -> torch.Tensor:
        """Run the 16 -> 4 -> 1 quaternion hierarchy."""
        q_input = self.encode_hierarchical_quaternions(x)
        level1_out, _ = self.level1_qlstm(q_input)
        level2_out, _ = self.level2_qlstm(level1_out)
        return level2_out.squeeze(2)


class HierarchicalQNNAttentionModel(HierarchicalQuaternionLSTMBase):
    """Hierarchical quaternion LSTM with final temporal attention."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_size: int = 16,
        num_features: int = 16,
        target_col: int = 3,
        norm_type: str = "revin",
        seq_len: int = 20,
        dish_init: str = "standard",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            input_size=input_size,
            num_features=num_features,
            target_col=target_col,
            norm_type=norm_type,
            seq_len=seq_len,
            dish_init=dish_init,
        )
        self.attention = TemporalAttention(hidden_size)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x = self.norm_layer(x, "norm")
        hierarchy_out = self.forward_hierarchy(x)
        projected = self.projection(hierarchy_out)

        if return_attention:
            context, attention_weights = self.attention(projected, return_weights=True)
        else:
            context = self.attention(projected)

        output = self.output_head(context)
        output = self.norm_layer.denorm_scalar(output, self.target_col)

        if return_attention:
            return output, attention_weights
        return output


class HierarchicalQuaternionLSTMNoAttention(HierarchicalQuaternionLSTMBase):
    """Hierarchical quaternion LSTM without temporal attention."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_layer(x, "norm")
        hierarchy_out = self.forward_hierarchy(x)
        last_hidden = hierarchy_out[:, -1]
        projected = self.projection(last_hidden)
        output = self.output_head(projected)
        output = self.norm_layer.denorm_scalar(output, self.target_col)
        return output
