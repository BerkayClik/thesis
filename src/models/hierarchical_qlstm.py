"""
Hierarchical Quaternion LSTM with configurable fusion strategies.

Processes 16 features as 4 semantic quaternion groups (4 features each),
each encoded into its own QLSTM. Group-level representations are combined
via a pluggable fusion module before the regression head.

Fusion strategies:
  - ConcatFusion:          concatenate + linear projection
  - GroupAttentionFusion:   learned attention weights over groups
  - MetaQuaternionFusion:   treat groups as quaternion components, fuse via Hamilton product
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quaternion_lstm import QuaternionLSTM
from .quaternion_ops import QuaternionLinear
from .attention import TemporalAttention
from .revin import RevIN
from .dish_ts import DishTS


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """Concatenate group representations and project to a shared space."""

    def __init__(self, hidden_size: int, num_groups: int = 4):
        super().__init__()
        self.projection = nn.Linear(num_groups * hidden_size, hidden_size)

    def forward(self, group_vectors: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            group_vectors: list of (batch, hidden_size) tensors, one per group.
        Returns:
            (batch, hidden_size)
        """
        return self.projection(torch.cat(group_vectors, dim=-1))


class GroupAttentionFusion(nn.Module):
    """Learned attention weights over semantic groups."""

    def __init__(self, hidden_size: int, num_groups: int = 4):
        super().__init__()
        self.score_fn = nn.Linear(hidden_size, 1)

    def forward(
        self,
        group_vectors: list[torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            group_vectors: list of (batch, hidden_size) tensors.
            return_weights: if True, also return (batch, num_groups) weights.
        Returns:
            (batch, hidden_size)  or  ((batch, hidden_size), (batch, num_groups))
        """
        stacked = torch.stack(group_vectors, dim=1)       # (batch, G, H)
        scores = self.score_fn(stacked).squeeze(-1)        # (batch, G)
        weights = F.softmax(scores, dim=-1)                # (batch, G)
        context = torch.bmm(
            weights.unsqueeze(1), stacked
        ).squeeze(1)                                       # (batch, H)
        if return_weights:
            return context, weights
        return context


class MetaQuaternionFusion(nn.Module):
    """
    Fuse 4 group representations via the Hamilton product.

    Each group's hidden vector becomes one quaternion component
    (Price=r, Market=i, Social=j, Sentiment=k).  A QuaternionLinear
    layer mixes inter-group relationships, and the result is projected
    back to real space.
    """

    def __init__(self, hidden_size: int, num_groups: int = 4):
        super().__init__()
        self.qlinear = QuaternionLinear(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, group_vectors: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            group_vectors: list of 4 tensors, each (batch, hidden_size).
        Returns:
            (batch, hidden_size)
        """
        # (batch, hidden_size, 4) — a quaternion per hidden feature
        q = torch.stack(group_vectors, dim=-1)
        fused = self.qlinear(q)                            # (batch, hidden_size, 4)
        flat = fused.reshape(fused.size(0), -1)            # (batch, hidden_size*4)
        return self.projection(flat)


# ---------------------------------------------------------------------------
# Hierarchical QLSTM model
# ---------------------------------------------------------------------------

_FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "concat": ConcatFusion,
    "group_attention": GroupAttentionFusion,
    "meta_quaternion": MetaQuaternionFusion,
}


class HierarchicalQLSTM(nn.Module):
    """
    Hierarchical Quaternion LSTM.

    Splits *num_groups* × *features_per_group* input features into semantic
    quaternion groups, processes each through an independent QLSTM, fuses
    group-level representations via *fusion_type*, and (optionally) applies
    per-group temporal attention before fusion.

    Args:
        hidden_size:  quaternion hidden dimension per group QLSTM.
        num_layers:   stacked QLSTM layers per group.
        dropout:      dropout between QLSTM layers.
        num_features: total input features (must equal num_groups × features_per_group).
        target_col:   index of the prediction target in the *selected* feature vector.
        norm_type:    ``'revin'`` or ``'dish_ts'``.
        seq_len:      lookback window length.
        dish_init:    Dish-TS init strategy (ignored when norm_type='revin').
        fusion_type:  ``'concat'``, ``'group_attention'``, or ``'meta_quaternion'``.
        use_temporal_attention: if True, apply per-group temporal attention
            before fusion; otherwise use last hidden state.
        num_groups:        number of semantic quaternion groups.
        features_per_group: features per group (must be 4 for quaternion encoding).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_features: int = 16,
        target_col: int = 3,
        norm_type: str = "revin",
        seq_len: int = 72,
        dish_init: str = "standard",
        fusion_type: str = "concat",
        use_temporal_attention: bool = False,
        num_groups: int = 4,
        features_per_group: int = 4,
        target_mode: str = "price",
    ):
        super().__init__()
        if num_features != num_groups * features_per_group:
            raise ValueError(
                f"num_features ({num_features}) must equal "
                f"num_groups ({num_groups}) × features_per_group ({features_per_group})"
            )
        if features_per_group != 4:
            raise ValueError("features_per_group must be 4 for quaternion encoding")

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.features_per_group = features_per_group
        self.target_col = target_col
        self.use_temporal_attention = use_temporal_attention
        self.fusion_type = fusion_type
        self.target_mode = target_mode

        # --- normalisation ---
        if norm_type == "dish_ts":
            self.norm_layer = DishTS(
                num_features=num_features, seq_len=seq_len, dish_init=dish_init,
            )
        else:
            self.norm_layer = RevIN(num_features)

        # --- per-group QLSTM + projection ---
        self.qlstms = nn.ModuleList([
            QuaternionLSTM(
                input_size=1, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout,
            )
            for _ in range(num_groups)
        ])
        projection_in = hidden_size * 4
        self.projections = nn.ModuleList([
            nn.Linear(projection_in, hidden_size)
            for _ in range(num_groups)
        ])

        # --- optional per-group temporal attention ---
        if use_temporal_attention:
            self.temporal_attentions = nn.ModuleList([
                TemporalAttention(hidden_size) for _ in range(num_groups)
            ])

        # --- fusion ---
        if fusion_type not in _FUSION_REGISTRY:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. "
                f"Choose from {list(_FUSION_REGISTRY)}"
            )
        self.fusion = _FUSION_REGISTRY[fusion_type](hidden_size, num_groups)

        # --- regression head ---
        self.output_head = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_groups(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Split (batch, seq, num_features) into per-group quaternion inputs.

        Returns list of (batch, seq, 1, 4) tensors ready for QLSTM.
        """
        batch, seq, _ = x.shape
        grouped = x.view(batch, seq, self.num_groups, self.features_per_group)
        return [grouped[:, :, g, :].unsqueeze(2) for g in range(self.num_groups)]

    def _process_groups(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run each group through its QLSTM and return one summary vector
        per group, shape (batch, hidden_size).
        """
        groups = self._split_groups(x)
        summaries: list[torch.Tensor] = []

        for g in range(self.num_groups):
            qlstm_out, _ = self.qlstms[g](groups[g])       # (B, S, H, 4)
            batch, seq, hidden, _ = qlstm_out.shape
            flat = qlstm_out.view(batch, seq, hidden * 4)   # (B, S, H*4)
            projected = self.projections[g](flat)            # (B, S, H)

            if self.use_temporal_attention:
                summary = self.temporal_attentions[g](projected)  # (B, H)
            else:
                summary = projected[:, -1]                        # (B, H)

            summaries.append(summary)

        return summaries

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_group_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, num_features) raw-scale input.
            return_group_weights: only used when fusion_type='group_attention'.

        Returns:
            (batch, 1) predicted price in original scale.
            Optionally (batch, num_groups) group-attention weights.
        """
        x = self.norm_layer(x, "norm")

        group_summaries = self._process_groups(x)

        if return_group_weights and self.fusion_type == "group_attention":
            fused, weights = self.fusion(group_summaries, return_weights=True)
        else:
            fused = self.fusion(group_summaries)
            weights = None

        output = self.output_head(fused)
        # Oracle O3: price-mode only; return-mode output stays in return scale.
        if self.target_mode == "price":
            output = self.norm_layer.denorm_scalar(output, self.target_col)

        if weights is not None:
            return output, weights
        return output
