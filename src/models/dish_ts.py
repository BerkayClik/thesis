"""
Dish-TS (Distribution Shift in Time Series) normalization module.

Provides learned normalization for time-series data that handles
distribution shift by using *different* statistics for the forward
(normalization) and inverse (denormalization) directions.  Unlike
RevIN, which uses the same per-instance mean/std for both directions,
Dish-TS learns a parametric mapping from the input to two sets of
location/scale parameters, allowing the model to capture asymmetric
shifts between the lookback window and the prediction horizon.

Reference: Fan et al., "Dish-TS: A General Paradigm for Alleviating
Distribution Shift in Time Series Forecasting", AAAI 2023.

This is an independent reimplementation adapted for single-step
scalar-output models (LSTM-based).  No code was copied; the
implementation follows the method described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    """
    Dish-TS normalization layer.

    Learns a per-feature, per-timestep weight matrix (``reduce_mlayer``)
    that maps an input window to two location estimates (phi_l, phi_h).
    Variance-like scales (xi_l, xi_h) are derived from these locations.
    Normalization uses (phi_l, xi_l); denormalization uses (phi_h, xi_h).

    Args:
        num_features: Number of input features (channels).  For OHLC
            data this is 4.
        seq_len: Length of the input sequence (lookback window size).
            This determines the shape of ``reduce_mlayer``.
        dish_init: Initialization strategy for ``reduce_mlayer``.
            One of ``'standard'`` (random / L), ``'avg'``
            (uniform 1 / L), or ``'uniform'`` (1 / L + random / L).
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        dish_init: str = "standard",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.eps = eps

        # Learnable mapping: (num_features, seq_len, 2)
        # Maps each feature's temporal profile to two scalars (phi_l, phi_h).
        if dish_init == "standard":
            self.reduce_mlayer = nn.Parameter(
                torch.rand(num_features, seq_len, 2) / seq_len
            )
        elif dish_init == "avg":
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len
            )
        elif dish_init == "uniform":
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len
                + torch.rand(num_features, seq_len, 2) / seq_len
            )
        else:
            raise ValueError(
                f"Unknown dish_init '{dish_init}'. "
                "Expected 'standard', 'avg', or 'uniform'."
            )

        # Learnable affine parameters (always enabled, matching RevIN default)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Per-instance statistics, computed during normalization and
        # reused during denormalization.  Not persistent state.
        self._phil: torch.Tensor | None = None
        self._phih: torch.Tensor | None = None
        self._xil: torch.Tensor | None = None
        self._xih: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public interface (matches RevIN)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply normalization or denormalization.

        Args:
            x: Input tensor.
                For ``'norm'``: shape ``(batch, seq_len, features)``.
                For ``'denorm'``: shape ``(batch, features)`` or
                ``(batch, seq_len, features)``.
            mode: ``'norm'`` to normalize, ``'denorm'`` to reverse.

        Returns:
            Transformed tensor with the same shape as *x*.
        """
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'norm' or 'denorm'."
            )

    def denorm_scalar(
        self, x: torch.Tensor, feature_idx: int
    ) -> torch.Tensor:
        """
        Denormalize a scalar prediction for a single feature.

        Used when the model outputs one value (e.g. predicted Close
        price) rather than the full feature vector.

        Args:
            x: Scalar predictions of shape ``(batch, 1)``.
            feature_idx: Index of the target feature (e.g. 0 for Close
                after OHLC selection).

        Returns:
            Denormalized predictions, shape ``(batch, 1)``.
        """
        if self._phih is None or self._xih is None:
            raise RuntimeError(
                "Cannot denormalize before normalizing. "
                "Call forward(x, 'norm') first."
            )

        # Extract per-feature statistics  (batch, 1, features) -> (batch, 1)
        feat_phih = self._phih[:, 0, feature_idx].unsqueeze(1)
        feat_xih = self._xih[:, 0, feature_idx].unsqueeze(1)
        feat_gamma = self.gamma[feature_idx]
        feat_beta = self.beta[feature_idx]

        # Undo affine, apply inverse scale/shift
        return (
            (x - feat_beta) / feat_gamma
        ) * torch.sqrt(feat_xih + self.eps) + feat_phih

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _preget(self, batch_x: torch.Tensor) -> None:
        """
        Compute the two sets of location/scale statistics from *batch_x*.

        ``batch_x`` has shape ``(B, L, D)``.

        The learned ``reduce_mlayer`` (shape ``(D, L, 2)``) is used as a
        batched matrix multiply over the feature dimension to produce
        ``theta`` of shape ``(B, 2, D)``, which is then split into the
        two location parameters ``phi_l`` and ``phi_h``.
        """
        # (B, L, D) -> (D, B, L)
        x_t = batch_x.permute(2, 0, 1)

        # Batched matmul: (D, B, L) @ (D, L, 2) -> (D, B, 2)
        theta = torch.bmm(x_t, self.reduce_mlayer)

        # (D, B, 2) -> (B, 2, D)
        theta = theta.permute(1, 2, 0)

        # Non-linearity on the learned statistics
        theta = F.gelu(theta)

        # Split into two location estimates, each (B, 1, D)
        self._phil = theta[:, :1, :]
        self._phih = theta[:, 1:, :]

        # Variance-like scale around each location
        L = batch_x.shape[1]
        self._xil = (
            torch.sum(torch.pow(batch_x - self._phil, 2), dim=1, keepdim=True)
            / (L - 1)
        )
        self._xih = (
            torch.sum(torch.pow(batch_x - self._phih, 2), dim=1, keepdim=True)
            / (L - 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute learned statistics and normalize.

        Uses ``phi_l`` (forward location) and ``xi_l`` (forward scale).
        """
        self._preget(x)

        out = (x - self._phil) / torch.sqrt(self._xil + self.eps)
        out = out * self.gamma + self.beta
        return out

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization using the *horizon* statistics.

        Uses ``phi_h`` (horizon location) and ``xi_h`` (horizon scale),
        which are intentionally different from the forward statistics.
        """
        if self._phih is None or self._xih is None:
            raise RuntimeError(
                "Cannot denormalize before normalizing. "
                "Call forward(x, 'norm') first."
            )

        out = ((x - self.beta) / self.gamma) * torch.sqrt(
            self._xih + self.eps
        ) + self._phih
        return out
