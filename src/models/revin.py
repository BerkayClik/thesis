"""
Reversible Instance Normalization (RevIN) module.

Provides per-instance normalization for time-series data, handling
distribution shift between training and inference. The model normalizes
input sequences at the start and reverses the transformation on outputs.

Inspired by: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.

This is an independent reimplementation following the method described in
the paper. No code was copied from external repositories.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time-series.

    Normalizes each input instance (sample) independently using its own
    mean and standard deviation, then provides a reverse operation to
    restore the original scale on model outputs.

    Optionally learns affine parameters (scale and shift) applied after
    normalization, similar to standard batch/layer normalization.

    Args:
        num_features: Number of input features (channels). For OHLC data
            this is 4 (Open, High, Low, Close).
        eps: Small constant for numerical stability in division.
        affine: If True, learns per-feature scale (gamma) and shift (beta)
            parameters applied after normalization.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            # Learnable per-feature scale and shift
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

        # Instance statistics, set during normalization and reused for denorm.
        # These are not persistent state -- they are recomputed each forward pass.
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply normalization or denormalization.

        Args:
            x: Input tensor. For 'norm' mode, shape is (batch, seq_len, features).
                For 'denorm' mode, shape is (batch, features) or (batch, seq_len, features).
            mode: Either 'norm' (normalize input) or 'denorm' (reverse normalization).

        Returns:
            Transformed tensor with the same shape as input.
        """
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'norm' or 'denorm'.")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-instance statistics and normalize.

        Input shape: (batch, seq_len, features)
        Statistics are computed over the seq_len dimension (dim=1), giving
        per-sample, per-feature mean and std.
        """
        # Compute mean and std over the temporal dimension
        # keepdim=True so shapes broadcast: (batch, 1, features)
        self._mean = x.mean(dim=1, keepdim=True).detach()
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        self._std = torch.sqrt(variance + self.eps).detach()

        # Standardize
        out = (x - self._mean) / self._std

        # Apply learned affine transformation
        if self.affine:
            out = out * self.gamma + self.beta

        return out

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization to restore original scale.

        Handles both (batch, features) and (batch, seq_len, features) inputs.
        Uses the statistics saved during the most recent _normalize call.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Cannot denormalize before normalizing. "
                "Call forward(x, 'norm') first."
            )

        # Undo learned affine
        if self.affine:
            x = (x - self.beta) / self.gamma

        mean = self._mean
        std = self._std

        # Restore original scale
        out = x * std + mean

        return out

    def denorm_scalar(self, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """
        Denormalize a scalar prediction corresponding to a single feature.

        This is used when the model outputs a single value (e.g., predicted
        Close price) rather than the full feature vector.

        Args:
            x: Scalar predictions of shape (batch, 1).
            feature_idx: Index of the feature to denormalize (e.g., 3 for Close).

        Returns:
            Denormalized predictions of shape (batch, 1).
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Cannot denormalize before normalizing. "
                "Call forward(x, 'norm') first."
            )

        # Extract statistics for the target feature
        # _mean shape: (batch, 1, features) -> (batch, 1)
        feat_mean = self._mean[:, 0, feature_idx].unsqueeze(1)
        feat_std = self._std[:, 0, feature_idx].unsqueeze(1)

        # Undo learned affine for this feature
        if self.affine:
            feat_gamma = self.gamma[feature_idx]
            feat_beta = self.beta[feature_idx]
            x = (x - feat_beta) / feat_gamma

        # Restore original scale for this feature
        return x * feat_std + feat_mean
