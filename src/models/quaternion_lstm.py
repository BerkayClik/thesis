"""
Quaternion LSTM module.

Implements LSTM using quaternion operations for gate computations.
Uses Hamilton product instead of matrix multiplication for processing
quaternion-encoded features.
"""

import torch
import torch.nn as nn
from .quaternion_ops import hamilton_product, QuaternionLinear


class QuaternionLSTMCell(nn.Module):
    """
    Quaternion LSTM cell using Hamilton product for gate computations.

    The cell maintains hidden state h and cell state c in quaternion space (4D).
    Gate computations use Hamilton product instead of matrix multiplication.

    Features for training stability:
    - Forget gate bias initialized to +1.0 (per Jozefowicz et al. 2015)
    - LayerNorm applied to cell and hidden states to prevent gradient explosion

    Args:
        input_size: Number of input quaternion features.
        hidden_size: Number of hidden quaternion features.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gate projections: input + hidden -> gate_size
        # Each gate needs quaternion linear transforms for input and hidden
        # Gates: input (i), forget (f), cell (g), output (o)

        # Input gate
        self.W_ii = QuaternionLinear(input_size, hidden_size)
        self.W_hi = QuaternionLinear(hidden_size, hidden_size)

        # Forget gate
        self.W_if = QuaternionLinear(input_size, hidden_size)
        self.W_hf = QuaternionLinear(hidden_size, hidden_size)

        # Cell gate (candidate)
        self.W_ig = QuaternionLinear(input_size, hidden_size)
        self.W_hg = QuaternionLinear(hidden_size, hidden_size)

        # Output gate
        self.W_io = QuaternionLinear(input_size, hidden_size)
        self.W_ho = QuaternionLinear(hidden_size, hidden_size)

        # LayerNorm for stabilizing cell and hidden states
        # Applied over the quaternion dimension (4 components)
        self.cell_norm = nn.LayerNorm(4)
        self.hidden_norm = nn.LayerNorm(4)

        # Initialize forget gate bias to +1.0 for better gradient flow
        # (per Jozefowicz et al. 2015 "An Empirical Exploration of RNN Architectures")
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self):
        """
        Initialize forget gate bias to +1.0 for better gradient flow.

        Per Jozefowicz et al. 2015, initializing forget gate bias to 1.0
        helps with learning long-term dependencies by keeping the forget
        gate initially open.
        """
        with torch.no_grad():
            # Add +1.0 to the real component of forget gate biases
            # This biases the sigmoid toward 1 (keeping the cell state)
            self.W_if.bias.data[..., 0] += 1.0
            self.W_hf.bias.data[..., 0] += 1.0

    def _quaternion_sigmoid(self, q: torch.Tensor) -> torch.Tensor:
        """
        Apply sigmoid to quaternion components.

        For gating, we apply sigmoid element-wise to all 4 components.
        This keeps values in [0, 1] for multiplicative gating.

        Args:
            q: Quaternion tensor of shape (..., 4).

        Returns:
            Tensor with sigmoid applied to each component.
        """
        return torch.sigmoid(q)

    def _quaternion_tanh(self, q: torch.Tensor) -> torch.Tensor:
        """
        Apply tanh to quaternion components.

        For cell state updates, we apply tanh element-wise.

        Args:
            q: Quaternion tensor of shape (..., 4).

        Returns:
            Tensor with tanh applied to each component.
        """
        return torch.tanh(q)

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple = None
    ) -> tuple:
        """
        Forward pass for single time step.

        Args:
            x: Input of shape (batch, input_size, 4).
            hx: Tuple of (h, c) hidden states, each of shape (batch, hidden_size, 4).
                If None, initializes to zeros.

        Returns:
            Tuple of (h_new, c_new) each of shape (batch, hidden_size, 4).
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # Initialize hidden states if not provided
        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, 4, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_size, 4, device=device, dtype=dtype)
        else:
            h, c = hx

        # Input gate: i = sigmoid(W_ii @ x + W_hi @ h)
        i = self._quaternion_sigmoid(self.W_ii(x) + self.W_hi(h))

        # Forget gate: f = sigmoid(W_if @ x + W_hf @ h)
        f = self._quaternion_sigmoid(self.W_if(x) + self.W_hf(h))

        # Cell candidate: g = tanh(W_ig @ x + W_hg @ h)
        g = self._quaternion_tanh(self.W_ig(x) + self.W_hg(h))

        # Output gate: o = sigmoid(W_io @ x + W_ho @ h)
        o = self._quaternion_sigmoid(self.W_io(x) + self.W_ho(h))

        # New cell state: c_new = f * c + i * g
        # Using Hamilton product for quaternion multiplication
        c_new = hamilton_product(f, c) + hamilton_product(i, g)
        # Apply LayerNorm to stabilize cell state magnitudes
        c_new = self.cell_norm(c_new)

        # New hidden state: h_new = o * tanh(c_new)
        h_new = hamilton_product(o, self._quaternion_tanh(c_new))
        # Apply LayerNorm to stabilize hidden state magnitudes
        h_new = self.hidden_norm(h_new)

        return h_new, c_new


class QuaternionLSTM(nn.Module):
    """
    Stacked Quaternion LSTM layers.

    Processes sequential quaternion data through multiple LSTM layers.

    Args:
        input_size: Number of input quaternion features.
        hidden_size: Number of hidden quaternion features.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between layers (applied if num_layers > 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        # Create stacked LSTM cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(QuaternionLSTMCell(cell_input_size, hidden_size))

        # Dropout layer (applied between layers, not after last layer)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple = None
    ) -> tuple:
        """
        Forward pass through all time steps and layers.

        Args:
            x: Input of shape (batch, seq_len, input_size, 4).
            hx: Initial hidden states. Tuple of (h_0, c_0) where each has
                shape (num_layers, batch, hidden_size, 4). If None, uses zeros.

        Returns:
            Tuple of (output, (h_n, c_n)):
                - output: shape (batch, seq_len, hidden_size, 4)
                - h_n: shape (num_layers, batch, hidden_size, 4)
                - c_n: shape (num_layers, batch, hidden_size, 4)
        """
        batch_size, seq_len, _, _ = x.size()
        device = x.device
        dtype = x.dtype

        # Initialize hidden states if not provided
        if hx is None:
            h = [torch.zeros(batch_size, self.hidden_size, 4, device=device, dtype=dtype)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, 4, device=device, dtype=dtype)
                 for _ in range(self.num_layers)]
        else:
            h_0, c_0 = hx
            h = [h_0[layer] for layer in range(self.num_layers)]
            c = [c_0[layer] for layer in range(self.num_layers)]

        # Pre-allocate output tensor to avoid list append/stack overhead
        output = torch.zeros(batch_size, seq_len, self.hidden_size, 4, device=device, dtype=dtype)

        # Process each time step
        for t in range(seq_len):
            # Input for first layer is x[:, t]
            layer_input = x[:, t]

            # Process through all layers
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(layer_input, (h[layer_idx], c[layer_idx]))
                layer_input = h[layer_idx]

                # Apply dropout between layers (not after last layer)
                if self.dropout > 0 and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout_layer(layer_input)

            # Output is hidden state from last layer
            output[:, t] = h[-1]

        # Stack final hidden states: (num_layers, batch, hidden_size, 4)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)
