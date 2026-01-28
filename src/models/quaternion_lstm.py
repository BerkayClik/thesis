"""
Quaternion LSTM module.

Implements LSTM using quaternion operations for gate computations.
Quaternion structure is used in the weight matrices (QuaternionLinear)
for parameter-efficient weight sharing via Hamilton product, while
gating operations use standard element-wise multiplication to preserve
LSTM's linear memory highway semantics.
"""

import torch
import torch.nn as nn
from .quaternion_ops import QuaternionLinear


class QuaternionLSTMCell(nn.Module):
    """
    Fused Quaternion LSTM cell with optimized gate computation.

    Uses fused gate computation to reduce kernel launches: instead of 8 separate
    QuaternionLinear calls per timestep (4 for input, 4 for hidden), uses only 2
    (one for input, one for hidden) with gates concatenated along the feature dim.

    The cell maintains hidden state h and cell state c in quaternion space (4D).
    Quaternion structure is in the weight matrices (QuaternionLinear uses Hamilton
    product for weight sharing). Gating uses standard element-wise multiplication
    to preserve the LSTM cell state as a linear memory highway.

    Features for training stability:
    - Forget gate bias initialized to +1.0 (per Jozefowicz et al. 2015)

    Args:
        input_size: Number of input quaternion features.
        hidden_size: Number of hidden quaternion features.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Fused gate projections: all 4 gates computed together
        # Output shape: (batch, 4 * hidden_size, 4) for [i, f, g, o] gates
        self.W_input = QuaternionLinear(input_size, 4 * hidden_size)
        self.W_hidden = QuaternionLinear(hidden_size, 4 * hidden_size)

        # Initialize forget gate bias to +1.0 for better gradient flow
        # (per Jozefowicz et al. 2015 "An Empirical Exploration of RNN Architectures")
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self):
        """
        Initialize forget gate bias to +1.0 on all quaternion components.

        Per Jozefowicz et al. 2015, initializing forget gate bias to 1.0
        helps with learning long-term dependencies by keeping the forget
        gate initially open.

        Gate layout in fused weights: [i, f, g, o] each of size hidden_size
        Forget gate is at index hidden_size:2*hidden_size

        With element-wise gating, each component is independently gated by
        sigmoid, so all 4 components need +1.0 to start with forget gate open.
        """
        with torch.no_grad():
            # Add +1.0 to ALL components of forget gate biases
            # Forget gate is the 2nd chunk: index [hidden_size:2*hidden_size]
            self.W_input.bias.data[self.hidden_size:2*self.hidden_size] += 1.0
            self.W_hidden.bias.data[self.hidden_size:2*self.hidden_size] += 1.0

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

        # Fused gate computation: 2 calls instead of 8
        gates = self.W_input(x) + self.W_hidden(h)  # (batch, 4*hidden_size, 4)

        # Split gates - chunk() is essentially free (view only)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        # Element-wise gating (standard LSTM semantics)
        # Quaternion structure is preserved in W_input/W_hidden (QuaternionLinear)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

    def forward_with_precomputed(
        self,
        gates_x: torch.Tensor,
        hx: tuple
    ) -> tuple:
        """
        Forward pass with pre-computed input projections.

        This method is used when input projections have been pre-computed
        for all timesteps, allowing significant speedup by avoiding
        redundant computation in the time loop.

        Args:
            gates_x: Pre-computed fused input projection (batch, 4*hidden_size, 4).
            hx: Tuple of (h, c) hidden states, each of shape (batch, hidden_size, 4).

        Returns:
            Tuple of (h_new, c_new) each of shape (batch, hidden_size, 4).
        """
        h, c = hx

        # Add hidden projection to pre-computed input projection
        gates = gates_x + self.W_hidden(h)  # (batch, 4*hidden_size, 4)

        # Split gates - chunk() is essentially free (view only)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        # Element-wise gating (standard LSTM semantics)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

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

        Uses pre-computed fused input projections for all layers to minimize
        kernel launches. Each layer pre-computes all gate projections in a
        single QuaternionLinear call before the time loop.

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
        batch_size, seq_len, input_size, _ = x.size()
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

        # PRE-COMPUTE: Fused input projection for first layer (all timesteps at once)
        # Flatten (batch, seq_len, input_size, 4) -> (batch*seq_len, input_size, 4)
        x_flat = x.view(batch_size * seq_len, input_size, 4)
        cell0 = self.cells[0]

        # Single fused projection for all gates: (batch*seq_len, 4*hidden_size, 4)
        gates_x_all = cell0.W_input(x_flat).view(batch_size, seq_len, 4 * self.hidden_size, 4)

        # Process each time step
        for t in range(seq_len):
            # First layer uses pre-computed fused input projection
            h[0], c[0] = cell0.forward_with_precomputed(
                gates_x_all[:, t],
                (h[0], c[0])
            )
            layer_input = h[0]

            # Process through subsequent layers (if any)
            for layer_idx in range(1, self.num_layers):
                # Apply dropout between layers
                if self.dropout > 0:
                    layer_input = self.dropout_layer(layer_input)
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](
                    layer_input, (h[layer_idx], c[layer_idx])
                )
                layer_input = h[layer_idx]

            # Output is hidden state from last layer
            output[:, t] = h[-1]

        # Stack final hidden states: (num_layers, batch, hidden_size, 4)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)
