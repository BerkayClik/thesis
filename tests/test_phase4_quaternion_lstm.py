"""
Phase 4 Tests: Quaternion LSTM

Tests QuaternionLSTMCell and QuaternionLSTM for:
- Output shape correctness
- Numerical stability
- Gradient flow
- Overfitting capability
"""

import pytest
import torch
import torch.nn as nn

from src.models.quaternion_lstm import QuaternionLSTMCell, QuaternionLSTM


class TestQuaternionLSTMCell:
    """Tests for QuaternionLSTMCell."""

    def test_output_shape(self):
        """Cell outputs correct shape."""
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4)  # (batch, input_size, 4)

        h, c = cell(x)

        assert h.shape == (16, 8, 4)
        assert c.shape == (16, 8, 4)

    def test_output_shape_with_hidden_state(self):
        """Cell handles provided hidden state."""
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4)
        h_prev = torch.randn(16, 8, 4)
        c_prev = torch.randn(16, 8, 4)

        h, c = cell(x, (h_prev, c_prev))

        assert h.shape == (16, 8, 4)
        assert c.shape == (16, 8, 4)

    def test_batch_independence(self):
        """Each batch element processed independently."""
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4)

        # Process all at once
        h_batch, c_batch = cell(x)

        # Process individually
        for i in range(16):
            h_single, c_single = cell(x[i : i + 1])
            assert torch.allclose(h_batch[i], h_single[0], atol=1e-5)
            assert torch.allclose(c_batch[i], c_single[0], atol=1e-5)

    def test_zero_hidden_state_initialization(self):
        """Cell initializes hidden state to zeros when not provided."""
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4)

        # Call without hidden state
        h1, c1 = cell(x)

        # Call with explicit zeros
        h_zeros = torch.zeros(16, 8, 4)
        c_zeros = torch.zeros(16, 8, 4)
        h2, c2 = cell(x, (h_zeros, c_zeros))

        assert torch.allclose(h1, h2, atol=1e-6)
        assert torch.allclose(c1, c2, atol=1e-6)

    def test_no_nan_in_output(self):
        """No NaN values in cell output."""
        torch.manual_seed(42)
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4)

        h, c = cell(x)

        assert not torch.isnan(h).any()
        assert not torch.isnan(c).any()

    def test_no_inf_in_output(self):
        """No Inf values in cell output."""
        torch.manual_seed(42)
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4)

        h, c = cell(x)

        assert not torch.isinf(h).any()
        assert not torch.isinf(c).any()

    def test_gradient_flow(self):
        """Gradients flow through cell."""
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4, requires_grad=True)

        h, c = cell(x)
        loss = h.sum() + c.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_in_gradients(self):
        """No NaN values in gradients."""
        torch.manual_seed(42)
        cell = QuaternionLSTMCell(input_size=4, hidden_size=8)
        x = torch.randn(16, 4, 4, requires_grad=True)

        h, c = cell(x)
        loss = h.pow(2).sum() + c.pow(2).sum()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        for param in cell.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestQuaternionLSTM:
    """Tests for QuaternionLSTM."""

    def test_output_shape_single_layer(self):
        """Single layer LSTM outputs correct shape."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4, 4)  # (batch, seq_len, input_size, 4)

        output, (h_n, c_n) = lstm(x)

        assert output.shape == (16, 20, 8, 4)
        assert h_n.shape == (1, 16, 8, 4)
        assert c_n.shape == (1, 16, 8, 4)

    def test_output_shape_multi_layer(self):
        """Multi-layer LSTM outputs correct shape."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=3)
        x = torch.randn(16, 20, 4, 4)

        output, (h_n, c_n) = lstm(x)

        assert output.shape == (16, 20, 8, 4)
        assert h_n.shape == (3, 16, 8, 4)
        assert c_n.shape == (3, 16, 8, 4)

    def test_output_shape_with_initial_hidden(self):
        """LSTM handles provided initial hidden state."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4)
        h_0 = torch.randn(2, 16, 8, 4)
        c_0 = torch.randn(2, 16, 8, 4)

        output, (h_n, c_n) = lstm(x, (h_0, c_0))

        assert output.shape == (16, 20, 8, 4)
        assert h_n.shape == (2, 16, 8, 4)
        assert c_n.shape == (2, 16, 8, 4)

    def test_sequence_length_one(self):
        """LSTM handles sequence length of 1."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=1)
        x = torch.randn(16, 1, 4, 4)

        output, (h_n, c_n) = lstm(x)

        assert output.shape == (16, 1, 8, 4)

    def test_batch_size_one(self):
        """LSTM handles batch size of 1."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=1)
        x = torch.randn(1, 20, 4, 4)

        output, (h_n, c_n) = lstm(x)

        assert output.shape == (1, 20, 8, 4)

    def test_last_output_equals_final_hidden(self):
        """Last output equals final hidden state of last layer."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4)

        output, (h_n, c_n) = lstm(x)

        # output[:, -1] should equal h_n[-1] (last layer's final hidden)
        assert torch.allclose(output[:, -1], h_n[-1], atol=1e-6)

    def test_no_nan_in_output(self):
        """No NaN values in LSTM output."""
        torch.manual_seed(42)
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4)

        output, (h_n, c_n) = lstm(x)

        assert not torch.isnan(output).any()
        assert not torch.isnan(h_n).any()
        assert not torch.isnan(c_n).any()

    def test_no_inf_in_output(self):
        """No Inf values in LSTM output."""
        torch.manual_seed(42)
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4)

        output, (h_n, c_n) = lstm(x)

        assert not torch.isinf(output).any()
        assert not torch.isinf(h_n).any()
        assert not torch.isinf(c_n).any()

    def test_gradient_flow(self):
        """Gradients flow through LSTM."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4, requires_grad=True)

        output, (h_n, c_n) = lstm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_in_gradients(self):
        """No NaN values in gradients."""
        torch.manual_seed(42)
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4, requires_grad=True)

        output, _ = lstm(x)
        loss = output.pow(2).sum()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        for param in lstm.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_dropout_training_mode(self):
        """Dropout is applied in training mode."""
        torch.manual_seed(42)
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=3, dropout=0.5)
        lstm.train()
        x = torch.randn(16, 20, 4, 4)

        # Run twice, should get different results due to dropout
        output1, _ = lstm(x)
        output2, _ = lstm(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)

    def test_dropout_eval_mode(self):
        """Dropout is not applied in eval mode."""
        torch.manual_seed(42)
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=3, dropout=0.5)
        lstm.eval()
        x = torch.randn(16, 20, 4, 4)

        # Run twice, should get same results
        output1, _ = lstm(x)
        output2, _ = lstm(x)

        assert torch.allclose(output1, output2)


class TestQuaternionLSTMOverfit:
    """Tests for overfitting capability."""

    def test_can_overfit_small_batch(self):
        """Quaternion LSTM can overfit a small batch."""
        torch.manual_seed(42)

        # Create small batch
        batch_size = 32
        seq_len = 10
        input_size = 1  # Single quaternion feature (OHLC encoded as 1 quaternion)
        hidden_size = 16

        x = torch.randn(batch_size, seq_len, input_size, 4)
        # Target: predict scalar from quaternion output
        target = torch.randn(batch_size, 1)

        # Model: QLSTM + projection to scalar
        lstm = QuaternionLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        # Project quaternion output (hidden_size * 4) to scalar
        fc = nn.Linear(hidden_size * 4, 1)

        optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=0.01)
        criterion = nn.MSELoss()

        # Initial loss
        lstm.train()
        output, _ = lstm(x)
        last_hidden = output[:, -1]  # (batch, hidden_size, 4)
        last_hidden_flat = last_hidden.view(batch_size, -1)  # (batch, hidden_size * 4)
        pred = fc(last_hidden_flat)
        initial_loss = criterion(pred, target).item()

        # Train for 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            output, _ = lstm(x)
            last_hidden = output[:, -1]
            last_hidden_flat = last_hidden.view(batch_size, -1)
            pred = fc(last_hidden_flat)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease significantly (at least 50%)
        assert final_loss < initial_loss * 0.5, f"Loss didn't decrease enough: {initial_loss} -> {final_loss}"

    def test_can_reduce_loss_on_pattern(self):
        """Quaternion LSTM can reduce loss on a pattern (learning capability)."""
        torch.manual_seed(42)

        batch_size = 32
        seq_len = 10
        input_size = 1
        hidden_size = 16

        # Simple target: random but fixed
        x = torch.randn(batch_size, seq_len, input_size, 4)
        target = torch.randn(batch_size, 1)

        lstm = QuaternionLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        fc = nn.Linear(hidden_size * 4, 1)

        optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=0.01)
        criterion = nn.MSELoss()

        # Get initial loss
        lstm.train()
        output, _ = lstm(x)
        last_hidden = output[:, -1]
        last_hidden_flat = last_hidden.view(batch_size, -1)
        pred = fc(last_hidden_flat)
        initial_loss = criterion(pred, target).item()

        # Train
        for _ in range(100):
            optimizer.zero_grad()
            output, _ = lstm(x)
            last_hidden = output[:, -1]
            last_hidden_flat = last_hidden.view(batch_size, -1)
            pred = fc(last_hidden_flat)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Should be able to reduce loss (learning is happening)
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_stability_with_small_inputs(self):
        """LSTM handles small input values."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4) * 1e-6

        output, (h_n, c_n) = lstm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_stability_with_large_inputs(self):
        """LSTM handles large input values."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4) * 10

        output, (h_n, c_n) = lstm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_stability_over_long_sequence(self):
        """LSTM stable over long sequences."""
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=1)
        x = torch.randn(4, 100, 4, 4)  # 100 time steps

        output, (h_n, c_n) = lstm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_stability(self):
        """Backward pass stable."""
        torch.manual_seed(42)
        lstm = QuaternionLSTM(input_size=4, hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, 4, requires_grad=True)

        output, _ = lstm(x)
        loss = output.pow(2).mean()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
