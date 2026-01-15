"""
Phase 5 Tests: Full Quaternion LSTM + Attention Model

Tests QNNAttentionModel and QuaternionLSTMNoAttention for:
- End-to-end forward pass
- Output shapes
- Gradient flow
- Attention weights
- Overfitting capability
"""

import pytest
import torch
import torch.nn as nn

from src.models.qnn_attention_model import QNNAttentionModel, QuaternionLSTMNoAttention


class TestQNNAttentionModelShapes:
    """Tests for QNNAttentionModel output shapes."""

    def test_output_shape(self):
        """Model outputs correct shape."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4)  # (batch, seq_len, OHLC)

        output = model(x)

        assert output.shape == (16, 1)

    def test_output_shape_with_attention(self):
        """Model outputs correct shape with attention weights."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4)

        output, attention = model(x, return_attention=True)

        assert output.shape == (16, 1)
        assert attention.shape == (16, 20)

    def test_different_batch_sizes(self):
        """Model handles various batch sizes."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 20, 4)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_different_sequence_lengths(self):
        """Model handles various sequence lengths."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)

        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(16, seq_len, 4)
            output = model(x)
            assert output.shape == (16, 1)

    def test_different_hidden_sizes(self):
        """Model works with various hidden sizes."""
        for hidden_size in [4, 8, 16, 32]:
            model = QNNAttentionModel(hidden_size=hidden_size, num_layers=1)
            x = torch.randn(16, 20, 4)
            output = model(x)
            assert output.shape == (16, 1)

    def test_multi_layer(self):
        """Model works with multiple layers."""
        model = QNNAttentionModel(hidden_size=8, num_layers=3)
        x = torch.randn(16, 20, 4)

        output = model(x)

        assert output.shape == (16, 1)


class TestQNNAttentionModelAttention:
    """Tests for attention mechanism."""

    def test_attention_weights_sum_to_one(self):
        """Attention weights sum to 1."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4)

        _, attention = model(x, return_attention=True)

        sums = attention.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)

    def test_attention_weights_positive(self):
        """Attention weights are positive (softmax output)."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4)

        _, attention = model(x, return_attention=True)

        assert (attention >= 0).all()

    def test_attention_weights_bounded(self):
        """Attention weights are bounded in [0, 1]."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4)

        _, attention = model(x, return_attention=True)

        assert (attention >= 0).all()
        assert (attention <= 1).all()


class TestQNNAttentionModelGradients:
    """Tests for gradient flow."""

    def test_gradient_flow(self):
        """Gradients flow through entire model."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_in_gradients(self):
        """No NaN values in gradients."""
        torch.manual_seed(42)
        model = QNNAttentionModel(hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4, requires_grad=True)

        output = model(x)
        loss = output.pow(2).sum()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradient_with_attention(self):
        """Gradients work when returning attention."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4, requires_grad=True)

        output, attention = model(x, return_attention=True)
        loss = output.sum() + attention.sum()
        loss.backward()

        assert x.grad is not None


class TestQNNAttentionModelNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_output(self):
        """No NaN in output."""
        torch.manual_seed(42)
        model = QNNAttentionModel(hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4)

        output = model(x)

        assert not torch.isnan(output).any()

    def test_no_inf_output(self):
        """No Inf in output."""
        torch.manual_seed(42)
        model = QNNAttentionModel(hidden_size=8, num_layers=2)
        x = torch.randn(16, 20, 4)

        output = model(x)

        assert not torch.isinf(output).any()

    def test_stability_with_small_inputs(self):
        """Model handles small input values."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4) * 1e-6

        output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_stability_with_large_inputs(self):
        """Model handles large input values."""
        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4) * 10

        output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestQNNAttentionModelOverfit:
    """Tests for overfitting capability."""

    def test_can_overfit_small_batch(self):
        """Model can overfit a small batch."""
        torch.manual_seed(42)

        batch_size = 32
        seq_len = 10

        x = torch.randn(batch_size, seq_len, 4)
        target = torch.randn(batch_size, 1)

        model = QNNAttentionModel(hidden_size=16, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.MSELoss()

        # Initial loss
        model.train()
        pred = model(x)
        initial_loss = criterion(pred, target).item()

        # Train longer with higher learning rate
        for _ in range(200):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"


class TestQuaternionLSTMNoAttention:
    """Tests for ablation model without attention."""

    def test_output_shape(self):
        """Model outputs correct shape."""
        model = QuaternionLSTMNoAttention(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4)

        output = model(x)

        assert output.shape == (16, 1)

    def test_gradient_flow(self):
        """Gradients flow through model."""
        model = QuaternionLSTMNoAttention(hidden_size=8, num_layers=1)
        x = torch.randn(16, 20, 4, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_can_overfit_small_batch(self):
        """Model can overfit a small batch."""
        torch.manual_seed(42)

        batch_size = 32
        seq_len = 10

        x = torch.randn(batch_size, seq_len, 4)
        target = torch.randn(batch_size, 1)

        model = QuaternionLSTMNoAttention(hidden_size=8, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Initial loss
        model.train()
        pred = model(x)
        initial_loss = criterion(pred, target).item()

        # Train
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        assert final_loss < initial_loss * 0.5


class TestModelComparison:
    """Tests comparing different model architectures."""

    def test_all_models_same_output_shape(self):
        """All models produce same output shape for same input."""
        from src.models import RealLSTM, RealLSTMAttention

        x = torch.randn(16, 20, 4)

        models = [
            RealLSTM(input_size=4, hidden_size=16, num_layers=1),
            RealLSTMAttention(input_size=4, hidden_size=16, num_layers=1),
            QuaternionLSTMNoAttention(hidden_size=8, num_layers=1),
            QNNAttentionModel(hidden_size=8, num_layers=1),
        ]

        for model in models:
            output = model(x)
            assert output.shape == (16, 1), f"Model {type(model).__name__} has wrong output shape"

    def test_quaternion_models_parameter_efficiency(self):
        """Quaternion models have reasonable parameter count."""
        # Compare parameter counts
        qnn = QNNAttentionModel(hidden_size=8, num_layers=1)
        qnn_no_att = QuaternionLSTMNoAttention(hidden_size=8, num_layers=1)

        qnn_params = sum(p.numel() for p in qnn.parameters())
        qnn_no_att_params = sum(p.numel() for p in qnn_no_att.parameters())

        # Models should have parameters (sanity check)
        assert qnn_params > 0
        assert qnn_no_att_params > 0

        # QNN with attention should have more params than without
        assert qnn_params > qnn_no_att_params


class TestDropout:
    """Tests for dropout behavior."""

    def test_dropout_training_mode(self):
        """Different outputs in training mode with dropout."""
        torch.manual_seed(42)
        model = QNNAttentionModel(hidden_size=8, num_layers=3, dropout=0.5)
        model.train()
        x = torch.randn(16, 20, 4)

        output1 = model(x)
        output2 = model(x)

        # Should be different due to dropout
        assert not torch.allclose(output1, output2)

    def test_dropout_eval_mode(self):
        """Same outputs in eval mode."""
        torch.manual_seed(42)
        model = QNNAttentionModel(hidden_size=8, num_layers=3, dropout=0.5)
        model.eval()
        x = torch.randn(16, 20, 4)

        output1 = model(x)
        output2 = model(x)

        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
