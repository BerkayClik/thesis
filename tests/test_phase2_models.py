"""
Phase 2 Baseline Models Tests.

Tests forward pass shapes, loss computation, and overfitting sanity check.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import RealLSTM, RealLSTMAttention, TemporalAttention
from src.training.losses import mse_loss


# Test Configuration Constants
BATCH_SIZE = 8
SEQ_LEN = 20
INPUT_SIZE = 4  # OHLC
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.1


class TestRealLSTMShapes:
    """Tests for RealLSTM forward pass shapes."""

    @pytest.fixture
    def model(self):
        """Create RealLSTM model."""
        return RealLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

    def test_output_shape(self, model, sample_input):
        """Verify output shape is (batch, 1)."""
        output = model(sample_input)
        assert output.shape == (BATCH_SIZE, 1)

    def test_batch_size_one(self, model):
        """Verify model works with batch size 1."""
        x = torch.randn(1, SEQ_LEN, INPUT_SIZE)
        output = model(x)
        assert output.shape == (1, 1)

    def test_different_sequence_lengths(self, model):
        """Verify model handles variable sequence lengths."""
        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(BATCH_SIZE, seq_len, INPUT_SIZE)
            output = model(x)
            assert output.shape == (BATCH_SIZE, 1)

    def test_parameter_count(self, model):
        """Verify model has trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0
        assert trainable_params == total_params


class TestRealLSTMAttentionShapes:
    """Tests for RealLSTMAttention forward pass shapes."""

    @pytest.fixture
    def model(self):
        """Create RealLSTMAttention model."""
        return RealLSTMAttention(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

    def test_output_shape(self, model, sample_input):
        """Verify output shape is (batch, 1)."""
        output = model(sample_input)
        assert output.shape == (BATCH_SIZE, 1)

    def test_output_with_attention_weights(self, model, sample_input):
        """Verify output and attention weights shapes."""
        output, attn_weights = model(sample_input, return_attention_weights=True)
        assert output.shape == (BATCH_SIZE, 1)
        assert attn_weights.shape == (BATCH_SIZE, SEQ_LEN)

    def test_attention_weights_sum_to_one(self, model, sample_input):
        """Verify attention weights sum to 1 for each sample."""
        _, attn_weights = model(sample_input, return_attention_weights=True)
        weight_sums = attn_weights.sum(dim=1)
        torch.testing.assert_close(
            weight_sums,
            torch.ones(BATCH_SIZE),
            rtol=1e-5,
            atol=1e-5
        )

    def test_batch_size_one(self, model):
        """Verify model works with batch size 1."""
        x = torch.randn(1, SEQ_LEN, INPUT_SIZE)
        output = model(x)
        assert output.shape == (1, 1)

    def test_different_sequence_lengths(self, model):
        """Verify model handles variable sequence lengths."""
        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(BATCH_SIZE, seq_len, INPUT_SIZE)
            output = model(x)
            assert output.shape == (BATCH_SIZE, 1)


class TestTemporalAttentionShapes:
    """Tests for standalone TemporalAttention shapes."""

    @pytest.fixture
    def attention(self):
        """Create TemporalAttention module."""
        return TemporalAttention(hidden_size=HIDDEN_SIZE)

    def test_context_shape(self, attention):
        """Verify context vector shape."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        context = attention(x)
        assert context.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_weights_shape(self, attention):
        """Verify attention weights shape."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        context, weights = attention(x, return_weights=True)
        assert weights.shape == (BATCH_SIZE, SEQ_LEN)


class TestLossComputation:
    """Tests for MSE loss computation with models."""

    @pytest.fixture
    def real_lstm(self):
        """Create RealLSTM model."""
        return RealLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    @pytest.fixture
    def real_lstm_attention(self):
        """Create RealLSTMAttention model."""
        return RealLSTMAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    @pytest.fixture
    def sample_batch(self):
        """Create sample input and target."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
        y = torch.randn(BATCH_SIZE)
        return x, y

    def test_mse_loss_real_lstm(self, real_lstm, sample_batch):
        """Verify MSE loss computes correctly for RealLSTM."""
        x, y = sample_batch
        pred = real_lstm(x)
        loss = mse_loss(pred.squeeze(), y)

        assert loss.shape == ()  # Scalar
        assert loss >= 0  # MSE is non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_mse_loss_real_lstm_attention(self, real_lstm_attention, sample_batch):
        """Verify MSE loss computes correctly for RealLSTMAttention."""
        x, y = sample_batch
        pred = real_lstm_attention(x)
        loss = mse_loss(pred.squeeze(), y)

        assert loss.shape == ()  # Scalar
        assert loss >= 0  # MSE is non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow_real_lstm(self, real_lstm, sample_batch):
        """Verify gradients flow through RealLSTM."""
        x, y = sample_batch
        pred = real_lstm(x)
        loss = mse_loss(pred.squeeze(), y)
        loss.backward()

        # Check all parameters have gradients
        for name, param in real_lstm.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_flow_real_lstm_attention(self, real_lstm_attention, sample_batch):
        """Verify gradients flow through RealLSTMAttention."""
        x, y = sample_batch
        pred = real_lstm_attention(x)
        loss = mse_loss(pred.squeeze(), y)
        loss.backward()

        # Check all parameters have gradients
        for name, param in real_lstm_attention.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestOverfittingSanityCheck:
    """
    Sanity check: verify models can overfit a small batch.

    This is a critical test - if a model cannot overfit a tiny dataset,
    something is fundamentally wrong with the architecture.
    """

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for overfitting test."""
        torch.manual_seed(42)
        n_samples = 32
        x = torch.randn(n_samples, SEQ_LEN, INPUT_SIZE)
        y = torch.randn(n_samples)
        return TensorDataset(x, y)

    def test_real_lstm_can_overfit(self, small_dataset):
        """RealLSTM should overfit small batch in 100 steps."""
        torch.manual_seed(42)

        model = RealLSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataloader = DataLoader(small_dataset, batch_size=32, shuffle=False)

        initial_loss = None
        final_loss = None

        model.train()
        for epoch in range(100):
            for x, y in dataloader:
                optimizer.zero_grad()
                pred = model(x)
                loss = mse_loss(pred.squeeze(), y)
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

        assert final_loss < initial_loss * 0.5, \
            f"RealLSTM failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_real_lstm_attention_can_overfit(self, small_dataset):
        """RealLSTMAttention should overfit small batch in 100 steps."""
        torch.manual_seed(42)

        model = RealLSTMAttention(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataloader = DataLoader(small_dataset, batch_size=32, shuffle=False)

        initial_loss = None
        final_loss = None

        model.train()
        for epoch in range(100):
            for x, y in dataloader:
                optimizer.zero_grad()
                pred = model(x)
                loss = mse_loss(pred.squeeze(), y)
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

        assert final_loss < initial_loss * 0.5, \
            f"RealLSTMAttention failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_loss_decreases_monotonically_early(self, small_dataset):
        """Loss should generally decrease in early training epochs."""
        torch.manual_seed(42)

        model = RealLSTMAttention(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataloader = DataLoader(small_dataset, batch_size=32, shuffle=False)

        losses = []
        model.train()
        for epoch in range(20):
            epoch_loss = 0.0
            for x, y in dataloader:
                optimizer.zero_grad()
                pred = model(x)
                loss = mse_loss(pred.squeeze(), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        assert losses[4] < losses[0], \
            f"Loss did not decrease in first 5 epochs: epoch1={losses[0]:.4f}, epoch5={losses[4]:.4f}"


class TestModelConfiguration:
    """Tests for model configuration options."""

    def test_real_lstm_different_configs(self):
        """Test RealLSTM with various configurations."""
        configs = [
            {'input_size': 4, 'hidden_size': 16, 'num_layers': 1, 'dropout': 0.0},
            {'input_size': 4, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1},
            {'input_size': 4, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2},
        ]

        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

        for config in configs:
            model = RealLSTM(**config)
            output = model(x)
            assert output.shape == (BATCH_SIZE, 1), f"Failed for config: {config}"

    def test_real_lstm_attention_different_configs(self):
        """Test RealLSTMAttention with various configurations."""
        configs = [
            {'input_size': 4, 'hidden_size': 16, 'num_layers': 1, 'dropout': 0.0},
            {'input_size': 4, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1},
            {'input_size': 4, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2},
        ]

        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

        for config in configs:
            model = RealLSTMAttention(**config)
            output = model(x)
            assert output.shape == (BATCH_SIZE, 1), f"Failed for config: {config}"


class TestModelIntegration:
    """Integration tests with data pipeline."""

    def test_models_with_sp500_dataset(self):
        """Test models work with SP500Dataset format."""
        from src.data import SP500Dataset

        # Simulate SP500 data
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

        models = [
            RealLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS),
            RealLSTMAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
        ]

        for model in models:
            model.eval()
            for x, y in dataloader:
                pred = model(x)
                loss = mse_loss(pred.squeeze(), y)
                assert not torch.isnan(loss)
                break  # Just test first batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
