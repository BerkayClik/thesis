"""
Phase 6 Tests: Training & Evaluation

Tests for:
- Trainer class (training loop, early stopping, checkpointing)
- Loss functions
- Evaluation metrics
- Directional accuracy
- End-to-end training pipeline
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

from src.training.trainer import Trainer
from src.training.losses import mse_loss
from src.evaluation.metrics import compute_mae, compute_mse
from src.evaluation.directional_accuracy import compute_directional_accuracy
from src.data.dataset import SP500Dataset
from src.models import RealLSTM, RealLSTMAttention, QNNAttentionModel


class TestMSELoss:
    """Tests for MSE loss function."""

    def test_zero_loss_for_identical(self):
        """Zero loss when predictions equal targets."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = mse_loss(pred, target)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_positive_loss(self):
        """Loss is positive for different values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = mse_loss(pred, target)
        assert loss > 0

    def test_known_value(self):
        """Test known MSE value."""
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 1.0])
        # MSE = (1^2 + 1^2) / 2 = 1.0
        loss = mse_loss(pred, target)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_symmetric(self):
        """MSE is symmetric."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        assert torch.isclose(mse_loss(a, b), mse_loss(b, a))


class TestMAE:
    """Tests for MAE computation."""

    def test_zero_for_identical(self):
        """Zero MAE when predictions equal targets."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        mae = compute_mae(pred, target)
        assert mae == pytest.approx(0.0)

    def test_known_value(self):
        """Test known MAE value."""
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 3.0])
        # MAE = (1 + 3) / 2 = 2.0
        mae = compute_mae(pred, target)
        assert mae == pytest.approx(2.0)

    def test_symmetric(self):
        """MAE is symmetric."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        assert compute_mae(a, b) == pytest.approx(compute_mae(b, a))


class TestMSE:
    """Tests for MSE computation."""

    def test_zero_for_identical(self):
        """Zero MSE when predictions equal targets."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        mse = compute_mse(pred, target)
        assert mse == pytest.approx(0.0)

    def test_known_value(self):
        """Test known MSE value."""
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 3.0])
        # MSE = (1^2 + 3^2) / 2 = 5.0
        mse = compute_mse(pred, target)
        assert mse == pytest.approx(5.0)


class TestDirectionalAccuracy:
    """Tests for directional accuracy."""

    def test_perfect_accuracy(self):
        """100% accuracy when all directions correct."""
        pred = torch.tensor([2.0, 4.0, 0.0])
        target = torch.tensor([3.0, 5.0, -1.0])  # Same direction as pred
        prev = torch.tensor([1.0, 3.0, 1.0])

        acc = compute_directional_accuracy(pred, target, prev)
        assert acc == pytest.approx(100.0)

    def test_zero_accuracy(self):
        """0% accuracy when all directions wrong."""
        pred = torch.tensor([2.0, 4.0, 0.0])
        target = torch.tensor([0.0, 2.0, 2.0])  # Opposite direction
        prev = torch.tensor([1.0, 3.0, 1.0])

        acc = compute_directional_accuracy(pred, target, prev)
        assert acc == pytest.approx(0.0)

    def test_fifty_percent_accuracy(self):
        """50% accuracy when half correct."""
        pred = torch.tensor([2.0, 4.0])
        target = torch.tensor([3.0, 2.0])  # First correct, second wrong
        prev = torch.tensor([1.0, 3.0])

        acc = compute_directional_accuracy(pred, target, prev)
        assert acc == pytest.approx(50.0)

    def test_no_change_case(self):
        """Handle case where no change (sign = 0)."""
        pred = torch.tensor([1.0])  # Same as prev
        target = torch.tensor([1.0])  # Same as prev
        prev = torch.tensor([1.0])

        acc = compute_directional_accuracy(pred, target, prev)
        # Both signs are 0, so they match
        assert acc == pytest.approx(100.0)


class TestSP500Dataset:
    """Tests for SP500Dataset."""

    def test_length(self):
        """Dataset has correct length."""
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)
        assert len(dataset) == 80  # 100 - 20

    def test_item_shapes(self):
        """Items have correct shapes."""
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)

        x, y = dataset[0]
        assert x.shape == (20, 4)
        assert y.shape == ()  # scalar

    def test_target_is_next_close(self):
        """Target is next-step close price."""
        data = torch.arange(100 * 4).float().view(100, 4)
        dataset = SP500Dataset(data, window_size=20, target_col=3)

        x, y = dataset[0]
        # Target should be close price at position 20
        expected = data[20, 3]
        assert y == expected

    def test_no_overlap_with_target(self):
        """Window doesn't overlap with target."""
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)

        x, y = dataset[0]
        # Last point in x should be at index 19
        # y should be from index 20
        assert torch.equal(x[-1], data[19])


class TestTrainer:
    """Tests for Trainer class."""

    def create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(4, 1)

    def create_data_loaders(self, batch_size=16, num_samples=100):
        """Create simple data loaders for testing."""
        x = torch.randn(num_samples, 4)
        y = torch.randn(num_samples)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size), DataLoader(dataset, batch_size=batch_size)

    def test_train_epoch(self):
        """Train epoch runs without error."""
        model = self.create_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))

        train_loader, _ = self.create_data_loaders()
        loss = trainer.train_epoch(train_loader)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate(self):
        """Validation runs without error."""
        model = self.create_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))

        _, val_loader = self.create_data_loaders()
        loss = trainer.validate(val_loader)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_returns_history(self):
        """Training returns history dictionary."""
        model = self.create_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))

        train_loader, val_loader = self.create_data_loaders()
        history = trainer.train(train_loader, val_loader, num_epochs=3, verbose=False)

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'best_epoch' in history
        assert len(history['train_loss']) == 3

    def test_early_stopping(self):
        """Early stopping works correctly."""
        model = self.create_simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # No learning -> no improvement
        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))

        train_loader, val_loader = self.create_data_loaders()
        history = trainer.train(train_loader, val_loader, num_epochs=100, patience=3, verbose=False)

        # Should stop early due to no improvement
        assert len(history['train_loss']) <= 4  # 1 initial + 3 patience

    def test_save_load_checkpoint(self):
        """Checkpoint saving and loading works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.create_simple_model()
            optimizer = torch.optim.Adam(model.parameters())
            trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'), checkpoint_dir=tmpdir)

            # Save
            trainer.save_checkpoint('test.pt')

            # Check file exists
            assert os.path.exists(os.path.join(tmpdir, 'test.pt'))

            # Modify weights
            with torch.no_grad():
                model.weight.fill_(999)

            # Load
            trainer.load_checkpoint('test.pt')

            # Weights should be restored
            assert not (model.weight == 999).all()


class TestEndToEndTraining:
    """End-to-end training tests with real models."""

    def test_real_lstm_training(self):
        """RealLSTM can be trained."""
        torch.manual_seed(42)

        # Create data
        data = torch.randn(200, 4)
        train_dataset = SP500Dataset(data[:150], window_size=20)
        val_dataset = SP500Dataset(data[150:], window_size=20)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Model
        model = RealLSTM(input_size=4, hidden_size=16, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))
        history = trainer.train(train_loader, val_loader, num_epochs=5, patience=10, verbose=False)

        assert len(history['train_loss']) == 5

    def test_real_lstm_attention_training(self):
        """RealLSTMAttention can be trained."""
        torch.manual_seed(42)

        data = torch.randn(200, 4)
        train_dataset = SP500Dataset(data[:150], window_size=20)
        val_dataset = SP500Dataset(data[150:], window_size=20)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        model = RealLSTMAttention(input_size=4, hidden_size=16, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))
        history = trainer.train(train_loader, val_loader, num_epochs=5, patience=10, verbose=False)

        assert len(history['train_loss']) == 5

    def test_qnn_attention_training(self):
        """QNNAttentionModel can be trained."""
        torch.manual_seed(42)

        data = torch.randn(200, 4)
        train_dataset = SP500Dataset(data[:150], window_size=20)
        val_dataset = SP500Dataset(data[150:], window_size=20)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        model = QNNAttentionModel(hidden_size=8, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, mse_loss, torch.device('cpu'))
        history = trainer.train(train_loader, val_loader, num_epochs=3, patience=10, verbose=False)

        assert len(history['train_loss']) == 3

    def test_full_evaluation_pipeline(self):
        """Full evaluation pipeline works."""
        torch.manual_seed(42)

        # Create data
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)
        loader = DataLoader(dataset, batch_size=16)

        # Model
        model = RealLSTM(input_size=4, hidden_size=16, num_layers=1)
        model.eval()

        # Collect predictions
        all_preds = []
        all_targets = []
        all_prevs = []

        with torch.no_grad():
            for x, y in loader:
                pred = model(x).squeeze()
                prev = x[:, -1, 3]  # Last close price in window

                all_preds.append(pred)
                all_targets.append(y)
                all_prevs.append(prev)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        prevs = torch.cat(all_prevs)

        # Compute metrics
        mae = compute_mae(preds, targets)
        mse = compute_mse(preds, targets)
        dir_acc = compute_directional_accuracy(preds, targets, prevs)

        assert mae >= 0
        assert mse >= 0
        assert 0 <= dir_acc <= 100


class TestNoLookAheadBias:
    """Tests to verify no look-ahead bias."""

    def test_trainer_uses_only_past_data(self):
        """Trainer doesn't use future data."""
        # This is verified by the SP500Dataset implementation
        # which only returns past data as X and future as y
        data = torch.arange(100 * 4).float().view(100, 4)
        dataset = SP500Dataset(data, window_size=20)

        x, y = dataset[50]

        # All values in x should be less than y's timestamp
        # x covers indices 50 to 69, y is from index 70
        max_x_value = x.max()
        assert max_x_value < y

    def test_validation_separate_from_training(self):
        """Validation doesn't leak into training."""
        torch.manual_seed(42)

        # Create train and val with different distributions
        train_data = torch.randn(100, 4)
        val_data = torch.randn(50, 4) + 10  # Different distribution

        train_dataset = SP500Dataset(train_data, window_size=20)
        val_dataset = SP500Dataset(val_data, window_size=20)

        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)

        model = nn.Linear(4 * 20, 1)
        optimizer = torch.optim.Adam(model.parameters())

        # Flatten model for testing
        class FlatModel(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x.view(x.size(0), -1))

        flat_model = FlatModel(model)
        trainer = Trainer(flat_model, optimizer, mse_loss, torch.device('cpu'))

        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        # Val loss should be different (different distribution)
        assert train_loss != val_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
