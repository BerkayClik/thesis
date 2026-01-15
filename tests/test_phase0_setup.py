"""
Phase 0 verification tests.

Tests that basic infrastructure works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SP500Dataset
from src.data.preprocessing import normalize_data, temporal_split, encode_quaternion
from src.models.real_lstm import RealLSTM
from src.models.attention import TemporalAttention
from src.training.trainer import Trainer
from src.training.losses import mse_loss
from src.evaluation.metrics import compute_mae, compute_mse
from src.evaluation.directional_accuracy import compute_directional_accuracy
from src.utils.config import load_config


def test_config_loading():
    """Test that config files load correctly."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base.yaml')
    config = load_config(config_path)

    assert 'data' in config
    assert 'model' in config
    assert 'training' in config
    assert config['data']['window_size'] == 20
    print("Config loading: PASSED")


def test_dataset():
    """Test that dataset class works correctly."""
    # Create dummy data
    data = torch.randn(100, 4)  # 100 samples, 4 features (OHLC)
    window_size = 10

    dataset = SP500Dataset(data, window_size)

    assert len(dataset) == 90  # 100 - 10
    x, y = dataset[0]
    assert x.shape == (10, 4)
    assert y.shape == ()
    print("Dataset: PASSED")


def test_preprocessing():
    """Test preprocessing functions."""
    data = torch.randn(100, 4) * 10 + 5

    # Normalize
    normalized, stats = normalize_data(data)
    assert 'mean' in stats
    assert 'std' in stats
    assert normalized.shape == data.shape

    # Temporal split
    train, val, test = temporal_split(data, 60, 80)
    assert train.shape[0] == 60
    assert val.shape[0] == 20
    assert test.shape[0] == 20

    # Quaternion encoding
    encoded = encode_quaternion(data)
    assert encoded.shape == data.shape
    print("Preprocessing: PASSED")


def test_real_lstm():
    """Test RealLSTM model forward pass."""
    model = RealLSTM(input_size=4, hidden_size=32, num_layers=2)
    x = torch.randn(8, 20, 4)  # batch=8, seq_len=20, features=4

    output = model(x)
    assert output.shape == (8, 1)
    print("RealLSTM: PASSED")


def test_attention():
    """Test TemporalAttention forward pass."""
    attention = TemporalAttention(hidden_size=32)
    x = torch.randn(8, 20, 32)  # batch=8, seq_len=20, hidden=32

    context = attention(x)
    assert context.shape == (8, 32)

    context, weights = attention(x, return_weights=True)
    assert weights.shape == (8, 20)
    print("TemporalAttention: PASSED")


def test_training_loop():
    """Test that training loop runs without errors."""
    # Create dummy data
    data = torch.randn(200, 4)
    window_size = 10

    # Split data
    train_data = data[:150]
    val_data = data[150:]

    # Create datasets
    train_dataset = SP500Dataset(train_data, window_size)
    val_dataset = SP500Dataset(val_data, window_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model and trainer
    model = RealLSTM(input_size=4, hidden_size=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cpu')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=mse_loss,
        device=device,
        checkpoint_dir='/tmp/test_checkpoints'
    )

    # Run a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        patience=5,
        verbose=True
    )

    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 3
    print("Training loop: PASSED")


def test_metrics():
    """Test evaluation metrics."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.1, 2.2, 2.8])
    prev = torch.tensor([0.9, 1.8, 2.9])

    mae = compute_mae(pred, target)
    mse = compute_mse(pred, target)
    da = compute_directional_accuracy(pred, target, prev)

    assert mae >= 0
    assert mse >= 0
    assert 0 <= da <= 100
    print("Metrics: PASSED")


def run_all_tests():
    """Run all Phase 0 tests."""
    print("=" * 50)
    print("Running Phase 0 Setup Tests")
    print("=" * 50)

    test_config_loading()
    test_dataset()
    test_preprocessing()
    test_real_lstm()
    test_attention()
    test_metrics()
    test_training_loop()

    print("=" * 50)
    print("All Phase 0 tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
