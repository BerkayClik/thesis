"""
Phase 7 Tests: Experiments & Ablation

Tests for experiment runner functionality:
- Model creation
- Experiment configuration loading
- Results aggregation
- End-to-end experiment execution
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_experiments import (
    load_config,
    set_seed,
    get_device,
    create_model,
    evaluate_model,
    print_results_table,
    save_results,
)
from src.models import RealLSTM, RealLSTMAttention, QNNAttentionModel
from src.models.qnn_attention_model import QuaternionLSTMNoAttention
from src.data.dataset import SP500Dataset
from torch.utils.data import DataLoader


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_base_config(self):
        """Base config loads correctly."""
        config = load_config('configs/base.yaml')
        assert 'data' in config
        assert 'model' in config
        assert 'training' in config

    def test_load_experiment_config(self):
        """Experiment config loads correctly."""
        config = load_config('configs/experiment.yaml')
        assert 'experiment' in config
        assert 'name' in config['experiment']
        assert 'variants' in config['experiment']


class TestSeedSetting:
    """Tests for seed setting."""

    def test_reproducibility(self):
        """Same seed produces same random values."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        assert torch.allclose(a, b)

    def test_different_seeds_different_values(self):
        """Different seeds produce different values."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(123)
        b = torch.randn(10)

        assert not torch.allclose(a, b)


class TestDeviceSelection:
    """Tests for device selection."""

    def test_cpu_device(self):
        """CPU device selection works."""
        config = {'device': 'cpu'}
        device = get_device(config)
        assert device.type == 'cpu'

    def test_default_cpu(self):
        """Default is CPU when not specified."""
        config = {}
        device = get_device(config)
        assert device.type == 'cpu'


class TestModelCreation:
    """Tests for model creation."""

    def test_create_real_lstm(self):
        """Real LSTM creation works."""
        model = create_model('real_lstm', hidden_size=32, num_layers=1)
        assert isinstance(model, RealLSTM)

    def test_create_real_lstm_attention(self):
        """Real LSTM + Attention creation works."""
        model = create_model('real_lstm_attention', hidden_size=32, num_layers=1)
        assert isinstance(model, RealLSTMAttention)

    def test_create_quaternion_lstm(self):
        """Quaternion LSTM creation works."""
        model = create_model('quaternion_lstm', hidden_size=8, num_layers=1)
        assert isinstance(model, QuaternionLSTMNoAttention)

    def test_create_quaternion_lstm_attention(self):
        """Quaternion LSTM + Attention creation works."""
        model = create_model('quaternion_lstm_attention', hidden_size=8, num_layers=1)
        assert isinstance(model, QNNAttentionModel)

    def test_create_unknown_model_raises(self):
        """Unknown model type raises error."""
        with pytest.raises(ValueError):
            create_model('unknown_model', hidden_size=32, num_layers=1)

    def test_model_forward_works(self):
        """Created models can do forward pass."""
        x = torch.randn(16, 20, 4)

        for model_type in ['real_lstm', 'real_lstm_attention', 'quaternion_lstm', 'quaternion_lstm_attention']:
            hidden_size = 8 if 'quaternion' in model_type else 32
            model = create_model(model_type, hidden_size=hidden_size, num_layers=1)
            output = model(x)
            assert output.shape == (16, 1)


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_evaluate_returns_metrics(self):
        """Evaluation returns all required metrics."""
        model = RealLSTM(input_size=4, hidden_size=16, num_layers=1)
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)
        loader = DataLoader(dataset, batch_size=16)

        metrics = evaluate_model(model, loader, torch.device('cpu'))

        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'directional_accuracy' in metrics

    def test_metrics_are_valid(self):
        """Metrics have valid values."""
        model = RealLSTM(input_size=4, hidden_size=16, num_layers=1)
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)
        loader = DataLoader(dataset, batch_size=16)

        metrics = evaluate_model(model, loader, torch.device('cpu'))

        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert 0 <= metrics['directional_accuracy'] <= 100


class TestResultsSaving:
    """Tests for results saving."""

    def test_save_results_creates_file(self):
        """Results are saved to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                'model_a': {
                    'aggregated': {
                        'mae': {'mean': 0.1, 'std': 0.01},
                        'mse': {'mean': 0.02, 'std': 0.002},
                        'directional_accuracy': {'mean': 55.0, 'std': 2.0}
                    }
                }
            }
            save_results(results, tmpdir, 'test_experiment')

            # Check file was created
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith('test_experiment_')
            assert files[0].endswith('.json')

    def test_saved_results_loadable(self):
        """Saved results can be loaded."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                'model_a': {
                    'aggregated': {
                        'mae': {'mean': 0.1, 'std': 0.01},
                        'mse': {'mean': 0.02, 'std': 0.002},
                        'directional_accuracy': {'mean': 55.0, 'std': 2.0}
                    }
                }
            }
            save_results(results, tmpdir, 'test')

            # Load and verify
            files = os.listdir(tmpdir)
            with open(os.path.join(tmpdir, files[0]), 'r') as f:
                loaded = json.load(f)

            assert loaded == results


class TestPrintResultsTable:
    """Tests for results table printing."""

    def test_print_does_not_crash(self, capsys):
        """Print results table works without error."""
        results = {
            'real_lstm': {
                'aggregated': {
                    'mae': {'mean': 0.1, 'std': 0.01},
                    'mse': {'mean': 0.02, 'std': 0.002},
                    'directional_accuracy': {'mean': 55.0, 'std': 2.0},
                    'sharpe_ratio': {'mean': 0.5, 'std': 0.1}
                }
            },
            'quaternion_lstm': {
                'aggregated': {
                    'mae': {'mean': 0.09, 'std': 0.02},
                    'mse': {'mean': 0.018, 'std': 0.003},
                    'directional_accuracy': {'mean': 57.0, 'std': 3.0},
                    'sharpe_ratio': {'mean': 0.6, 'std': 0.15}
                }
            }
        }

        print_results_table(results)
        captured = capsys.readouterr()

        assert 'EXPERIMENT RESULTS' in captured.out
        assert 'real_lstm' in captured.out
        assert 'quaternion_lstm' in captured.out


class TestAllModelsIntegration:
    """Integration tests for all model types."""

    @pytest.mark.parametrize("model_type,hidden_size", [
        ("real_lstm", 32),
        ("real_lstm_attention", 32),
        ("quaternion_lstm", 8),
        ("quaternion_lstm_attention", 8),
    ])
    def test_model_training_evaluation_pipeline(self, model_type, hidden_size):
        """Each model type can be trained and evaluated."""
        torch.manual_seed(42)

        # Create model
        model = create_model(model_type, hidden_size=hidden_size, num_layers=1)

        # Create data
        data = torch.randn(100, 4)
        dataset = SP500Dataset(data, window_size=20)
        loader = DataLoader(dataset, batch_size=16)

        # Evaluate (no training, just check it works)
        metrics = evaluate_model(model, loader, torch.device('cpu'))

        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert 0 <= metrics['directional_accuracy'] <= 100


class TestExperimentMatrix:
    """Tests for experiment matrix coverage."""

    def test_all_model_types_defined(self):
        """All required model types can be created."""
        model_types = [
            'real_lstm',
            'real_lstm_attention',
            'quaternion_lstm',
            'quaternion_lstm_attention'
        ]

        for model_type in model_types:
            hidden_size = 8 if 'quaternion' in model_type else 32
            model = create_model(model_type, hidden_size=hidden_size, num_layers=1)
            assert model is not None

    def test_config_has_all_variants(self):
        """Experiment config has all 4 model variants."""
        config = load_config('configs/experiment.yaml')
        variants = config['experiment']['variants']

        variant_names = [v['name'] for v in variants]
        assert 'real_lstm' in variant_names
        assert 'real_lstm_attention' in variant_names
        assert 'quaternion_lstm' in variant_names
        assert 'quaternion_lstm_attention' in variant_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
