"""
Experiment runner module.

Provides configurable experiment execution with results logging.
Runs all model variants and ablation studies specified in config.
"""

import torch
import yaml
import os
import sys
import json
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_sp500_data, dataframe_to_tensor
from src.data.preprocessing import preprocess_data, preprocess_data_ratio, normalize_data
from src.data.dataset import SP500Dataset
from src.models import RealLSTM, RealLSTMAttention, QNNAttentionModel
from src.models.qnn_attention_model import QuaternionLSTMNoAttention
from src.training.trainer import Trainer
from src.training.losses import mse_loss
from src.evaluation.metrics import compute_mape
from src.evaluation.directional_accuracy import compute_directional_accuracy, compute_directional_accuracy_3class
from src.evaluation.sharpe_ratio import compute_sharpe_ratio, compute_sharpe_ratio_3class


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int, fast_mode: bool = False):
    """
    Set random seeds for reproducibility with deterministic behavior.

    Args:
        seed: Random seed for reproducibility.
        fast_mode: If True, enables TF32 for faster training (less reproducible).
                   If False (default), uses IEEE precision for strict reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if fast_mode:
        # Fast mode: enable TF32 for 2-3x speedup on Ampere+ GPUs
        # Less reproducible but suitable for quick iteration
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # New API (PyTorch 2.9+)
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
        # Legacy API fallback
        elif hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        # Strict reproducibility mode (default)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # New API (PyTorch 2.9+)
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
        # Legacy API fallback
        elif hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        # For PyTorch 1.11+, enable deterministic algorithms globally
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions don't have warn_only parameter
                pass


def get_device(config: Dict) -> torch.device:
    """Get compute device from config with automatic fallback."""
    device_name = config.get('device', 'cpu')

    if device_name == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            print("CUDA not available, using MPS")
            return torch.device('mps')
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device('cpu')
    elif device_name == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS not available, falling back to CPU")
            return torch.device('cpu')

    return torch.device(device_name)


class NaiveBaseline(torch.nn.Module):
    """Naive baseline - predicts last observed close (persistence)."""

    def __init__(self):
        super().__init__()
        # Dummy parameter so optimizer doesn't complain
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        # Return the last close price in the window (persistence model)
        return x[:, -1, 3:4]  # Last close in window, shape (batch, 1) for consistency


def create_model(model_type: str, hidden_size: int, num_layers: int, dropout: float = 0.0):
    """Create model based on type string."""
    if model_type == "naive_zero":
        return NaiveBaseline()
    elif model_type == "real_lstm":
        return RealLSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "real_lstm_attention":
        return RealLSTMAttention(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "quaternion_lstm":
        return QuaternionLSTMNoAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_features=4,
            target_col=3
        )
    elif model_type == "quaternion_lstm_attention":
        return QNNAttentionModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_features=4,
            target_col=3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def denormalize(x: torch.Tensor, stats: Dict, col: int = 3) -> torch.Tensor:
    """Denormalize data using stored statistics."""
    return x * stats['std'][col] + stats['mean'][col]


def compute_statistical_significance(
    results: Dict,
    baseline_model: str = "real_lstm"
) -> Dict[str, Dict[str, Dict]]:
    """
    Compute statistical significance tests between models.

    Args:
        results: Results dictionary from run_experiment.
        baseline_model: Name of the baseline model for comparison.

    Returns:
        Dictionary with significance test results for each metric and model pair.
    """
    if baseline_model not in results:
        return {}

    significance_results = {}
    metrics = ['mape', 'directional_accuracy', 'sharpe_ratio',
               'directional_accuracy_3class', 'sharpe_ratio_3class']

    # Get baseline values
    baseline_runs = results[baseline_model]['individual_runs']
    baseline_values = {
        metric: [r['test_metrics'][metric] for r in baseline_runs]
        for metric in metrics
    }

    for model_name, model_results in results.items():
        if model_name == baseline_model:
            continue

        model_runs = model_results['individual_runs']
        model_values = {
            metric: [r['test_metrics'][metric] for r in model_runs]
            for metric in metrics
        }

        significance_results[model_name] = {}
        for metric in metrics:
            baseline_vals = baseline_values[metric]
            model_vals = model_values[metric]

            # Paired t-test (if same number of seeds)
            if len(baseline_vals) == len(model_vals) and len(baseline_vals) >= 2:
                t_stat, p_value = stats.ttest_rel(model_vals, baseline_vals)
                # Cohen's d effect size
                diff = np.array(model_vals) - np.array(baseline_vals)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
            else:
                # Independent t-test as fallback
                t_stat, p_value = stats.ttest_ind(model_vals, baseline_vals)
                pooled_std = np.sqrt(
                    (np.var(model_vals, ddof=1) + np.var(baseline_vals, ddof=1)) / 2
                )
                cohens_d = (np.mean(model_vals) - np.mean(baseline_vals)) / pooled_std if pooled_std > 0 else 0

            significance_results[model_name][metric] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant_0.05': p_value < 0.05,
                'significant_0.01': p_value < 0.01
            }

    return significance_results


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    norm_stats: Dict,
    flat_threshold_fraction: float = 0.5,
    needs_denorm: bool = False
) -> Dict:
    """
    Evaluate model on a dataset.

    Returns dict with MAPE, directional accuracy (binary and 3-class),
    and Sharpe ratio (binary and 3-class).
    All metrics are computed on original-scale prices.

    Two data paths:
    - Quaternion models (needs_denorm=False): use raw data with internal RevIN,
      outputs are already in original price scale.
    - Real LSTM / naive models (needs_denorm=True): use Z-score normalized data,
      outputs are denormalized back to original price scale here.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Compute device.
        norm_stats: Dict with 'mean', 'std', and 'return_std' for denormalization
                    and flat threshold computation.
        flat_threshold_fraction: Fraction of training return_std for FLAT zone (default: 0.5).
        needs_denorm: If True, denormalize predictions/targets/prevs from Z-score scale.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_prevs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            pred = model(x).cpu()
            # Handle different output shapes: (batch,), (batch, 1), etc.
            if pred.dim() > 1:
                pred = pred.squeeze(-1)  # Only squeeze last dim to preserve batch
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)  # Handle single-sample batch
            prev = x[:, -1, 3].cpu()  # Last close in window (raw scale)
            all_preds.append(pred)
            all_targets.append(y)
            all_prevs.append(prev)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    prevs = torch.cat(all_prevs)

    if needs_denorm:
        # Real LSTM / naive: outputs are in normalized scale, denormalize to original prices
        preds = denormalize(preds, norm_stats, col=3)
        targets = denormalize(targets, norm_stats, col=3)
        prevs = denormalize(prevs, norm_stats, col=3)
    # Quaternion models: outputs already in original price scale via RevIN

    # Compute flat threshold from training return std
    return_std = norm_stats.get('return_std', 0.0)
    flat_threshold = flat_threshold_fraction * return_std

    return {
        'mape': compute_mape(preds, targets),
        'directional_accuracy': compute_directional_accuracy(
            preds, targets, prevs
        ),
        'sharpe_ratio': compute_sharpe_ratio(
            preds, targets, prevs
        ),
        'directional_accuracy_3class': compute_directional_accuracy_3class(
            preds, targets, prevs, flat_threshold=flat_threshold
        ),
        'sharpe_ratio_3class': compute_sharpe_ratio_3class(
            preds, targets, prevs, flat_threshold=flat_threshold
        ),
        'flat_threshold': flat_threshold,
        'return_std': return_std,
        # Store predictions for visualization
        'predictions': preds.numpy().tolist(),
        'targets': targets.numpy().tolist(),
        'prev_closes': prevs.numpy().tolist()
    }


def run_single_experiment(
    config: Dict,
    model_config: Dict,
    variant: Dict,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    norm_stats: Dict,
    verbose: bool = True,
    debug: bool = False,
    variant_name: str = "default",
    fast_mode: bool = False,
    flat_threshold_fraction: float = 0.5,
    needs_denorm: bool = False
) -> Dict:
    """
    Run a single experiment with one model configuration and seed.

    Returns results dictionary with training history and test metrics.
    """
    set_seed(seed, fast_mode=fast_mode)

    # Create model
    model = create_model(
        model_type=model_config['type'],
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.0)
    )

    # Compile model for faster execution (PyTorch 2.0+)
    # Skip torch.compile for:
    # - MPS: Metal shader buffer limits can be exceeded
    # - Quaternion models: sequential time loop causes excessive recompilation
    should_compile = (
        hasattr(torch, 'compile')
        and device.type != 'mps'
        and 'quaternion' not in model_config['type']
    )
    if should_compile:
        try:
            model = torch.compile(model)
            if verbose:
                print(f"    Model compiled with torch.compile")
        except Exception as e:
            if verbose:
                print(f"    torch.compile failed, using eager mode: {e}")

    # Count parameters for debugging
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"    Model parameters: {num_params:,}")

    # Skip training for naive baseline (no trainable parameters)
    if model_config['type'] == 'naive_zero':
        if verbose:
            print(f"  Evaluating naive baseline (no training)...")
        model.to(device)
        history = {'train_loss': [], 'val_loss': [], 'best_epoch': 0}
    else:
        # Check for per-variant learning rate override
        variant_lr = variant.get('learning_rate', config['training']['learning_rate'])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=variant_lr
        )

        # Trainer - use unique checkpoint dir per variant and seed to avoid conflicts
        checkpoint_dir = os.path.join(
            config.get('output', {}).get('results_dir', 'experiments/results'),
            'checkpoints',
            variant_name,
            f'seed_{seed}'
        )

        # Training stability settings from config
        training_config = config.get('training', {})
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        scheduler_config = training_config.get('scheduler', None)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=mse_loss,
            device=device,
            checkpoint_dir=checkpoint_dir,
            max_grad_norm=max_grad_norm,
            scheduler_config=scheduler_config,
            debug=debug
        )

        # Train
        if verbose:
            print(f"  Training with seed {seed}...")

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            patience=config['training']['patience'],
            verbose=1 if verbose else 0
        )

        # Load best model for evaluation
        checkpoint_path = os.path.join(trainer.checkpoint_dir, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint('best_model.pt')
            if verbose:
                print(f"    Loaded best model checkpoint")
        else:
            if verbose:
                print(f"    Warning: No checkpoint found, using current model state")

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device, norm_stats,
                                  flat_threshold_fraction=flat_threshold_fraction,
                                  needs_denorm=needs_denorm)

    result = {
        'seed': seed,
        'history': history,
        'test_metrics': test_metrics,
        'best_epoch': history['best_epoch'],
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'num_parameters': num_params
    }

    # Add gradient norm summary if available
    if 'grad_norms' in history:
        grad_norms = history['grad_norms']
        if grad_norms:
            result['grad_norm_summary'] = {
                'first_epoch': grad_norms[0] if grad_norms else None,
                'last_epoch': grad_norms[-1] if grad_norms else None,
                'mean_across_epochs': sum(g['mean'] for g in grad_norms) / len(grad_norms)
            }

    # Memory cleanup to prevent OOM during multi-seed/variant runs
    del model
    if 'trainer' in locals():
        del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_experiment(
    config: Dict,
    experiment_config: Dict,
    verbose: bool = True,
    debug: bool = False,
    output_dir: str = None,
    experiment_name: str = None
) -> Dict:
    """
    Run a complete experiment with multiple seeds.

    Returns aggregated results with mean and std across seeds.

    Args:
        config: Base configuration dictionary.
        experiment_config: Experiment-specific configuration.
        verbose: Whether to print progress information.
        debug: Whether to enable debug mode.
        output_dir: Directory for intermediate results (enables rolling save).
        experiment_name: Experiment name for intermediate results filename.
    """
    device = get_device(config)

    # Load and preprocess data
    if verbose:
        print("Loading data...")

    df = load_sp500_data(
        ticker=config['data']['ticker'],
        start_date=config['data'].get('start_date'),
        end_date=config['data'].get('end_date'),
        cache_dir=config['data'].get('cache_dir', 'data/cache'),
        interval=config['data'].get('interval', '1d'),
        period=config['data'].get('period'),
        resample_interval=config['data'].get('resample_interval')
    )

    if verbose:
        interval_str = config['data'].get('interval', '1d')
        resample_str = config['data'].get('resample_interval')
        if resample_str:
            print(f"  Loaded {len(df)} data points ({interval_str} -> {resample_str} resampled)")
        else:
            print(f"  Loaded {len(df)} data points ({interval_str} interval)")

    # Convert to tensor with dates
    data, dates = dataframe_to_tensor(df)

    # Preprocess data - use ratio-based or year-based splitting
    if 'train_ratio' in config['data']:
        # Ratio-based splitting (for hourly data)
        processed = preprocess_data_ratio(
            data,
            dates=dates,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio']
        )
        if verbose:
            split_info = processed['split_info']
            print(f"  Split (ratio-based): Train={split_info['train_size']}, "
                  f"Val={split_info['val_size']}, Test={split_info['test_size']}")
    else:
        # Year-based splitting (for daily data)
        processed = preprocess_data(
            data,
            dates=dates,
            train_end_year=config['data']['train_end_year'],
            val_end_year=config['data']['val_end_year']
        )
        if verbose:
            split_info = processed['split_info']
            print(f"  Split (year-based): Train={split_info['train_size']}, "
                  f"Val={split_info['val_size']}, Test={split_info['test_size']}")
    train_data = processed['train_data']
    val_data = processed['val_data']
    test_data = processed['test_data']
    norm_stats = processed['norm_stats']

    # Z-score normalize data using training stats (for real LSTM / naive models)
    stats = {'mean': norm_stats['mean'], 'std': norm_stats['std']}
    train_norm, _ = normalize_data(train_data, stats=stats)
    val_norm, _ = normalize_data(val_data, stats=stats)
    test_norm, _ = normalize_data(test_data, stats=stats)

    # Create datasets
    window_size = config['data']['window_size']

    # Raw datasets (for quaternion models with internal RevIN)
    train_dataset_raw = SP500Dataset(train_data, window_size=window_size)
    val_dataset_raw = SP500Dataset(val_data, window_size=window_size)
    test_dataset_raw = SP500Dataset(test_data, window_size=window_size)

    # Normalized datasets (for real LSTM / naive models)
    train_dataset_norm = SP500Dataset(train_norm, window_size=window_size)
    val_dataset_norm = SP500Dataset(val_norm, window_size=window_size)
    test_dataset_norm = SP500Dataset(test_norm, window_size=window_size)

    # Create data loaders for both paths
    batch_size = config['training']['batch_size']
    train_loader_raw = DataLoader(train_dataset_raw, batch_size=batch_size, shuffle=False)
    val_loader_raw = DataLoader(val_dataset_raw, batch_size=batch_size)
    test_loader_raw = DataLoader(test_dataset_raw, batch_size=batch_size)

    train_loader_norm = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=False)
    val_loader_norm = DataLoader(val_dataset_norm, batch_size=batch_size)
    test_loader_norm = DataLoader(test_dataset_norm, batch_size=batch_size)

    # Get evaluation settings
    flat_threshold_fraction = config.get('evaluation', {}).get('flat_threshold_fraction', 0.5)

    # Get seeds
    seeds = experiment_config.get('seeds', [config['training']['seed']])

    # Run for each variant
    all_results = {}

    variants = experiment_config.get('variants', [])
    for variant in variants:
        variant_name = variant['name']
        model_config = variant['model']

        if verbose:
            print(f"\nRunning variant: {variant_name}")

        # Select data path based on model type
        uses_revin = 'quaternion' in model_config['type']
        needs_denorm = not uses_revin

        if uses_revin:
            cur_train_loader = train_loader_raw
            cur_val_loader = val_loader_raw
            cur_test_loader = test_loader_raw
        else:
            cur_train_loader = train_loader_norm
            cur_val_loader = val_loader_norm
            cur_test_loader = test_loader_norm

        variant_results = []
        fast_mode = config.get('training', {}).get('fast_mode', False)
        for seed in seeds:
            result = run_single_experiment(
                config=config,
                model_config=model_config,
                variant=variant,
                seed=seed,
                train_loader=cur_train_loader,
                val_loader=cur_val_loader,
                test_loader=cur_test_loader,
                device=device,
                norm_stats=norm_stats,
                verbose=verbose,
                debug=debug,
                variant_name=variant_name,
                fast_mode=fast_mode,
                flat_threshold_fraction=flat_threshold_fraction,
                needs_denorm=needs_denorm
            )
            variant_results.append(result)

        # Aggregate results
        test_mapes = [r['test_metrics']['mape'] for r in variant_results]
        test_das = [r['test_metrics']['directional_accuracy'] for r in variant_results]
        test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in variant_results]
        test_das_3c = [r['test_metrics']['directional_accuracy_3class'] for r in variant_results]
        test_sharpes_3c = [r['test_metrics']['sharpe_ratio_3class'] for r in variant_results]

        all_results[variant_name] = {
            'individual_runs': variant_results,
            'aggregated': {
                'mape': {'mean': np.mean(test_mapes), 'std': np.std(test_mapes)},
                'directional_accuracy': {'mean': np.mean(test_das), 'std': np.std(test_das)},
                'sharpe_ratio': {'mean': np.mean(test_sharpes), 'std': np.std(test_sharpes)},
                'directional_accuracy_3class': {'mean': np.mean(test_das_3c), 'std': np.std(test_das_3c)},
                'sharpe_ratio_3class': {'mean': np.mean(test_sharpes_3c), 'std': np.std(test_sharpes_3c)}
            }
        }

        # Rolling save after each variant
        if output_dir and experiment_name:
            save_intermediate_results(all_results, output_dir, experiment_name)

        if verbose:
            print(f"  Results (mean ± std over {len(seeds)} seeds):")
            print(f"    MAPE: {np.mean(test_mapes):.2f}% ± {np.std(test_mapes):.2f}%")
            print(f"    Dir Acc (binary): {np.mean(test_das):.2f}% ± {np.std(test_das):.2f}%")
            print(f"    Dir Acc (3-class): {np.mean(test_das_3c):.2f}% ± {np.std(test_das_3c):.2f}%")
            print(f"    Sharpe (binary): {np.mean(test_sharpes):.3f} ± {np.std(test_sharpes):.3f}")
            print(f"    Sharpe (3-class): {np.mean(test_sharpes_3c):.3f} ± {np.std(test_sharpes_3c):.3f}")

    # Compute statistical significance against baseline
    significance_results = compute_statistical_significance(all_results, baseline_model="real_lstm")

    return {
        'model_results': all_results,
        'statistical_significance': significance_results
    }


def print_results_table(results: Dict):
    """Print results in a formatted table with statistical significance."""
    model_results = results.get('model_results', results)
    significance = results.get('statistical_significance', {})

    print("\n" + "=" * 120)
    print("EXPERIMENT RESULTS")
    print("=" * 120)

    # Header
    print(f"{'Model':<30} {'MAPE (%)':<18} {'Dir Acc (%)':<18} {'Sharpe':<14} {'DA 3-cls (%)':<18} {'Sharpe 3-cls':<14}")
    print("-" * 120)

    # Rows
    for model_name, model_data in model_results.items():
        agg = model_data['aggregated']
        mape_str = f"{agg['mape']['mean']:.2f} ± {agg['mape']['std']:.2f}"
        da_str = f"{agg['directional_accuracy']['mean']:.2f} ± {agg['directional_accuracy']['std']:.2f}"
        sharpe_str = f"{agg['sharpe_ratio']['mean']:.3f} ± {agg['sharpe_ratio']['std']:.3f}"
        da3_str = f"{agg['directional_accuracy_3class']['mean']:.2f} ± {agg['directional_accuracy_3class']['std']:.2f}"
        sharpe3_str = f"{agg['sharpe_ratio_3class']['mean']:.3f} ± {agg['sharpe_ratio_3class']['std']:.3f}"
        print(f"{model_name:<30} {mape_str:<18} {da_str:<18} {sharpe_str:<14} {da3_str:<18} {sharpe3_str:<14}")

    print("=" * 120)

    # Print statistical significance if available
    if significance:
        print("\nSTATISTICAL SIGNIFICANCE (vs real_lstm baseline)")
        print("-" * 80)
        print(f"{'Model':<35} {'Metric':<24} {'p-value':<12} {'Cohens d':<12} {'Significant':<12}")
        print("-" * 80)

        for model_name, metrics in significance.items():
            for metric, stats in metrics.items():
                sig_marker = "**" if stats['significant_0.01'] else ("*" if stats['significant_0.05'] else "")
                print(f"{model_name:<35} {metric:<24} {stats['p_value']:<12.4f} {stats['cohens_d']:<12.3f} {sig_marker:<12}")

        print("-" * 80)
        print("* p < 0.05, ** p < 0.01")
        print("=" * 80)


def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def save_intermediate_results(results: Dict, output_dir: str, experiment_name: str):
    """Save intermediate results after each variant (overwrites same file)."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{experiment_name}_intermediate.json"
    filepath = os.path.join(output_dir, filename)

    # Statistical significance not computed yet for intermediate results
    intermediate = {
        'model_results': results,
        'status': 'in_progress',
        'completed_variants': list(results.keys())
    }

    serializable_results = convert_to_serializable(intermediate)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"  [Intermediate results saved to {filepath}]")


def save_results(results: Dict, output_dir: str, experiment_name: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    serializable_results = convert_to_serializable(results)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {filepath}")

    # Delete intermediate file after final save
    intermediate_path = os.path.join(output_dir, f"{experiment_name}_intermediate.json")
    if os.path.exists(intermediate_path):
        os.remove(intermediate_path)
        print(f"Intermediate file removed: {intermediate_path}")


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--base-config', type=str, default='configs/base.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--experiment-config', type=str, default='configs/experiment.yaml',
                        help='Path to experiment configuration file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (track gradients, weight stats)')
    args = parser.parse_args()

    # Load configs
    base_config = load_config(args.base_config)
    experiment_config = load_config(args.experiment_config)

    # Merge experiment config overrides into base config
    # This allows quick_experiment.yaml to override training settings
    if 'training' in experiment_config:
        for key, value in experiment_config['training'].items():
            base_config['training'][key] = value
            print(f"Override: training.{key} = {value}")

    print("=" * 80)
    print("Quaternion Neural Networks - Experiment Runner")
    print("=" * 80)
    print(f"Experiment: {experiment_config['experiment']['name']}")
    print(f"Description: {experiment_config['experiment']['description']}")
    print(f"Seeds: {experiment_config['experiment']['seeds']}")
    print()

    if args.debug:
        print("DEBUG MODE ENABLED - Tracking gradients and weight statistics")
        print()

    # Get output settings
    output_dir = experiment_config.get('output', {}).get('results_dir', 'experiments/results')
    experiment_name = experiment_config['experiment']['name']

    # Run experiments with rolling save
    results = run_experiment(
        config=base_config,
        experiment_config=experiment_config['experiment'],
        verbose=not args.quiet,
        debug=args.debug,
        output_dir=output_dir,
        experiment_name=experiment_name
    )

    # Print results table
    print_results_table(results)

    # Save final results (also cleans up intermediate file)
    save_results(results, output_dir, experiment_name)


if __name__ == "__main__":
    main()
