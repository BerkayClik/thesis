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
from datetime import datetime
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_sp500_data, dataframe_to_tensor
from src.data.preprocessing import preprocess_data
from src.data.dataset import SP500Dataset
from src.models import RealLSTM, RealLSTMAttention, QNNAttentionModel
from src.models.qnn_attention_model import QuaternionLSTMNoAttention
from src.training.trainer import Trainer
from src.training.losses import mse_loss
from src.evaluation.metrics import compute_mae, compute_mse
from src.evaluation.directional_accuracy import compute_directional_accuracy
from src.evaluation.sharpe_ratio import compute_sharpe_ratio


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Dict) -> torch.device:
    """Get compute device from config."""
    device_name = config.get('device', 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    elif device_name == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device_name = 'cpu'
    return torch.device(device_name)


def create_model(model_type: str, hidden_size: int, num_layers: int, dropout: float = 0.0):
    """Create model based on type string."""
    if model_type == "real_lstm":
        return RealLSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "real_lstm_attention":
        return RealLSTMAttention(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "quaternion_lstm":
        return QuaternionLSTMNoAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "quaternion_lstm_attention":
        return QNNAttentionModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Evaluate model on a dataset.

    Returns dict with MAE, MSE, and directional accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_prevs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            pred = model(x).squeeze().cpu()
            prev = x[:, -1, 3].cpu()  # Last close price in window

            all_preds.append(pred)
            all_targets.append(y)
            all_prevs.append(prev)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    prevs = torch.cat(all_prevs)

    return {
        'mae': compute_mae(preds, targets),
        'mse': compute_mse(preds, targets),
        'directional_accuracy': compute_directional_accuracy(preds, targets, prevs),
        'sharpe_ratio': compute_sharpe_ratio(preds, targets, prevs)
    }


def run_single_experiment(
    config: Dict,
    model_config: Dict,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Dict:
    """
    Run a single experiment with one model configuration and seed.

    Returns results dictionary with training history and test metrics.
    """
    set_seed(seed)

    # Create model
    model = create_model(
        model_type=model_config['type'],
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.0)
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=mse_loss,
        device=device,
        checkpoint_dir=os.path.join(config.get('output', {}).get('results_dir', 'experiments/results'), 'checkpoints')
    )

    # Train
    if verbose:
        print(f"  Training with seed {seed}...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        verbose=False
    )

    # Load best model for evaluation
    try:
        trainer.load_checkpoint('best_model.pt')
    except FileNotFoundError:
        pass  # Use current model if no checkpoint

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)

    return {
        'seed': seed,
        'history': history,
        'test_metrics': test_metrics,
        'best_epoch': history['best_epoch'],
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
    }


def run_experiment(config: Dict, experiment_config: Dict, verbose: bool = True) -> Dict:
    """
    Run a complete experiment with multiple seeds.

    Returns aggregated results with mean and std across seeds.
    """
    device = get_device(config)

    # Load and preprocess data
    if verbose:
        print("Loading data...")

    df = load_sp500_data(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        cache_dir=config['data'].get('cache_dir', 'data/cache')
    )

    # Convert to tensor with dates
    data, dates = dataframe_to_tensor(df)

    # Preprocess
    processed = preprocess_data(
        data,
        dates=dates,
        train_end_year=config['data']['train_end_year'],
        val_end_year=config['data']['val_end_year']
    )
    train_data = processed['train_data']
    val_data = processed['val_data']
    test_data = processed['test_data']

    # Create datasets
    window_size = config['data']['window_size']
    train_dataset = SP500Dataset(train_data, window_size=window_size)
    val_dataset = SP500Dataset(val_data, window_size=window_size)
    test_dataset = SP500Dataset(test_data, window_size=window_size)

    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

        variant_results = []
        for seed in seeds:
            result = run_single_experiment(
                config=config,
                model_config=model_config,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                verbose=verbose
            )
            variant_results.append(result)

        # Aggregate results
        test_maes = [r['test_metrics']['mae'] for r in variant_results]
        test_mses = [r['test_metrics']['mse'] for r in variant_results]
        test_das = [r['test_metrics']['directional_accuracy'] for r in variant_results]
        test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in variant_results]

        all_results[variant_name] = {
            'individual_runs': variant_results,
            'aggregated': {
                'mae': {'mean': np.mean(test_maes), 'std': np.std(test_maes)},
                'mse': {'mean': np.mean(test_mses), 'std': np.std(test_mses)},
                'directional_accuracy': {'mean': np.mean(test_das), 'std': np.std(test_das)},
                'sharpe_ratio': {'mean': np.mean(test_sharpes), 'std': np.std(test_sharpes)}
            }
        }

        if verbose:
            print(f"  Results (mean ± std over {len(seeds)} seeds):")
            print(f"    MAE: {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
            print(f"    MSE: {np.mean(test_mses):.4f} ± {np.std(test_mses):.4f}")
            print(f"    Dir Acc: {np.mean(test_das):.2f}% ± {np.std(test_das):.2f}%")
            print(f"    Sharpe: {np.mean(test_sharpes):.3f} ± {np.std(test_sharpes):.3f}")

    return all_results


def print_results_table(results: Dict):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS")
    print("=" * 100)

    # Header
    print(f"{'Model':<28} {'MAE':<18} {'MSE':<18} {'Dir Acc (%)':<16} {'Sharpe':<16}")
    print("-" * 100)

    # Rows
    for model_name, model_results in results.items():
        agg = model_results['aggregated']
        mae_str = f"{agg['mae']['mean']:.4f} ± {agg['mae']['std']:.4f}"
        mse_str = f"{agg['mse']['mean']:.4f} ± {agg['mse']['std']:.4f}"
        da_str = f"{agg['directional_accuracy']['mean']:.2f} ± {agg['directional_accuracy']['std']:.2f}"
        sharpe_str = f"{agg['sharpe_ratio']['mean']:.3f} ± {agg['sharpe_ratio']['std']:.3f}"
        print(f"{model_name:<28} {mae_str:<18} {mse_str:<18} {da_str:<16} {sharpe_str:<16}")

    print("=" * 100)


def save_results(results: Dict, output_dir: str, experiment_name: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--base-config', type=str, default='configs/base.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--experiment-config', type=str, default='configs/experiment.yaml',
                        help='Path to experiment configuration file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    args = parser.parse_args()

    # Load configs
    base_config = load_config(args.base_config)
    experiment_config = load_config(args.experiment_config)

    print("=" * 80)
    print("Quaternion Neural Networks - Experiment Runner")
    print("=" * 80)
    print(f"Experiment: {experiment_config['experiment']['name']}")
    print(f"Description: {experiment_config['experiment']['description']}")
    print(f"Seeds: {experiment_config['experiment']['seeds']}")
    print()

    # Run experiments
    results = run_experiment(
        config=base_config,
        experiment_config=experiment_config['experiment'],
        verbose=not args.quiet
    )

    # Print results table
    print_results_table(results)

    # Save results
    output_dir = experiment_config.get('output', {}).get('results_dir', 'experiments/results')
    experiment_name = experiment_config['experiment']['name']
    save_results(results, output_dir, experiment_name)


if __name__ == "__main__":
    main()
