"""
Visualization script for experiment results.

Reads JSON results file and generates publication-ready figures for thesis.

Usage:
    python experiments/visualize_results.py --results experiments/results/experiment_YYYYMMDD_HHMMSS.json
    python experiments/visualize_results.py --results experiments/results/experiment_YYYYMMDD_HHMMSS.json --output figures/
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Color palette - colorblind friendly
COLORS = {
    'real_lstm': '#1f77b4',              # Blue
    'real_lstm_attention': '#2ca02c',    # Green
    'quaternion_lstm_param_matched': '#ff7f0e',  # Orange
    'quaternion_lstm_attention_param_matched': '#d62728',  # Red
    'quaternion_lstm': '#9467bd',        # Purple
    'quaternion_lstm_attention': '#8c564b',  # Brown
    'naive_zero': '#7f7f7f',             # Gray
}

# Shorter display names for plots
DISPLAY_NAMES = {
    'naive_zero': 'Naive (Zero)',
    'real_lstm': 'Real LSTM',
    'real_lstm_attention': 'Real LSTM + Attn',
    'quaternion_lstm_param_matched': 'Quat LSTM (PM)',
    'quaternion_lstm_attention_param_matched': 'Quat LSTM + Attn (PM)',
    'quaternion_lstm': 'Quat LSTM (LM)',
    'quaternion_lstm_attention': 'Quat LSTM + Attn (LM)',
}


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_metric_stats(model_data: Dict, metric: str) -> tuple:
    """
    Get mean and std for a metric, handling different JSON structures.

    Supports both:
    - Flat: model_data['mape_mean'], model_data['mape_std']
    - Nested: model_data['aggregated']['mape']['mean'], model_data['aggregated']['mape']['std']
    """
    # Try nested structure first (aggregated)
    if 'aggregated' in model_data:
        agg = model_data['aggregated']
        if metric in agg:
            return agg[metric].get('mean', 0), agg[metric].get('std', 0)

    # Try flat structure
    mean = model_data.get(f'{metric}_mean', 0)
    std = model_data.get(f'{metric}_std', 0)

    return mean, std


def plot_training_curves(results: Dict, output_dir: str, figsize=(14, 10)):
    """
    Plot training and validation loss curves for all models.

    Creates a grid of subplots, one per model variant.
    """
    model_results = results.get('model_results', results)

    # Filter out naive_zero (no training)
    models = [m for m in model_results.keys() if m != 'naive_zero']
    n_models = len(models)

    if n_models == 0:
        print("No trainable models found for training curves.")
        return

    # Create subplot grid
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        model_data = model_results[model_name]

        # Plot each seed's training curve
        for run in model_data['individual_runs']:
            seed = run['seed']
            history = run['history']
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            epochs = range(1, len(train_loss) + 1)

            ax.plot(epochs, train_loss, alpha=0.7, linestyle='-',
                   label=f'Train (seed={seed})')
            ax.plot(epochs, val_loss, alpha=0.7, linestyle='--',
                   label=f'Val (seed={seed})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(DISPLAY_NAMES.get(model_name, model_name))
        ax.legend(fontsize=8)
        ax.set_yscale('log')  # Log scale often better for loss

    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Training and Validation Loss Curves', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_metric_comparison(results: Dict, output_dir: str, figsize=(18, 10)):
    """
    Bar chart comparing all metrics across models with error bars.
    """
    model_results = results.get('model_results', results)

    metrics = ['mape', 'directional_accuracy_3class', 'sharpe_ratio_3class']
    metric_labels = {
        'mape': 'MAPE % (Lower is Better)',
        'directional_accuracy_3class': 'Dir. Acc. 3-Class % (Higher is Better)',
        'sharpe_ratio_3class': 'Sharpe 3-Class (Higher is Better)'
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = axes.flatten()

    models = list(model_results.keys())
    x = np.arange(len(models))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = []
        stds = []
        colors = []

        for model_name in models:
            model_data = model_results[model_name]
            mean, std = get_metric_stats(model_data, metric)
            means.append(mean)
            stds.append(std)
            colors.append(COLORS.get(model_name, '#333333'))

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

        ax.set_ylabel(metric_labels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models],
                          rotation=45, ha='right')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'metric_comparison.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_predictions_vs_actual(results: Dict, output_dir: str,
                                model_name: str = 'quaternion_lstm_param_matched',
                                seed_idx: int = 0,
                                figsize=(14, 6)):
    """
    Plot predicted vs actual values for a specific model.

    Shows time series of predictions overlaid on actual values.
    """
    model_results = results.get('model_results', results)

    if model_name not in model_results:
        print(f"Model {model_name} not found. Available: {list(model_results.keys())}")
        return

    model_data = model_results[model_name]
    runs = model_data['individual_runs']

    if seed_idx >= len(runs):
        print(f"Seed index {seed_idx} out of range. Available: {len(runs)} seeds")
        return

    run = runs[seed_idx]
    test_metrics = run['test_metrics']

    if 'predictions' not in test_metrics or 'targets' not in test_metrics:
        print("Predictions not saved in results. Re-run experiments to include predictions.")
        return

    predictions = np.array(test_metrics['predictions'])
    targets = np.array(test_metrics['targets'])

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # Plot 1: Time series comparison
    ax1 = axes[0]
    time_idx = np.arange(len(targets))

    ax1.plot(time_idx, targets, label='Actual', color='#1f77b4', alpha=0.8, linewidth=1.5)
    ax1.plot(time_idx, predictions, label='Predicted', color='#ff7f0e', alpha=0.8, linewidth=1.5)

    ax1.set_xlabel('Time Step (Test Set)')
    ax1.set_ylabel('Close Price (Denormalized)')
    ax1.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)} - Predictions vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction error over time
    ax2 = axes[1]
    errors = predictions - targets

    ax2.fill_between(time_idx, errors, 0, alpha=0.5,
                     color='green', where=(errors >= 0), label='Overpredict')
    ax2.fill_between(time_idx, errors, 0, alpha=0.5,
                     color='red', where=(errors < 0), label='Underpredict')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Time Step (Test Set)')
    ax2.set_ylabel('Prediction Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, f'predictions_vs_actual_{model_name}.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_predictions_all_models(results: Dict, output_dir: str,
                                 seed_idx: int = 0, figsize=(16, 12)):
    """
    Plot predictions vs actual for all models in a grid.
    """
    model_results = results.get('model_results', results)

    # Filter models with predictions
    models_with_preds = []
    for model_name, model_data in model_results.items():
        runs = model_data['individual_runs']
        if runs and 'predictions' in runs[0].get('test_metrics', {}):
            models_with_preds.append(model_name)

    if not models_with_preds:
        print("No models have predictions saved.")
        return

    n_models = len(models_with_preds)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_name in enumerate(models_with_preds):
        ax = axes[idx]

        model_data = model_results[model_name]
        run = model_data['individual_runs'][seed_idx]
        test_metrics = run['test_metrics']

        predictions = np.array(test_metrics['predictions'])
        targets = np.array(test_metrics['targets'])
        time_idx = np.arange(len(targets))

        ax.plot(time_idx, targets, label='Actual', color='#1f77b4', alpha=0.7, linewidth=1)
        ax.plot(time_idx, predictions, label='Predicted', color='#ff7f0e', alpha=0.7, linewidth=1)

        # Add metrics to title
        mape = test_metrics['mape']
        da = test_metrics.get('directional_accuracy_3class', test_metrics.get('directional_accuracy', 0))
        ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}\nMAPE: {mape:.2f}%, DA 3C: {da:.1f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Predictions vs Actual - All Models', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'predictions_all_models.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_box_plots(results: Dict, output_dir: str, figsize=(18, 10)):
    """
    Box plots showing distribution of metrics across seeds.
    """
    model_results = results.get('model_results', results)

    metrics = ['mape', 'directional_accuracy_3class', 'sharpe_ratio_3class']
    metric_labels = {
        'mape': 'MAPE (%)',
        'directional_accuracy_3class': 'Dir. Acc. 3-Class (%)',
        'sharpe_ratio_3class': 'Sharpe 3-Class'
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = axes.flatten()

    models = list(model_results.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        data = []
        labels = []
        colors = []

        for model_name in models:
            model_data = model_results[model_name]
            runs = model_data['individual_runs']
            values = [run['test_metrics'][metric] for run in runs]
            data.append(values)
            labels.append(DISPLAY_NAMES.get(model_name, model_name))
            colors.append(COLORS.get(model_name, '#333333'))

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(metric_labels[metric])
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Metric Distribution Across Seeds', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'box_plots.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_significance_heatmap(results: Dict, output_dir: str, figsize=(12, 8)):
    """
    Heatmap showing p-values for statistical significance tests.
    """
    significance = results.get('statistical_significance', {})

    if not significance:
        print("No statistical significance data found.")
        return

    metrics = ['mape', 'directional_accuracy_3class', 'sharpe_ratio_3class']
    models = list(significance.keys())

    # Create p-value matrix
    p_values = np.zeros((len(models), len(metrics)))

    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            if metric in significance[model]:
                p_values[i, j] = significance[model][metric].get('p_value', 1.0)
            else:
                p_values[i, j] = 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: green for significant, red for not significant
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(p_values, annot=True, fmt='.3f', cmap=cmap,
                xticklabels=[m.upper() for m in metrics],
                yticklabels=[DISPLAY_NAMES.get(m, m) for m in models],
                vmin=0, vmax=0.1, center=0.05,
                cbar_kws={'label': 'p-value'},
                ax=ax)

    ax.set_title('Statistical Significance vs Real LSTM Baseline\n(p < 0.05 is significant)')

    # Add significance markers
    for i in range(len(models)):
        for j in range(len(metrics)):
            p = p_values[i, j]
            if p < 0.01:
                ax.text(j + 0.5, i + 0.7, '**', ha='center', va='center',
                       fontsize=14, fontweight='bold')
            elif p < 0.05:
                ax.text(j + 0.5, i + 0.7, '*', ha='center', va='center',
                       fontsize=14, fontweight='bold')

    plt.tight_layout()

    filepath = os.path.join(output_dir, 'significance_heatmap.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_parameter_efficiency(results: Dict, output_dir: str, figsize=(10, 6)):
    """
    Scatter plot showing performance vs parameter count.
    """
    model_results = results.get('model_results', results)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    metrics = [('mape', 'MAPE % (Lower is Better)'),
               ('directional_accuracy_3class', 'Dir. Acc. 3-Class % (Higher is Better)')]

    for ax, (metric, label) in zip(axes, metrics):
        for model_name, model_data in model_results.items():
            if model_name == 'naive_zero':
                continue

            runs = model_data['individual_runs']
            if not runs:
                continue

            num_params = runs[0].get('num_parameters', 0)
            mean_val, std_val = get_metric_stats(model_data, metric)

            color = COLORS.get(model_name, '#333333')
            ax.errorbar(num_params / 1000, mean_val, yerr=std_val,
                       fmt='o', markersize=10, capsize=5,
                       color=color, label=DISPLAY_NAMES.get(model_name, model_name))

        ax.set_xlabel('Parameters (K)')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Parameter Efficiency Analysis', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'parameter_efficiency.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def plot_radar_chart(results: Dict, output_dir: str, figsize=(10, 10)):
    """
    Radar/spider chart comparing models across all metrics.
    Uses 3-class variants as primary directional/sharpe metrics.
    """
    model_results = results.get('model_results', results)

    metrics = ['mape', 'directional_accuracy_3class', 'sharpe_ratio_3class']
    metric_labels = ['MAPE', 'Dir. Acc. (3-Class)', 'Sharpe (3-Class)']

    # Normalize metrics to 0-1 scale (inverted for lower-is-better metrics)
    all_values = {metric: [] for metric in metrics}
    for model_data in model_results.values():
        for metric in metrics:
            mean_val, _ = get_metric_stats(model_data, metric)
            all_values[metric].append(mean_val)

    # Calculate min/max for normalization
    ranges = {}
    for metric in metrics:
        vals = all_values[metric]
        ranges[metric] = (min(vals), max(vals))

    def normalize(value, metric):
        min_val, max_val = ranges[metric]
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        # Invert for lower-is-better metrics
        if metric in ['mape']:
            normalized = 1 - normalized
        return normalized

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for model_name, model_data in model_results.items():
        values = [normalize(get_metric_stats(model_data, m)[0], m) for m in metrics]
        values += values[:1]  # Complete the circle

        color = COLORS.get(model_name, '#333333')
        ax.plot(angles, values, 'o-', linewidth=2,
               label=DISPLAY_NAMES.get(model_name, model_name), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Comparison (Normalized, Higher is Better)')

    plt.tight_layout()

    filepath = os.path.join(output_dir, 'radar_chart.png')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.png', '.pdf'))
    print(f"Saved: {filepath}")
    plt.close()


def generate_all_figures(results: Dict, output_dir: str):
    """Generate all figures."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating figures in: {output_dir}")
    print("=" * 50)

    # Generate all plots
    plot_training_curves(results, output_dir)
    plot_metric_comparison(results, output_dir)
    plot_box_plots(results, output_dir)
    plot_significance_heatmap(results, output_dir)
    plot_parameter_efficiency(results, output_dir)
    plot_radar_chart(results, output_dir)

    # Predictions plots (if available)
    plot_predictions_all_models(results, output_dir)

    # Individual prediction plots for key models
    for model in ['real_lstm', 'quaternion_lstm_param_matched']:
        plot_predictions_vs_actual(results, output_dir, model_name=model)

    print("=" * 50)
    print("All figures generated successfully!")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='experiments/figures',
                        help='Output directory for figures')
    parser.add_argument('--only', type=str, default=None,
                        help='Generate only specific plot (training, metrics, predictions, box, significance, efficiency, radar)')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)

    if args.only:
        os.makedirs(args.output, exist_ok=True)
        plot_funcs = {
            'training': plot_training_curves,
            'metrics': plot_metric_comparison,
            'predictions': plot_predictions_all_models,
            'box': plot_box_plots,
            'significance': plot_significance_heatmap,
            'efficiency': plot_parameter_efficiency,
            'radar': plot_radar_chart,
        }
        if args.only in plot_funcs:
            plot_funcs[args.only](results, args.output)
        else:
            print(f"Unknown plot type: {args.only}")
            print(f"Available: {list(plot_funcs.keys())}")
    else:
        generate_all_figures(results, args.output)


if __name__ == '__main__':
    main()
