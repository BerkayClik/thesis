"""visualize_results must render figures even with a single non-naive variant.

The original code did `axes.flatten() if n_models > 1 else [axes]`, which for one
model returned `[ndarray_of_3_axes]` so `axes[0]` was an ndarray, not an Axes,
crashing with `'numpy.ndarray' object has no attribute 'plot'`.
"""

import os

import experiments.visualize_results as V


def _one_variant_results():
    return {
        "model_results": {
            "naive_zero": {
                "individual_runs": [{
                    "seed": 42,
                    "history": {"train_loss": [], "val_loss": []},
                    "test_metrics": {
                        "mape": 0.7, "directional_accuracy": 50.0,
                        "sharpe_ratio": 0.0, "directional_accuracy_3class": 50.0,
                        "sharpe_ratio_3class": 0.0,
                        "predictions": [100.0, 101.0, 102.0],
                        "targets": [100.5, 101.5, 101.0],
                        "prev_closes": [99.5, 100.0, 101.5],
                    },
                    "num_parameters": 0,
                }],
                "aggregated": {
                    m: {"mean": v, "std": 0.0} for m, v in
                    [("mape", 0.7), ("directional_accuracy", 50.0), ("sharpe_ratio", 0.0),
                     ("directional_accuracy_3class", 50.0), ("sharpe_ratio_3class", 0.0)]
                },
            },
            "hier_qlstm_concat": {
                "individual_runs": [{
                    "seed": 42,
                    "history": {"train_loss": [0.01, 0.008, 0.007], "val_loss": [0.012, 0.009, 0.008]},
                    "test_metrics": {
                        "mape": 2.0, "directional_accuracy": 51.0,
                        "sharpe_ratio": 0.02, "directional_accuracy_3class": 31.0,
                        "sharpe_ratio_3class": 0.01,
                        "predictions": [100.2, 101.3, 101.8],
                        "targets": [100.5, 101.5, 101.0],
                        "prev_closes": [99.5, 100.0, 101.5],
                    },
                    "num_parameters": 227553,
                }],
                "aggregated": {
                    m: {"mean": v, "std": 0.0} for m, v in
                    [("mape", 2.0), ("directional_accuracy", 51.0), ("sharpe_ratio", 0.02),
                     ("directional_accuracy_3class", 31.0), ("sharpe_ratio_3class", 0.01)]
                },
            },
        }
    }


def test_training_curves_single_variant(tmp_path):
    # naive_zero is filtered out -> exactly 1 trainable model (the bug trigger)
    V.plot_training_curves(_one_variant_results(), str(tmp_path))
    assert os.path.getsize(tmp_path / "training_curves.png") > 1024


def test_predictions_all_models_single_variant(tmp_path):
    V.plot_predictions_all_models(_one_variant_results(), str(tmp_path))
    assert os.path.getsize(tmp_path / "predictions_all_models.png") > 1024
