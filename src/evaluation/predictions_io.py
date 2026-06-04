"""Per-bar predictions CSV writer for leakage-free backtest alignment (Oracle O1).

Each row pins a single prediction to BOTH the decision bar (``decision_time``,
the last observed bar t) and the target bar (``target_time``, the predicted bar
t+1), alongside the raw prev/pred/true closes and their derived returns. The
backtest engine consumes this file to build next-open-execution signals without
inferring timestamps from dropped index information.
"""

import csv

import numpy as np
import pandas as pd

HEADER = [
    "decision_time", "target_time", "prev_close",
    "pred_close", "true_close", "pred_return", "true_return",
]


def _iso_times(values):
    """Render any timestamp-like array (pandas Timestamp/Index or numpy
    datetime64) to a list of ISO-8601 strings for CSV output."""
    return [pd.Timestamp(v).isoformat() for v in values]


def write_predictions_csv(
    path,
    decision_times,
    target_times,
    prev_closes,
    pred_closes,
    true_closes,
):
    prev = np.asarray(prev_closes, dtype=np.float64)
    pred = np.asarray(pred_closes, dtype=np.float64)
    true = np.asarray(true_closes, dtype=np.float64)

    lengths = {
        len(decision_times), len(target_times), len(prev), len(pred), len(true),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"All inputs must share one length; got "
            f"{[len(decision_times), len(target_times), len(prev), len(pred), len(true)]}"
        )

    decision_iso = _iso_times(decision_times)
    target_iso = _iso_times(target_times)
    pred_return = pred / prev - 1.0
    true_return = true / prev - 1.0

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i in range(len(prev)):
            writer.writerow([
                decision_iso[i],
                target_iso[i],
                f"{prev[i]:.10g}",
                f"{pred[i]:.10g}",
                f"{true[i]:.10g}",
                f"{pred_return[i]:.10g}",
                f"{true_return[i]:.10g}",
            ])
