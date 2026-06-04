"""Per-bar predictions CSV writer (Oracle O1 timestamp export).

Schema (exact column order):
    decision_time, target_time, prev_close, pred_close, true_close,
    pred_return, true_return
"""

import csv

import numpy as np

from src.evaluation.predictions_io import write_predictions_csv

EXPECTED_HEADER = [
    "decision_time", "target_time", "prev_close",
    "pred_close", "true_close", "pred_return", "true_return",
]


def _read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_csv_schema_and_rows(tmp_path):
    out = tmp_path / "preds.csv"
    write_predictions_csv(
        str(out),
        decision_times=np.array(["2024-01-01T00", "2024-01-01T04", "2024-01-01T08"], dtype="datetime64[h]"),
        target_times=np.array(["2024-01-01T04", "2024-01-01T08", "2024-01-01T12"], dtype="datetime64[h]"),
        prev_closes=[100.0, 105.0, 110.0],
        pred_closes=[103.0, 108.0, 112.0],
        true_closes=[105.0, 110.0, 109.0],
    )
    rows = _read(str(out))
    assert rows[0] == EXPECTED_HEADER
    assert len(rows) == 4  # header + 3 data rows


def test_pred_and_true_return_derived(tmp_path):
    out = tmp_path / "preds.csv"
    write_predictions_csv(
        str(out),
        decision_times=np.array(["2024-01-01T00"], dtype="datetime64[h]"),
        target_times=np.array(["2024-01-01T04"], dtype="datetime64[h]"),
        prev_closes=[100.0],
        pred_closes=[105.0],
        true_closes=[110.0],
    )
    rows = _read(str(out))
    header, data = rows[0], rows[1]
    rec = dict(zip(header, data))
    # pred_return = 105/100 - 1 = 0.05 ; true_return = 110/100 - 1 = 0.10
    assert abs(float(rec["pred_return"]) - 0.05) < 1e-9
    assert abs(float(rec["true_return"]) - 0.10) < 1e-9


def test_target_time_strictly_after_decision_time(tmp_path):
    out = tmp_path / "preds.csv"
    write_predictions_csv(
        str(out),
        decision_times=np.array(["2024-01-01T00", "2024-01-01T04"], dtype="datetime64[h]"),
        target_times=np.array(["2024-01-01T04", "2024-01-01T08"], dtype="datetime64[h]"),
        prev_closes=[100.0, 105.0],
        pred_closes=[103.0, 108.0],
        true_closes=[105.0, 110.0],
    )
    rows = _read(str(out))[1:]
    for r in rows:
        dt = np.datetime64(r[0])
        tt = np.datetime64(r[1])
        assert tt > dt


def test_length_mismatch_raises(tmp_path):
    out = tmp_path / "preds.csv"
    try:
        write_predictions_csv(
            str(out),
            decision_times=np.array(["2024-01-01T00"], dtype="datetime64[h]"),
            target_times=np.array(["2024-01-01T04"], dtype="datetime64[h]"),
            prev_closes=[100.0, 105.0],  # mismatched length
            pred_closes=[103.0],
            true_closes=[105.0],
        )
        assert False, "expected ValueError on length mismatch"
    except ValueError:
        pass
