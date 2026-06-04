"""Oracle O5 alignment invariants over a predictions CSV.

check_alignment_invariants returns a list of violation strings (empty = OK):
  - target_time strictly after decision_time on every row
  - constant bar gap when require_constant_gap=True
  - pred_close == prev_close * (1 + pred_return)   (reconstruction identity)
  - true_close == prev_close * (1 + true_return)
"""

import pandas as pd

from src.evaluation.verify_alignment import check_alignment_invariants


def _frame(rows):
    return pd.DataFrame(rows)


def _good_rows():
    return _frame({
        "decision_time": pd.to_datetime(["2024-01-01T00:00", "2024-01-01T04:00"]),
        "target_time": pd.to_datetime(["2024-01-01T04:00", "2024-01-01T08:00"]),
        "prev_close": [100.0, 105.0],
        "pred_close": [103.0, 108.15],
        "true_close": [105.0, 110.0],
        "pred_return": [0.03, 0.03],
        "true_return": [0.05, 0.047619047619],
    })


def test_clean_frame_has_no_violations():
    violations = check_alignment_invariants(_good_rows())
    assert violations == []


def test_detects_target_not_after_decision():
    df = _good_rows()
    # break ordering on row 1
    df.loc[1, "target_time"] = df.loc[1, "decision_time"]
    violations = check_alignment_invariants(df)
    assert any("target_time" in v for v in violations)


def test_detects_non_constant_gap():
    df = _good_rows()
    df.loc[1, "target_time"] = pd.Timestamp("2024-01-01T10:00")  # 6h gap vs 4h
    violations = check_alignment_invariants(df, require_constant_gap=True)
    assert any("gap" in v.lower() for v in violations)


def test_detects_broken_pred_reconstruction():
    df = _good_rows()
    df.loc[0, "pred_close"] = 999.0  # no longer prev*(1+pred_return)
    violations = check_alignment_invariants(df)
    assert any("pred_close" in v for v in violations)


def test_detects_broken_true_reconstruction():
    df = _good_rows()
    df.loc[0, "true_close"] = 999.0
    violations = check_alignment_invariants(df)
    assert any("true_close" in v for v in violations)
