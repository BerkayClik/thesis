"""Oracle O5 alignment invariants for a predictions CSV.

These checks are the gate before any backtest: a single off-by-one in the
decision/target timestamp alignment, or a broken price reconstruction, would
silently invalidate every downstream Sharpe / drawdown / return number.
"""

import numpy as np
import pandas as pd


def check_alignment_invariants(df, require_constant_gap=False, rtol=1e-6):
    """Return a list of human-readable violation strings (empty list = OK).

    Args:
        df: DataFrame with columns decision_time, target_time, prev_close,
            pred_close, true_close, pred_return, true_return.
        require_constant_gap: if True, every target-minus-decision gap must
            equal the first gap (fixed-frequency bars).
        rtol: relative tolerance for the reconstruction identities.
    """
    violations = []

    decision = pd.to_datetime(df["decision_time"])
    target = pd.to_datetime(df["target_time"])

    if not (target > decision).all():
        bad = int((~(target > decision)).sum())
        violations.append(f"target_time not strictly after decision_time on {bad} row(s)")

    if require_constant_gap and len(df) > 0:
        gaps = (target - decision).dt.total_seconds().to_numpy()
        if not np.allclose(gaps, gaps[0]):
            uniq = sorted(set(np.round(gaps / 3600.0, 6)))
            violations.append(f"non-constant bar gap (hours): {uniq}")

    prev = df["prev_close"].to_numpy(dtype=np.float64)
    pred = df["pred_close"].to_numpy(dtype=np.float64)
    true = df["true_close"].to_numpy(dtype=np.float64)
    pred_ret = df["pred_return"].to_numpy(dtype=np.float64)
    true_ret = df["true_return"].to_numpy(dtype=np.float64)

    if not np.allclose(pred, prev * (1.0 + pred_ret), rtol=rtol):
        n = int((~np.isclose(pred, prev * (1.0 + pred_ret), rtol=rtol)).sum())
        violations.append(f"pred_close != prev_close*(1+pred_return) on {n} row(s)")

    if not np.allclose(true, prev * (1.0 + true_ret), rtol=rtol):
        n = int((~np.isclose(true, prev * (1.0 + true_ret), rtol=rtol)).sum())
        violations.append(f"true_close != prev_close*(1+true_return) on {n} row(s)")

    return violations
