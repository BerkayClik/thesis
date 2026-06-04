"""Oracle O5 gate: print 5 hand-verifiable rows from a predictions CSV and
enforce the alignment invariants. Exits non-zero on any violation.

Usage:
    python scripts/verify_alignment.py <predictions.csv> [--rows N]

This is the human checkpoint before any backtest is built: a reviewer reads the
printed rows and confirms decision/target timestamps and the price
reconstruction line up bar-by-bar.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.evaluation.verify_alignment import check_alignment_invariants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--constant-gap", action="store_true",
                        help="require a fixed bar gap (intraday data)")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)

    cols = ["decision_time", "target_time", "prev_close",
            "pred_close", "true_close", "pred_return", "true_return"]
    print(f"\nFile: {args.predictions_csv}")
    print(f"Rows: {len(df)}\n")
    print("First", args.rows, "rows (hand-verify decision->target gap and reconstruction):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].head(args.rows).to_string(index=False))

    violations = check_alignment_invariants(df, require_constant_gap=args.constant_gap)
    print()
    if violations:
        print("ALIGNMENT VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("ALIGNMENT OK: all invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
