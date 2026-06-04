"""Smoke test for the isolated vectorbt backtest environment.

Run with the dedicated backtest venv (NOT the main thesis 3.13 env):

    .venv-backtest/bin/python scripts/backtest_env_smoke.py

Expected: prints "vbt OK <version>" and a finite Total Return [%] from a
minimal ``Portfolio.from_signals`` run on synthetic data, then exits 0.

In the main thesis env (numpy 2.x) this import FAILS by design, proving the
two environments are isolated.
"""

import sys


def main() -> int:
    try:
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
    except Exception as exc:  # noqa: BLE001 - smoke test reports any import failure
        print(f"IMPORT FAILED: {type(exc).__name__}: {exc}")
        return 1

    # Minimal synthetic price series (100 bars).
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    rng = np.random.default_rng(42)
    close = pd.Series(
        100.0 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="close"
    )

    # Simple alternating entries/exits to exercise the engine.
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    entries.iloc[::10] = True
    exits.iloc[5::10] = True

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=10_000,
        fees=0.001,
        slippage=0.0005,
        freq="4h",
    )

    total_return = float(pf.total_return()) * 100.0
    if not np.isfinite(total_return):
        print(f"NON-FINITE Total Return: {total_return}")
        return 1

    print(f"vbt OK {vbt.__version__}")
    print(f"numpy {np.__version__}  pandas {pd.__version__}")
    print(f"Total Return [%]: {total_return:.4f}")
    print(f"Sharpe Ratio: {float(pf.sharpe_ratio()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
