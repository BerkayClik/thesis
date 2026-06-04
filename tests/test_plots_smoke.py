"""Equity curve + drawdown PNG rendering from an equity series."""

import os

import numpy as np
import pandas as pd

from src.backtesting.plots import plot_equity_and_drawdown


def test_equity_and_drawdown_pngs_created(tmp_path):
    idx = pd.date_range("2024-01-01", periods=100, freq="4h")
    rng = np.random.default_rng(0)
    equity = pd.Series(10_000 + np.cumsum(rng.normal(0, 50, 100)), index=idx)

    eq_png, dd_png = plot_equity_and_drawdown(
        equity, outdir=str(tmp_path), label="test"
    )

    assert os.path.exists(eq_png)
    assert os.path.exists(dd_png)
    assert os.path.getsize(eq_png) > 1024
    assert os.path.getsize(dd_png) > 1024


def test_reads_equity_csv(tmp_path):
    idx = pd.date_range("2024-01-01", periods=50, freq="4h")
    equity = pd.Series(10_000 + np.arange(50) * 10.0, index=idx)
    csv = tmp_path / "equity.csv"
    equity.to_csv(csv, header=["value"])

    eq_png, dd_png = plot_equity_and_drawdown(
        str(csv), outdir=str(tmp_path), label="fromcsv"
    )
    assert os.path.getsize(eq_png) > 1024
    assert os.path.getsize(dd_png) > 1024
