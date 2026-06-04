"""Comparison figures across variants: backtest bars, signal grid, equity overlay."""

import os

import numpy as np
import pandas as pd

from src.backtesting.compare import (
    plot_backtest_comparison,
    plot_signal_quality_grid,
    plot_equity_overlay,
)


def _stats(tr, sharpe, mdd, wr, trades=10):
    return {
        "Total Return [%]": tr, "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": mdd, "Win Rate [%]": wr,
        "Benchmark Return [%]": -26.0, "# Trades": trades,
    }


def test_backtest_comparison_bars(tmp_path):
    per_variant = {
        "real_lstm": _stats(-8.9, -1.4, 19.8, 52.9),
        "hier_concat": _stats(-16.0, -2.4, 24.3, 35.9),
        "quaternion": _stats(2.1, 0.3, 12.0, 55.0),
    }
    p = plot_backtest_comparison(per_variant, outdir=str(tmp_path), label="cmp")
    assert os.path.exists(p)
    assert os.path.getsize(p) > 1024


def test_backtest_comparison_handles_none_and_inf(tmp_path):
    # A non-trading variant yields None/NaN sharpe + missing fields; must not crash.
    per_variant = {
        "traded": _stats(-8.9, -1.4, 19.8, 52.9),
        "no_trade": {
            "Total Return [%]": 0.0, "Sharpe Ratio": None,
            "Max Drawdown [%]": None, "Win Rate [%]": None,
            "Benchmark Return [%]": -26.0,
        },
    }
    p = plot_backtest_comparison(per_variant, outdir=str(tmp_path), label="cmpnan")
    assert os.path.getsize(p) > 1024


def test_signal_quality_grid(tmp_path):
    rng = np.random.default_rng(0)
    per_variant = {}
    for name in ["a", "b", "c"]:
        n = 100
        tr = rng.normal(0, 0.01, n)
        pr = tr * 0.3 + rng.normal(0, 0.01, n)
        per_variant[name] = pd.DataFrame({"pred_return": pr, "true_return": tr})
    p = plot_signal_quality_grid(per_variant, outdir=str(tmp_path), label="sig")
    assert os.path.getsize(p) > 1024


def test_equity_overlay(tmp_path):
    idx = pd.date_range("2024-01-01", periods=50, freq="4h")
    per_variant = {
        "a": pd.Series(10000 + np.arange(50) * 5.0, index=idx),
        "b": pd.Series(10000 - np.arange(50) * 3.0, index=idx),
    }
    benchmark = pd.Series(10000 + np.sin(np.arange(50) / 5) * 200, index=idx)
    p = plot_equity_overlay(per_variant, benchmark=benchmark, outdir=str(tmp_path), label="eq")
    assert os.path.getsize(p) > 1024


def test_equity_overlay_no_benchmark(tmp_path):
    idx = pd.date_range("2024-01-01", periods=20, freq="4h")
    per_variant = {"a": pd.Series(10000 + np.arange(20) * 5.0, index=idx)}
    p = plot_equity_overlay(per_variant, benchmark=None, outdir=str(tmp_path), label="eq2")
    assert os.path.getsize(p) > 1024
