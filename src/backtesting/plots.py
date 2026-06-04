"""Equity curve and drawdown PNG rendering for backtest results."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _as_series(equity):
    if isinstance(equity, pd.Series):
        return equity
    df = pd.read_csv(equity, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0]


def plot_equity_and_drawdown(equity, outdir, label="backtest"):
    """Render equity (linear + log) and drawdown PNGs; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    eq = _as_series(equity)

    running_max = eq.cummax()
    drawdown = (eq / running_max - 1.0) * 100.0

    eq_png = os.path.join(outdir, f"{label}_equity.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(eq.index, eq.to_numpy(), color="#1f77b4", linewidth=1.4)
    ax1.set_ylabel("Equity")
    ax1.set_title(f"{label} — Equity Curve")
    ax1.grid(True, alpha=0.3)
    ax2.plot(eq.index, eq.to_numpy(), color="#1f77b4", linewidth=1.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("Equity (log)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(eq_png, dpi=120)
    plt.close(fig)

    dd_png = os.path.join(outdir, f"{label}_drawdown.png")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0.0,
                    color="#d62728", alpha=0.4)
    ax.plot(drawdown.index, drawdown.to_numpy(), color="#d62728", linewidth=1.0)
    ax.set_ylabel("Drawdown [%]")
    ax.set_xlabel("Time")
    ax.set_title(f"{label} — Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dd_png, dpi=120)
    plt.close(fig)

    return eq_png, dd_png
