"""Cross-variant comparison figures: backtest bars, signal-quality grid,
equity overlay vs buy-and-hold. Pure matplotlib (no vectorbt dependency)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _num(v):
    """Coerce None / non-finite stats to NaN so matplotlib bars render."""
    if v is None:
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


def plot_backtest_comparison(per_variant_stats, outdir, label="backtest"):
    """Grouped bar charts of Total Return / Sharpe / Max Drawdown / Win Rate
    across variants. ``per_variant_stats`` maps variant name -> stats dict."""
    os.makedirs(outdir, exist_ok=True)
    names = list(per_variant_stats)
    metrics = [
        ("Total Return [%]", "Total Return [%]", "#1f77b4"),
        ("Sharpe Ratio", "Sharpe Ratio", "#ff7f0e"),
        ("Max Drawdown [%]", "Max Drawdown [%]", "#d62728"),
        ("Win Rate [%]", "Win Rate [%]", "#2ca02c"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, (key, title, color) in zip(axes, metrics):
        vals = [_num(per_variant_stats[n].get(key)) for n in names]
        ax.bar(range(len(names)), vals, color=color, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        if key == "Total Return [%]":
            bench = _num(per_variant_stats[names[0]].get("Benchmark Return [%]"))
            if np.isfinite(bench):
                ax.axhline(bench, color="grey", ls="--", lw=1.2,
                           label=f"Buy&Hold {bench:.1f}%")
                ax.legend(fontsize=8)
    fig.suptitle(f"{label} — Backtest comparison", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, f"{label}_backtest_comparison.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_signal_quality_grid(per_variant_preds, outdir, label="signal"):
    """Grid of predicted-vs-true return scatters, one per variant, each
    annotated with correlation + directional agreement. ``per_variant_preds``
    maps variant name -> DataFrame with pred_return/true_return columns."""
    os.makedirs(outdir, exist_ok=True)
    names = list(per_variant_preds)
    n = len(names)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                             squeeze=False)
    axes = axes.flatten()
    for ax, name in zip(axes, names):
        df = per_variant_preds[name]
        tr = np.asarray(df["true_return"], float) * 100
        pr = np.asarray(df["pred_return"], float) * 100
        ax.scatter(tr, pr, s=8, alpha=0.4, color="#2ca02c")
        lim = max(np.abs(tr).max(), np.abs(pr).max(), 1e-9)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
        corr = np.corrcoef(tr, pr)[0, 1] if tr.std() and pr.std() else float("nan")
        da = (np.sign(pr) == np.sign(tr)).mean() * 100
        ax.set_title(f"{name}\ncorr={corr:.3f}, dir={da:.0f}%", fontsize=9)
        ax.set_xlabel("True ret [%]", fontsize=8)
        ax.set_ylabel("Pred ret [%]", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{label} — Signal quality (pred vs true return)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, f"{label}_signal_quality_grid.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_equity_overlay(per_variant_equity, outdir, benchmark=None, label="equity"):
    """All strategy equity curves on one axis, optionally with a buy-and-hold
    benchmark. ``per_variant_equity`` maps variant name -> equity Series."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, eq in per_variant_equity.items():
        ax.plot(eq.index, eq.to_numpy(), lw=1.2, alpha=0.85, label=name)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.to_numpy(), color="k", lw=1.6,
                ls="--", label="Buy & Hold")
    ax.set_ylabel("Equity"); ax.set_xlabel("Time")
    ax.set_title(f"{label} — Strategy equity vs buy & hold")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f"{label}_equity_overlay.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path
