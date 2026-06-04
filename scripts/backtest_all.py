"""Backtest every variant in a return-mode experiment results dir (one seed),
then render cross-variant comparison figures.

Runs in the isolated backtest env (needs vectorbt):

    .venv-backtest/bin/python scripts/backtest_all.py \
        --results-dir experiments/results/4hourly_btc_hier_return \
        --ohlc data/cache/lunarcrush_btc_4hour_full.csv \
        --seed 42

For each variant it runs a single-coin next-open backtest, writes per-variant
stats/equity, and emits:
  - <label>_backtest_comparison.png  (Return/Sharpe/MaxDD/WinRate bars)
  - <label>_signal_quality_grid.png  (pred vs true return scatter per variant)
  - <label>_equity_overlay.png       (all strategies vs buy & hold)
  - <label>_summary.csv              (one row per variant)
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.backtesting.vbt_adapter import load_ohlc, run_single_coin_backtest, _stats_dict
from src.backtesting.compare import (
    plot_backtest_comparison,
    plot_signal_quality_grid,
    plot_equity_overlay,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--ohlc", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--fees", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--init-cash", type=float, default=10_000.0)
    p.add_argument("--freq", default="4h")
    args = p.parse_args()

    label = os.path.basename(args.results_dir.rstrip("/"))
    outdir = os.path.join(args.results_dir, "backtest_all")
    os.makedirs(outdir, exist_ok=True)

    csvs = sorted(glob.glob(f"{args.results_dir}/*_seed{args.seed}_predictions.csv"))
    if not csvs:
        print(f"No seed-{args.seed} predictions in {args.results_dir}")
        return 1

    ohlc = load_ohlc(args.ohlc)

    per_stats, per_preds, per_equity = {}, {}, {}
    rows = []
    benchmark_equity = None

    for csv in csvs:
        variant = os.path.basename(csv).replace(f"_seed{args.seed}_predictions.csv", "")
        preds = pd.read_csv(csv, parse_dates=["decision_time", "target_time"])
        try:
            pf, sig = run_single_coin_backtest(
                preds, ohlc, threshold=args.threshold, init_cash=args.init_cash,
                fees=args.fees, slippage=args.slippage, freq=args.freq,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue across variants
            print(f"[{variant}] backtest failed: {exc}")
            continue

        stats = _stats_dict(pf)
        per_stats[variant] = stats
        per_preds[variant] = preds
        eq = pf.value()
        per_equity[variant] = eq

        # Buy & hold benchmark equity (same for all; build once from first variant)
        if benchmark_equity is None:
            exec_price = pd.Series(sig["exec_price"].to_numpy(), index=sig.index)
            benchmark_equity = args.init_cash * exec_price / exec_price.iloc[0]

        pr = np.asarray(preds["pred_return"], float)
        tr = np.asarray(preds["true_return"], float)
        corr = np.corrcoef(pr, tr)[0, 1] if pr.std() and tr.std() else float("nan")
        da = float((np.sign(pr) == np.sign(tr)).mean() * 100)
        rows.append({
            "variant": variant,
            "total_return_pct": stats.get("Total Return [%]"),
            "benchmark_return_pct": stats.get("Benchmark Return [%]"),
            "sharpe": stats.get("Sharpe Ratio"),
            "max_drawdown_pct": stats.get("Max Drawdown [%]"),
            "win_rate_pct": stats.get("Win Rate [%]"),
            "trades": stats.get("# Trades"),
            "corr": round(corr, 4),
            "dir_agree_pct": round(da, 1),
        })
        def _fmt(x, d=2):
            return f"{x:.{d}f}" if isinstance(x, (int, float)) else "n/a"
        print(f"[{variant}] return={_fmt(stats.get('Total Return [%]'))}%  "
              f"sharpe={_fmt(stats.get('Sharpe Ratio'))}  corr={_fmt(corr, 3)}")

    if not per_stats:
        print("No successful backtests.")
        return 1

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(outdir, f"{label}_summary.csv"), index=False)

    fig1 = plot_backtest_comparison(per_stats, outdir, label=label)
    fig2 = plot_signal_quality_grid(per_preds, outdir, label=label)
    fig3 = plot_equity_overlay(per_equity, outdir, benchmark=benchmark_equity, label=label)

    print(f"\nVariants backtested: {len(per_stats)}")
    print(f"  summary -> {outdir}/{label}_summary.csv")
    print(f"  figures -> {fig1}\n             {fig2}\n             {fig3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
