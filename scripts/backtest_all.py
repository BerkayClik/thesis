"""Backtest every variant in a return-mode experiment results dir, then render
cross-variant comparison figures.

Runs in the isolated backtest env (needs vectorbt):

    .venv-backtest/bin/python scripts/backtest_all.py \
        --results-dir experiments/results/4hourly_btc_hier_return \
        --ohlc data/cache/lunarcrush_btc_4hour_full.csv \
        --seeds 42,123,2024 --strategy long_short --threshold auto

Strategies:
  long_only   entries when pred_return > threshold, flat otherwise (legacy).
  long_short  +1 / -1 / flat target positions with a symmetric dead band and
              hysteresis (see build_position_signals).

Threshold:
  a float    fixed dead-band half-width on pred_return.
  "auto"     per variant+seed, swept on the *validation* predictions CSV
             (<variant>_seed<seed>_val_predictions.csv) and frozen before
             touching test. Falls back to 0.0 with a warning if no val CSV.

For each variant it writes per-variant stats/equity (first seed), and emits:
  - <label>_backtest_comparison.png  (Return/Sharpe/MaxDD/WinRate bars)
  - <label>_signal_quality_grid.png  (pred vs true return scatter per variant)
  - <label>_equity_overlay.png       (all strategies vs buy & hold)
  - <label>_summary.csv              (one row per variant x seed)
  - <label>_summary_agg.csv          (mean +/- std across seeds per variant)
  - <label>_fee_sensitivity.csv      (variant x fee grid, when --fee-grid set)
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.backtesting.vbt_adapter import (
    load_ohlc,
    run_single_coin_backtest,
    run_position_backtest,
    _stats_dict,
)
from src.backtesting.compare import (
    plot_backtest_comparison,
    plot_signal_quality_grid,
    plot_equity_overlay,
)


def _run_backtest(preds, ohlc, args, threshold, fees=None):
    """Dispatch to the configured strategy; returns (portfolio, signals)."""
    fees = args.fees if fees is None else fees
    if args.strategy == "long_short":
        return run_position_backtest(
            preds, ohlc, threshold=threshold, allow_short=True,
            exit_mode=args.exit_mode, init_cash=args.init_cash,
            fees=fees, slippage=args.slippage, freq=args.freq,
        )
    return run_single_coin_backtest(
        preds, ohlc, threshold=threshold, init_cash=args.init_cash,
        fees=fees, slippage=args.slippage, freq=args.freq,
    )


def _n_trades(stats):
    # vectorbt names this "Total Trades" (older builds used "# Trades")
    return stats.get("Total Trades", stats.get("# Trades"))


def select_threshold_on_val(val_csv, ohlc, args):
    """Sweep the dead-band threshold on validation predictions; return the
    threshold with the best validation Sharpe (ties -> higher total return).
    The grid adapts to the prediction scale via quantiles of |pred_return|,
    so collapsed near-zero predictors still get a meaningful sweep."""
    val = pd.read_csv(val_csv, parse_dates=["decision_time", "target_time"])
    abs_pred = np.abs(np.asarray(val["pred_return"], float))
    qs = np.quantile(abs_pred, [0.25, 0.5, 0.75, 0.9])
    grid = sorted(set([0.0] + [float(q) for q in qs if q > 0]))

    best = (None, -np.inf, -np.inf)  # (threshold, sharpe, return)
    for c in grid:
        try:
            pf, _ = _run_backtest(val, ohlc, args, threshold=c)
        except ValueError:
            continue
        stats = _stats_dict(pf)
        sharpe = stats.get("Sharpe Ratio")
        ret = stats.get("Total Return [%]")
        sharpe = -np.inf if sharpe is None else sharpe
        ret = -np.inf if ret is None else ret
        if (sharpe, ret) > (best[1], best[2]):
            best = (c, sharpe, ret)
    return 0.0 if best[0] is None else best[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--ohlc", required=True)
    p.add_argument("--seeds", default="42",
                   help="comma-separated seeds to backtest (default: 42)")
    p.add_argument("--seed", type=int, default=None,
                   help="deprecated single-seed alias for --seeds")
    p.add_argument("--strategy", choices=["long_only", "long_short"],
                   default="long_only")
    p.add_argument("--threshold", default="0.0",
                   help="dead-band half-width on pred_return, or 'auto' to "
                        "select per variant+seed on validation predictions")
    p.add_argument("--exit-mode", choices=["hold", "flat"], default="hold",
                   help="long_short only: behavior inside the dead band")
    p.add_argument("--fees", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--fee-grid", default=None,
                   help="comma-separated per-side fee rates for a sensitivity "
                        "table, e.g. 0,0.0005,0.001,0.0025")
    p.add_argument("--init-cash", type=float, default=10_000.0)
    p.add_argument("--freq", default="4h")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.seed is not None:  # legacy alias
        seeds = [args.seed]
    auto_threshold = str(args.threshold).strip().lower() == "auto"
    fixed_threshold = 0.0 if auto_threshold else float(args.threshold)
    fee_grid = ([float(f) for f in args.fee_grid.split(",")]
                if args.fee_grid else None)

    label = os.path.basename(args.results_dir.rstrip("/"))
    outdir = os.path.join(args.results_dir, "backtest_all")
    os.makedirs(outdir, exist_ok=True)

    ohlc = load_ohlc(args.ohlc)

    per_stats, per_preds, per_equity = {}, {}, {}  # first-seed, for figures
    rows, fee_rows = [], []
    benchmark_equity = None
    first_seed = seeds[0]

    for seed in seeds:
        csvs = sorted(glob.glob(
            f"{args.results_dir}/*_seed{seed}_predictions.csv"))
        csvs = [c for c in csvs
                if not c.endswith(f"_seed{seed}_val_predictions.csv")]
        if not csvs:
            print(f"No seed-{seed} predictions in {args.results_dir}")
            continue

        for csv in csvs:
            variant = os.path.basename(csv).replace(
                f"_seed{seed}_predictions.csv", "")
            preds = pd.read_csv(csv, parse_dates=["decision_time", "target_time"])

            threshold = fixed_threshold
            if auto_threshold:
                val_csv = csv.replace("_predictions.csv", "_val_predictions.csv")
                if os.path.exists(val_csv):
                    threshold = select_threshold_on_val(val_csv, ohlc, args)
                else:
                    print(f"[{variant} seed {seed}] no val predictions CSV, "
                          f"threshold 'auto' falls back to 0.0")

            try:
                pf, sig = _run_backtest(preds, ohlc, args, threshold=threshold)
            except Exception as exc:  # noqa: BLE001 - report and continue across variants
                print(f"[{variant} seed {seed}] backtest failed: {exc}")
                continue

            stats = _stats_dict(pf)
            if seed == first_seed:
                per_stats[variant] = stats
                per_preds[variant] = preds
                per_equity[variant] = pf.value()
                if benchmark_equity is None:
                    exec_price = pd.Series(
                        sig["exec_price"].to_numpy(), index=sig.index)
                    benchmark_equity = (
                        args.init_cash * exec_price / exec_price.iloc[0])

            pr = np.asarray(preds["pred_return"], float)
            tr = np.asarray(preds["true_return"], float)
            degenerate = pr.std() == 0.0  # e.g. naive_zero predicts exactly 0
            corr = float("nan") if degenerate or not tr.std() else \
                float(np.corrcoef(pr, tr)[0, 1])
            da = float("nan") if degenerate else \
                float((np.sign(pr) == np.sign(tr)).mean() * 100)
            rows.append({
                "variant": variant,
                "seed": seed,
                "threshold": threshold,
                "total_return_pct": stats.get("Total Return [%]"),
                "benchmark_return_pct": stats.get("Benchmark Return [%]"),
                "sharpe": stats.get("Sharpe Ratio"),
                "max_drawdown_pct": stats.get("Max Drawdown [%]"),
                "win_rate_pct": stats.get("Win Rate [%]"),
                "trades": _n_trades(stats),
                "corr": round(corr, 4) if corr == corr else corr,
                "dir_agree_pct": round(da, 1) if da == da else da,
            })

            if fee_grid:
                for fee in fee_grid:
                    try:
                        pf_f, _ = _run_backtest(
                            preds, ohlc, args, threshold=threshold, fees=fee)
                    except Exception:  # noqa: BLE001
                        continue
                    s = _stats_dict(pf_f)
                    fee_rows.append({
                        "variant": variant,
                        "seed": seed,
                        "fee_per_side": fee,
                        "total_return_pct": s.get("Total Return [%]"),
                        "sharpe": s.get("Sharpe Ratio"),
                        "trades": _n_trades(s),
                    })

            def _fmt(x, d=2):
                return f"{x:.{d}f}" if isinstance(x, (int, float)) else "n/a"
            print(f"[{variant} seed {seed}] thr={threshold:.5g}  "
                  f"return={_fmt(stats.get('Total Return [%]'))}%  "
                  f"sharpe={_fmt(stats.get('Sharpe Ratio'))}  "
                  f"corr={_fmt(corr, 3)}")

    if not rows:
        print("No successful backtests.")
        return 1

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(outdir, f"{label}_summary.csv"), index=False)

    # Mean +/- std across seeds per variant
    num_cols = ["total_return_pct", "sharpe", "max_drawdown_pct",
                "win_rate_pct", "trades", "corr", "dir_agree_pct"]
    agg = summary.groupby("variant")[num_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", summary.groupby("variant")["seed"].nunique())
    agg.round(4).to_csv(os.path.join(outdir, f"{label}_summary_agg.csv"))

    if fee_rows:
        pd.DataFrame(fee_rows).to_csv(
            os.path.join(outdir, f"{label}_fee_sensitivity.csv"), index=False)

    fig1 = plot_backtest_comparison(per_stats, outdir, label=label)
    fig2 = plot_signal_quality_grid(per_preds, outdir, label=label)
    fig3 = plot_equity_overlay(per_equity, outdir,
                               benchmark=benchmark_equity, label=label)

    print(f"\nVariant-seed combinations backtested: {len(rows)}")
    print(f"  summary     -> {outdir}/{label}_summary.csv")
    print(f"  aggregated  -> {outdir}/{label}_summary_agg.csv")
    if fee_rows:
        print(f"  fee grid    -> {outdir}/{label}_fee_sensitivity.csv")
    print(f"  figures -> {fig1}\n             {fig2}\n             {fig3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
