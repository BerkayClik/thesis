# Returns-Mode Training & Portfolio Backtesting

This document covers two additive capabilities layered on top of the existing
price-based pipeline:

1. **`target_mode`** — train models to predict **returns** instead of the raw
   next close, without touching the existing price workflow.
2. **vectorbt portfolio backtesting** — turn model predictions into a
   leakage-free, fee-aware portfolio and measure real performance
   (total return, Sharpe, Sortino, max drawdown, win rate, equity curve).

Everything here is **additive**. The default `target_mode` is `price`, and every
existing config/result is byte-for-byte unchanged.

---

## 1. `target_mode` (price | return | log_return)

Add one optional key under `data:` in any data config:

```yaml
data:
  # ... existing fields ...
  target_mode: return        # price (default) | return | log_return
```

| mode | target the model learns | reconstruction at eval |
|------|-------------------------|------------------------|
| `price` (default) | next close (absolute) | none (legacy path) |
| `return` | `close[t+1] / close[t] - 1` | `pred_close = prev_close * (1 + r)` |
| `log_return` | `log(close[t+1]) - log(close[t])` | `pred_close = prev_close * exp(r)` |

Key correctness properties (verified by tests):

- **No leakage.** Targets are computed from raw closes only; the input window
  for real/LSTM models is still z-scored, but the return target and the
  `prev_close` used for reconstruction come from raw prices (Design B).
- **No price-stat denormalization of returns.** In return-mode the
  RevIN / Dish-TS price-scale denormalization is bypassed for the prediction —
  a predicted return is never inverse-transformed with price statistics.
- **`naive_zero` stays a true persistence baseline.** In return-mode it predicts
  a zero return, i.e. `pred_close == prev_close`.
- **Existing metrics keep working.** MAPE / directional accuracy / the legacy
  toy Sharpe all operate on reconstructed prices, so price-mode and return-mode
  numbers are directly comparable.

Run an experiment in return-mode exactly like any other run:

```bash
python experiments/run_experiments.py \
    --base-config configs/data/4hourly/btc_hier.yaml \
    --experiment-config configs/experiments/4hourly_hierarchical.yaml
```

(set `target_mode: return` in the base config first).

### Per-bar predictions CSV

Every run now also writes, beside the results JSON, one CSV per variant/seed:

```
<results_dir>/<variant>_seed<seed>_predictions.csv
```

with columns:

```
decision_time, target_time, prev_close, pred_close, true_close, pred_return, true_return
```

- `decision_time` = last observed bar **t** (the model's information cutoff).
- `target_time`   = predicted bar **t+1** (where the trade executes).

This CSV is the bridge to the backtester. The results JSON also gains two
top-level keys per run: `target_mode` and `predictions_csv_path`.

> Note: predictions CSVs live under `experiments/results/` and are git-ignored
> regenerable artifacts, not committed.

---

## 2. Isolated backtest environment (uv + Python 3.11)

vectorbt depends on numba and requires `numpy < 2` / `pandas < 3`, which is
**incompatible** with the main `thesis` env (numpy 2.x, torch 2.9). Keep them
separate. The main training env is never modified.

```bash
# one-time setup
uv venv --python 3.11 .venv-backtest
uv pip install --python .venv-backtest -r requirements-backtest.txt

# smoke test (must print "vbt OK <version>" and a finite Total Return)
.venv-backtest/bin/python scripts/backtest_env_smoke.py
```

`requirements-backtest.txt` pins `vectorbt==0.27.3` (pulls numpy 1.26.4,
pandas 2.3.x, numba 0.65.x on macOS arm64).

---

## 3. Alignment gate (run this before trusting any backtest)

Before backtesting, confirm the predictions CSV is correctly aligned. This
prints rows for hand-verification and enforces the invariants
(target_time after decision_time, fixed bar gap, price reconstruction):

```bash
python scripts/verify_alignment.py \
    experiments/results/<run>/hier_qlstm_concat_seed42_predictions.csv \
    --rows 5 --constant-gap
```

Exit code 0 = all invariants hold. Non-zero = a violation is printed; do not
proceed to backtest.

---

## 4. Single-coin backtest

Runs in the backtest env. Signals are **long-only** (`pred_return > threshold`)
and **executed at the next bar's open** (`open` at `target_time`) — never on the
same bar the model observed.

```bash
.venv-backtest/bin/python -m src.backtesting.vbt_adapter \
    --predictions experiments/results/<run>/hier_qlstm_concat_seed42_predictions.csv \
    --ohlc data/cache/lunarcrush_btc_4hour_full.csv \
    --outdir experiments/results/<run>/backtest \
    --label btc_hier_4h --freq 4h \
    --fees 0.001 --slippage 0.0005 --init-cash 10000
```

Outputs in `--outdir`:

- `<label>_stats.json` — total return, Sharpe, Sortino, max drawdown, win rate, …
- `<label>_trades.csv` — per-trade records
- `<label>_equity.csv` — equity curve series

### Batch backtests: long/short, dead band, multi-seed, fee grid

`scripts/backtest_all.py` backtests every variant in a results dir and supports
two strategies:

- `--strategy long_only` (default, legacy): long when `pred_return > threshold`,
  flat otherwise.
- `--strategy long_short`: target positions +1 / -1 / 0 with a symmetric dead
  band of half-width `threshold` around zero. Inside the band, `--exit-mode hold`
  (default) keeps the previous position (hysteresis, cuts fee churn);
  `--exit-mode flat` exits to cash. Orders fire only on position *changes*, so
  holding pays no fees.

```bash
.venv-backtest/bin/python scripts/backtest_all.py \
    --results-dir experiments/results/daily_btc_ohlc_return \
    --ohlc data/cache/lunarcrush_btc_day_full.csv \
    --seeds 42,123,2024 --strategy long_short --threshold auto \
    --exit-mode hold --freq 1d --fee-grid 0,0.0005,0.001,0.0025
```

- `--seeds` backtests every listed seed; `<label>_summary.csv` gets one row per
  variant x seed and `<label>_summary_agg.csv` the mean ± std across seeds.
- `--threshold auto` sweeps the dead band per variant+seed on the **validation**
  predictions CSV (`<variant>_seed<seed>_val_predictions.csv`, written by
  `run_experiments.py` alongside the test CSV) and freezes the winner before
  touching test — no test-set tuning. Runs that predate val CSVs fall back to
  0.0 with a warning.
- `--fee-grid` adds `<label>_fee_sensitivity.csv` (variant x per-side fee →
  net return / Sharpe / trades), for "signal supports trading below X bps"
  claims instead of a single fee assumption.

Caveat for the thesis: the long/short backtest assumes shorts are available at
symmetric cost (no funding/borrow fees) — state this simplification explicitly.

### Equity & drawdown plots

```bash
python -c "from src.backtesting.plots import plot_equity_and_drawdown; \
    plot_equity_and_drawdown('experiments/results/<run>/backtest/btc_hier_4h_equity.csv', \
    outdir='experiments/results/<run>/backtest/figs', label='btc_hier_4h')"
```

Produces `*_equity.png` (linear + log panels) and `*_drawdown.png`.

---

## 5. Multi-coin basket (pooled portfolio)

A true pooled-NAV portfolio across coins via `cash_sharing=True`. Each bar, every
coin with `pred_return > threshold` gets an equal target weight `1/|active|`;
inactive coins are flat; if none are active the bar is all-cash.

```bash
.venv-backtest/bin/python -m src.backtesting.basket \
    --coin "btc:experiments/results/<btc_run>/hier_qlstm_concat_seed42_predictions.csv:data/cache/lunarcrush_btc_4hour_full.csv" \
    --coin "eth:experiments/results/<eth_run>/hier_qlstm_concat_seed42_predictions.csv:data/cache/lunarcrush_eth_4hour_full.csv" \
    --outdir experiments/results/basket --label btc_eth_basket --freq 4h \
    --fees 0.001 --slippage 0.0005 --init-cash 100000
```

The equity series has a single `group` column — one pooled NAV, not independent
per-coin runs.

---

## Recommended validation order

1. `configs/data/4hourly/btc_hier.yaml` (single coin) — run the alignment gate.
2. Same config with `target_mode: return` — confirm CSV + reconstruction.
3. Single-coin backtest (Section 4).
4. Multi-coin basket across BTC + an alt (Section 5).

---

## Important notes

- **The legacy `test_metrics.sharpe_ratio` in the results JSON is the toy
  sign-of-(pred−prev) Sharpe**, kept untouched for backward comparability. The
  real, fee-aware portfolio Sharpe lives in the backtest `stats.json`, not the
  experiment JSON.
- **Close-to-close execution is not used.** Execution is next-bar open to avoid
  the optimism of filling on the same bar the prediction was made.
- A low MAPE on prices can be misleading (a persistence model scores well on
  MAPE but has no tradeable edge). The portfolio backtest is the honest judge.
- **Empirical status (June 2026):** the daily and 4h return-mode backtests to
  date lose money — predictions have ≈ zero correlation with realized returns,
  the test windows are bear markets (daily buy & hold −32.5%), and fees add
  5–20 pts of drag but are not the root cause (gross backtests also lose).
  See `docs/FINDINGS.md` before quoting any backtest number in the thesis.
- The long/short dead-band strategy, `--threshold auto`, `--seeds`, and
  `--fee-grid` were added in direct response to those findings (long-only is
  structurally handicapped in a bear test window; thresholds must come from
  validation, not test).
