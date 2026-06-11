# Empirical Findings (as of June 2026)

Status summary of what the experiments and backtests have actually shown so
far. This is the honest-results companion to the methodology docs
(ARCHITECTURE.md, BACKTESTING.md). Update this file whenever a major run
completes.

---

## Completed runs

| Run | Status | Where |
|-----|--------|-------|
| Daily BTC all-methods (4 experiments × 13 variants × 3 seeds: OHLC / OHLC-return / hier / hier-return) + backtests | **Complete** (2026-06-08) | Drive `thesis_results_daily_backtest/`, archive `20260608_132917`; local copy `drive_results/daily_20260608_132917/` |
| 4-hourly BTC full-data run (data-starvation hypothesis: 365-day subset vs full ~13k-bar history, 8 experiments) | **Incomplete** — stalled at experiment 4/8 on 2026-06-08; the 4 full-data experiments and all backtests never ran. Notebook is resumable from the Drive checkpoint. | Drive `thesis_results_all_full_backtest_4h/checkpoint/`; local copy `drive_results/4h_checkpoint/` |
| RevIN ablation (daily + 4-hourly) | **Awaiting full Colab run** — only a 3-epoch smoke test so far | Drive folder `thesis_results_revin_ablation` |

---

## Key findings from the daily + 4h backtest investigation (2026-06-11)

1. **No exploitable signal in return mode.** Across all models and seeds,
   corr(predicted return, true return) ≈ 0 (|corr| < 0.05), and the
   prediction std is 3–10× smaller than the true return std — the models
   collapse toward persistence (predicting ~0). `real_lstm` in return mode
   collapses to "always slightly positive" (the training-period drift),
   which puts it 100% in-market and makes it exactly track the benchmark.

2. **Daily hierarchical return-mode models are seed-consistently
   anti-predictive**: corr between −0.01 and −0.06 across all 6 hier
   variants × 3 seeds. Systematic, not noise — a plausible explanation is
   momentum learned on the training period meeting mean-reversion in the
   test period. Worth a dedicated investigation.

3. **Price-mode quaternion models show the only seed-consistent positive
   correlation** (~0.09–0.13 daily). However, the batch backtests only cover
   the return-mode result dirs, and a manual fee-aware backtest of the
   price-mode predictions still lost money (too many trades).

4. **The test windows are bear markets.** Daily test =
   2024-12-20 → 2026-02-11 (buy & hold −32.5%); 4h test =
   2025-12-06 → 2026-02-12 (−26.7%). Long-only strategies are structurally
   handicapped here, and "beating the benchmark" mostly means being out of
   the market. This motivated the long/short dead-band strategy in
   `scripts/backtest_all.py` (see BACKTESTING.md §4).

5. **Fees matter but are not the root cause.** Gross (zero-fee) backtests
   still lose; the cost drag is 5–20 percentage points at 0.1% fee + 0.05%
   slippage with 50–190 trades over the 419-bar daily test set.

6. **The pipeline itself was audited and is clean**: train-only scaler
   statistics, raw `prev_close` for return reconstruction, sequential
   splits, next-bar-open execution, and RevIN denormalization correctly
   skipped in return mode.

---

## The RevIN-vs-Z-score confound (resolved by design, run pending)

The original design gave quaternion models per-window RevIN while real LSTM
baselines got static Z-score normalization — so the real-vs-quaternion
comparison confounded **architecture** with **normalization**. In price
mode, test prices far outside the training range (BTC at $65k+ vs a training
range that tops out much lower) are out-of-distribution for the Z-scored
baseline *by construction*.

The ablation (commit `12974d0`) adds `real_lstm_revin` /
`real_lstm_attention_revin` — the same real backbones wrapped in the exact
RevIN protocol of the quaternion models. A 3-epoch smoke test showed
**real_lstm_revin MAPE 2.3% vs 17.6% for the Z-scored baseline**, strongly
suggesting the headline price-mode gap was the normalization confound, not
quaternion algebra. The full 3-seed run
(`configs/experiments/daily_revin_ablation_3seed.yaml`,
`4hourly_revin_ablation_3seed.yaml`,
`notebooks/RevIN_Ablation_BTC_Daily_4Hourly.ipynb`) is still pending.

Any thesis claim about quaternion superiority in price mode must be made
against `real_lstm_revin`, not against the Z-scored `real_lstm`.

---

## Open issues / known limitations

- **Training DataLoaders use `shuffle=False`** — batches are temporally
  ordered during training, which is unusual and may hurt optimization.
  Not yet fixed because changing it invalidates comparability with all
  completed runs; fix and re-run together.
- **Walk-forward (rolling-window) evaluation is not implemented.** All
  results come from a single fixed temporal split; with the test window
  being a bear market (finding 4), conclusions are regime-dependent.
- **Long/short backtests assume symmetric shorting costs** (no
  funding/borrow fees) — state this simplification explicitly in the
  thesis.
- The 4h full-data run (data-starvation hypothesis) needs to be re-run to
  completion before any claims about 365-day vs full-history training.

---

## Where the raw results live

Drive access is via the rclone remote `gdrive:` (results are produced by
the Colab notebooks and saved to Drive):

- `gdrive:thesis_results_daily_backtest/` — daily all-methods run
- `gdrive:thesis_results_all_full_backtest_4h/` — incomplete 4h full run
- `gdrive:thesis_results_revin_ablation/` — RevIN ablation (pending)
- Local snapshots: `drive_results/` (git-ignored)
