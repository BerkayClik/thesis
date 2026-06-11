# REPO_STRUCTURE.md

The repository follows this structure:

```
src/
  data/
    dataset.py                 # Sliding-window PyTorch Dataset
    preprocessing.py           # Splits, normalization stats, return targets
    loader.py                  # LunarCrush / Yahoo Finance loading & caching
    lunarcrush_api.py          # LunarCrush API client

  models/
    real_lstm.py               # Real LSTM baseline (Z-score normalized input)
    real_lstm_attention.py     # Real LSTM + temporal attention
    real_lstm_revin.py         # RevIN-wrapped real baselines (normalization ablation)
    attention.py               # Temporal attention module
    quaternion_ops.py          # Hamilton product, QuaternionLinear
    quaternion_lstm.py         # Quaternion LSTM cell & stacked layer
    qnn_attention_model.py     # Quaternion LSTM (+ attention) models
    hierarchical_qlstm.py      # Hierarchical QLSTM (4 groups x 3 fusion strategies)
    revin.py                   # Reversible Instance Normalization
    dish_ts.py                 # Dish-TS normalization (RevIN alternative)

  training/
    trainer.py                 # Training loop, early stopping, LR scheduling
    losses.py

  evaluation/
    metrics.py                 # MAPE
    directional_accuracy.py    # Binary & 3-class
    sharpe_ratio.py            # Binary & 3-class (legacy toy Sharpe)

  backtesting/                 # Runs in the isolated .venv-backtest env (numpy<2)
    vbt_adapter.py             # Single-coin vectorbt backtest (long-only & long/short)
    signals.py                 # Position signal construction (dead band, hysteresis)
    basket.py                  # Pooled multi-coin portfolio (cash_sharing)
    plots.py                   # Equity / drawdown figures
    compare.py                 # Cross-run comparison tables

  utils/
    config.py                  # Config loading and merging

configs/
  data/
    daily/                     # btc_ohlc(.yaml/_return), btc_hier(_return), btc_lunar,
                               # btc_single, eth/sol/xrp/bnb (_ohlc/_lunar), sp500, gold
    hourly/                    # btc_ohlc, btc_hier, *_730d variants, alts, sp500, gold
    4hourly/                   # btc_ohlc(_return/_full), btc_hier(_return/_full), alts
  experiments/
    daily_comparison_3seed.yaml        # 7 OHLC variants x 3 seeds
    daily_hierarchical_3seed.yaml      # 6 hierarchical variants x 3 seeds
    4hourly_comparison_3seed.yaml
    4hourly_hierarchical_3seed.yaml
    daily_revin_ablation_3seed.yaml    # real_lstm(+attn) with/without RevIN vs quaternion
    4hourly_revin_ablation_3seed.yaml
    daily_dishts.yaml / 4hourly_dishts.yaml / hourly_dishts.yaml   # Dish-TS ablation
    hourly_comparison(.yaml/_730d), hourly_hierarchical.yaml, hourly_real_only_730d.yaml
    full_comparison.yaml, quick_test.yaml

experiments/
  run_experiments.py           # Runner; writes results JSON + test/val predictions CSVs
  visualize_results.py
  results/                     # git-ignored regenerable artifacts

scripts/
  backtest_all.py              # Batch backtests: long-only / long_short dead band,
                               # --seeds, --threshold auto (val-selected), --fee-grid
  verify_alignment.py          # Alignment gate for predictions CSVs
  backtest_env_smoke.py        # Sanity check for the backtest env
  download_lunarcrush.py
  resample_4hourly.py          # 1h -> 4h resampling
  feature_selection.py / explore_lunarcrush.py / probe_lunarcrush.py

notebooks/                     # Colab notebooks; checkpoint to Drive, resumable
  All_Methods_Backtest_BTC_Daily.ipynb
  All_Methods_Comparison_BTC_4Hourly.ipynb
  All_Methods_FullData_Backtest_BTC_4Hourly.ipynb
  RevIN_Ablation_BTC_Daily_4Hourly.ipynb
  Returns_Backtest_BTC_4Hourly.ipynb
  Hierarchical_QLSTM_BTC_Hourly.ipynb
  4Hourly_Hierarchical_QLSTM_BTC.ipynb

docs/
  ARCHITECTURE.md              # Model & pipeline technical guide
  BACKTESTING.md               # Returns-mode training & portfolio backtesting
  FINDINGS.md                  # Empirical results & open issues
  SPEC.md, LITERATURE_SCOPE.md, IMPLEMENTATION_PHASES.md, REPO_STRUCTURE.md

data/cache/                    # LunarCrush CSV caches (btc/eth/sol/xrp/bnb, day/1h/4h)
drive_results/                 # git-ignored local snapshots of Drive run outputs
.venv-backtest/                # Isolated Python 3.11 env for vectorbt (numpy<2)
```
