# IMPLEMENTATION_PHASES.md

Phases 0–8 are complete. Phases 9–12 reflect the 2026 extensions; current
status of runs and findings is tracked in FINDINGS.md.

## Phase 0 – Setup
- Initialize repo structure
- Setup config system
- Verify empty training loop runs

---

## Phase 1 – Data Pipeline
- Load S&P 500 OHLC data
- Implement sliding window dataset
- Implement temporal split
- Apply z-score normalization (train-only stats)

---

## Phase 2 – Baseline Models
- Implement real-valued LSTM
- Implement temporal attention
- Validate forward pass and loss

---

## Phase 3 – Quaternion Core
- Implement Hamilton product
- Implement QuaternionLinear
- Unit-test quaternion ops

---

## Phase 4 – Quaternion LSTM
- Implement Quaternion LSTM cell
- Validate shapes and stability

---

## Phase 5 – Full Model
- Quaternion encoder
- Quaternion → real projection
- Temporal attention
- Regression head

---

## Phase 6 – Evaluation
- Directional accuracy
- MAPE
- Rolling-window validation

---

## Phase 7 – Ablation
- Real vs Quaternion
- With vs Without Attention
- Daily vs Hourly

---

## Phase 8 – Hierarchical Extension
- Extend from 4 OHLC features to 16 LunarCrush features (price, market, social, sentiment)
- Group 16 features into 4 semantic quaternions (4 features each)
- Implement 4 independent QLSTMs (one per group) with configurable fusion
- Implement 3 fusion strategies: Concat, Group Attention, Meta-Quaternion
- Add optional per-group temporal attention
- Integrate 6 hierarchical variants into experiment runner
- Create data configs for hourly and 4-hourly hierarchical experiments
- Create Colab notebooks for reproducible execution

---

## Phase 9 – LunarCrush Migration & Multi-Asset Data
- Replace Yahoo Finance with LunarCrush as the primary BTC data source
  (daily / hourly / 4-hourly, 2020–2026)
- Add data configs for ETH, SOL, XRP, BNB
- Feature selection over the 18 LunarCrush columns (4 OHLC / 16 hierarchical)
- 1h → 4h resampling script; data caching in `data/cache/`

## Phase 10 – Returns-Mode Training & Portfolio Backtesting
- `target_mode: price | return | log_return` with leakage-free price
  reconstruction
- Per-bar test **and validation** predictions CSVs from the runner
- Isolated `.venv-backtest` env (vectorbt needs numpy<2)
- Single-coin vectorbt backtest (next-bar-open execution), multi-coin
  pooled basket, alignment-gate script
- Batch backtesting (`scripts/backtest_all.py`): long-only and long/short
  dead-band strategies (hysteresis), multi-seed aggregation,
  validation-selected thresholds (`--threshold auto`), fee-sensitivity grid

## Phase 11 – Normalization Ablations
- Dish-TS as a RevIN alternative for quaternion models (`norm_type: dish_ts`)
- RevIN-wrapped real baselines (`real_lstm_revin`,
  `real_lstm_attention_revin`) to de-confound architecture vs normalization
- Ablation configs (daily + 4-hourly, 3 seeds) and the
  `RevIN_Ablation_BTC_Daily_4Hourly` Colab notebook
- **Status:** smoke-tested; full Colab run pending

## Phase 12 – Robustness & Honest Evaluation (in progress)
- Re-run the incomplete 4-hourly full-data run (data-starvation hypothesis)
- Complete the RevIN ablation run
- Investigate the seed-consistent negative correlation of daily
  hierarchical return-mode models
- Planned: walk-forward (rolling-window) evaluation; fix
  `shuffle=False` in training DataLoaders (together with a re-run, since
  it breaks comparability with existing results)
