# SPEC.md
## Quaternion Neural Networks with Temporal Attention for Financial Time-Series Forecasting

### Objective
Implement and evaluate Quaternion Neural Networks (QNNs) with temporal
attention for financial time-series forecasting, with Bitcoin as the
primary asset.

Primary evaluation metric is directional accuracy; tradeability is judged
by a fee-aware portfolio backtest (see docs/BACKTESTING.md).

> Historical note: the project originally targeted S&P 500 daily OHLC
> (Yahoo Finance, year-based 2000–2024 splits). It has since migrated to
> BTC on LunarCrush data as the primary instrument; S&P 500 and Gold remain
> as secondary assets with year-based splits.

---

### Problem Definition
Given a window of market data, predict the next-step close (price mode) or
next-step return (`target_mode: return` / `log_return`) and evaluate
direction correctness and portfolio performance.

---

### Dataset
- Primary instrument: Bitcoin (LunarCrush API; cached CSVs in `data/cache/`)
- Period: 2020-02-14 → 2026-02-12
- Frequencies: Daily (primary), 4-hourly, hourly
- Features:
  - OHLC experiments: 4 price columns selected from the 18 LunarCrush columns
  - Hierarchical experiments: 16 features (price, market, social, sentiment)
- Secondary instruments: ETH/SOL/XRP/BNB (LunarCrush), S&P 500 and Gold
  (Yahoo Finance)
- Temporal split only (no shuffling across splits):
  - BTC and alts: sequential ratio split 70 / 10 / 20
  - S&P 500 / Gold daily: year-based (train ≤ 2021, val 2022, test ≥ 2023)

---

### Data Encoding
Each timestep's 4-feature group is encoded as a quaternion:

q_t = O_t + H_t * i + L_t * j + C_t * k

The hierarchical extension groups 16 features into 4 semantic quaternions
(Price, Market, Social, Sentiment), each processed by an independent QLSTM
and combined by a fusion module (concat / group attention / meta-quaternion).

---

### Models

#### Baselines
1. Naive persistence (`naive_zero`)
2. Real-valued LSTM (Z-score normalized)
3. Real-valued LSTM + Temporal Attention
4. Quaternion LSTM (no attention)

#### Proposed Models
- Quaternion LSTM + Real-valued Temporal Attention
- Hierarchical QLSTM (16 features, 4 quaternion groups, 3 fusion strategies)

Design principle:
- Feature correlation → quaternion space
- Temporal importance → real-valued space

#### Normalization ablations (required for valid conclusions)
Quaternion models normalize per-window with RevIN while real baselines use
static Z-score — a confound. The comparison set therefore also includes:
- `real_lstm_revin` / `real_lstm_attention_revin` (real backbones + RevIN)
- Dish-TS variants of the quaternion models (`norm_type: dish_ts`)

Architecture claims must be made against the RevIN-matched real baseline.

---

### Training
- Loss: MSE
- Optimizer: Adam (lower LR for quaternion variants)
- Fixed random seeds; 3 seeds per variant for headline runs
- Early stopping on validation loss; gradient clipping (max_norm = 1.0)

---

### Evaluation
Primary:
- Directional Accuracy (binary and 3-class with flat zone)

Secondary:
- MAPE (on reconstructed prices in return mode)
- Fee-aware portfolio backtest: total return, Sharpe, Sortino, max
  drawdown vs buy & hold; next-bar-open execution; thresholds selected on
  validation only

The legacy in-experiment `sharpe_ratio` metric is a frictionless toy and is
kept only for backward comparability.

---

### Validation Strategy
- Strict temporal splits; no look-ahead bias allowed (train-only
  normalization statistics, raw prev-close for return reconstruction).
- Rolling / expanding-window (walk-forward) validation is **specified but
  not yet implemented** — current results come from a single fixed split
  whose test window is a bear market (see docs/FINDINGS.md).

---

### Non-goals
- No market-beating claims
- No technical indicators in core experiments
