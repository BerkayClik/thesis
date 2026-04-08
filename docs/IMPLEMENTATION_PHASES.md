# IMPLEMENTATION_PHASES.md

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
