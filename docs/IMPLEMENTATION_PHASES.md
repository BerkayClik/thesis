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
