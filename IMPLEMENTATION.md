# IMPLEMENTATION.md

## Quaternion Neural Networks with Temporal Attention for S&P 500 Forecasting

### Progress Tracking

This document tracks implementation progress for all phases. Mark tasks with `[x]` when complete.

---

## Phase 0 – Setup

**Goal:** Initialize repository and verify basic infrastructure.

### Tasks
- [x] Create directory structure per REPO_STRUCTURE.md
  - [x] `src/data/`
  - [x] `src/models/`
  - [x] `src/training/`
  - [x] `src/evaluation/`
  - [x] `configs/`
  - [x] `experiments/`
- [x] Create placeholder files
  - [x] `src/data/dataset.py`
  - [x] `src/data/preprocessing.py`
  - [x] `src/models/real_lstm.py`
  - [x] `src/models/attention.py`
  - [x] `src/models/quaternion_ops.py`
  - [x] `src/models/quaternion_lstm.py`
  - [x] `src/models/qnn_attention_model.py`
  - [x] `src/training/trainer.py`
  - [x] `src/training/losses.py`
  - [x] `src/evaluation/metrics.py`
  - [x] `src/evaluation/directional_accuracy.py`
  - [x] `configs/base.yaml`
  - [x] `configs/experiment.yaml`
  - [x] `experiments/run_experiments.py`
- [x] Setup config system (YAML loading)
- [x] Verify empty training loop runs without errors

**Notes:**
```
- Created pyenv virtualenv 'thesis' with Python 3.13.3
- Added src/utils/config.py for YAML loading
- Added tests/test_phase0_setup.py for verification
- All tests pass successfully
```

---

## Phase 1 – Data Pipeline

**Goal:** Load and preprocess S&P 500 OHLC data.

### Dataset Specification
- **Instrument:** S&P 500 Index
- **Features:** Open, High, Low, Close
- **Frequency:** Daily (primary), Hourly (ablation)
- **Split:** Temporal only (no shuffling)

| Split      | Period    |
|------------|-----------|
| Train      | 2000–2018 |
| Validation | 2019–2021 |
| Test       | 2022–2024 |

### Quaternion Encoding
```
q_t = O_t + H_t * i + L_t * j + C_t * k
```

### Tasks
- [x] Download/load S&P 500 OHLC data (2000–2024)
- [x] Implement `src/data/dataset.py`
  - [x] Sliding window dataset class
  - [x] Configurable window size
  - [x] Returns (X, y) pairs where y is next-step close
- [x] Implement `src/data/preprocessing.py`
  - [x] Z-score normalization using train-only statistics
  - [x] Temporal split function (no shuffling)
  - [x] Quaternion encoding function
- [x] Verify no look-ahead bias in data pipeline
- [x] Unit tests for data loading and splitting

**Notes:**
```
- Added src/data/loader.py for yfinance data downloading with caching support
- Updated temporal_split to use year-based boundaries (train<=2018, val=2019-2021, test>=2022)
- Added preprocess_data() convenience function for complete pipeline
- Created comprehensive pytest test suite (tests/test_phase1_data.py) with 28 tests
- All tests verify no look-ahead bias in normalization and splitting
- Updated configs/base.yaml with data source configuration (ticker, dates, cache_dir)
```

---

## Phase 2 – Baseline Models

**Goal:** Implement real-valued baselines for comparison.

### Models
1. Real-valued LSTM
2. Real-valued LSTM + Temporal Attention

### Tasks
- [ ] Implement `src/models/real_lstm.py`
  - [ ] Standard LSTM encoder
  - [ ] Configurable hidden size, num layers
  - [ ] Regression output head
- [ ] Implement `src/models/attention.py`
  - [ ] Temporal attention mechanism
  - [ ] Attention weights visualization support
- [ ] Combine LSTM + Attention baseline
- [ ] Validate forward pass shapes
- [ ] Validate loss computation (MSE)
- [ ] Sanity check: model can overfit small batch

**Notes:**
```
```

---

## Phase 3 – Quaternion Core

**Goal:** Implement fundamental quaternion operations.

### Mathematical Background
Hamilton product for quaternions p = (a, b, c, d) and q = (e, f, g, h):
```
p * q = (ae - bf - cg - dh,
         af + be + ch - dg,
         ag - bh + ce + df,
         ah + bg - cf + de)
```

### Tasks
- [ ] Implement `src/models/quaternion_ops.py`
  - [ ] Hamilton product
  - [ ] Quaternion conjugate
  - [ ] Quaternion norm
  - [ ] QuaternionLinear layer
- [ ] Unit tests for quaternion operations
  - [ ] Test Hamilton product correctness
  - [ ] Test associativity: (p * q) * r = p * (q * r)
  - [ ] Test norm preservation properties
- [ ] Verify gradient flow through quaternion ops

**Notes:**
```
```

---

## Phase 4 – Quaternion LSTM

**Goal:** Implement Quaternion LSTM cell.

### Architecture
Quaternion LSTM applies Hamilton product instead of matrix multiplication in gates.

### Tasks
- [ ] Implement `src/models/quaternion_lstm.py`
  - [ ] QuaternionLSTMCell
  - [ ] QuaternionLSTM (stacked cells)
  - [ ] Proper hidden state initialization
- [ ] Validate output shapes match specification
- [ ] Test numerical stability
  - [ ] No NaN/Inf in forward pass
  - [ ] No NaN/Inf in gradients
- [ ] Sanity check: quaternion LSTM can overfit small batch

**Notes:**
```
```

---

## Phase 5 – Full Model

**Goal:** Implement complete Quaternion LSTM + Temporal Attention model.

### Design Principle
- **Feature correlation** → Quaternion space
- **Temporal importance** → Real-valued space

### Architecture
```
Input (OHLC) → Quaternion Encoding → Quaternion LSTM → Projection → Temporal Attention → Regression Head → Output
```

### Tasks
- [ ] Implement `src/models/qnn_attention_model.py`
  - [ ] Quaternion encoder (OHLC → quaternion)
  - [ ] Quaternion LSTM backbone
  - [ ] Quaternion → real projection layer
  - [ ] Temporal attention on real-valued features
  - [ ] Regression head (predict next close)
- [ ] Verify end-to-end forward pass
- [ ] Verify gradient flow through all components
- [ ] Sanity check: full model can overfit small batch

**Notes:**
```
```

---

## Phase 6 – Training & Evaluation

**Goal:** Implement training loop and evaluation metrics.

### Training Configuration
- **Loss:** MSE
- **Optimizer:** Adam
- **Early stopping:** On validation loss
- **Seeds:** Fixed for reproducibility

### Evaluation Metrics
| Metric               | Type      |
|----------------------|-----------|
| Directional Accuracy | Primary   |
| MAE                  | Secondary |
| MSE                  | Secondary |

### Tasks
- [ ] Implement `src/training/trainer.py`
  - [ ] Training loop with early stopping
  - [ ] Validation loop
  - [ ] Checkpoint saving/loading
  - [ ] Logging (loss, metrics per epoch)
- [ ] Implement `src/training/losses.py`
  - [ ] MSE loss wrapper
- [ ] Implement `src/evaluation/metrics.py`
  - [ ] MAE computation
  - [ ] MSE computation
- [ ] Implement `src/evaluation/directional_accuracy.py`
  - [ ] Direction correctness: sign(pred - prev) == sign(actual - prev)
- [ ] Implement rolling/expanding window validation
- [ ] Verify no look-ahead bias in evaluation

**Notes:**
```
```

---

## Phase 7 – Experiments & Ablation

**Goal:** Run systematic experiments and ablation studies.

### Experiment Matrix

| Model                           | Attention | Encoding   |
|---------------------------------|-----------|------------|
| Real LSTM                       | No        | Real       |
| Real LSTM + Attention           | Yes       | Real       |
| Quaternion LSTM                 | No        | Quaternion |
| Quaternion LSTM + Attention     | Yes       | Quaternion |

### Ablation Studies
1. **Real vs Quaternion:** Compare encoding effectiveness
2. **With vs Without Attention:** Measure attention contribution
3. **Daily vs Hourly:** Test frequency sensitivity

### Tasks
- [ ] Implement `experiments/run_experiments.py`
  - [ ] Configurable model selection
  - [ ] Results logging to file
  - [ ] Statistical summary (mean, std over seeds)
- [ ] Create `configs/experiment.yaml` variations
- [ ] Run all baseline experiments
  - [ ] Real LSTM
  - [ ] Real LSTM + Attention
  - [ ] Quaternion LSTM (no attention)
- [ ] Run proposed model experiment
  - [ ] Quaternion LSTM + Attention
- [ ] Run ablation studies
  - [ ] Real vs Quaternion comparison
  - [ ] Attention vs No Attention comparison
  - [ ] Daily vs Hourly comparison
- [ ] Compile results table
- [ ] Generate visualizations (if needed)

**Notes:**
```
```

---

## Summary Checklist

| Phase | Description           | Status |
|-------|-----------------------|--------|
| 0     | Setup                 | [x]    |
| 1     | Data Pipeline         | [x]    |
| 2     | Baseline Models       | [ ]    |
| 3     | Quaternion Core       | [ ]    |
| 4     | Quaternion LSTM       | [ ]    |
| 5     | Full Model            | [ ]    |
| 6     | Training & Evaluation | [ ]    |
| 7     | Experiments & Ablation| [ ]    |

---

## Change Log

| Date | Phase | Change Description |
|------|-------|--------------------|
| 2026-01-15 | 0 | Completed Phase 0 setup: directory structure, placeholder files, config system, training loop verification |
| 2026-01-15 | 1 | Completed Phase 1 data pipeline: yfinance loader, year-based temporal split, preprocessing pipeline, 28 pytest tests |

---

## References

- SPEC.md – Project specification
- REPO_STRUCTURE.md – Directory structure
- IMPLEMENTATION_PHASES.md – Phase descriptions
- LITERATURE_SCOPE.md – Allowed references
- CLAUDE.md – Implementation rules
