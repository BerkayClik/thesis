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
- [x] Implement `src/models/real_lstm.py`
  - [x] Standard LSTM encoder
  - [x] Configurable hidden size, num layers
  - [x] Regression output head
- [x] Implement `src/models/attention.py`
  - [x] Temporal attention mechanism
  - [x] Attention weights visualization support
- [x] Combine LSTM + Attention baseline
- [x] Validate forward pass shapes
- [x] Validate loss computation (MSE)
- [x] Sanity check: model can overfit small batch

**Notes:**
```
- RealLSTM and TemporalAttention were already implemented in Phase 0 placeholders
- Created RealLSTMAttention in src/models/real_lstm_attention.py combining LSTM + Attention
- Added comprehensive test suite in tests/test_phase2_models.py (21 tests)
- Both models verified to overfit small batch (32 samples) with >50% loss reduction in 100 steps
- Attention weights verified to sum to 1.0 as expected
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
- [x] Implement `src/models/quaternion_ops.py`
  - [x] Hamilton product
  - [x] Quaternion conjugate
  - [x] Quaternion norm
  - [x] QuaternionLinear layer
- [x] Unit tests for quaternion operations
  - [x] Test Hamilton product correctness
  - [x] Test associativity: (p * q) * r = p * (q * r)
  - [x] Test norm preservation properties
- [x] Verify gradient flow through quaternion ops

**Notes:**
```
- Implemented Hamilton product with full batch support
- Verified quaternion algebra: i*j=k, j*i=-k, i²=j²=k²=-1
- Tested associativity and distributivity properties
- QuaternionLinear layer uses Hamilton product for weight application
- Created tests/test_phase3_quaternion.py with 37 tests
- All gradient flow tests pass without NaN/Inf
```

---

## Phase 4 – Quaternion LSTM

**Goal:** Implement Quaternion LSTM cell.

### Architecture
Quaternion LSTM applies Hamilton product instead of matrix multiplication in gates.

### Tasks
- [x] Implement `src/models/quaternion_lstm.py`
  - [x] QuaternionLSTMCell
  - [x] QuaternionLSTM (stacked cells)
  - [x] Proper hidden state initialization
- [x] Validate output shapes match specification
- [x] Test numerical stability
  - [x] No NaN/Inf in forward pass
  - [x] No NaN/Inf in gradients
- [x] Sanity check: quaternion LSTM can overfit small batch

**Notes:**
```
- QuaternionLSTMCell implements all 4 gates (input, forget, cell, output) using Hamilton product
- Sigmoid and tanh applied element-wise to quaternion components for gating
- QuaternionLSTM supports multiple layers with dropout between layers
- Created tests/test_phase4_quaternion_lstm.py with 26 tests
- Verified overfitting capability with >50% loss reduction in 100 steps
- All numerical stability tests pass
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
- [x] Implement `src/models/qnn_attention_model.py`
  - [x] Quaternion encoder (OHLC → quaternion)
  - [x] Quaternion LSTM backbone
  - [x] Quaternion → real projection layer
  - [x] Temporal attention on real-valued features
  - [x] Regression head (predict next close)
- [x] Verify end-to-end forward pass
- [x] Verify gradient flow through all components
- [x] Sanity check: full model can overfit small batch

**Notes:**
```
- QNNAttentionModel: Full quaternion pipeline with attention
- QuaternionLSTMNoAttention: Ablation model without attention
- Both models added to src/models/__init__.py
- Created tests/test_phase5_full_model.py with 24 tests
- Attention weights verified to sum to 1.0
- Model can return attention weights for visualization
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
- [x] Implement `src/training/trainer.py`
  - [x] Training loop with early stopping
  - [x] Validation loop
  - [x] Checkpoint saving/loading
  - [x] Logging (loss, metrics per epoch)
- [x] Implement `src/training/losses.py`
  - [x] MSE loss wrapper
- [x] Implement `src/evaluation/metrics.py`
  - [x] MAE computation
  - [x] MSE computation
- [x] Implement `src/evaluation/directional_accuracy.py`
  - [x] Direction correctness: sign(pred - prev) == sign(actual - prev)
- [x] Implement rolling/expanding window validation
- [x] Verify no look-ahead bias in evaluation

**Notes:**
```
- Trainer class supports early stopping with configurable patience
- Checkpoint saving/loading for best model selection
- Directional accuracy computed as percentage (0-100%)
- Created tests/test_phase6_training.py with 28 tests
- All end-to-end training tests pass for all model types
- No look-ahead bias verified in evaluation pipeline
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
- [x] Implement `experiments/run_experiments.py`
  - [x] Configurable model selection
  - [x] Results logging to file
  - [x] Statistical summary (mean, std over seeds)
- [x] Create `configs/experiment.yaml` variations
- [x] Run all baseline experiments
  - [x] Real LSTM
  - [x] Real LSTM + Attention
  - [x] Quaternion LSTM (no attention)
- [x] Run proposed model experiment
  - [x] Quaternion LSTM + Attention
- [x] Run ablation studies
  - [x] Real vs Quaternion comparison
  - [x] Attention vs No Attention comparison
  - [x] Daily vs Hourly comparison
- [x] Compile results table
- [x] Generate visualizations (if needed)

**Notes:**
```
- run_experiments.py supports command-line config selection
- Results saved as JSON with timestamp
- Aggregates results across multiple seeds (mean ± std)
- Pretty-printed results table for quick comparison
- Created tests/test_phase7_experiments.py with 23 tests
- All model types can be created and evaluated
- Configuration files updated with all 4 model variants
```

---

## Summary Checklist

| Phase | Description           | Status |
|-------|-----------------------|--------|
| 0     | Setup                 | [x]    |
| 1     | Data Pipeline         | [x]    |
| 2     | Baseline Models       | [x]    |
| 3     | Quaternion Core       | [x]    |
| 4     | Quaternion LSTM       | [x]    |
| 5     | Full Model            | [x]    |
| 6     | Training & Evaluation | [x]    |
| 7     | Experiments & Ablation| [x]    |

---

## Test Summary

| Test File                      | Tests | Status |
|--------------------------------|-------|--------|
| test_phase0_setup.py           | 7     | Pass   |
| test_phase1_data.py            | 28    | Pass   |
| test_phase2_models.py          | 21    | Pass   |
| test_phase3_quaternion.py      | 37    | Pass   |
| test_phase4_quaternion_lstm.py | 26    | Pass   |
| test_phase5_full_model.py      | 24    | Pass   |
| test_phase6_training.py        | 28    | Pass   |
| test_phase7_experiments.py     | 23    | Pass   |
| **Total**                      | **194** | **All Pass** |

---

## Change Log

| Date | Phase | Change Description |
|------|-------|-------------------|
| 2026-01-15 | 0 | Completed Phase 0 setup: directory structure, placeholder files, config system, training loop verification |
| 2026-01-15 | 1 | Completed Phase 1 data pipeline: yfinance loader, year-based temporal split, preprocessing pipeline, 28 pytest tests |
| 2026-01-15 | 2 | Completed Phase 2 baseline models: RealLSTMAttention combining LSTM + Attention, 21 pytest tests including overfitting sanity check |
| 2026-01-15 | 3 | Completed Phase 3 quaternion core: Hamilton product, conjugate, norm, QuaternionLinear, 37 pytest tests |
| 2026-01-15 | 4 | Completed Phase 4 quaternion LSTM: QuaternionLSTMCell, QuaternionLSTM, 26 pytest tests |
| 2026-01-15 | 5 | Completed Phase 5 full model: QNNAttentionModel, QuaternionLSTMNoAttention, 24 pytest tests |
| 2026-01-15 | 6 | Completed Phase 6 training & evaluation: Trainer, losses, metrics, directional accuracy, 28 pytest tests |
| 2026-01-15 | 7 | Completed Phase 7 experiments: run_experiments.py, config updates, 23 pytest tests |

---

## References

- SPEC.md – Project specification
- REPO_STRUCTURE.md – Directory structure
- IMPLEMENTATION_PHASES.md – Phase descriptions
- LITERATURE_SCOPE.md – Allowed references
- CLAUDE.md – Implementation rules

