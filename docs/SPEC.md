# SPEC.md
## Quaternion Neural Networks with Temporal Attention for S&P 500 Forecasting

### Objective
Implement and evaluate a Quaternion Neural Network (QNN) with temporal attention for financial time-series forecasting using S&P 500 OHLC data.

Primary evaluation metric is directional accuracy.

---

### Problem Definition
Given a window of OHLC prices, predict the next-step close price and evaluate direction correctness.

---

### Dataset
- Instrument: S&P 500 Index
- Features: Open, High, Low, Close
- Frequency: Daily (primary), Hourly (ablation)
- Temporal split only (no shuffling)

Train: 2000–2018  
Validation: 2019–2021  
Test: 2022–2024

---

### Data Encoding
Each timestep is encoded as a quaternion:

q_t = O_t + H_t * i + L_t * j + C_t * k

---

### Models

#### Baselines
1. Real-valued LSTM
2. Real-valued LSTM + Temporal Attention
3. Quaternion LSTM (no attention)

#### Proposed Model
Quaternion LSTM + Real-valued Temporal Attention

Design principle:
- Feature correlation → quaternion space
- Temporal importance → real-valued space

---

### Training
- Loss: MSE
- Optimizer: Adam
- Fixed random seeds
- Early stopping on validation loss

---

### Evaluation
Primary:
- Directional Accuracy

Secondary:
- MAPE

---

### Validation Strategy
Rolling / expanding window validation.
No look-ahead bias allowed.

---

### Non-goals
- No market-beating claims
- No technical indicators in core experiments
