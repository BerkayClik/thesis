# Thesis Plan: Quaternion Neural Networks with Attention for Financial Time-Series

## 1. Problem Definition

The goal of this thesis is to investigate whether **Quaternion Neural Networks (QNNs)** combined with a **temporal attention mechanism** can improve financial time-series prediction compared to traditional real-valued neural networks.

The task focuses on **short-horizon price prediction** using historical market data. While the model predicts future prices (regression), the primary evaluation emphasizes **directional correctness (up/down movement)**, which is more robust in noisy financial environments.

---

## 2. Motivation

Financial time-series data are inherently:

* Multivariate
* Highly correlated
* Noisy and non-stationary

Traditional real-valued neural networks process features independently, often failing to explicitly model inter-feature relationships (e.g., Open, High, Low, Close prices).

Quaternion Neural Networks provide a structured way to group related features into a single mathematical entity, enabling efficient modeling of inter-feature dependencies. Temporal attention mechanisms complement this by identifying which time steps are most informative, addressing noise and regime changes.

---

## 3. Dataset and Data Construction

### 3.1 Market Selection

* **Market:** S&P 500
* **Assets:** Index-level data and selected constituent stocks

### 3.2 Data Source

* Publicly available OHLC data (e.g., Yahoo Finance)
* Dataset will be constructed manually to ensure reproducibility and consistent preprocessing

### 3.3 Time Resolution

* **Primary resolution:** Hourly data
* **Secondary comparison:** Daily data

Hourly data is chosen to better exploit the benefits of temporal attention, while daily data is used as a robustness comparison.

---

## 4. Feature Representation

### 4.1 Quaternion Mapping

At each time step, market data is represented as a quaternion:

```
q_t = (Open_t, High_t, Low_t, Close_t)
```

This mapping treats OHLC prices as a single structured entity, allowing quaternion-valued layers to model their internal relationships directly.

### 4.2 Rationale

* OHLC prices describe different aspects of the same market event
* Quaternion representation preserves this structure
* Avoids manual feature engineering at the initial stage

---

## 5. Model Architecture

### 5.1 Overall Architecture

```
Input (Quaternion OHLC)
        ↓
Quaternion LSTM Encoder
        ↓
Real-Valued Temporal Attention
        ↓
Prediction Head (Price Output)
```

### 5.2 Quaternion Encoder

* Quaternion Linear layers
* Quaternion LSTM for temporal encoding
* Captures inter-feature dependencies within each time step

### 5.3 Temporal Attention

* Real-valued attention applied over encoded hidden states
* Learns which time steps contribute most to the prediction
* Improves robustness to noise and irrelevant historical data

### 5.4 Design Choice Justification

Full quaternion-valued attention is avoided to maintain computational feasibility and interpretability. The hybrid design separates feature correlation modeling (quaternion space) from temporal importance modeling (real-valued attention).

---

## 6. Prediction Task

* **Primary task:** Next-step price prediction (regression)
* **Primary evaluation:** Directional accuracy (up/down movement)
* **Secondary metrics:** MAE, MSE

This dual formulation ensures both numerical accuracy and financially meaningful evaluation.

---

## 7. Experimental Setup

### 7.1 Baseline Models

* Real-valued LSTM
* Quaternion LSTM without attention

### 7.2 Proposed Model

* Quaternion LSTM encoder + temporal attention

### 7.3 Validation Strategy

* Rolling window validation
* Train/validation splits respect temporal ordering

### 7.4 Ablation Studies

* Quaternion vs real-valued models
* Attention vs no-attention
* Hourly vs daily resolution

---

## 8. Implementation Details

* **Programming language:** Python
* **Framework:** PyTorch
* **Codebase:** Forked open-source Quaternion Neural Network repository
* Modular structure for data processing, models, and experiments

---

## 9. Expected Contributions

* Application of Quaternion Neural Networks to S&P 500 time-series data
* A hybrid quaternion-attention architecture for financial prediction
* Empirical analysis of quaternion representations under different temporal resolutions
* Insights into the practical benefits and limitations of hypercomplex models in finance

---

## 10. Thesis Type and Scope

* Experimental and applied
* Focus on architectural design and empirical evaluation
* Emphasis on interpretability and reproducibility

---

## 11. Summary

This thesis proposes a structured and feasible approach to financial time-series modeling by combining quaternion-valued feature representations with temporal attention mechanisms. The study aims to provide empirical evidence on whether such representations offer measurable advantages over traditional real-valued neural networks in noisy financial environments.
