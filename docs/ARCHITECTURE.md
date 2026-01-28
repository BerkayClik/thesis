# Model Architecture Guide

A technical guide for CS students explaining the neural network architectures in this project.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [The Problem We're Solving](#the-problem-were-solving)
3. [Background: LSTM Networks](#background-lstm-networks)
4. [Background: Quaternions](#background-quaternions)
5. [Model 1: Real LSTM (Baseline)](#model-1-real-lstm-baseline)
6. [Model 2: Real LSTM + Attention](#model-2-real-lstm--attention)
7. [Model 3: Quaternion LSTM](#model-3-quaternion-lstm)
8. [Model 4: Quaternion LSTM + Attention](#model-4-quaternion-lstm--attention)
9. [Comparison Summary](#comparison-summary)
   - [4 Model Architectures](#4-model-architectures)
   - [7 Experimental Variants](#7-experimental-variants)
   - [Parameter Count Comparison](#parameter-count-comparison)
10. [Data Flow Diagram](#data-flow-diagram-complete)
11. [Training, Validation, and Testing](#training-validation-and-testing)
    - [Data Splitting Strategy](#data-splitting-strategy)
    - [Phase 1: Training](#phase-1-training)
    - [Phase 2: Validation](#phase-2-validation)
    - [Phase 3: Testing](#phase-3-testing)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Complete Training Workflow](#complete-training-workflow)
    - [Multi-Seed Experiments](#multi-seed-experiments)

---

## Prerequisites

You should be familiar with:
- Basic Python and PyTorch
- What a neural network is (layers, weights, forward pass)
- What activation functions do (sigmoid, tanh, ReLU)
- Basic linear algebra (matrix multiplication)

---

## The Problem We're Solving

**Input:** 20 days of stock data, where each day has 4 values:
- Open (O) - price when market opened
- High (H) - highest price that day
- Low (L) - lowest price that day
- Close (C) - price when market closed

**Output:** Predicted normalized Close price for the next day

**Shape:** `(batch_size, 20, 4)` → `(batch_size, 1)`

---

## Background: LSTM Networks

### Why LSTM?

Regular neural networks process each input independently. But stock prices are **sequential** - what happened yesterday affects today. LSTMs (Long Short-Term Memory) are designed for sequences.

### The LSTM Cell

An LSTM cell has two states:
- **Hidden state (h):** Short-term memory, output at each step
- **Cell state (c):** Long-term memory, carries information across many steps

```
        ┌─────────────────────────────────┐
        │           LSTM Cell             │
        │                                 │
x_t ───►│  ┌───┐ ┌───┐ ┌───┐ ┌───┐      │
        │  │ f │ │ i │ │ g │ │ o │      │───► h_t
h_{t-1}─►│  └───┘ └───┘ └───┘ └───┘      │
        │  forget input cell  output     │
c_{t-1}─►│  gate   gate  gate  gate      │───► c_t
        │                                 │
        └─────────────────────────────────┘
```

### The Four Gates

Each gate controls information flow:

```python
# Forget gate: What to remove from cell state
f = sigmoid(W_f @ [h_{t-1}, x_t])   # Values between 0-1

# Input gate: What new info to add
i = sigmoid(W_i @ [h_{t-1}, x_t])

# Cell candidate: New potential values
g = tanh(W_g @ [h_{t-1}, x_t])      # Values between -1 and 1

# Output gate: What to output
o = sigmoid(W_o @ [h_{t-1}, x_t])
```

### State Updates

```python
# Update cell state: forget old + add new
c_t = f * c_{t-1} + i * g

# Compute output
h_t = o * tanh(c_t)
```

The `*` here is element-wise multiplication.

---

## Background: Quaternions

### What is a Quaternion?

A quaternion is a 4-dimensional number:

```
q = a + bi + cj + dk
```

Where `i`, `j`, `k` are imaginary units with special properties:
- `i² = j² = k² = -1`
- `ij = k`, `jk = i`, `ki = j`
- `ji = -k`, `kj = -i`, `ik = -j`

### Why Use Quaternions for OHLC?

We have 4 values per day (O, H, L, C). Instead of treating them as 4 separate numbers, we encode them as one quaternion:

```
q_t = O_t + H_t·i + L_t·j + C_t·k
```

This preserves relationships between OHLC values through quaternion multiplication.

### The Hamilton Product

When multiplying two quaternions, we use the **Hamilton product**:

```python
def hamilton_product(p, q):
    """
    p = (a, b, c, d)
    q = (e, f, g, h)

    Returns p * q (non-commutative: p*q ≠ q*p)
    """
    a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    e, f, g, h = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r = a*e - b*f - c*g - d*h  # real part
    i = a*f + b*e + c*h - d*g  # i component
    j = a*g - b*h + c*e + d*f  # j component
    k = a*h + b*g - c*f + d*e  # k component

    return stack([r, i, j, k])
```

Key insight: This mixes all 4 components together, capturing cross-feature interactions that element-wise operations would miss.

---

## Model 1: Real LSTM (Baseline)

**File:** `src/models/real_lstm.py`

This is the simplest model - a standard PyTorch LSTM.

### Architecture

```
Input: (batch, 20, 4)
         │
         ▼
    ┌─────────┐
    │  LSTM   │  PyTorch's nn.LSTM
    │ layers  │  hidden_size=64, num_layers=2
    └────┬────┘
         │ Output: (batch, 20, 64)
         │
         ▼ Take last time step
    ┌─────────┐
    │ Linear  │  64 → 1
    └────┬────┘
         │
         ▼
Output: (batch, 1)
```

### Key Code

```python
class RealLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        self.lstm = nn.LSTM(
            input_size=input_size,      # 4 (OHLC)
            hidden_size=hidden_size,    # 64
            num_layers=num_layers,      # 2
            batch_first=True,
            dropout=dropout
        )
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)      # (batch, 20, 64)
        last_out = lstm_out[:, -1, :]   # (batch, 64) - last timestep
        return self.output_head(last_out)  # (batch, 1)
```

### Limitation

Only uses the **last** hidden state. Information from earlier days might be lost.

---

## Model 2: Real LSTM + Attention

**File:** `src/models/real_lstm_attention.py`

Adds an attention mechanism to weight all time steps.

### What is Attention?

Instead of only using the last hidden state, attention computes a **weighted average** of ALL hidden states:

```
                    Attention Weights
Day 1:  h_1  ────── 0.05 ─────┐
Day 2:  h_2  ────── 0.08 ─────┤
Day 3:  h_3  ────── 0.12 ─────┤
  ...                         ├───► Weighted Sum ───► context
Day 18: h_18 ────── 0.15 ─────┤
Day 19: h_19 ────── 0.25 ─────┤
Day 20: h_20 ────── 0.35 ─────┘
                   (sum = 1.0)
```

### Attention Mechanism

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        self.attention = nn.Linear(hidden_size, 1)  # Learns importance

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)

        # Score each time step
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Convert to probabilities (sum to 1)
        weights = softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum: weights @ x
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, seq_len)
            x                       # (batch, seq_len, hidden)
        ).squeeze(1)               # (batch, hidden)

        return context
```

### Architecture

```
Input: (batch, 20, 4)
         │
         ▼
    ┌─────────┐
    │  LSTM   │
    └────┬────┘
         │ (batch, 20, 64) - ALL time steps
         ▼
    ┌─────────┐
    │Attention│  Learns which days matter most
    └────┬────┘
         │ (batch, 64) - weighted combination
         ▼
    ┌─────────┐
    │ Linear  │
    └────┬────┘
         │
         ▼
Output: (batch, 1)
```

### Advantage

Can learn that "3 days ago was important for this prediction" rather than always focusing on the most recent day.

---

## Model 3: Quaternion LSTM

**File:** `src/models/quaternion_lstm.py`

Replaces standard operations with quaternion operations.

### Quaternion Input Encoding

OHLC data is encoded as a single quaternion feature (`input_size=1`). This approach:

- Maps the 4 OHLC values (Open, High, Low, Close) directly to the 4 quaternion components (r, i, j, k)
- Captures inter-OHLC relationships through the Hamilton product during gate computations
- Follows the approach used in quaternion neural network literature (Gaudet & Maida 2018)

```python
# Input shape transformation
raw_input: (batch, seq_len, 4)      # OHLC features
q_input:   (batch, seq_len, 1, 4)   # 1 quaternion feature with 4 components
```

This is why `QNNAttentionModel` uses `input_size=1` when instantiating the QuaternionLSTM.

### Key Difference: Quaternion Linear Layer

In a normal linear layer:
```python
output = W @ x + b  # Matrix multiplication
```

In a quaternion linear layer:
```python
output = hamilton_product(W, x) + b  # Hamilton product
```

### Quaternion Linear Layer

```python
class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        # Weight is a quaternion: (out, in, 4)
        self.weight = Parameter(torch.empty(out_features, in_features, 4))
        self.bias = Parameter(torch.zeros(out_features, 4))

    def forward(self, x):
        # x: (batch, in_features, 4)
        # Compute Hamilton product for each in/out pair
        x_expanded = x.unsqueeze(1)           # (batch, 1, in, 4)
        w_expanded = self.weight.unsqueeze(0)  # (1, out, in, 4)

        products = hamilton_product(x_expanded, w_expanded)  # (batch, out, in, 4)
        output = products.sum(dim=2) + self.bias            # (batch, out, 4)
        return output
```

### Quaternion LSTM Cell

Same gates as regular LSTM, but using quaternion operations:

```python
class QuaternionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        # 8 quaternion linear layers (2 per gate)
        self.W_ii = QuaternionLinear(input_size, hidden_size)
        self.W_hi = QuaternionLinear(hidden_size, hidden_size)
        # ... (forget, cell, output gates similar)

        # LayerNorm for training stability
        self.cell_norm = nn.LayerNorm(4)
        self.hidden_norm = nn.LayerNorm(4)
        self.gate_norm_i = nn.LayerNorm(4)  # Per-gate normalization
        # ... (gate_norm_f, gate_norm_g, gate_norm_o similar)

    def forward(self, x, hx):
        h, c = hx  # Both are quaternions: (batch, hidden, 4)

        # Gates with pre-activation normalization
        i = sigmoid(gate_norm_i(self.W_ii(x) + self.W_hi(h)))  # input gate
        f = sigmoid(gate_norm_f(self.W_if(x) + self.W_hf(h)))  # forget gate
        g = tanh(gate_norm_g(self.W_ig(x) + self.W_hg(h)))     # cell candidate
        o = sigmoid(gate_norm_o(self.W_io(x) + self.W_ho(h)))  # output gate

        # State updates use Hamilton product + LayerNorm
        c_new = hamilton_product(f, c) + hamilton_product(i, g)
        c_new = self.cell_norm(c_new)

        h_new = hamilton_product(o, tanh(c_new))
        h_new = self.hidden_norm(h_new)

        return h_new, c_new
```

### Shape Flow

```
Input: (batch, 20, 4)
         │
         ▼ Add quaternion dimension
    (batch, 20, 1, 4)   # 1 quaternion feature per timestep
         │
         ▼
    ┌──────────────┐
    │ Quaternion   │
    │    LSTM      │  Each hidden unit is a quaternion
    └──────┬───────┘
           │ (batch, 20, hidden_size, 4)
           │
           ▼ Flatten quaternions
    (batch, 20, hidden_size * 4)
           │
           ▼ Take last timestep
    (batch, hidden_size * 4)
           │
           ▼
    ┌─────────┐
    │ Linear  │  hidden*4 → hidden → 1
    └────┬────┘
         │
         ▼
Output: (batch, 1)
```

### Why This Might Work Better

The Hamilton product mixes all 4 OHLC values in a structured way. Instead of treating O, H, L, C as independent features, the model processes them as a single mathematical object where the relationships are preserved.

### Design Decision: Element-wise Activation Functions

In standard LSTM, sigmoid and tanh are applied element-wise to gate values. For quaternion LSTMs, there are two main approaches:

**Option A: Pure Quaternion Activations**
- Use quaternion-specific activation (e.g., split activation, quaternion exponential)
- Mathematically principled but complex to implement and train

**Option B: Element-wise Activations (Our Approach)**
- Apply sigmoid/tanh independently to each quaternion component (r, i, j, k)
- Simpler implementation, proven effective in practice

**Why We Chose Option B:**

1. **Literature Support:** Gaudet & Maida (2018) and Parcollet et al. (2019) successfully use element-wise activations in quaternion networks for speech and image tasks.

2. **Gating Semantics:** For LSTM gates, we need values in [0,1] (sigmoid) or [-1,1] (tanh) for multiplicative control. Element-wise application preserves this bounded range for each component.

3. **Training Stability:** Pure quaternion activations can cause gradient issues. Element-wise operations have well-understood gradient flow.

4. **Empirical Success:** This approach works well in our experiments without the complexity of quaternion-specific activations.

**Important Note:** This means our "quaternion gating" is not mathematically equivalent to pure quaternion operations. The key quaternion benefit comes from the Hamilton product in state updates (c_new, h_new), not from the activation functions.

**References:**
- Gaudet, C. & Maida, A. (2018). Deep Quaternion Networks. IJCNN.
- Parcollet, T. et al. (2019). Quaternion Recurrent Neural Networks. ICLR.

### Training Stability Features

The Quaternion LSTM includes several features for stable training:

**1. LayerNorm on Gates and States:**
- Pre-activation normalization on each gate prevents magnitude explosion from Hamilton product
- Post-update normalization on cell state (c) and hidden state (h)

```python
# Gate normalization before activation
i = sigmoid(gate_norm_i(W_ii(x) + W_hi(h)))

# State normalization after update
c_new = cell_norm(hamilton_product(f, c) + hamilton_product(i, g))
h_new = hidden_norm(hamilton_product(o, tanh(c_new)))
```

**2. Forget Gate Bias Initialization:**
- Forget gate bias initialized to +1.0 (per Jozefowicz et al. 2015)
- Biases sigmoid toward 1, keeping cell state information initially
- Helps with learning long-term dependencies

**3. Xavier-like Weight Initialization:**
- Quaternion weights use adapted Xavier uniform initialization
- Accounts for 4D quaternion structure in fan calculation

---

## Model 4: Quaternion LSTM + Attention

**File:** `src/models/qnn_attention_model.py`

Combines both innovations: quaternion operations AND attention.

### Design Philosophy

- **Quaternion space:** Captures OHLC feature correlations via Hamilton product
- **Real-valued attention:** Learns temporal importance (which days matter)

### Architecture

```
Input: (batch, 20, 4)
         │
         ▼ Quaternion encoding
    (batch, 20, 1, 4)
         │
         ▼
    ┌──────────────┐
    │ Quaternion   │
    │    LSTM      │
    └──────┬───────┘
           │ (batch, 20, hidden, 4)
           │
           ▼ Flatten: hidden*4
    (batch, 20, hidden*4)
           │
           ▼
    ┌─────────────┐
    │ Projection  │  hidden*4 → hidden
    └──────┬──────┘
           │ (batch, 20, hidden) - Now in real space
           │
           ▼
    ┌──────────────┐
    │  Attention   │  Learn which days matter
    └──────┬───────┘
           │ (batch, hidden)
           │
           ▼
    ┌──────────────┐
    │ Linear Head  │  hidden → 1
    └──────┬───────┘
           │
           ▼
Output: (batch, 1)
```

### Key Code

```python
class QNNAttentionModel(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        self.qlstm = QuaternionLSTM(
            input_size=1,        # 1 quaternion feature
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Quaternion → Real projection
        self.projection = nn.Linear(hidden_size * 4, hidden_size)

        # Attention in real space
        self.attention = TemporalAttention(hidden_size)

        # Single linear output (matches Real LSTM simplicity)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Quaternion encoding
        q_input = x.unsqueeze(2)  # (batch, 20, 1, 4)

        # Quaternion LSTM
        qlstm_out, _ = self.qlstm(q_input)  # (batch, 20, hidden, 4)

        # Flatten and project to real space
        qlstm_flat = qlstm_out.view(batch, 20, -1)  # (batch, 20, hidden*4)
        projected = self.projection(qlstm_flat)     # (batch, 20, hidden)

        # Attention over time
        context = self.attention(projected)  # (batch, hidden)

        # Final prediction
        return self.output_head(context)  # (batch, 1)
```

---

## Comparison Summary

### 4 Model Architectures

| Model | OHLC Encoding | Sequence Processing | Time Aggregation |
|-------|---------------|---------------------|------------------|
| Real LSTM | Independent features | Standard LSTM | Last timestep |
| Real LSTM + Attention | Independent features | Standard LSTM | Learned weights |
| Quaternion LSTM | Single quaternion | Hamilton product LSTM | Last timestep |
| **Quaternion LSTM + Attention** | Single quaternion | Hamilton product LSTM | Learned weights |

### 7 Experimental Variants

We run **7 variants** in experiments. This includes a naive baseline plus 6 model variants. Quaternion models have ~3-4x more parameters at the same hidden size, so we test them in two configurations for fair comparison:

| # | Variant Name | Architecture | Hidden | Purpose |
|---|--------------|--------------|--------|---------|
| 0 | `naive_zero` | Always predicts 0 | N/A | Naive baseline (no learning) |
| 1 | `real_lstm` | Real LSTM | 64 | Baseline |
| 2 | `real_lstm_attention` | Real LSTM + Attention | 64 | Baseline |
| 3 | `quaternion_lstm` | Quaternion LSTM | 64 | Layer-matched |
| 4 | `quaternion_lstm_attention` | Quaternion LSTM + Attention | 64 | Layer-matched |
| 5 | `quaternion_lstm_param_matched` | Quaternion LSTM | 32 | Parameter-matched |
| 6 | `quaternion_lstm_attention_param_matched` | Quaternion LSTM + Attention | 32 | Parameter-matched |

**Naive baseline:** Always predicts zero (no price change). Establishes that models are learning something meaningful.

**Layer-matched (hidden=64):** Same architecture depth as real LSTM, but quaternion has more parameters. Tests if quaternion math itself helps.

**Parameter-matched (hidden=32):** Reduced hidden size so parameter count matches real LSTM (~51K). Tests if improvements come from quaternion math or just having more parameters.

### Parameter Count Comparison

**Layer-matched (hidden_size=64):**

| Model | Approximate Parameters |
|-------|----------------------|
| Real LSTM | ~51K |
| Real LSTM + Attention | ~51K |
| Quaternion LSTM | ~174K |
| Quaternion LSTM + Attention | ~179K |

**Parameter-matched (hidden_size=32):**

| Model | Approximate Parameters |
|-------|----------------------|
| Quaternion LSTM (param-matched) | ~51K |
| Quaternion LSTM + Attention (param-matched) | ~53K |

Quaternion models have more parameters because each weight is 4D instead of 1D.

---

### Architectural Note: Projection Layer

Quaternion models include an additional projection layer that Real models don't have:

```
Quaternion: QLSTM → Flatten(hidden*4) → Projection(hidden*4 → hidden) → Linear(hidden → 1)
Real:       LSTM  → Last timestep(hidden) → Linear(hidden → 1)
```

**Output Head:** Both Real and Quaternion models use the same single `Linear(hidden → 1)` output layer. The projection layer in Quaternion models transforms the flattened quaternion output (hidden*4) back to real space (hidden) before the output head.

### What We're Testing

The research question: **Does quaternion encoding help predict stock prices?**

- Compare all models vs naive_zero → Verify models learn meaningful patterns
- Compare Real LSTM vs Quaternion LSTM → Effect of quaternion encoding
- Compare without attention vs with attention → Effect of temporal attention
- Compare layer-matched vs parameter-matched → Is it the math or just more parameters?
- Compare all 7 → Find the best combination

---

## Data Flow Diagram (Complete)

```
Raw OHLC Data (2010-2024)
         │
         ▼
┌─────────────────────────────┐
│      Preprocessing          │
│  1. Temporal split (raw)    │
│  2. Z-score normalize       │
│  3. Sliding windows         │
└──────────┬──────────────────┘
           │
           ▼
    X: (batch, 20, 4) normalized OHLC
    y: next-day normalized Close price
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
Real Models   Quaternion Models
    │             │
    │        unsqueeze(2)
    │             │
    ▼             ▼
(batch,20,4)  (batch,20,1,4)
    │             │
    ▼             ▼
  LSTM       Quaternion LSTM
    │             │
    ▼             ▼
(batch,20,H)  (batch,20,H,4)
    │             │
    │         flatten
    │             │
    │             ▼
    │        (batch,20,H*4)
    │             │
    │          project
    │             │
    ▼             ▼
(batch,20,H)  (batch,20,H)
    │             │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
Last step    Attention
    │             │
    ▼             ▼
(batch, H)   (batch, H)
    │             │
    └──────┬──────┘
           │
           ▼
      Linear Head
           │
           ▼
      (batch, 1)
           │
           ▼
   Predicted Close Price
        (normalized)
```

---

## Training, Validation, and Testing

This section explains the complete training pipeline from data splitting to final evaluation.

### Data Splitting Strategy

We use **temporal splitting** to prevent look-ahead bias - a critical requirement for financial time series:

```
Timeline: 2010 ─────────────────────────────────────────────► 2024

          │◄────── TRAIN ──────►│◄─ VAL ─►│◄──── TEST ────►│
          │      2010-2021      │  2022   │   2023-2024    │
          │       12 years      │ 1 year  │    2 years     │
```

**Key Implementation Details:**

| Split | Years | Purpose | Normalization |
|-------|-------|---------|---------------|
| Train | 2010-2021 | Model learning | Stats computed here |
| Validation | 2022 | Early stopping & hyperparameter tuning | Uses train stats |
| Test | 2023-2024 | Final unbiased evaluation | Uses train stats |

**Critical:** Normalization statistics (mean, std) are computed from training data **only**. This prevents data leakage from validation/test sets.

### Preprocessing Pipeline Order

```python
# Step 1: Temporal split (on RAW data)
train_raw, val_raw, test_raw = temporal_split(raw_data, dates)

# Step 2: Z-score normalize OHLC features
# CRITICAL: Stats computed from TRAIN only to prevent data leakage
train_mean = train_raw.mean()
train_std = train_raw.std()

train_normalized = (train_raw - train_mean) / train_std
val_normalized = (val_raw - train_mean) / train_std    # Uses TRAIN stats
test_normalized = (test_raw - train_mean) / train_std  # Uses TRAIN stats

# Step 3: Create sliding window datasets
dataset = SP500Dataset(train_normalized, window_size=20)
```

### Sliding Window Creation

Data is converted to supervised learning format:

```
Input Window (X): 20 consecutive days of NORMALIZED OHLC
Target (y): Next-day NORMALIZED Close price

Day:    1   2   3  ...  19  20  │ 21
        └─ X (normalized) ─────┘  └── y = NormalizedClose₂₁
```

---

### Phase 1: Training

**File:** `src/training/trainer.py`

The training loop processes batches with gradient updates:

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING EPOCH                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FOR each batch in train_loader:                           │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────┐                                           │
│  │ Forward Pass │  pred = model(x)                         │
│  └──────┬──────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │ Compute Loss │  loss = MSE(pred, target)                │
│  └──────┬──────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Backward Pass │  loss.backward()                        │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌────────────────┐                                        │
│  │ Gradient Clip  │  clip_grad_norm_(params, max_norm=1.0) │
│  └──────┬─────────┘                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌────────────────┐                                        │
│  │ Optimizer Step │  optimizer.step()                      │
│  └────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Training Components:**

| Component | Configuration | Purpose |
|-----------|--------------|---------|
| Loss Function | MSE | Penalizes prediction errors |
| Optimizer | Adam (lr=0.001) | Adaptive learning rate |
| Gradient Clipping | max_norm=1.0 | Prevents gradient explosion |
| Batch Size | 32 | Memory vs. convergence balance |
| LR Scheduler | ReduceLROnPlateau | Reduce LR by 0.5× on plateau |

---

### Phase 2: Validation

After each training epoch, the model is evaluated on the validation set **without gradient updates**:

```python
def validate(self, dataloader: DataLoader) -> float:
    self.model.eval()  # Disable dropout, batchnorm training mode

    with torch.no_grad():  # No gradient computation
        for x, y in dataloader:
            pred = self.model(x)
            loss = mse_loss(pred, y)

    return average_validation_loss
```

**Early Stopping Mechanism:**

```
Epoch 1:  val_loss = 0.0045  → Best! Save checkpoint. patience = 0
Epoch 2:  val_loss = 0.0042  → Best! Save checkpoint. patience = 0
Epoch 3:  val_loss = 0.0041  → Best! Save checkpoint. patience = 0
Epoch 4:  val_loss = 0.0043  → No improvement.       patience = 1
Epoch 5:  val_loss = 0.0044  → No improvement.       patience = 2
...
Epoch 13: val_loss = 0.0046  → No improvement.       patience = 10 → STOP!

Best model from Epoch 3 is loaded for testing.
```

**Configuration:**
- `patience = 10`: Stop if no improvement for 10 epochs
- `max_epochs = 100`: Maximum training iterations
- Best checkpoint saved to `checkpoints/best_model.pt`

---

### Phase 3: Testing

After training completes, the **best checkpoint** is loaded and evaluated on the held-out test set:

```
┌────────────────────────────────────────────────────────────┐
│                    TEST EVALUATION                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Load best checkpoint (lowest validation loss)          │
│                                                            │
│  2. Set model to eval mode                                 │
│                                                            │
│  3. Forward pass on test set (no gradients)                │
│     └── predictions = model(test_data)                     │
│                                                            │
│  4. Compute evaluation metrics:                            │
│     ├── MAE (Mean Absolute Error)                          │
│     ├── MSE (Mean Squared Error)                           │
│     ├── Directional Accuracy                               │
│     └── Sharpe Ratio                                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | `mean(\|pred - target\|)` | Average prediction error magnitude |
| MSE | `mean((pred - target)²)` | Penalizes large errors more heavily |
| Directional Accuracy | `% where sign(pred - prev) == sign(target - prev)` | Did we predict up/down correctly? |
| Sharpe Ratio | `mean(strategy_returns) / std(strategy_returns)` | Risk-adjusted trading performance |

**Directional Accuracy:**
```python
# Compare predicted vs actual DIRECTION relative to previous day
pred_direction = sign(pred - prev)      # Did model predict up or down?
target_direction = sign(target - prev)  # Did price actually go up or down?
accuracy = (pred_direction == target_direction).mean() * 100
```

**Sharpe Ratio (Trading Strategy):**
```python
# Strategy: Long when predicted direction is up, short when down
actual_returns = (target - prev) / prev
pred_direction = sign(pred - prev)
strategy_returns = pred_direction * actual_returns
sharpe = mean(strategy_returns) / std(strategy_returns)
```

---

### Complete Training Workflow

```
                         START
                           │
                           ▼
            ┌──────────────────────────────┐
            │     Load & Preprocess Data    │
            │  ─────────────────────────── │
            │  • Download OHLC data        │
            │  • Temporal split            │
            │  • Z-score normalize         │
            │  • Create sliding windows    │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │    Create DataLoaders        │
            │  ─────────────────────────── │
            │  • train_loader (shuffle=F)  │
            │  • val_loader (shuffle=F)    │
            │  • test_loader (shuffle=F)   │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │     Initialize Training      │
            │  ─────────────────────────── │
            │  • Create model              │
            │  • Setup optimizer           │
            │  • Setup LR scheduler        │
            │  • Initialize best_loss = ∞  │
            └──────────────┬───────────────┘
                           │
                           ▼
         ┌────────────────────────────────────┐
         │           TRAINING LOOP            │
         │  ────────────────────────────────  │
         │                                    │
         │  ┌──────────────────────────────┐ │
    ┌───►│  │      Train Epoch             │ │
    │    │  │  • Forward/backward pass     │ │
    │    │  │  • Gradient clipping         │ │
    │    │  │  • Optimizer step            │ │
    │    │  └──────────────┬───────────────┘ │
    │    │                 │                  │
    │    │                 ▼                  │
    │    │  ┌──────────────────────────────┐ │
    │    │  │      Validate                │ │
    │    │  │  • Forward pass (no grad)    │ │
    │    │  │  • Compute val_loss          │ │
    │    │  └──────────────┬───────────────┘ │
    │    │                 │                  │
    │    │                 ▼                  │
    │    │  ┌──────────────────────────────┐ │
    │    │  │   Update Best / Patience     │ │
    │    │  │  • If improved: save ckpt    │ │
    │    │  │  • Else: patience++          │ │
    │    │  └──────────────┬───────────────┘ │
    │    │                 │                  │
    │    └─────────────────┼──────────────────┘
    │                      │
    │              ┌───────┴───────┐
    │              │ patience < 10 │
    │              │ & epoch < 100 │
    │              └───────┬───────┘
    │                 Yes  │  No
    └──────────────────────┘   │
                               ▼
            ┌──────────────────────────────┐
            │      Load Best Checkpoint    │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │       Test Evaluation        │
            │  ─────────────────────────── │
            │  • Forward pass on test set  │
            │  • Compute MAE, MSE          │
            │  • Compute Dir. Accuracy     │
            │  • Compute Sharpe Ratio      │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │       Save Results           │
            │  ─────────────────────────── │
            │  • Metrics JSON              │
            │  • Training history          │
            │  • Model checkpoint          │
            └──────────────────────────────┘
                           │
                           ▼
                          END
```

### Multi-Seed Experiments

For statistical validity, each model variant is trained with multiple random seeds:

```
Variant: quaternion_lstm_attention
├── Seed 42  → test_mae=0.0234, dir_acc=54.2%
├── Seed 123 → test_mae=0.0241, dir_acc=53.8%
└── Seed 456 → test_mae=0.0228, dir_acc=55.1%
    ────────────────────────────────────
    Mean ± Std: MAE=0.0234±0.0005, Dir=54.4±0.5%
```

Statistical significance is computed using:
- **Paired t-test:** Compare model vs baseline
- **Cohen's d:** Effect size magnitude
- **p-values:** Significance at 0.05 and 0.01 levels

---

## Further Reading

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Colah's Blog
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original attention paper
- [Quaternion Neural Networks](https://arxiv.org/abs/1903.08478) - Quaternion DNNs paper
