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

**Output:** Predicted Close price for the next day (original price scale)

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

The `QuaternionLinear` layer uses an **optimized matmul-based** implementation of the Hamilton product. Instead of broadcasting element-wise quaternion multiplications, it concatenates quaternion components and performs 4 standard matrix multiplications (leveraging cuBLAS):

```python
class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        # Weight is a quaternion: (out, in, 4)
        self.weight = Parameter(torch.empty(out_features, in_features, 4))
        self.bias = Parameter(torch.zeros(out_features, 4))

    def forward(self, x):
        # x: (batch, in_features, 4)
        # Extract quaternion components
        x_r, x_i, x_j, x_k = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        W_r, W_i, W_j, W_k = (self.weight[..., 0], self.weight[..., 1],
                               self.weight[..., 2], self.weight[..., 3])

        # Concatenate input components: (batch, 4*in_features)
        x_flat = torch.cat([x_r, x_i, x_j, x_k], dim=-1)

        # Build combined weight matrices for each output component
        # Each: (out_features, 4*in_features)
        W_for_r = torch.cat([ W_r, -W_i, -W_j, -W_k], dim=1)
        W_for_i = torch.cat([ W_i,  W_r,  W_k, -W_j], dim=1)
        W_for_j = torch.cat([ W_j, -W_k,  W_r,  W_i], dim=1)
        W_for_k = torch.cat([ W_k,  W_j, -W_i,  W_r], dim=1)

        # 4 matmuls instead of element-wise broadcast + sum
        y_r = x_flat @ W_for_r.T
        y_i = x_flat @ W_for_i.T
        y_j = x_flat @ W_for_j.T
        y_k = x_flat @ W_for_k.T

        return torch.stack([y_r, y_i, y_j, y_k], dim=-1) + self.bias
```

This approach encodes the full Hamilton product multiplication rules into concatenated weight matrices, turning quaternion algebra into standard dense matmuls that hardware accelerators handle efficiently.

### Quaternion LSTM Cell

Same gates as regular LSTM, but using quaternion linear layers for weight transformations. The cell uses **fused gate computation** — 2 `QuaternionLinear` calls produce all 4 gates at once, instead of 8 separate layers:

```python
class QuaternionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        # Fused gate projections: all 4 gates computed together
        # Output shape: (batch, 4 * hidden_size, 4) for [i, f, g, o] gates
        self.W_input = QuaternionLinear(input_size, 4 * hidden_size)
        self.W_hidden = QuaternionLinear(hidden_size, 4 * hidden_size)

        # Forget gate bias initialized to +1.0 (Jozefowicz et al. 2015)
        self._init_forget_gate_bias()

    def forward(self, x, hx):
        h, c = hx  # Both are quaternions: (batch, hidden, 4)

        # Fused gate computation: 2 calls instead of 8
        gates = self.W_input(x) + self.W_hidden(h)  # (batch, 4*hidden, 4)

        # Split into 4 gates (chunk is essentially free — view only)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Apply activations
        i = sigmoid(i_gate)
        f = sigmoid(f_gate)
        g = tanh(g_gate)
        o = sigmoid(o_gate)

        # Element-wise gating (standard LSTM semantics)
        # Quaternion structure is in W_input/W_hidden (QuaternionLinear)
        c_new = f * c + i * g
        h_new = o * tanh(c_new)

        return h_new, c_new
```

**Key design decisions:**

- **Fused gates:** A single `QuaternionLinear(input_size, 4 * hidden_size)` computes all 4 gate projections at once, reducing kernel launches from 8 to 2.
- **Element-wise state updates:** The cell state update uses standard element-wise `f * c + i * g` rather than `hamilton_product(f, c) + hamilton_product(i, g)`. The quaternion structure is captured in the weight matrices (via `QuaternionLinear`), while gating preserves the LSTM cell state as a linear memory highway.
- **No LayerNorm:** Gate outputs are naturally bounded by sigmoid [0,1] and tanh [-1,1], and gradient clipping handles stability.

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

**Important Note:** This means our "quaternion gating" is not mathematically equivalent to pure quaternion operations. The key quaternion benefit comes from the Hamilton product in the **weight matrices** (`QuaternionLinear`), which mix all 4 OHLC components when computing gate pre-activations. The state updates themselves (c_new, h_new) use standard element-wise operations.

**References:**
- Gaudet, C. & Maida, A. (2018). Deep Quaternion Networks. IJCNN.
- Parcollet, T. et al. (2019). Quaternion Recurrent Neural Networks. ICLR.

### Training Stability Features

The Quaternion LSTM includes several features for stable training:

**1. Forget Gate Bias Initialization:**
- Forget gate bias initialized to +1.0 (per Jozefowicz et al. 2015)
- Biases sigmoid toward 1, keeping cell state information initially
- Helps with learning long-term dependencies

```python
# Gate layout in fused weights: [i, f, g, o] each of size hidden_size
# Forget gate is at index hidden_size:2*hidden_size
self.W_input.bias.data[hidden_size:2*hidden_size] += 1.0
self.W_hidden.bias.data[hidden_size:2*hidden_size] += 1.0
```

**2. Glorot Normal Weight Initialization:**
- Quaternion weights use Glorot (Xavier) **normal** initialization
- Fan counts are scaled by 4 to account for the Hamilton product structure, where each output scalar sums over `4 * in_features` terms

```python
fan_in = 4 * self.in_features
fan_out = 4 * self.out_features
stdv = math.sqrt(2.0 / (fan_in + fan_out))
nn.init.normal_(self.weight, 0, stdv)
```

**3. Gradient Clipping:**
- `clip_grad_norm_(params, max_norm=1.0)` applied during training to prevent gradient explosion
- Gate activations (sigmoid, tanh) naturally bound outputs, so no LayerNorm is needed on cell/hidden states

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
| 0 | `naive_zero` | Persistence (last close) | N/A | Naive baseline (no learning) |
| 1 | `real_lstm` | Real LSTM | 64 | Baseline |
| 2 | `real_lstm_attention` | Real LSTM + Attention | 64 | Baseline |
| 3 | `quaternion_lstm` | Quaternion LSTM | 64 | Layer-matched |
| 4 | `quaternion_lstm_attention` | Quaternion LSTM + Attention | 64 | Layer-matched |
| 5 | `quaternion_lstm_param_matched` | Quaternion LSTM | 32 | Parameter-matched |
| 6 | `quaternion_lstm_attention_param_matched` | Quaternion LSTM + Attention | 32 | Parameter-matched |

**Naive baseline:** A persistence model that predicts the last observed close price as the next value (random walk hypothesis). Establishes that models are learning something meaningful beyond simple persistence.

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
Raw OHLC Data (e.g., BTC 2014-2024)
         │
         ▼
┌─────────────────────────────┐
│      Preprocessing          │
│  1. Temporal split (raw)    │
│  2. Sliding windows         │
│  (No Z-score — RevIN inside │
│   models handles this)      │
└──────────┬──────────────────┘
           │
           ▼
    X: (batch, 20, 4) raw OHLC
    y: next-day raw Close price
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
Real Models   Quaternion Models
    │             │
    ▼             ▼
  RevIN norm    RevIN norm        ← Instance normalization
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
    RevIN denorm              ← Restores original price scale
           │
           ▼
   Predicted Close Price
      (original scale)
```

---

## Training, Validation, and Testing

This section explains the complete training pipeline from data splitting to final evaluation.

### Data Splitting Strategy

We use **temporal splitting** to prevent look-ahead bias - a critical requirement for financial time series:

```
Daily data (year-based splitting):

Timeline: 2014 ─────────────────────────────────────────────► 2024

          │◄────── TRAIN ──────►│◄─ VAL ─►│◄──── TEST ────►│
          │      2014-2021      │  2022   │   2023-2024    │
          │       7+ years      │ 1 year  │    2 years     │

Hourly/4-hourly data uses ratio-based splitting (70/10/20) instead of year boundaries.
```

**Key Implementation Details:**

| Split | Years (Daily BTC) | Purpose | Normalization |
|-------|-------------------|---------|---------------|
| Train | 2014-2021 | Model learning | RevIN (per-window) |
| Validation | 2022 | Early stopping & hyperparameter tuning | RevIN (per-window) |
| Test | 2023-2024 | Final unbiased evaluation | RevIN (per-window) |

For hourly/4-hourly data, ratio-based splitting (70/10/20) is used instead of year boundaries.

**Critical:** RevIN normalizes each window independently using its own statistics, so there is no cross-sample leakage. Training return statistics (return_std) are still computed from training data only for the 3-class evaluation threshold.

### Preprocessing Pipeline Order

```python
# Step 1: Temporal split (on RAW data)
train_raw, val_raw, test_raw = temporal_split(raw_data, dates)

# Step 2: Compute training return statistics (for 3-class evaluation threshold)
train_returns = compute_returns(train_raw[:, 3])  # Close column
norm_stats = {'return_std': train_returns.std()}

# Step 3: Create sliding window datasets (raw OHLC, no Z-score)
dataset = SP500Dataset(train_raw, window_size=20)

# Normalization is handled inside each model by RevIN (see below)
```

### RevIN: Reversible Instance Normalization

Normalization is performed **inside each model** using RevIN rather than as
a static preprocessing step. RevIN computes per-instance (per-window) mean
and standard deviation at the input, normalizes, and reverses the
transformation on the model output to restore the original price scale.

This approach addresses **distribution shift** -- the non-stationary nature
of financial time series where the price distribution changes over time.
Static Z-score normalization computed from training data can become stale for
test data from a different time period. RevIN normalizes each window
independently, making the model invariant to the local scale and offset.

Key properties:
- Normalizes each input window by its own mean/std (instance normalization)
- Learns optional per-feature affine parameters (scale and shift)
- Reverses normalization on model output to produce original-scale predictions
- Eliminates the need for global normalization statistics at preprocessing time

Inspired by: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.
Reimplemented independently; no code copied from external repositories.

### Sliding Window Creation

Data is converted to supervised learning format:

```
Input Window (X): 20 consecutive days of raw OHLC
Target (y): Next-day raw Close price

Day:    1   2   3  ...  19  20  │ 21
        └─── X (raw OHLC) ─────┘  └── y = Close₂₁
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
│     ├── MAPE (Mean Absolute Percentage Error)              │
│     ├── Directional Accuracy (binary & 3-class)            │
│     └── Sharpe Ratio (binary & 3-class)                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAPE | `mean(\|target - pred\| / \|target\|) × 100` | Scale-independent percentage error |
| Directional Accuracy | `% where sign(pred - prev) == sign(target - prev)` | Did we predict up/down correctly? |
| Directional Accuracy 3-class | `% where class(pred) == class(target)` (UP/FLAT/DOWN) | Accounts for small "flat" moves |
| Sharpe Ratio | `mean(strategy_returns) / std(strategy_returns)` | Risk-adjusted trading performance |
| Sharpe Ratio 3-class | Same, but no position in FLAT zone | Avoids trading noise |

**Directional Accuracy (Binary):**
```python
# Compare predicted vs actual DIRECTION relative to previous day
pred_direction = sign(pred - prev)      # Did model predict up or down?
target_direction = sign(target - prev)  # Did price actually go up or down?
accuracy = (pred_direction == target_direction).mean() * 100
```

**Directional Accuracy (3-class):**
```python
# Classify returns into UP (+1), FLAT (0), DOWN (-1)
# flat_threshold = flat_threshold_fraction * training_return_std (default: 0.5)
pred_return = (pred - prev) / (abs(prev) + 1e-8)
target_return = (target - prev) / (abs(prev) + 1e-8)

# UP if return > threshold, DOWN if < -threshold, FLAT otherwise
pred_class = classify(pred_return, flat_threshold)
target_class = classify(target_return, flat_threshold)
accuracy = (pred_class == target_class).mean() * 100
```

**Sharpe Ratio (Binary Trading Strategy):**
```python
# Strategy: Long when predicted direction is up, short when down
actual_returns = (target - prev) / prev
pred_direction = sign(pred - prev)
strategy_returns = pred_direction * actual_returns
sharpe = mean(strategy_returns) / std(strategy_returns)
```

**Sharpe Ratio (3-class Trading Strategy):**
```python
# Strategy: Long when UP, short when DOWN, no position when FLAT
# Only active (non-FLAT) periods contribute to Sharpe computation
position = classify(pred_return, flat_threshold)  # +1, 0, or -1
active_returns = position[active] * actual_returns[active]
sharpe = mean(active_returns) / std(active_returns)
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
            │  • Create sliding windows    │
            │  (RevIN normalizes in model) │
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
            │  • Compute MAPE              │
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
├── Seed 42  → test_mape=2.34%, dir_acc=54.2%
├── Seed 123 → test_mape=2.41%, dir_acc=53.8%
└── Seed 456 → test_mape=2.28%, dir_acc=55.1%
    ────────────────────────────────────
    Mean ± Std: MAPE=2.34±0.05%, Dir=54.4±0.5%
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
