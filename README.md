# Quaternion Neural Networks for Financial Time Series Forecasting

<div align="center">

**A Novel Approach to Bitcoin Return Prediction Using Quaternion-Valued LSTMs**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Abstract

This thesis investigates the application of **Quaternion Neural Networks (QNNs)** to financial time series forecasting. By encoding OHLC (Open, High, Low, Close) price data as quaternions, we hypothesize that the Hamilton product can capture cross-feature correlations that traditional real-valued networks miss.

We compare four model architectures across seven experimental variants, using Bitcoin (BTC-USD) as our test asset over a 10-year period (2015-2024).

---

## Key Research Question

> **Does quaternion encoding improve stock return prediction compared to traditional real-valued neural networks?**

---

## Model Architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Real-Valued Models              Quaternion Models                          │
│   ──────────────────              ─────────────────                          │
│                                                                              │
│   OHLC as 4 features              OHLC as 1 quaternion                       │
│   [O, H, L, C]                    q = O + Hi + Lj + Ck                       │
│        │                                  │                                  │
│        ▼                                  ▼                                  │
│   ┌─────────┐                      ┌─────────────┐                          │
│   │  LSTM   │                      │ Quaternion  │                          │
│   │ Layers  │                      │    LSTM     │                          │
│   └────┬────┘                      └──────┬──────┘                          │
│        │                                  │                                  │
│   Standard                           Hamilton                                │
│   Matrix Mult                        Product                                 │
│        │                                  │                                  │
│        ▼                                  ▼                                  │
│   ┌─────────┐                      ┌─────────────┐                          │
│   │Attention│ (optional)           │  Attention  │ (optional)               │
│   └────┬────┘                      └──────┬──────┘                          │
│        │                                  │                                  │
│        └──────────────┬───────────────────┘                                  │
│                       ▼                                                      │
│                  Predicted                                                   │
│                   Return                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Four Models

| Model | OHLC Encoding | Sequence Processing | Attention |
|-------|---------------|---------------------|-----------|
| Real LSTM | 4 independent features | Standard LSTM | No |
| Real LSTM + Attention | 4 independent features | Standard LSTM | Yes |
| Quaternion LSTM | Single quaternion | Hamilton product | No |
| Quaternion LSTM + Attention | Single quaternion | Hamilton product | Yes |

---

## Experimental Design

### Seven Variants for Fair Comparison

Since quaternion layers have ~4× more parameters at equal hidden size, we test both **layer-matched** and **parameter-matched** configurations:

| Variant | Model | Hidden Size | Parameters | Purpose |
|---------|-------|-------------|------------|---------|
| `naive_zero` | Always predicts 0 | N/A | 0 | Sanity check baseline |
| `real_lstm` | Real LSTM | 64 | ~51K | Primary baseline |
| `real_lstm_attention` | Real LSTM + Attention | 64 | ~51K | Attention baseline |
| `quaternion_lstm_param_matched` | Quaternion LSTM | 32 | ~56K | Fair comparison |
| `quaternion_lstm_attention_param_matched` | Quaternion LSTM + Attn | 32 | ~56K | Fair comparison |
| `quaternion_lstm` | Quaternion LSTM | 64 | ~174K | Capacity test |
| `quaternion_lstm_attention` | Quaternion LSTM + Attn | 64 | ~179K | Capacity test |

### Data Configuration

```yaml
Asset:        Bitcoin (BTC-USD)
Period:       2015-01-01 to 2024-12-31 (10 years)
Features:     Open, High, Low, Close (OHLC)
Target:       Next-day percentage return
Window:       20 trading days
```

### Temporal Split (No Look-Ahead Bias)

```
Timeline: 2015 ──────────────────────────────────────────────► 2024

          │◄─────── TRAIN ───────►│◄── VAL ──►│◄──── TEST ────►│
          │       2015-2021       │    2022    │   2023-2024    │
          │        7 years        │   1 year   │    2 years     │
```

---

## Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Prediction accuracy |
| **MSE** | Mean Squared Error | Penalizes large errors |
| **Directional Accuracy** | % correct up/down predictions | Trading signal quality |
| **Sharpe Ratio** | Risk-adjusted returns | Real-world profitability |

> **Note:** MAPE is intentionally excluded as it produces unstable results when target values (returns) are near zero.

---

## Project Structure

```
thesis/
├── configs/
│   ├── base.yaml              # Default hyperparameters
│   └── experiment.yaml        # Experiment variants & seeds
│
├── src/
│   ├── data/
│   │   ├── loader.py          # Data downloading & caching
│   │   ├── preprocessing.py   # Normalization & return computation
│   │   └── dataset.py         # PyTorch Dataset class
│   │
│   ├── models/
│   │   ├── real_lstm.py           # Standard LSTM baseline
│   │   ├── real_lstm_attention.py # LSTM + Temporal Attention
│   │   ├── quaternion_ops.py      # Hamilton product & QuaternionLinear
│   │   ├── quaternion_lstm.py     # Quaternion LSTM cell & layer
│   │   └── qnn_attention_model.py # Full Quaternion + Attention model
│   │
│   ├── training/
│   │   ├── trainer.py         # Training loop with early stopping
│   │   └── losses.py          # Loss functions
│   │
│   └── evaluation/
│       ├── metrics.py             # MAE, MSE
│       ├── directional_accuracy.py
│       └── sharpe_ratio.py
│
├── experiments/
│   ├── run_experiments.py     # Main experiment runner
│   └── results/               # JSON results & checkpoints
│
│
└── docs/
    ├── ARCHITECTURE.md        # Detailed technical documentation
    ├── SPEC.md                # Project specification
    └── LITERATURE_SCOPE.md    # Research background
```

---

## Quick Start

### Prerequisites

- Python 3.13+
- PyTorch 2.0+
- CUDA (optional, MPS supported for Apple Silicon)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thesis.git
cd thesis

# Create virtual environment (using pyenv)
pyenv virtualenv 3.13.3 thesis
pyenv local thesis

# Install dependencies
pip install torch pyyaml pandas numpy scipy
```

### Running Experiments

```bash
# Run full experiment suite (7 variants × 3 seeds)
python experiments/run_experiments.py \
    --base-config configs/base.yaml \
    --experiment-config configs/experiment.yaml

# Run with debug mode (gradient tracking)
python experiments/run_experiments.py --debug
```

---

## The Hamilton Product

The key innovation is replacing standard matrix multiplication with the **Hamilton product** for quaternion-valued weights:

```python
def hamilton_product(p, q):
    """
    Compute quaternion multiplication: p ⊗ q

    p = a + bi + cj + dk
    q = e + fi + gj + hk

    Result mixes all components, capturing cross-feature correlations.
    """
    a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    e, f, g, h = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        a*e - b*f - c*g - d*h,  # real
        a*f + b*e + c*h - d*g,  # i
        a*g - b*h + c*e + d*f,  # j
        a*h + b*g - c*f + d*e   # k
    ], dim=-1)
```

This structured mixing of OHLC components may capture relationships that element-wise operations miss.

---

## Sample Results

```
====================================================================================================
EXPERIMENT RESULTS
====================================================================================================
Model                               MAE                MSE                Dir Acc (%)      Sharpe
----------------------------------------------------------------------------------------------------
naive_zero                          0.0225 ± 0.0000    0.0009 ± 0.0000    50.00 ± 0.00     0.000 ± 0.000
real_lstm                           0.0225 ± 0.0012    0.0009 ± 0.0001    48.50 ± 1.09    -0.036 ± 0.001
real_lstm_attention                 0.0212 ± 0.0009    0.0009 ± 0.0001    47.75 ± 1.00    -0.079 ± 0.018
quaternion_lstm_param_matched       ...                ...                 ...              ...
...
====================================================================================================

STATISTICAL SIGNIFICANCE (vs real_lstm baseline)
----------------------------------------------------------------------------------------------------
Model                               Metric                   p-value      Cohens d     Significant
----------------------------------------------------------------------------------------------------
quaternion_lstm_param_matched       directional_accuracy     0.0234       0.892        *
...
----------------------------------------------------------------------------------------------------
* p < 0.05, ** p < 0.01
====================================================================================================
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed model architecture & data flow |
| [SPEC.md](docs/SPEC.md) | Project specification & requirements |
| [LITERATURE_SCOPE.md](docs/LITERATURE_SCOPE.md) | Research background & references |
| [IMPLEMENTATION_PHASES.md](docs/IMPLEMENTATION_PHASES.md) | Development phases & milestones |

---

## Technical Highlights

- **Reproducibility:** Deterministic training with fixed seeds and disabled TF32
- **Gradient Stability:** Gradient clipping (max_norm=1.0) prevents explosion in quaternion layers
- **Fair Comparison:** Parameter-matched variants ensure differences come from quaternion math, not capacity
- **Proper Preprocessing:** Returns computed from raw prices *before* normalization to avoid instability
- **Multi-Seed Evaluation:** 3 seeds per variant with statistical significance testing (paired t-test, Cohen's d)

---

## References

- Parcollet, T., et al. (2019). *Quaternion Recurrent Neural Networks*. ICLR.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

---

## Author

**Berkay** — M.Sc. Thesis Project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
