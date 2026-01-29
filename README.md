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

We compare four model architectures across seven experimental variants, using Bitcoin (BTC-USD) as our primary test asset with data from 2014-2024. The framework supports multiple assets (BTC, S&P 500, Gold) and data frequencies (daily, hourly, 4-hourly).

---

## Key Research Question

> **Does quaternion encoding improve stock return prediction compared to traditional real-valued neural networks?**

---

## Model Architectures

```
+-----------------------------------------------------------------------------+
|                         ARCHITECTURE COMPARISON                              |
+-----------------------------------------------------------------------------+
|                                                                              |
|   Real-Valued Models              Quaternion Models                          |
|   ------------------              -----------------                          |
|                                                                              |
|   OHLC as 4 features              OHLC as 1 quaternion                       |
|   [O, H, L, C]                    q = O + Hi + Lj + Ck                       |
|        |                                  |                                  |
|        v                                  v                                  |
|   +---------+                      +-------------+                          |
|   |  LSTM   |                      | Quaternion  |                          |
|   | Layers  |                      |    LSTM     |                          |
|   +----+----+                      +------+------+                          |
|        |                                  |                                  |
|   Standard                           Hamilton                                |
|   Matrix Mult                        Product                                 |
|        |                                  |                                  |
|        v                                  v                                  |
|   +---------+                      +-------------+                          |
|   |Attention| (optional)           |  Attention  | (optional)               |
|   +----+----+                      +------+------+                          |
|        |                                  |                                  |
|        +----------------+-----------------+                                  |
|                         v                                                    |
|                    Predicted                                                 |
|                     Return                                                   |
|                                                                              |
+-----------------------------------------------------------------------------+
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

Since quaternion layers have ~4x more parameters at equal hidden size, we test both **layer-matched** and **parameter-matched** configurations:

| Variant | Model | Hidden Size | Parameters | Purpose |
|---------|-------|-------------|------------|---------|
| `naive_zero` | Persistence (predicts last close) | N/A | 0 | Sanity check baseline |
| `real_lstm` | Real LSTM | 64 | ~51K | Primary baseline |
| `real_lstm_attention` | Real LSTM + Attention | 64 | ~51K | Attention baseline |
| `quaternion_lstm_param_matched` | Quaternion LSTM | 32 | ~56K | Fair comparison |
| `quaternion_lstm_attention_param_matched` | Quaternion LSTM + Attn | 32 | ~56K | Fair comparison |
| `quaternion_lstm` | Quaternion LSTM | 64 | ~174K | Capacity test |
| `quaternion_lstm_attention` | Quaternion LSTM + Attn | 64 | ~179K | Capacity test |

### Data Configuration

The primary configuration uses daily BTC data:

```yaml
Asset:        Bitcoin (BTC-USD)
Period:       2014-09-17 to 2024-12-31
Features:     Open, High, Low, Close (OHLC)
Target:       Next-day normalized Close price
Window:       20 trading days
```

Multiple data frequencies are supported:

| Frequency | Window Size | Splitting | Config Example |
|-----------|-------------|-----------|----------------|
| Daily | 20 days | Year-based (train/val/test by year boundaries) | `configs/data/daily/btc.yaml` |
| Hourly | 72 bars (3 days) | Ratio-based (70/10/20) | `configs/data/hourly/btc.yaml` |
| 4-Hourly | 30 bars (5 days) | Ratio-based (70/10/20) | `configs/data/4hourly/btc.yaml` |

### Temporal Split (No Look-Ahead Bias)

**Daily data** uses year-based splitting:

```
Timeline: 2014 ────────────────────────────────────────────────> 2024

          |<─────── TRAIN ───────>|<── VAL ──>|<──── TEST ────>|
          |      2014-2021        |    2022    |   2023-2024    |
          |       7+ years        |   1 year   |    2 years     |
```

**Hourly/4-hourly data** uses ratio-based splitting (70/10/20) since year boundaries don't apply to rolling-window intraday data.

---

## Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **MAPE** | Mean Absolute Percentage Error | Scale-independent prediction accuracy |
| **Directional Accuracy** | % correct up/down predictions (binary) | Trading signal quality |
| **Directional Accuracy 3-class** | % correct up/flat/down predictions | Accounts for small moves in a "flat" zone |
| **Sharpe Ratio** | Risk-adjusted returns (long/short strategy) | Real-world profitability |
| **Sharpe Ratio 3-class** | Risk-adjusted returns (long/flat/short) | Avoids trading in the flat zone |

The 3-class metrics classify returns into UP, FLAT, or DOWN based on a configurable threshold (`flat_threshold_fraction * training_return_std`). The 3-class Sharpe ratio only computes returns over active (non-flat) periods.

---

## Project Structure

```
thesis/
├── configs/
│   ├── data/                         # Per-asset, per-frequency data configs
│   │   ├── daily/
│   │   │   ├── btc.yaml              # BTC daily (2014-2024, year-based split)
│   │   │   ├── btc_single.yaml       # BTC daily single-seed variant
│   │   │   ├── sp500.yaml            # S&P 500 daily
│   │   │   └── gold.yaml             # Gold daily
│   │   ├── hourly/
│   │   │   ├── btc.yaml              # BTC hourly (ratio-based split)
│   │   │   ├── sp500.yaml            # S&P 500 hourly
│   │   │   └── gold.yaml             # Gold hourly
│   │   └── 4hourly/
│   │       └── btc.yaml              # BTC 4-hourly (resampled from 1h)
│   └── experiments/                  # Experiment variant definitions
│       ├── full_comparison.yaml      # Full 7-variant × 3-seed comparison
│       ├── quick_test.yaml           # Quick single-seed iteration
│       ├── daily_btc_single.yaml     # Daily BTC single-seed
│       └── 4hourly_btc.yaml          # 4-hourly BTC experiments
│
├── src/
│   ├── data/
│   │   ├── loader.py                 # Data downloading & caching
│   │   ├── preprocessing.py          # Normalization, splitting, quaternion encoding
│   │   └── dataset.py                # Sliding window PyTorch Dataset
│   │
│   ├── models/
│   │   ├── real_lstm.py              # Standard LSTM baseline
│   │   ├── real_lstm_attention.py    # LSTM + Temporal Attention
│   │   ├── quaternion_ops.py         # Hamilton product & QuaternionLinear
│   │   ├── quaternion_lstm.py        # Quaternion LSTM cell & stacked layer
│   │   ├── qnn_attention_model.py    # Quaternion LSTM + Attention model
│   │   └── attention.py              # Temporal attention mechanism
│   │
│   ├── training/
│   │   ├── trainer.py                # Training loop with early stopping
│   │   └── losses.py                 # Loss functions
│   │
│   ├── evaluation/
│   │   ├── metrics.py                # MAPE
│   │   ├── directional_accuracy.py   # Binary & 3-class directional accuracy
│   │   └── sharpe_ratio.py           # Binary & 3-class Sharpe ratio
│   │
│   └── utils/
│       └── config.py                 # Config loading and merging
│
├── experiments/
│   ├── run_experiments.py            # Main experiment runner
│   ├── visualize_results.py          # Results visualization
│   └── results/                      # JSON results & checkpoints
│
└── docs/
    ├── ARCHITECTURE.md               # Detailed technical documentation
    ├── SPEC.md                       # Project specification
    ├── LITERATURE_SCOPE.md           # Research background
    ├── IMPLEMENTATION_PHASES.md      # Development phases & milestones
    └── REPO_STRUCTURE.md             # Repository structure reference
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
# Run full experiment suite (7 variants x 3 seeds) on daily BTC data
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc.yaml \
    --experiment-config configs/experiments/full_comparison.yaml

# Quick iteration (fewer epochs, single seed)
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc.yaml \
    --experiment-config configs/experiments/quick_test.yaml

# Run on 4-hourly BTC data
python experiments/run_experiments.py \
    --base-config configs/data/4hourly/btc.yaml \
    --experiment-config configs/experiments/4hourly_btc.yaml

# Run with debug mode (gradient tracking)
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc.yaml \
    --experiment-config configs/experiments/full_comparison.yaml \
    --debug
```

**Config structure:** The `--base-config` provides data source, window size, split boundaries, training hyperparameters, and evaluation settings. The `--experiment-config` defines which model variants to run and with which seeds.

---

## The Hamilton Product

The key innovation is replacing standard matrix multiplication with the **Hamilton product** for quaternion-valued weights:

```python
def hamilton_product(p, q):
    """
    Compute quaternion multiplication: p * q

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
================================================================================
EXPERIMENT RESULTS
================================================================================
Model                          MAPE (%)           Dir Acc (%)        Sharpe
--------------------------------------------------------------------------------
naive_zero                     X.XX +- 0.00       50.00 +- 0.00      0.000 +- 0.000
real_lstm                      X.XX +- X.XX       XX.XX +- X.XX      X.XXX +- X.XXX
real_lstm_attention            X.XX +- X.XX       XX.XX +- X.XX      X.XXX +- X.XXX
quaternion_lstm_param_matched  ...                 ...                ...
...
================================================================================

STATISTICAL SIGNIFICANCE (vs real_lstm baseline)
--------------------------------------------------------------------------------
Model                               Metric                   p-value      Cohens d     Significant
--------------------------------------------------------------------------------
quaternion_lstm_param_matched       directional_accuracy     X.XXXX       X.XXX        *
...
--------------------------------------------------------------------------------
* p < 0.05, ** p < 0.01
================================================================================
```

*(Placeholder format -- actual values from experiment runs)*

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed model architecture & data flow |
| [SPEC.md](docs/SPEC.md) | Project specification & requirements |
| [LITERATURE_SCOPE.md](docs/LITERATURE_SCOPE.md) | Research background & references |
| [IMPLEMENTATION_PHASES.md](docs/IMPLEMENTATION_PHASES.md) | Development phases & milestones |
| [REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) | Repository structure reference |

---

## Technical Highlights

- **Reproducibility:** Deterministic training with fixed seeds and configurable TF32 (disabled by default for reproducibility, enabled in fast mode)
- **Gradient Stability:** Gradient clipping (max_norm=1.0) prevents explosion in quaternion layers
- **Fair Comparison:** Parameter-matched variants ensure differences come from quaternion math, not capacity
- **Proper Preprocessing:** Normalization statistics computed from training data only to prevent look-ahead bias
- **Multi-Seed Evaluation:** 3 seeds per variant with statistical significance testing (paired t-test, Cohen's d)
- **Optimized Quaternion Ops:** QuaternionLinear uses matmul-based Hamilton product (4 matrix multiplications) instead of naive broadcast, and QuaternionLSTMCell uses fused gate computation (2 QuaternionLinear calls instead of 8)
- **3-Class Metrics:** Directional accuracy and Sharpe ratio with configurable flat zone threshold based on training return standard deviation
- **Multi-Frequency Support:** Daily (year-based split), hourly and 4-hourly (ratio-based split) data configurations

---

## References

- Parcollet, T., et al. (2019). *Quaternion Recurrent Neural Networks*. ICLR.
- Gaudet, C. & Maida, A. (2018). *Deep Quaternion Networks*. IJCNN.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Jozefowicz, R., et al. (2015). *An Empirical Exploration of Recurrent Network Architectures*. ICML.

---

## Author

**Berkay** -- M.Sc. Thesis Project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
