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

We compare five model architectures across thirteen core experimental variants (plus normalization-ablation variants), using Bitcoin as the primary test asset with LunarCrush data from 2020-02-14 to 2026-02-12. The framework supports multiple assets (BTC, ETH, SOL, XRP, BNB via LunarCrush; S&P 500 and Gold via Yahoo Finance), data frequencies (daily, hourly, 4-hourly), price- and return-mode targets, and fee-aware portfolio backtesting.

> **Current empirical status:** see [docs/FINDINGS.md](docs/FINDINGS.md). In short: return-mode models show no exploitable signal so far, the apparent price-mode gap between quaternion and real models is largely a normalization confound (RevIN vs Z-score — now isolated by a dedicated ablation), and the test windows are bear markets, which handicaps long-only backtests.

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

### The Five Model Families

| Model | Feature Encoding | Sequence Processing | Attention |
|-------|-----------------|---------------------|-----------|
| Real LSTM | 4 independent features | Standard LSTM | No |
| Real LSTM + Attention | 4 independent features | Standard LSTM | Yes |
| Quaternion LSTM | 4 features → 1 quaternion | Hamilton product | No |
| Quaternion LSTM + Attention | 4 features → 1 quaternion | Hamilton product | Yes |
| **Hierarchical QLSTM** | **16 features → 4 quaternion groups** | **4 independent QLSTMs → fusion** | **Optional per-group** |

### Hierarchical Extension

The hierarchical model extends the quaternion approach from 4 OHLC features to **16 LunarCrush features** (price + market + social + sentiment), grouped into 4 semantic quaternions:

```
Input (batch, seq, 16) → RevIN(16) → Split into 4 groups:
  ├─ Q₁ (Price):     [open, high, low, close]        → QLSTM₁
  ├─ Q₂ (Market):    [vol, mcap, dominance, supply]   → QLSTM₂
  ├─ Q₃ (Social):    [contrib_active, created, ...]   → QLSTM₃
  └─ Q₄ (Sentiment): [sentiment, galaxy, ...]         → QLSTM₄
                                                            ↓
                                              Fusion (concat / group_attn / meta_quat)
                                                            ↓
                                                      Output → denorm
```

Three fusion strategies tested as ablation:
- **Concat → Linear**: simplest baseline
- **Group Attention**: learned importance weights per group (interpretable)
- **Meta-Quaternion**: fuse via Hamilton product (novel algebraic approach)

---

## Experimental Design

### Thirteen Variants for Fair Comparison

Since quaternion layers have ~4x more parameters at equal hidden size, we test both **layer-matched** and **parameter-matched** configurations. The hierarchical extension adds 6 variants testing 3 fusion strategies × 2 attention modes:

| Variant | Model | Hidden Size | Parameters | Purpose |
|---------|-------|-------------|------------|---------|
| `naive_zero` | Persistence (predicts last close) | N/A | 0 | Sanity check baseline |
| `real_lstm` | Real LSTM | 64 | ~51K | Primary baseline |
| `real_lstm_attention` | Real LSTM + Attention | 64 | ~51K | Attention baseline |
| `quaternion_lstm_param_matched` | Quaternion LSTM | 32 | ~56K | Fair comparison |
| `quaternion_lstm_attention_param_matched` | Quaternion LSTM + Attn | 32 | ~56K | Fair comparison |
| `quaternion_lstm` | Quaternion LSTM | 64 | ~174K | Capacity test |
| `quaternion_lstm_attention` | Quaternion LSTM + Attn | 64 | ~179K | Capacity test |
| `hier_qlstm_concat` | Hierarchical QLSTM (concat) | 32 | ~228K | Hierarchical baseline |
| `hier_qlstm_concat_attn` | Hierarchical QLSTM (concat+attn) | 32 | ~228K | + temporal attention |
| `hier_qlstm_group_attn` | Hierarchical QLSTM (group attn) | 32 | ~223K | Interpretable fusion |
| `hier_qlstm_group_attn_temporal` | Hierarchical QLSTM (group+temporal) | 32 | ~224K | Dual attention |
| `hier_qlstm_meta_quat` | Hierarchical QLSTM (meta-quat) | 32 | ~232K | Novel quaternion fusion |
| `hier_qlstm_meta_quat_attn` | Hierarchical QLSTM (meta-quat+attn) | 32 | ~232K | Full model |

**Ablation variants** (beyond the 13 core variants):

- `real_lstm_revin` / `real_lstm_attention_revin` — the real LSTM backbones wrapped in the same per-window RevIN normalization the quaternion models use. This de-confounds architecture from normalization (the standard comparison gives RevIN only to quaternion models). Configs: `configs/experiments/daily_revin_ablation_3seed.yaml`, `4hourly_revin_ablation_3seed.yaml`.
- `qlstm_*_dishts_*` — quaternion variants using **Dish-TS** instead of RevIN (`norm_type: dish_ts`). Configs: `configs/experiments/daily_dishts.yaml`, `4hourly_dishts.yaml`, `hourly_dishts.yaml`.

### Data Configuration

The primary configuration uses daily BTC data from LunarCrush (18 raw columns: OHLC + market + social + sentiment; 4 selected for OHLC experiments, 16 for hierarchical):

```yaml
Asset:        Bitcoin (LunarCrush, data/cache/lunarcrush_btc_day_full.csv)
Period:       2020-02-14 to 2026-02-12 (~2,190 daily bars; ~13,150 4-hourly bars)
Features:     OHLC (4 of 18) or 16-feature hierarchical selection
Target:       Next-bar Close — price, return, or log_return (target_mode)
Window:       20 bars (daily), 30 bars (4-hourly), 72 bars (hourly)
```

Multiple data frequencies are supported, all with ratio-based 70/10/20 splits for BTC (year-based splitting remains available for the long-history Yahoo Finance assets, e.g. `configs/data/daily/sp500.yaml`):

| Frequency | Window Size | Config Example |
|-----------|-------------|----------------|
| Daily OHLC | 20 bars | `configs/data/daily/btc_ohlc.yaml` (+ `_return` variant) |
| Daily Hierarchical (16 features) | 20 bars | `configs/data/daily/btc_hier.yaml` (+ `_return`) |
| Hourly | 72 bars | `configs/data/hourly/btc_ohlc.yaml` |
| 4-Hourly OHLC | 30 bars | `configs/data/4hourly/btc_ohlc.yaml` (+ `_return`, `_full`) |
| 4-Hourly Hierarchical | 30 bars | `configs/data/4hourly/btc_hier.yaml` (+ `_return`, `_full`) |

The standard 4-hourly configs train on the **last 365 days** (`last_n_days: 365`); the `*_full.yaml` configs use the full ~13k-bar history to test the data-starvation hypothesis.

### Temporal Split (No Look-Ahead Bias)

All BTC configs use sequential ratio-based splitting (70/10/20):

```
Timeline: 2020-02 ──────────────────────────────────────────> 2026-02

          |<─────── TRAIN (70%) ───────>|< VAL (10%) >|< TEST (20%) >|
```

With the full daily range this puts the test window at **2024-12-20 → 2026-02-11** — a bear-market period (buy & hold −32.5%); see [docs/FINDINGS.md](docs/FINDINGS.md) for why that matters when interpreting backtests. Year-based splitting (train ≤ 2021 / val 2022 / test ≥ 2023) is used for S&P 500 and Gold daily configs.

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
│   │   ├── daily/                    # btc_ohlc, btc_ohlc_return, btc_hier, btc_hier_return,
│   │   │                             # btc_lunar, eth/sol/xrp/bnb (_ohlc/_lunar), sp500, gold
│   │   ├── hourly/                   # btc_ohlc, btc_hier, btc_*_730d, alts, sp500, gold
│   │   └── 4hourly/                  # btc_ohlc/_return/_full, btc_hier/_return/_full, alts
│   └── experiments/                  # Experiment variant definitions
│       ├── daily_comparison_3seed.yaml        # 7 OHLC variants × 3 seeds
│       ├── daily_hierarchical_3seed.yaml      # 6 hierarchical variants × 3 seeds
│       ├── 4hourly_comparison_3seed.yaml      # 4-hourly equivalents
│       ├── 4hourly_hierarchical_3seed.yaml
│       ├── daily_revin_ablation_3seed.yaml    # real_lstm ± RevIN vs quaternion
│       ├── 4hourly_revin_ablation_3seed.yaml
│       ├── daily_dishts.yaml / 4hourly_dishts.yaml / hourly_dishts.yaml  # Dish-TS ablation
│       └── quick_test.yaml                    # Quick single-seed iteration
│
├── src/
│   ├── data/
│   │   ├── loader.py                 # Data loading & caching (LunarCrush / Yahoo)
│   │   ├── lunarcrush_api.py         # LunarCrush API client
│   │   ├── preprocessing.py          # Normalization, splitting, return targets
│   │   └── dataset.py                # Sliding window PyTorch Dataset
│   │
│   ├── models/
│   │   ├── real_lstm.py              # Standard LSTM baseline (Z-score)
│   │   ├── real_lstm_attention.py    # LSTM + Temporal Attention
│   │   ├── real_lstm_revin.py        # RevIN-wrapped real baselines (ablation)
│   │   ├── revin.py                  # Reversible Instance Normalization
│   │   ├── dish_ts.py                # Dish-TS normalization (RevIN alternative)
│   │   ├── quaternion_ops.py         # Hamilton product & QuaternionLinear
│   │   ├── quaternion_lstm.py        # Quaternion LSTM cell & stacked layer
│   │   ├── qnn_attention_model.py    # Quaternion LSTM + Attention model
│   │   ├── hierarchical_qlstm.py     # Hierarchical QLSTM (4 groups × 3 fusions)
│   │   └── attention.py              # Temporal attention mechanism
│   │
│   ├── training/                     # Trainer (early stopping) & losses
│   ├── evaluation/                   # MAPE, directional accuracy, Sharpe (binary & 3-class)
│   ├── backtesting/                  # vectorbt adapter, signals, basket, plots, compare
│   └── utils/                        # Config loading and merging
│
├── experiments/
│   ├── run_experiments.py            # Main experiment runner (writes test + val predictions CSVs)
│   ├── visualize_results.py          # Results visualization
│   └── results/                      # JSON results, predictions CSVs (git-ignored)
│
├── scripts/
│   ├── backtest_all.py               # Batch backtests: long-only / long-short dead band,
│   │                                 # multi-seed, auto threshold, fee grid
│   ├── verify_alignment.py           # Predictions-CSV alignment gate
│   ├── backtest_env_smoke.py         # Backtest env sanity check
│   ├── download_lunarcrush.py        # Data download
│   ├── resample_4hourly.py           # 1h → 4h resampling
│   └── feature_selection.py          # LunarCrush feature selection analysis
│
├── notebooks/                        # Colab notebooks (resumable from Drive checkpoints)
│   ├── All_Methods_Backtest_BTC_Daily.ipynb
│   ├── All_Methods_Comparison_BTC_4Hourly.ipynb
│   ├── All_Methods_FullData_Backtest_BTC_4Hourly.ipynb
│   ├── RevIN_Ablation_BTC_Daily_4Hourly.ipynb
│   ├── Returns_Backtest_BTC_4Hourly.ipynb
│   ├── Hierarchical_QLSTM_BTC_Hourly.ipynb
│   └── 4Hourly_Hierarchical_QLSTM_BTC.ipynb
│
└── docs/
    ├── ARCHITECTURE.md               # Detailed technical documentation
    ├── BACKTESTING.md                # Returns-mode training & portfolio backtesting
    ├── FINDINGS.md                   # Empirical results & open issues (June 2026)
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
# OHLC comparison (7 variants x 3 seeds) on daily BTC data
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc_ohlc.yaml \
    --experiment-config configs/experiments/daily_comparison_3seed.yaml

# Same in return mode (the backtest-relevant target)
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc_ohlc_return.yaml \
    --experiment-config configs/experiments/daily_comparison_3seed.yaml

# Hierarchical QLSTM (16 LunarCrush features, 6 variants x 3 seeds)
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc_hier.yaml \
    --experiment-config configs/experiments/daily_hierarchical_3seed.yaml

# RevIN ablation (de-confound architecture vs normalization)
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc_ohlc.yaml \
    --experiment-config configs/experiments/daily_revin_ablation_3seed.yaml

# Quick iteration (fewer epochs, single seed)
python experiments/run_experiments.py \
    --base-config configs/data/daily/btc_ohlc.yaml \
    --experiment-config configs/experiments/quick_test.yaml
```

Useful flags: `--debug` (gradient tracking), `--quiet`, `--results-dir <dir>` (override output location). Full-scale runs are executed in Colab via the notebooks in `notebooks/`, which checkpoint to Google Drive and are resumable across disconnects.

**Config structure:** The `--base-config` provides data source, window size, split boundaries, training hyperparameters, and evaluation settings. The `--experiment-config` defines which model variants to run and with which seeds.

### Returns-Mode Training & Portfolio Backtesting

Two additive capabilities (the default price workflow above is unchanged):

- **Train on returns** — set `target_mode: return` (or `log_return`) under `data:` in any base config. The model predicts a return; prices are reconstructed for the existing metrics. Default is `price`. Each run writes per-bar **test and validation predictions CSVs** alongside the results JSON.
- **Portfolio backtest** — feed model predictions into a leakage-free, fee-aware vectorbt portfolio (next-bar-open execution) and measure real performance (total return, Sharpe, Sortino, max drawdown, equity curve), single-coin or as a pooled multi-coin basket. `scripts/backtest_all.py` supports **long-only and long/short dead-band strategies** (with hysteresis), multi-seed aggregation (`--seeds`), validation-selected thresholds (`--threshold auto` — no test-set tuning), and fee-sensitivity grids (`--fee-grid`).

The backtester runs in an **isolated env** (vectorbt needs `numpy<2`, incompatible with the main env):

```bash
uv venv --python 3.11 .venv-backtest
uv pip install --python .venv-backtest -r requirements-backtest.txt
.venv-backtest/bin/python scripts/backtest_env_smoke.py   # verify
```

See **[BACKTESTING.md](docs/BACKTESTING.md)** for the full workflow (alignment gate, single-coin, basket, plots).

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

## Results So Far

Full detail in **[docs/FINDINGS.md](docs/FINDINGS.md)**. Headline findings from the June 2026 daily + 4-hourly runs:

- **Return mode has no exploitable signal yet** — corr(pred, true) ≈ 0 across all models and seeds; models collapse toward persistence. Daily hierarchical return-mode models are seed-consistently *anti*-predictive (corr −0.01 to −0.06).
- **Price-mode quaternion models show the only seed-consistent positive correlation** (~0.09–0.13 daily), but fee-aware backtests of those predictions still lose money.
- **The price-mode MAPE gap is largely a normalization confound, not quaternion algebra**: a smoke test of the RevIN ablation shows `real_lstm_revin` at MAPE 2.3% vs 17.6% for the Z-scored `real_lstm`. The full 3-seed ablation run is pending.
- **Test windows are bear markets** (daily: 2024-12 → 2026-02, buy & hold −32.5%), so long-only backtests are structurally handicapped — hence the long/short dead-band strategy in `scripts/backtest_all.py`.
- Pending: full RevIN ablation run, and re-running the incomplete 4-hourly full-data (data-starvation) experiment.

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed model architecture & data flow |
| [SPEC.md](docs/SPEC.md) | Project specification & requirements |
| [LITERATURE_SCOPE.md](docs/LITERATURE_SCOPE.md) | Research background & references |
| [IMPLEMENTATION_PHASES.md](docs/IMPLEMENTATION_PHASES.md) | Development phases & milestones |
| [REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) | Repository structure reference |
| [BACKTESTING.md](docs/BACKTESTING.md) | Returns-mode training & vectorbt portfolio backtesting |
| [FINDINGS.md](docs/FINDINGS.md) | Empirical results, open issues, and run status (June 2026) |

---

## Technical Highlights

- **Reproducibility:** Deterministic training with fixed seeds and configurable TF32 (disabled by default for reproducibility, enabled in fast mode)
- **Gradient Stability:** Gradient clipping (max_norm=1.0) prevents explosion in quaternion layers
- **Fair Comparison:** Parameter-matched variants ensure differences come from quaternion math, not capacity
- **Proper Preprocessing:** Normalization statistics computed from training data only to prevent look-ahead bias
- **Multi-Seed Evaluation:** 3 seeds per variant with statistical significance testing (paired t-test, Cohen's d)
- **Optimized Quaternion Ops:** QuaternionLinear uses matmul-based Hamilton product (4 matrix multiplications) instead of naive broadcast, and QuaternionLSTMCell uses fused gate computation (2 QuaternionLinear calls instead of 8)
- **3-Class Metrics:** Directional accuracy and Sharpe ratio with configurable flat zone threshold based on training return standard deviation
- **Multi-Frequency Support:** Daily, hourly and 4-hourly configurations (ratio-based 70/10/20 splits for BTC; year-based splits for long-history S&P 500 / Gold)
- **Hierarchical Multi-Feature Support:** 16 LunarCrush features (price, market, social, sentiment) grouped into 4 semantic quaternions with 3 fusion strategies (concat, group attention, meta-quaternion)
- **Normalization Ablations:** RevIN-wrapped real baselines (`real_lstm_revin`) and Dish-TS variants isolate the normalization effect from the architecture effect
- **Honest Evaluation:** Fee-aware vectorbt backtests with next-bar-open execution, validation-selected thresholds, and fee-sensitivity grids — the legacy in-sample Sharpe is kept only for comparability

---

## References

- Parcollet, T., et al. (2019). *Quaternion Recurrent Neural Networks*. ICLR.
- Gaudet, C. & Maida, A. (2018). *Deep Quaternion Networks*. IJCNN.
- Kim, T., et al. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR.
- Fan, W., et al. (2023). *Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting*. AAAI.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Jozefowicz, R., et al. (2015). *An Empirical Exploration of Recurrent Network Architectures*. ICML.

---

## Author

**Berkay** -- M.Sc. Thesis Project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
