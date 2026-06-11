# CLAUDE.md

## Environment Setup

**IMPORTANT:** This project uses a dedicated pyenv virtual environment.

- **Python Environment:** `thesis` (pyenv virtualenv)
- **Python Version:** 3.13.3
- **Activation:** The `.python-version` file in the project root automatically activates the environment
- **Always use `python` command** - pyenv will automatically use the correct environment when in this directory

### Installing Dependencies
```bash
pip install torch pyyaml pandas numpy
```

### Backtesting Environment (separate!)
vectorbt requires `numpy<2`, incompatible with the main env. Backtesting code
(`src/backtesting/`, `scripts/backtest_all.py`) runs in the isolated
`.venv-backtest` (Python 3.11, `uv venv` + `requirements-backtest.txt`):
```bash
.venv-backtest/bin/python scripts/backtest_all.py ...
```
Never install vectorbt/numba into the main `thesis` env. See docs/BACKTESTING.md.

---

## General Rules
- Follow SPEC.md strictly
- Do not introduce extra features, indicators, or datasets
- Do not copy-paste code from external repositories
- Literature may be used for inspiration only

---

## Allowed
- Reimplement known architectures
- Adapt standard training loops
- Use PyTorch best practices

---

## Forbidden
- Exact code replication
- Using results from papers
- Changing evaluation metrics without permission

---

## Documentation
- Comment all inspired components
- Keep code readable and modular
- Prefer explicit over implicit logic
- **Update `docs/ARCHITECTURE.md`** when making changes to:
  - Data preprocessing pipeline (normalization, return computation, splitting)
  - Model architectures (LSTM, attention, quaternion layers)
  - Training loop (loss functions, optimizers, early stopping)
  - Evaluation metrics or methodology
- **Update `docs/FINDINGS.md`** when a major experiment or backtest run
  completes (or fails) — it is the single source of truth for empirical
  status and open issues
- **Update `docs/BACKTESTING.md`** when changing anything under
  `src/backtesting/` or `scripts/backtest_all.py`
