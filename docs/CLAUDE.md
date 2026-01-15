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
