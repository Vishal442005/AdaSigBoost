# AdaSigBoost

**Task-Driven Adaptive Signal Decomposition with Joint Second-Order Gradient Boosting**

[![CI](https://github.com/Vishal442005/AdaSigBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/Vishal442005/AdaSigBoost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

AdaSigBoost embeds differentiable multi-physics spectral-temporal decomposition directly inside the second-order functional gradient boosting loop. Instead of relying on open-loop unsupervised preprocessing (FFT, DWT, EMD, VMD), each boosting stage:
1. **Decomposes** signals via unit-$L^2$-normalized composite filterbanks (Gabor-Morlet, smooth trend, and rational spectral atoms) with simplex channel mixing, initialized at cross-spectral residual peaks and fitted out-of-fold (OOF) to prevent noise memorization.
2. **Evaluates** each sub-signal using dedicated C++17/OpenMP histogram trees with exact coordinate-adjusted gradients $\tilde{g}_{i,m,k}$.
3. **Weights & Prunes** modes via closed-form Elastic-Net / group-sparse Newton updates, zeroing out uncorrelated distractors.

## Quickstart

```python
import numpy as np
from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor

# Multi-channel signals X: (N, C, T) + optional tabular covariates Z: (N, P)
X = np.random.randn(200, 3, 512)
Z = np.random.randn(200, 4)
y = np.random.randint(0, 3, size=200)

clf = AdaSigBoostClassifier(n_estimators=40, n_modes=8, lambda1=0.5, random_state=42)
clf.fit(X, y, Z=Z)
probs = clf.predict_proba(X, Z=Z)
contribs = clf.predict_mode_contributions(X, Z=Z)  # (N, K, n_classes)
```

## Repository Architecture

- `adasigboost/` — Core C++17/OpenMP (`csrc/`) + Python `scikit-learn`-compatible library (`decomposition/`, `descriptors/`, `core/`, `interpretability/`).
- `docs/theory/` — Formal mathematical proofs (`AdaSigBoost_Theory_Note.tex` / `.pdf`).
- `docs/PERFECTION_PLAN.md` — Complete end-to-end research & engineering blueprint.
- `benchmarks/` — Synthetic colored-noise suite, leak-free multi-domain loaders, Optuna tuning, ablations (`ABL-1`–`ABL-7`), and Friedman/Nemenyi/Holm-Bonferroni statistical tests.
- `tests/` — Unit tests covering C++ KKT optimality, Theorem 1 descent, Proposition 1 distractor pruning, and 100% `sklearn` `check_estimator` compliance.
- `paper/` — LaTeX manuscript (`main.tex`, `appendix.tex`) and automated table generators (`paper/scripts/generate_tables.py`).

## Build, Verify & Reproduce

```bash
make build       # Compile C++17/OpenMP extension and editable install
make test        # Run full pytest suite (including sklearn check_estimator)
make lint        # Run ruff checks
make typecheck   # Run mypy static type checks
make theory      # Compile Phase I Theory Note PDF
```
