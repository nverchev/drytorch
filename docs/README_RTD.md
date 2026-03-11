# DRYTorch's Documentation
[![PyPI version](https://img.shields.io/pypi/v/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![Python](https://img.shields.io/pypi/pyversions/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![License](https://img.shields.io/pypi/l/drytorch.svg)](https://github.com/nverchev/drytorch/blob/master/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-nverchev%2Fdrytorch-blue?logo=github)](https://github.com/nverchev/drytorch)

Reproducible machine learning experiments with PyTorch.

## Design
Applies Don't Repeat Yourself principles: replicable, documented, reusable.

* **Reproducibility:** experimental isolation to prevent unintended dependencies, data leakage, and misconfiguration.
* **Modularity:** flexible protocols preserving type inference in custom implementations.
* **Decoupled Tracking:** execution independent of tracking events (logging, plotting, and storing metadata).
* **Optional Dependencies:** support for external libraries (Hydra, W&B, TensorBoard, etc.) but minimal requirements.
* **Self-Documentation:** automatic metadata extraction and standardization.
* **Ready-to-use:** high-level implementations for advanced applications and workflows.


## Installation

**Requirements:**
- The library only requires recent versions of **PyTorch** and **NumPy**.
- **PyYAML** and **tqdm** are recommended.

**pip:**
```bash
pip install drytorch
```

**UV:**
```bash
uv add drytorch
```
