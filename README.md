![drytorch_logo.png](https://raw.githubusercontent.com/nverchev/drytorch/main/docs/_static/drytorch_logo.png)
[![PyPI version](https://img.shields.io/pypi/v/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![Total Downloads](https://img.shields.io/pypi/dm/drytorch?label=downloads&style=flat)](https://pypi.org/project/drytorch/)
[![Python](https://img.shields.io/pypi/pyversions/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![License](https://img.shields.io/pypi/l/drytorch.svg)](https://github.com/nverchev/drytorch/blob/master/LICENSE)
[![CI Status](https://github.com/nverchev/drytorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/nverchev/drytorch/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/github/nverchev/drytorch/graph/badge.svg?token=CZND67KAW1)](https://codecov.io/github/nverchev/drytorch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![Documentation Status](https://readthedocs.org/projects/drytorch/badge/?version=latest)](https://drytorch.readthedocs.io/en/latest/?badge=latest)
# DRYTorch
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

## Package Structure
Modules are organized into the following subpackages:

- **`core`:** internal routines and the interfaces for library and custom components.
- **`lib`:** reusable implementations and abstract classes of the protocols.
- **`tracker`:** optional tracker plugins that integrate via the event system.
- **`contrib`:** community-driven extensions and support for external libraries.
- **`utils`:** general utilities independent to the framework.

## Documentation

**[Read the full documentation on Read the Docs →](https://drytorch.readthedocs.io/)**

The documentation includes:
- **[Tutorials](https://drytorch.readthedocs.io/latest/tutorials.html)**: walkthrough through the core features.
- **[API Reference](https://drytorch.readthedocs.io/latest/api.html)**: detailed API documentation.
- **[Architecture Overview](https://drytorch.readthedocs.io/latest/architecture.html)**: design structure.


## See also
- **[Changelog](https://github.com/nverchev/drytorch/blob/main/CHANGELOG.md)**
