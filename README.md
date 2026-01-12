![drytorch_logo.png](https://raw.githubusercontent.com/nverchev/drytorch/main/docs/_static/drytorch_logo.png)
[![PyPI version](https://img.shields.io/pypi/v/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![Total Downloads](https://img.shields.io/pypi/dm/drytorch?label=downloads&style=flat)](https://pypi.org/project/drytorch/)
[![Python](https://img.shields.io/pypi/pyversions/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![License](https://img.shields.io/github/license/nverchev/drytorch.svg)](https://github.com/nverchev/drytorch/blob/master/LICENSE)
[![CI Status](https://github.com/nverchev/drytorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/nverchev/drytorch/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/github/nverchev/drytorch/graph/badge.svg?token=CZND67KAW1)](https://codecov.io/github/nverchev/drytorch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![Documentation Status](https://readthedocs.org/projects/drytorch/badge/?version=latest)](https://drytorch.readthedocs.io/en/latest/?badge=latest)
# DRYTorch

## 💡 Design Philosophy
By adhering to the Don't Repeat Yourself (DRY) principle, this library makes your machine-learning projects easier to replicate, document, and reuse.

## ✨ Features at a Glance
* **Experimental Scope:**  All logic runs within a controlled scope, preventing unintended dependencies, data leakage, and misconfiguration.
* **Modularity:** Components communicate via defined protocols, providing type safety and flexibility for custom implementations.
* **Decoupled Tracking:** Logging, plotting, and metadata are handled by an event system that separates execution from tracking.
* **Lean Dependencies:** Minimal core requirements while supporting optional external libraries (Hydra, W&B, TensorBoard, etc.).
* **Self-Documentation:** Metadata is automatically extracted in a standardized and robust manner.
* **Ready-to-Use Implementations:** Advanced functionalities with minimal boilerplate, suitable for a wide range of ML applications.


## 📦 Installation

**Requirements**
The library only requires recent versions of **PyTorch** and **NumPy**. Tracker dependencies are optional.

**Commands**

```bash
pip install drytorch
```
or:
```bash
uv add drytorch
```

## 🗂️ Library Organization
Folders are organized as follows:

- **Core (`core`):** The library kernel. Contains the **Event System**, **Protocols** for component communication, and internal safety **Checks**.
- **Standard Library (`lib`):** Reusable implementations and abstract classes of the protocols.
- **Trackers (`tracker`):** Optional tracker plugins that integrate via the event system.
- **Contributions (`contrib`):** Dedicated space for community-driven extensions.
- **Utilities (`utils`):** Functions and classes independent to the framework.

## 📚 Documentation

**[Read the full documentation on Read the Docs →](https://drytorch.readthedocs.io/)**

The documentation includes:
- **[Tutorials](https://drytorch.readthedocs.io/en/latest/tutorials.html)** - Complete walkthrough
- **[API Reference](https://drytorch.readthedocs.io/en/latest/api.html)** - Detailed API documentation
- **[Architecture Overview](https://drytorch.readthedocs.io/en/latest/architecture.html)** - Design principles and structure


## 📝 **[Changelog](https://github.com/nverchev/drytorch/blob/master/)**
