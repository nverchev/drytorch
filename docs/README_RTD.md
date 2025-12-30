# DRYTorch's Documentation
[![PyPI version](https://img.shields.io/pypi/v/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![Python](https://img.shields.io/pypi/pyversions/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![License](https://img.shields.io/github/license/nverchev/drytorch.svg)](https://github.com/nverchev/drytorch/blob/master/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-nverchev%2Fdrytorch-blue?logo=github)](https://github.com/nverchev/drytorch)

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
