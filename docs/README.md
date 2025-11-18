![drytorch_logo.png](resources/drytorch_logo.png)
[![PyPI version](https://img.shields.io/pypi/v/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![Total Downloads](https://img.shields.io/pypi/dm/drytorch?label=downloads&style=flat)](https://pypi.org/project/drytorch/)
[![Python](https://img.shields.io/pypi/pyversions/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![License](https://img.shields.io/github/license/nverchev/drytorch.svg)](LICENSE)
[![CI Status](https://github.com/nverchev/drytorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/nverchev/drytorch/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/github/nverchev/drytorch/graph/badge.svg?token=CZND67KAW1)](https://codecov.io/github/nverchev/drytorch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
# DRYTorch
A lean, minimal framework for machine-learning experiments. It enforces best practices and maximizes code reusability via a modular, decoupled design.

## 🌟 Features at a Glance

* **DRY Training Cycle:** Minimal boilerplate for a wide range of ML applications. Includes a hook-based training cycle, adaptive data loading, and simplified checkpointing.
* **Protocol-Driven Modularity:** Components communicate via defined protocols and abstract classes, ensuring type safety and flexibility for custom classes.
* **Decoupled Tracking:** An event system notifies optional external libraries (Hydra, W&B, TensorBoard, etc.). Trackers do not impact the core experiment.
* **Enforced Best Practices:** All logic runs within an **experimental scope** to discourage dependencies, prevent data leakage, and ensure correct configuration.

## 🛠️ Installation

**Requirements**
The library only requires recent versions of **PyTorch** and **NumPy**. Tracker dependencies are optional.

**Installation**
Using pip:
```bash
pip install drytorch
```

Using uv
```bash
uv add drytorch
```

## 🏗️ Library Organization

The library uses a microkernel (plugin) architecture to separate concerns.

1.  **Core (`core`):** The library kernel. Contains the **Event System**, **Protocols** for component communication, and internal safety **Checks**.
2.  **Standard Library (`lib`):** Reusable implementations and abstract classes of the protocols.
3.  **Trackers (`tracker`):** Optional tracker plugins that integrate via the event system.
4.  **Contributions (`contrib`):** Dedicated space for community-driven extensions.
5. **Utilities (`utils`):**
Functions and classes independent to the framework.

## 🚀 Usage

Dive into the full, runnable examples:

<div style="display: flex; justify-content: space-between; align-items: center;">
    <p style="margin: 0;">
        ➡️ <strong><a href="https://github.com/nverchev/drytorch/blob/main/docs/tutorials/getting_started.ipynb">Getting Started Notebook</a></strong>
    </p>
    <a target="_blank" href="https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/getting_started.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle;"/>
    </a>
</div>
