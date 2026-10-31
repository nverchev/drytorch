![drytorch_logo.png](https://raw.githubusercontent.com/nverchev/drytorch/main/docs/_static/drytorch_logo.png)
[![PyPI version](https://img.shields.io/pypi/v/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![Total Downloads](https://img.shields.io/pypi/dm/drytorch?label=downloads&style=flat)](https://pypi.org/project/drytorch/)
[![Python](https://img.shields.io/pypi/pyversions/drytorch.svg?style=flat)](https://pypi.org/project/drytorch/)
[![License](https://img.shields.io/pypi/l/drytorch.svg)](https://github.com/nverchev/drytorch/blob/master/LICENSE)
[![CI Status](https://github.com/nverchev/drytorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/nverchev/drytorch/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/nverchev/drytorch/graph/badge.svg?token=CZND67KAW1)](https://codecov.io/github/nverchev/drytorch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![Documentation Status](https://readthedocs.org/projects/drytorch/badge/?version=latest)](https://drytorch.readthedocs.io/latest/?badge=latest)
# DRYTorch
Reproducible machine learning experiments with PyTorch.

## Design
Don't Repeat Yourself (DRY) principles: replicable, documented, reusable.

* **Reproducibility:** experimental isolation to prevent unintended dependencies, data leakage, and misconfiguration.
* **Modularity:** flexible protocols preserving type inference in custom implementations.
* **Type Checking:** support for type checkers (such as basedpyright) for static detection of incompatible components.
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

**uv:**
```bash
uv add drytorch
```

## Project Template

To start a new project, use
**[drytorch-template](https://github.com/nverchev/drytorch_template)**. It is a
complete machine learning project, with Hydra configuration, custom trackers, Optuna
tuning and distributed training, and its README explains which files to change
for your task.

## Package Structure
Modules are organized into the following subpackages:

- **`core`:** internal routines and the interfaces for library and custom components.
- **`lib`:** reusable implementations and abstract classes of the protocols.
- **`trackers`:** optional tracker plugins that integrate via the event system.
- **`contrib`:** community-driven extensions and support for external libraries.
- **`utils`:** general utilities independent to the framework.


## Documentation

Comprehensive documentation is available on **[Read the Docs](https://drytorch.readthedocs.io/)**.

### Tutorials

| Tutorial | Description |
| :--- | :--- |
| [Getting Started](https://drytorch.readthedocs.io/latest/tutorials/getting_started.html) | End-to-end tutorial on deep image prior super-resolution. |
| [Experiments and Runs](https://drytorch.readthedocs.io/latest/tutorials/experiments_and_runs.html) | Defining experiments, run sessions, and registries. |
| [Data](https://drytorch.readthedocs.io/latest/tutorials/data.html) | Data loaders and dataset splitting. |
| [Models](https://drytorch.readthedocs.io/latest/tutorials/models.html) | Module wrappers and checkpointing. |
| [Objectives](https://drytorch.readthedocs.io/latest/tutorials/objectives.html) | Losses, metrics, and accumulators. |
| [Learning](https://drytorch.readthedocs.io/latest/tutorials/learning.html) | Optimizers, learning rate schemas, and schedulers. |
| [Training](https://drytorch.readthedocs.io/latest/tutorials/training.html) | Trainer configuration, validation, hooks, and Optuna tuning. |
| [Trackers](https://drytorch.readthedocs.io/latest/tutorials/trackers.html) | Logging metrics, plots, and run states. |
| [Events](https://drytorch.readthedocs.io/latest/tutorials/events.html) | Domain-specific signals, event types, and tracker integration. |
| [Distributed Training](https://drytorch.readthedocs.io/latest/tutorials/distributed_training.html) | Multi-process distributed training with DDP. |
| [Typing Support](https://drytorch.readthedocs.io/latest/tutorials/typing_support.html) | Multi-modal inputs, custom model outputs, and type safety. |

### Reference

- **[API Reference](https://drytorch.readthedocs.io/latest/api.html)**: detailed API documentation.
- **[Protocol Overview](https://drytorch.readthedocs.io/latest/protocols.html)**: design structure and protocols.
- **[Changelog](https://github.com/nverchev/drytorch/blob/main/CHANGELOG.md)**: release history and migration notes.
