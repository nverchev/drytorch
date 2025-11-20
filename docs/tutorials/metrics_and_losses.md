---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Metrics and Losses

DRYTorch helps you **standardize and document** your model's metrics and loss.

### Terminology

An **objective** is a criterion for model performance evaluation. We distinguish between two types:

* **Metric:** Assesses model performance as a proxy for the overall goal.
* **Loss:** Optimizes the model parameters to improve metric assessments.

DRYTorch allows, and often **implements** by default, using losses as metrics.

### Protocols

DRYTorch defines an `ObjectiveProtocol`, used by classes that implement the validation and testing of a model, and a `LossProtocol`, used for its training.

+++

## Compatibility with Existing Libraries

DRYTorch **does not re-implement** common metrics or losses. Instead, it defines **protocols** to ensure **full compatibility** with classes from popular existing libraries.


### For Validation and Testing

The `ObjectiveProtocol` is compatible with `Metric` classes from:
* [**TorchMetrics**](https://lightning.ai/docs/torchmetrics/stable/)
* [**TorchEval**](https://docs.pytorch.org/torcheval/stable/)

You can use instances of these third-party metric classes **directly** when defining a DRYTorch validation or test step.

### For Training

The `LossProtocol` is designed to accept any class that meets its requirements, including some metrics. You can therefore use differentiable metrics from libraries like **TorchMetrics** directly when building a DRYTorch training class.


**TorchMetrics** also offers a `CompositionalMetric`, with support for algebra operations, which inspired part of the DRYTorch own implementation. To make it compatible with the framework and better documentation, you can use
`from_torchmetrics`.

```{code-cell} ipython3
import torch
import torchmetrics

from torcheval import metrics as eval_metrics

from drytorch.core import protocols as p


torch_metric = torchmetrics.MeanSquaredError()
eval_metric = eval_metrics.MeanSquaredError()


def supported_for_validation(
    metric: p.ObjectiveProtocol[torch.Tensor, torch.Tensor],
) -> bool:
    """Test metric follows the Objective protocol."""
    return isinstance(metric, p.ObjectiveProtocol)


supported_for_validation(eval_metric) and supported_for_validation(torch_metric)
```

```{code-cell} ipython3
def supported_for_training(
    metric: p.LossProtocol[torch.Tensor, torch.Tensor],
) -> bool:
    """Test metric follows the Loss protocol."""
    return isinstance(metric, p.LossProtocol)


supported_for_training(torch_metric)
```

```{code-cell} ipython3
from drytorch.contrib.torchmetrics import from_torchmetrics


new_metric = torch_metric + torch_metric


imported_metric = from_torchmetrics(new_metric)
```

## DRYTorch implementation
DRYTorch objective classes act as wrappers around **user-defined** metric and loss callables.


These callables must accept model outputs and targets as arguments and return a scalar PyTorch Tensor for an aggregated mini-batch value or a vector of batched values (recommended for more precise averaging across batches of varying sizes). The abstract `Objective` class handles **calling the logic, documenting, and correctly aggregating** the results across batches.


### The Metric and MetricCollection classes
The `Metric` class is to define a single metric. You can document it by giving it an explicit name and specify whether it is better when higher or lower. You can also concatenate different Metric instances with compatible signatures into a `MetricCollection` instance, or creating one directly from a dictionary of named metric functions.

```{code-cell} ipython3
from torch.nn.functional import mse_loss as mse_loss_fn  # returns scalar value

from drytorch.lib.objectives import Metric


def mae_loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Returns batched Meas Absolute Error (MAE) values."""
    return torch.abs(outputs - targets).flatten().mean(1)


mse_metric = Metric(mse_loss_fn, name='MSE', higher_is_better=False)
mae_metric = Metric(mae_loss_fn, 'MAE', higher_is_better=False)
metric_collection = mse_metric | mae_metric
```

### Define a Custom Metric class

You can subclass the abstract `Objective` class by overriding the `calculate` method. In this example, we slightly reduce the calculation overhead to obtain the previous metrics.

```{code-cell} ipython3
from typing_extensions import override

from drytorch.lib.objectives import Objective


class MyMetrics(Objective[torch.Tensor, torch.Tensor]):
    """Class to calculate MSE and MAE more efficiently."""

    @override
    def calculate(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        diff = outputs - targets
        return {
            'MSE': torch.pow(diff, 2).flatten().mean(1),
            'MAE': torch.abs(diff).flatten().mean(1),
        }


my_metrics = MyMetrics()
```

## LossBase, Loss and CompositionalLoss

`LossBase` is the abstract class for concrete loss classes, such as `Loss` and `CompositionalLoss`.

`Loss` is equivalent to Metric and accepts a single callable that is used both as a criterion for backpropagation for the loss and as a metric.


The `CompositionalLoss` class extends this idea by evaluating other metrics besides the main optimization criterion. This allows you to easily document and track the performance of the **single components** that make up a more complex, composed loss function.


It is possible to create a compositional loss by using simple algebraic operations between a `LossBase` instance and an integer, float, or another `LossBase` instance. The resulting object's `formula` attribute documents the specific operations and component losses utilized.

```{code-cell} ipython3
from torch.nn.functional import mse_loss as mse_loss_fn  # returns scalar value

from drytorch.lib.objectives import Loss


def mae_loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Returns batched Meas Absolute Error (MAE) values."""
    return torch.abs(outputs - targets).flatten().mean(1)


mse_loss = Loss(mse_loss_fn, name='MSE', higher_is_better=False)
mae_loss = Loss(mae_loss_fn, 'MAE', higher_is_better=False)
composed_loss = mse_loss**2 + 0.5 * mae_loss

composed_loss.formula
```
