---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Objectives

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/objectives.ipynb)

DRYTorch helps you **standardize and document** your model's metrics and loss.

## Design
The modular design extends to metrics and losses. DRYTorch provides a common
interface for both, allowing you to easily switch between different libraries.

### Terminology
An **objective** is a criterion for model performance evaluation, categorized
into two types:
* **Metric:** Assesses model performance as a proxy for the overall goal.
* **Loss:** Optimizes the model parameters to improve metric assessments.

DRYTorch allows using losses as metrics, but not vice versa.

### Protocols
DRYTorch defines an [`ObjectiveProtocol`](#drytorch.core.protocols.ObjectiveProtocol), used by classes that implement the
validation and testing of a model, and a [`LossProtocol`](#drytorch.core.protocols.LossProtocol), used for its training.

```{code-cell} ipython3
:tags: [skip-execution]

from __future__ import annotations


! uv pip install drytorch torchmetrics torcheval
```

```{code-cell} ipython3
import torch

from drytorch import Experiment


experiment = Experiment(None, name='Objectives', par_dir='experiments')
run = experiment.create_run()
run.start()

tensor_a = torch.ones(1, 1, dtype=torch.float)
tensor_b = 3 * torch.ones(1, 1, dtype=torch.float)
```

## Library classes
DRYTorch objective classes act as wrappers around metric and loss callables.
These callables accept model outputs and targets and return one value per
sample: a tensor of shape `(B,)`, where `B` is the batch size. The objective
averages over all samples, so a batch counts as many times as it has samples.

### The Metric and MetricCollection classes
The [`Metric`](#drytorch.lib.objectives.Metric) class defines a single metric. You can document it by giving it an
explicit name and specifying whether higher or lower values are better. You can
also combine different [`Metric`](#drytorch.lib.objectives.Metric) instances with compatible signatures into a
[`MetricCollection`](#drytorch.lib.objectives.MetricCollection) using the pipe operator (`|`).

```{code-cell} ipython3
from drytorch.lib.objectives import Metric


def mae_loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return batched Mean Absolute Error (MAE) values.

    Args:
        outputs: predicted model outputs.
        targets: ground truth targets.

    Returns:
        MAE values per sample.
    """
    return torch.abs(outputs - targets).flatten(1).mean(1)


mse_metric = Metric(
    torch.nn.functional.mse_loss, name='MSE', higher_is_better=False
)
mae_metric = Metric(mae_loss_fn, name='MAE', higher_is_better=False)
metric_collection = mse_metric | mae_metric
metric_collection.update(tensor_a, tensor_b)
metric_collection.compute()
expected_metric_collection = {
    'MSE': torch.tensor(4.0),
    'MAE': torch.tensor(2.0),
}
if not metric_collection.compute() == expected_metric_collection:
    raise AssertionError('Metrics values should be as expected.')
```

### LossWithMetrics and Loss
[`LossWithMetrics`](#drytorch.lib.objectives.LossWithMetrics) is a collection of metrics with a criterion that extracts the
loss value from them.

[`Loss`](#drytorch.lib.objectives.Loss) is a subclass equivalent to [`Metric`](#drytorch.lib.objectives.Metric): it accepts a single callable that
is used both as the criterion for backpropagation and as a metric.

Algebraic operations between a [`LossWithMetrics`](#drytorch.lib.objectives.LossWithMetrics) instance and a scalar
or another [`LossWithMetrics`](#drytorch.lib.objectives.LossWithMetrics) instance create a new [`LossWithMetrics`](#drytorch.lib.objectives.LossWithMetrics). The new
loss also evaluates its components as metrics, so you can track the performance
of the single components that make up a composed loss. Its `formula` attribute
documents the operations and component losses used.

```{code-cell} ipython3
from drytorch.lib.objectives import Loss


mse_loss = Loss(torch.nn.functional.mse_loss, name='MSE')
mae_loss = Loss(mae_loss_fn, name='MAE')
composed_loss = mse_loss**2 + 0.5 * mae_loss
composed_loss.update(tensor_a, tensor_b)
expected_metrics_from_loss = {
    'Combined Loss': torch.tensor(17.0),
    'MSE': torch.tensor(4.0),
    'MAE': torch.tensor(2.0),
}
if not composed_loss.compute() == expected_metrics_from_loss:
    raise AssertionError('Metrics values should be as expected.')

if composed_loss.formula != '[MSE]^2 + 0.5 x [MAE]':
    raise AssertionError('Formula mismatch.')
```

### Joining objectives with join
The [`join`](#drytorch.lib.objectives.join) function combines multiple objectives. If the first argument is a
loss, [`join(loss, *metrics)`](#drytorch.lib.objectives.join) returns a loss that computes the metrics
alongside the criterion; otherwise, [`join(*metrics)`](#drytorch.lib.objectives.join) returns a metric
collection. In both cases, the original objectives do not change.

Objectives that are not metric collections (such as third-party metrics) are
wrapped in [`JoinedLoss`](#drytorch.lib.objectives.JoinedLoss) or [`JoinedMetrics`](#drytorch.lib.objectives.JoinedMetrics).

```{code-cell} ipython3
from drytorch.lib.objectives import Metric, join


l1_metric = Metric(
    lambda out, tgt: torch.abs(out - tgt).flatten(1).mean(1),
    name='L1',
    higher_is_better=False,
)
joined_loss = join(composed_loss, l1_metric)
joined_loss.update(tensor_a, tensor_b)
expected_joined = {
    'Combined Loss': torch.tensor(17.0),
    'MSE': torch.tensor(4.0),
    'MAE': torch.tensor(2.0),
    'L1': torch.tensor(2.0),
}
if not joined_loss.compute() == expected_joined:
    raise AssertionError('Joined metrics values should be as expected.')
```

### The DictLoss class
[`DictLoss`](#drytorch.lib.objectives.DictLoss) is a subclass of [`LossWithMetrics`](#drytorch.lib.objectives.LossWithMetrics) for functions that calculate the
loss alongside auxiliary metrics in a single callable.

The `name` parameter specifies which key in the returned dictionary provides
the loss value for gradient optimization. All other entries are recorded as
auxiliary metrics.

```{code-cell} ipython3
from drytorch.lib.objectives import DictLoss


def multi_task_fn(
    outputs: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute combined loss and separate error metrics.

    Args:
        outputs: predicted model outputs.
        targets: ground truth targets.

    Returns:
        Dictionary mapping metric names to batch values.
    """
    mse = torch.pow(outputs - targets, 2).flatten(1).mean(1)
    mae = torch.abs(outputs - targets).flatten(1).mean(1)
    return {'Loss': mse + 0.1 * mae, 'MSE': mse, 'MAE': mae}


dict_loss = DictLoss(multi_task_fn, name='Loss')
dict_loss.update(tensor_a, tensor_b)
computed_dict_loss = dict_loss.compute()
expected_dict_loss = {
    'Loss': torch.tensor(4.2),
    'MSE': torch.tensor(4.0),
    'MAE': torch.tensor(2.0),
}
if not torch.allclose(computed_dict_loss['Loss'], expected_dict_loss['Loss']):
    raise AssertionError('Loss value does not match.')

if not torch.allclose(computed_dict_loss['MSE'], expected_dict_loss['MSE']):
    raise AssertionError('Auxiliary MSE does not match.')

if not torch.allclose(computed_dict_loss['MAE'], expected_dict_loss['MAE']):
    raise AssertionError('Auxiliary MAE does not match.')
```

## External classes
DRYTorch does not re-implement common metrics or losses. Instead, its
protocols ensure full compatibility with classes from popular existing
libraries like [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)
and [TorchEval](https://docs.pytorch.org/torcheval/stable/).

### Compatibility with TorchMetrics and TorchEval
The [`ObjectiveProtocol`](#drytorch.core.protocols.ObjectiveProtocol) is compatible with [`Metric`](#torchmetrics.Metric) classes from TorchMetrics (such as
[`MeanSquaredError`](#torchmetrics.MeanSquaredError)) and TorchEval (such as [`MeanSquaredError`](#torcheval.metrics.MeanSquaredError)).
You can use instances of these third-party metric classes directly when defining a DRYTorch validation or test step.

The [`LossProtocol`](#drytorch.core.protocols.LossProtocol) is designed to accept any callable or class that computes a
differentiable scalar, including differentiable metrics from TorchMetrics.

```{code-cell} ipython3
import torchmetrics

from torcheval import metrics as eval_metrics

from drytorch.core import protocols as p


torch_metric = torchmetrics.MeanSquaredError()
eval_metric = eval_metrics.MeanSquaredError()


def is_valid_objective(
    metric: p.ObjectiveProtocol[torch.Tensor, torch.Tensor],
) -> bool:
    """Test metric follows the Objective protocol.

    Args:
        metric: objective instance to test.

    Returns:
        True if the metric follows the ObjectiveProtocol.
    """
    return isinstance(metric, p.ObjectiveProtocol)


def is_valid_loss(
    metric: p.LossProtocol[torch.Tensor, torch.Tensor],
) -> bool:
    """Test metric follows the Loss protocol.

    Args:
        metric: loss instance to test.

    Returns:
        True if the metric follows the LossProtocol.
    """
    return isinstance(metric, p.LossProtocol)


torch_metric.update(tensor_a, tensor_b)
eval_metric.update(tensor_a, tensor_b)
if not torch.isclose(torch_metric.compute(), eval_metric.compute()):
    raise AssertionError('Metrics values should match.')

if not (is_valid_objective(eval_metric) and is_valid_objective(torch_metric)):
    raise AssertionError('These objects should follow the ObjectiveProtocol.')

if not is_valid_loss(torch_metric):
    raise AssertionError('This object should also follow the LossProtocol.')
```

### Compositional metrics with from_torchmetrics
TorchMetrics offers `CompositionalMetric` with support for algebraic
operations. To wrap it with DRYTorch naming, component inspection, and
documentation, use [`from_torchmetrics`](#drytorch.contrib.torchmetrics.from_torchmetrics):

```{code-cell} ipython3
from drytorch.contrib.torchmetrics import from_torchmetrics


new_metric = 1 + torch_metric
imported_metric = from_torchmetrics(new_metric)
imported_metric.update(tensor_a, tensor_b)
expected_metrics_from_torchmetrics = {
    'Loss': torch.tensor(5.0),
    'MeanSquaredError': torch.tensor(4.0),
}
if not imported_metric.compute() == expected_metrics_from_torchmetrics:
    raise AssertionError('Metrics values should be as expected.')
```

### Wrapping TorchEval with from_torcheval
To name TorchEval metrics explicitly, specify whether higher is better, and
return a metric dictionary from [`compute()`](#drytorch.core.protocols.ObjectiveProtocol.compute), use [`from_torcheval`](#drytorch.contrib.torcheval.from_torcheval):

```{code-cell} ipython3
from drytorch.contrib.torcheval import from_torcheval


wrapped_eval = from_torcheval(
    eval_metrics.MeanSquaredError(),
    name='TorchEval_MSE',
    higher_is_better=False,
)
wrapped_eval.update(tensor_a, tensor_b)
expected_eval = {'TorchEval_MSE': torch.tensor(4.0)}
if not wrapped_eval.compute() == expected_eval:
    raise AssertionError('Wrapped metric value mismatch.')
```

## Custom classes
To implement custom objective or accumulation logic, DRYTorch provides base
classes that handle calling logic, metadata extraction, and batch aggregation.

### Custom metric class
To define a custom running average metric, subclass [`AverageObjective`](#drytorch.lib.objectives.AverageObjective) and
override [`calculate`](#drytorch.lib.objectives.LossWithMetrics.calculate). For custom aggregation strategies, subclass [`Objective`](#drytorch.lib.objectives.Objective)
and override `_get_accumulator_cls`:

```{code-cell} ipython3
from typing_extensions import override

from drytorch.lib.objectives import AverageObjective


class MyMetrics(AverageObjective[torch.Tensor, torch.Tensor]):
    """Class to calculate MSE and MAE more efficiently."""

    @override
    def calculate(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        diff = outputs - targets
        return {
            'MSE': torch.pow(diff, 2).flatten(1).mean(1),
            'MAE': torch.abs(diff).flatten(1).mean(1),
        }


my_metrics = MyMetrics()
my_metrics.update(tensor_a, tensor_b)
my_metrics.compute()
if not my_metrics.compute() == expected_metric_collection:
    raise AssertionError('Metrics values should be as before.')
```

### Custom accumulators
[`Objective`](#drytorch.lib.objectives.Objective) subclasses aggregate batch values using an [`AbstractAccumulator`](#drytorch.lib.accumulators.AbstractAccumulator).
[`AverageObjective`](#drytorch.lib.objectives.AverageObjective) uses [`TorchMeanAccumulator`](#drytorch.lib.accumulators.TorchMeanAccumulator) to calculate a sample-weighted
arithmetic mean across batches.

To define a custom aggregation strategy, subclass [`AbstractAccumulator`](#drytorch.lib.accumulators.AbstractAccumulator) and
implement:
- `from_value(cls, value)`: creates an accumulator from a batch value.
- `merge(self, other)`: merges another accumulator of the same class.
- `reduce(self)`: computes the final reduced metric.
- [`sync(self)`](#drytorch.lib.accumulators.AbstractAccumulator.sync): synchronizes state across distributed processes.

Override `_get_accumulator_cls` on an [`Objective`](#drytorch.lib.objectives.Objective) subclass to return your
custom accumulator class.

```{code-cell} ipython3
import dataclasses

from typing import Self

from torch import distributed as dist
from typing_extensions import override

from drytorch.lib.accumulators import AbstractAccumulator
from drytorch.lib.objectives import Objective


@dataclasses.dataclass(slots=True)
class MaxAccumulator(AbstractAccumulator[torch.Tensor, torch.Tensor]):
    """Accumulator tracking the maximum value."""

    max_val: torch.Tensor

    @override
    def merge(self, other: MaxAccumulator) -> None:
        self.max_val = torch.maximum(self.max_val, other.max_val)
        return

    @override
    def reduce(self) -> torch.Tensor:
        return self.max_val

    @override
    def sync(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.max_val, op=dist.ReduceOp.MAX)

        return

    @classmethod
    @override
    def from_value(cls, value: torch.Tensor) -> Self:
        return cls(max_val=value.detach().max())


class MaxObjective(Objective[torch.Tensor, torch.Tensor]):
    """Objective tracking maximum absolute error."""

    @classmethod
    @override
    def _get_accumulator_cls(cls) -> type[MaxAccumulator]:
        return MaxAccumulator

    @override
    def calculate(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            'MaxError': torch.abs(outputs - targets).flatten(1).max(1).values
        }


max_obj = MaxObjective()
max_obj.update(torch.tensor([[1.0]]), torch.tensor([[4.0]]))
max_obj.update(torch.tensor([[1.0]]), torch.tensor([[7.0]]))
if not torch.isclose(max_obj.compute()['MaxError'], torch.tensor(6.0)):
    raise AssertionError('Max accumulator should track the maximum value.')
```

## Stop the run

```{code-cell} ipython3
run.stop()
```

For multi-process metric synchronization across distributed processes, see the
[distributed_training](distributed_training.md) tutorial.
