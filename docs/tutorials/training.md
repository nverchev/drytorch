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

# Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/training.ipynb)

This tutorial introduces model training, evaluation and testing workflows in DRYTorch.

## Design
A training workflow coordinates sources (such as trainers and evaluations) and
lifecycle hooks through modular, composable components that report metrics
through a unified event system.

### Terminology
* **Source ([`Source`](#drytorch.lib.runners.Source)):** The execution unit that pairs a model with a data
  loader and an objective, running batches and acting as the origin of
  [`MetricEvent`](#drytorch.core.log_events.MetricEvent)s dispatched to trackers.
* **Evaluation ([`Evaluation`](#drytorch.lib.evaluations.Evaluation)):** The base evaluation source that runs the
  model in inference mode for diagnostics and metrics computation.
* **Validation ([`Validation`](#drytorch.lib.evaluations.Validation)):** An evaluation source designed for held-out
  data during training (steering early stopping, hyperparameter tuning,
  and checkpoint selection).
* **Test ([`Test`](#drytorch.lib.evaluations.Test)):** An evaluation source satisfying [`TestProtocol`](#drytorch.core.protocols.TestProtocol) executed
  outside the training loop, publishing test events.
* **Trainer ([`Trainer`](#drytorch.lib.training.Trainer)):** A training source satisfying [`TrainerProtocol`](#drytorch.core.protocols.TrainerProtocol) that
  optimizes model parameters across training epochs, manages gradient updates,
  and coordinates hooks.
* **Hook ([`TrainerHook`](#drytorch.lib.hooks.TrainerHook)):** A callback satisfying [`HookProtocol`](#drytorch.core.protocols.HookProtocol)
  that binds to the trainer and executes before or after training epochs.

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch optuna
```

## Set up
The examples train a linear model on noisy samples of a line, with a
mean-squared-error loss and a mean-absolute-error metric (as in the
[objectives](objectives.md)
tutorial).

```{code-cell} ipython3
:tags: [hide-input]

import torch

from torch.utils import data

from drytorch import (
    DataLoader,
    Experiment,
    LearningSchema,
    Loss,
    Metric,
    Model,
    Trainer,
)


class LineDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Noisy samples of the line y = 2x + 1."""

    def __init__(self, n_samples: int) -> None:
        """Initialize.

        Args:
            n_samples: number of samples to generate.
        """
        self.inputs: torch.Tensor = torch.rand(n_samples, 1)
        self.targets: torch.Tensor = (
            2 * self.inputs + 1 + 0.1 * torch.randn(n_samples, 1)
        )
        return

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input and the target of a sample."""
        return self.inputs[index], self.targets[index]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.inputs)


def squared_error(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the squared error of each sample.

    Args:
        outputs: predicted model outputs.
        targets: ground truth targets.

    Returns:
        Squared error per sample.
    """
    return (outputs - targets).pow(2).flatten(1).mean(1)


def absolute_error(
    outputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return the absolute error of each sample.

    Args:
        outputs: predicted model outputs.
        targets: ground truth targets.

    Returns:
        Absolute error per sample.
    """
    return (outputs - targets).abs().flatten(1).mean(1)


def make_model() -> Model[torch.Tensor, torch.Tensor]:
    """Return a new linear model.

    Returns:
        Model instance wrapping a linear layer.
    """
    return Model(torch.nn.Linear(1, 1))


experiment = Experiment(None, name='Training', par_dir='experiments')
run = experiment.create_run()
run.start()
train_loader = DataLoader(LineDataset(64), batch_size=8)
val_loader = DataLoader(LineDataset(16), batch_size=8)
test_loader = DataLoader(LineDataset(16), batch_size=8)
loss = Loss(squared_error, name='MSE')
mae = Metric(absolute_error, name='MAE', higher_is_better=False)
learning_schema = LearningSchema.adam(base_lr=0.05)
```

## Evaluation
Evaluation sources execute a model over a data loader in inference mode and
publish metric events. They pair a model with a data loader and an objective
satisfying [`ObjectiveProtocol`](#drytorch.core.protocols.ObjectiveProtocol) (such as a [`Metric`](#drytorch.lib.objectives.Metric) or [`MetricCollection`](#drytorch.lib.objectives.MetricCollection)) to
compute and report metrics without modifying model parameters:

```{code-cell} ipython3
from drytorch import Test


eval_model = make_model()
test = Test(eval_model, loader=test_loader, metric=mae)
test()
```

## Trainer
A [`Trainer`](#drytorch.lib.training.Trainer) executes the optimization loop over training batches. Because it
must backpropagate gradients and update model weights, a [`Trainer`](#drytorch.lib.training.Trainer) requires the
stricter [`LossProtocol`](#drytorch.core.protocols.LossProtocol) (such as [`Loss`](#drytorch.lib.objectives.Loss), [`LossWithMetrics`](#drytorch.lib.objectives.LossWithMetrics), or [`DictLoss`](#drytorch.lib.objectives.DictLoss)). See
the [models](models.md),
[learning](learning.md), and
[data](data.md) tutorials for
details on configuring models, learning schemas, and data loaders.

* `train(n_epochs)` trains for a specified number of epochs, advancing
  `trainer.model.epoch`.
* `train_until(epoch)` trains until the model reaches a given epoch count.
* `update_learning_rate(...)` adjusts the base learning rate or active scheduler.
* [`save_checkpoint()`](#drytorch.core.protocols.TrainerProtocol.save_checkpoint) and [`load_checkpoint(epoch=-1)`](#drytorch.core.protocols.TrainerProtocol.load_checkpoint) save and restore model
  weights, optimizer state, and the current epoch (see the
  [models](models.md)
  tutorial).
* `terminate_training(reason)` permanently stops training and records the event.

```{code-cell} ipython3
trainer = Trainer(
    make_model(),
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
)
trainer.train(2)
trainer.train_until(4)
if trainer.model.epoch != 4:
    raise AssertionError('The model should have reached epoch 4.')

trainer.update_learning_rate(base_lr=0.01)
trainer.terminate_training('The direct trainer example is complete.')
if not trainer.terminated:
    raise AssertionError('The trainer should be terminated.')
```

## Builder pattern and validation
Constructing a trainer via [`Trainer.builder`](#drytorch.lib.training.Trainer.builder) is intuitive when assembling
components incrementally from configuration flags or hyperparameter sweeps.
More importantly, the builder wires sensible defaults across components:
[`add_validation`](#drytorch.lib.training.TrainerBuilder.add_validation) attaches periodic validation as a post-epoch hook, while
[`builder.watch()`](#drytorch.lib.training.TrainerBuilder.watch) automatically targets that validation's metric, providing a
ready-to-use default for monitoring hooks. Once all settings and hooks are
collected, [`build()`](#drytorch.lib.training.TrainerBuilder.build) returns the final immutable [`Trainer`](#drytorch.lib.training.Trainer).

```{code-cell} ipython3
builder = Trainer.builder(
    make_model(),
    name='BuiltTrainer',
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
)
builder.add_validation(val_loader, validate_every=2, val_name='BuiltValidation')
built_trainer = builder.build()
built_trainer.train(2)
```

## Lifecycle hooks
A hook performs actions at specific points in the training lifecycle.

### Execution order
Hooks are executed in the order they are registered:
* **Pre-epoch hooks (`pre_epoch_hooks`)**: run before each epoch begins.
* **Post-epoch hooks (`post_epoch_hooks`)**: run after each epoch finishes.

### Built-in hook helpers
The builder provides convenient methods to configure common hooks:
* `add_saving(save_every=...)`: periodically saves checkpoints.
* `add_early_stopping(patience=..., min_delta=...)`: stops training when
  performance ceases improving.
* `add_reduce_lr_on_plateau(factor=..., patience=...)`: reduces the
  learning rate when a metric plateaus.
* `add_pruning(thresholds=...)`: prunes unpromising trials that fail
  intermediate performance thresholds.

By default, monitoring hooks configured through these helpers target
[`builder.watch()`](#drytorch.lib.training.TrainerBuilder.watch), automatically tracking the validation metric (or the
training loss if no validation was added):

```{code-cell} ipython3
helper_builder = Trainer.builder(
    make_model(),
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
)
helper_builder.add_validation(val_loader, validate_every=1)
helper_builder.add_early_stopping(patience=2)
helper_builder.add_saving(save_every=2)
helper_trainer = helper_builder.build()
helper_trainer.train(2)
```

## Watch settings and condition logic
The `when` argument of any hook takes a [`Condition`](#drytorch.lib.monitoring.Condition) that determines whether the
hook runs at each invocation.

### Built-in conditions
The `monitoring` module provides standard condition primitives:
* [`Every`](#drytorch.lib.monitoring.Every)`(interval, start)`: runs every `interval` epochs from `start`, and when
  training terminates.
* [`After`](#drytorch.lib.monitoring.After)`(epoch)`: runs from a given epoch onward.
* [`AtEpochs`](#drytorch.lib.monitoring.AtEpochs)`(epochs)`: runs only at specified epochs.
* [`OnTermination`](#drytorch.lib.monitoring.OnTermination)`()`: runs when training is terminated.
* [`Cooldown`](#drytorch.lib.monitoring.Cooldown)`(epochs)`: runs after a specified interval of inactivity.

### Logical operators
Conditions can be combined using standard Python bitwise operators:
* `condition1 & condition2`: holds when both conditions hold (AND).
* `condition1 | condition2`: holds when either condition holds (OR).
* `~condition`: holds when the condition does not hold (NOT).

```{code-cell} ipython3
from drytorch.lib import hooks, monitoring


save_every_two = hooks.SavingHook(when=monitoring.Every(2))
save_after_three = hooks.SavingHook(
    when=monitoring.After(3) & monitoring.Every(2)
)
```

### Following a metric with Watch
Some conditions inspect metrics published by a source:
* [`Plateau`](#drytorch.lib.monitoring.Plateau)`(patience, min_delta)`: holds when a metric fails to improve over
  `patience` evaluation steps.
* [`WorseThan`](#drytorch.lib.monitoring.WorseThan)`(thresholds)`: holds when a metric misses defined numerical
  thresholds at specific epochs.

A [`Watch`](#drytorch.lib.monitoring.Watch) object specifies what to follow:
* `source`: the source computing the metric (defaults to the trainer).
* `metric_name`: the name of the metric (needed when the objective yields multiple metrics).
* `best_is`: whether higher or lower values are better (needed when not specified by the source's objective).
* `filter_fn`: how recent values are aggregated (e.g. trailing means from
  `drytorch.utils.averages`).

The helper [`builder.watch(...)`](#drytorch.lib.training.TrainerBuilder.watch) configures a [`Watch`](#drytorch.lib.monitoring.Watch) bound to the builder's
validation (or to training if `training=True`). Note that hooks monitoring a
validation must run after that validation in the post-epoch sequence.

```{code-cell} ipython3
from drytorch.utils import averages


builder = Trainer.builder(
    make_model(),
    name='MonitoredTrainer',
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
)
builder.add_validation(val_loader, val_name='MonitoredValidation')
builder.add_reduce_lr_on_plateau(factor=0.5, patience=1)
builder.add_early_stopping(
    patience=2, watch=builder.watch(filter_fn=averages.get_trailing_mean(2))
)
builder.add_pruning(thresholds={3: 100.0})
builder.add_post_epoch_hook(save_every_two)
builder.add_post_epoch_hook(save_after_three)
monitored_trainer = builder.build()
monitored_trainer.train(6)
```

Without the builder, you can pass the source directly to [`monitoring.Watch`](#drytorch.lib.monitoring.Watch) and
register an [`EvaluationHook`](#drytorch.lib.hooks.EvaluationHook) before the monitoring hooks:

```{code-cell} ipython3
from drytorch import Validation


direct_model = make_model()
direct_validation = Validation(
    direct_model, name='DirectValidation', loader=val_loader, metric=loss.copy()
)
early_stopping = hooks.EarlyStopping(
    patience=2, watch=monitoring.Watch(source=direct_validation)
)
direct_monitored = Trainer(
    direct_model,
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
    post_epoch_hooks=[hooks.EvaluationHook(direct_validation), early_stopping],
)
direct_monitored.train(2)
```

## Custom hooks and conditions
When built-in callbacks are not enough, you can extend the training lifecycle
with custom hooks and trigger conditions.

### Custom hook
A hook must satisfy [`HookProtocol`](#drytorch.core.protocols.HookProtocol), for example by subclassing [`TrainerHook`](#drytorch.lib.hooks.TrainerHook):
it binds the trainer to `self.trainer` and checks `when` conditions
automatically.

Once registered, hooks cannot be modified, and a hook bound to one trainer
cannot be rebound to another (raising `HookAlreadyBoundError`).

This example defines a custom hook that records epochs as they finish:

```{code-cell} ipython3
from drytorch.core import exceptions


class RecordEpoch(hooks.TrainerHook[monitoring.Condition]):
    """Record the epoch the model has reached."""

    def __init__(
        self,
        when: monitoring.Condition = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            when: condition controlling when the hook runs.
        """
        super().__init__(when=when)
        self._epochs: list[int] = []
        return

    @property
    def epochs(self) -> tuple[int, ...]:
        """Epochs reached so far."""
        return tuple(self._epochs)

    def run(self) -> None:
        """Add the current epoch."""
        self._epochs.append(self.trainer.model.epoch)
        return


record_hook = RecordEpoch()
hook_trainer = Trainer(
    make_model(),
    name='HookTrainer',
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
    post_epoch_hooks=[record_hook],
)
hook_trainer.train(2)
if record_hook.epochs != (1, 2):
    raise AssertionError('The hook should run after each epoch.')

try:
    Trainer(
        make_model(),
        loader=train_loader,
        loss=loss,
        learning_schema=learning_schema,
        post_epoch_hooks=[record_hook],
    )
except exceptions.HookAlreadyBoundError:
    pass
else:
    raise AssertionError('A hook cannot belong to multiple trainers.')
```

### Custom condition
To define a custom condition, subclass [`monitoring.Condition`](#drytorch.lib.monitoring.Condition) and implement
`__call__(self, hook)`. Conditions can also override [`bind_trainer`](#drytorch.lib.monitoring.Condition.bind_trainer), which is
called once when the trainer binds the hook, and [`observe`](#drytorch.lib.monitoring.Condition.observe), called on each
invocation before `__call__`.

In this example, `TimeBudget` holds when elapsed wall-clock time exceeds a
budget. By registering [`TerminateTraining`](#drytorch.lib.hooks.TerminateTraining) as a pre-epoch hook with a zero
budget, training terminates immediately before epoch 1 runs:

```{code-cell} ipython3
import time

from typing import Any

from typing_extensions import override

from drytorch.core import protocols as p


class TimeBudget(monitoring.Condition):
    """Hold after a given number of seconds have elapsed."""

    def __init__(self, seconds: float) -> None:
        """Initialize.

        Args:
            seconds: time budget in seconds.
        """
        self.seconds: float = seconds
        self._start: float = 0.0
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the budget is spent.

        Args:
            hook: hook being evaluated.

        Returns:
            True if budget is spent.
        """
        return time.perf_counter() - self._start > self.seconds

    @override
    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Start the clock.

        Args:
            trainer: trainer instance binding the condition.
        """
        super().bind_trainer(trainer)
        self._start = time.perf_counter()
        return


budget_trainer = Trainer(
    make_model(),
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
    pre_epoch_hooks=[hooks.TerminateTraining(when=TimeBudget(0.0))],
)
budget_trainer.train(3)
if budget_trainer.model.epoch != 0:
    raise AssertionError('The budget should stop training before epoch 1.')
```

## Hyperparameter optimization with Optuna
[`TrialCallback`](#drytorch.contrib.optuna.TrialCallback) (in `drytorch.contrib.optuna`) reports intermediate metrics to
an [Optuna](https://optuna.org) [`Trial`](#optuna.trial.Trial) and terminates training when Optuna
prunes the run. Each trial trains a fresh model within an isolated run.

[`get_final_value`](#drytorch.contrib.optuna.get_final_value) retrieves the trial's evaluation result (by default, the best
metric value according to the [`Study`](#optuna.study.Study)'s optimization direction) for use with
[`study.optimize`](#optuna.study.Study.optimize).

```{code-cell} ipython3
import optuna

from drytorch.contrib.optuna import TrialCallback, get_final_value


optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial) -> float:
    """Train with the suggested learning rate and return its validation loss.

    Args:
        trial: the trial suggesting the values.

    Returns:
        The value optuna compares across trials.
    """
    base_lr = trial.suggest_float('base_lr', 1e-3, 1e-1, log=True)
    builder = Trainer.builder(
        make_model(),
        name=f'Trial{trial.number}',
        loader=train_loader,
        loss=loss,
        learning_schema=LearningSchema.adam(base_lr=base_lr),
    )
    builder.add_validation(val_loader)
    builder.add_post_epoch_hook(TrialCallback(trial, watch=builder.watch()))
    builder.build().train(3)
    return get_final_value(trial)


study = optuna.create_study(
    direction='minimize', pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=3)
if 'base_lr' not in study.best_params:
    raise AssertionError('The study should find a learning rate.')
```

## Stop the run

```{code-cell} ipython3
run.stop()
```
