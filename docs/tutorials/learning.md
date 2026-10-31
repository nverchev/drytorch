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

# Learning Schema

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/learning.ipynb)

A [`LearningSchema`](#drytorch.lib.learning.LearningSchema) specifies how a trainer updates the model.

## Design
A learning schema satisfies [`LearningProtocol`](#drytorch.core.protocols.LearningProtocol), specifying the optimizer class
and its arguments, the base learning rate, a scheduler, and a gradient operation.
The learning rate can still change during training through the trainer, which
records the change (see the
[training](training.md)
tutorial).

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Set up
The examples train a two-layer model on samples of a line.

```{code-cell} ipython3
:tags: [hide-input]

import torch

from torch.utils import data

from drytorch import DataLoader, Experiment, Loss, Model, Trainer
from drytorch.core import exceptions


class EncoderHead(torch.nn.Module):
    """A linear encoder followed by a linear head."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.encoder: torch.nn.Linear = torch.nn.Linear(1, 4)
        self.head: torch.nn.Linear = torch.nn.Linear(4, 1)
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the inputs and apply the head.

        Args:
            inputs: input tensor.

        Returns:
            Output tensor.
        """
        return self.head(self.encoder(inputs))


class LineDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Samples of the line y = 2x + 1."""

    def __init__(self, n_samples: int) -> None:
        """Initialize.

        Args:
            n_samples: number of samples to generate.
        """
        self.inputs: torch.Tensor = torch.rand(n_samples, 1)
        self.targets: torch.Tensor = 2 * self.inputs + 1
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


experiment = Experiment(None, name='Learning', par_dir='experiments')
run = experiment.create_run()
run.start()
loader = DataLoader(LineDataset(32), batch_size=8)
loss = Loss(squared_error, name='MSE')
```

## The LearningSchema class
[`LearningSchema`](#drytorch.lib.learning.LearningSchema) provides the standard implementation of [`LearningProtocol`](#drytorch.core.protocols.LearningProtocol),
combining the optimizer class, the base learning rate, a scheduler, and a
gradient operation:

```{code-cell} ipython3
import torch

from drytorch import LearningSchema
from drytorch.lib import gradient_ops, schedulers


schema = LearningSchema(
    optimizer_cls=torch.optim.Adam,
    base_lr=0.01,
    optimizer_defaults={'betas': (0.9, 0.999)},
    scheduler=schedulers.CosineScheduler(decay_steps=20),
    gradient_op=gradient_ops.NormClipper(1.0),
)
```

### Convenience constructors
The class methods [`adam`](#drytorch.lib.learning.LearningSchema.adam), [`adam_w`](#drytorch.lib.learning.LearningSchema.adam_w), [`r_adam`](#drytorch.lib.learning.LearningSchema.r_adam), and [`sgd`](#drytorch.lib.learning.LearningSchema.sgd) preconfigure standard
optimizers with their default options:

```{code-cell} ipython3
adam_schema = LearningSchema.adam(
    base_lr=0.01,
    scheduler=schedulers.CosineScheduler(decay_steps=20),
    gradient_op=gradient_ops.NormClipper(1.0),
)
adam_w = LearningSchema.adam_w(base_lr=0.001, weight_decay=1e-2)
sgd = LearningSchema.sgd(base_lr=0.01, momentum=0.9, nesterov=True)
```

### Learning rates by submodule
`base_lr` also accepts a mapping from the names of the submodules to their
learning rates. The submodules must contain the same parameters as the module:
otherwise, creating the trainer raises `MissingParamError`.

```{code-cell} ipython3
by_submodule = LearningSchema.adam(base_lr={'encoder': 0.01, 'head': 0.001})
trainer = Trainer(
    Model(EncoderHead()),
    loader=loader,
    loss=loss,
    learning_schema=by_submodule,
)
trainer.train(1)
encoder_only = LearningSchema.adam(base_lr={'encoder': 0.01})
try:
    Trainer(
        Model(EncoderHead()),
        loader=loader,
        loss=loss,
        learning_schema=encoder_only,
    )
except exceptions.MissingParamError:
    pass
else:
    raise AssertionError('The head parameters have no learning rate.')
```

## Schedulers
A scheduler takes the base learning rate and the epoch and returns the
learning rate to use. The trainer calls it at the start of each epoch, with the
epoch the model is about to reach.

The `schedulers` module provides:

* [`ConstantScheduler`](#drytorch.lib.schedulers.ConstantScheduler): the default, which keeps the base learning rate.
* [`ExponentialScheduler`](#drytorch.lib.schedulers.ExponentialScheduler): multiplies the learning rate by a factor each epoch.
* [`CosineScheduler`](#drytorch.lib.schedulers.CosineScheduler): follows a cosine curve for a number of epochs.
* [`PolynomialScheduler`](#drytorch.lib.schedulers.PolynomialScheduler): decays with a power of the remaining epochs.
* [`StepScheduler`](#drytorch.lib.schedulers.StepScheduler): multiplies the learning rate by a factor at given epochs.

Decaying schedulers accept `min_decay`, the minimum fraction of the base rate.

```{code-cell} ipython3
import math

from drytorch.lib import schedulers


step = schedulers.StepScheduler(gamma=0.1, milestones=[10, 20])
cosine = schedulers.CosineScheduler(decay_steps=20, min_decay=0.01)
if not math.isclose(step(1.0, 15), 0.1):
    raise AssertionError('One milestone should have passed.')

if not math.isclose(cosine(1.0, 20), 0.01):
    raise AssertionError('The cosine decay should end at the minimum.')
```

### Transformations
The [`compose`](#drytorch.lib.schedulers.AbstractScheduler.compose) method applies a transformation to a scheduler:

* [`warmup`](#drytorch.lib.schedulers.warmup)`(warmup_steps)` raises the learning rate linearly from zero for a
  number of epochs, then starts the schedule.
* [`restart`](#drytorch.lib.schedulers.restart)`(restart_interval, restart_fraction, max_restart)` resets the
  schedule every `restart_interval` epochs, scaling the base rate by
  `restart_fraction` at each restart (up to `max_restart` times).
* [`rescale`](#drytorch.lib.schedulers.rescale)`(factor)` multiplies the learning rate by a factor.

The result is the transformation, which wraps the scheduler it changes.
Composing several transformations nests them in order, and the metadata
records each one with its own settings.

```{code-cell} ipython3
warm_cosine = cosine.compose(schedulers.warmup(5))
restarting = schedulers.ExponentialScheduler(exp_decay=0.9).compose(
    schedulers.restart(10, restart_fraction=0.5)
)
if not math.isclose(warm_cosine(1.0, 0), 0.0):
    raise AssertionError('The warmup should start from zero.')

if not math.isclose(warm_cosine(1.0, 5), 1.0):
    raise AssertionError('The schedule should start after the warmup.')

if not math.isclose(restarting(1.0, 11), 0.5 * 0.9):
    raise AssertionError('The restart should use half the learning rate.')

if 'warmup_steps=5' not in repr(warm_cosine):
    raise AssertionError('The representation should list the warmup.')
```

### Custom schedulers
A scheduler must satisfy [`SchedulerProtocol`](#drytorch.core.protocols.SchedulerProtocol), for example by subclassing
[`AbstractScheduler`](#drytorch.lib.schedulers.AbstractScheduler), which provides [`compose`](#drytorch.lib.schedulers.AbstractScheduler.compose) and argument validation.

```{code-cell} ipython3
import dataclasses

from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class HalvingScheduler(schedulers.AbstractScheduler):
    """Halve the learning rate at a fixed interval."""

    interval: int

    @override
    def _compute(self, base_lr: float, epoch: int) -> float:
        return base_lr * 0.5 ** (epoch // self.interval)


halving = HalvingScheduler(interval=10).compose(schedulers.warmup(2))
halving_schema = LearningSchema.sgd(base_lr=0.1, scheduler=halving)
if not math.isclose(halving(1.0, 22), 0.25):
    raise AssertionError('Two halvings should have passed after the warmup.')
```

## Gradient operations
A gradient operation receives the parameters of the model after the backward
pass and before the optimizer step, and changes their gradients. The default,
[`NoOp`](#drytorch.lib.gradient_ops.NoOp), leaves them as they are. The `gradient_ops` module provides:

* [`NormClipper`](#drytorch.lib.gradient_ops.NormClipper) and [`ValueClipper`](#drytorch.lib.gradient_ops.ValueClipper): clip the norm of the gradients, or
  each of their values, to a threshold.
* [`ParamNormalizer`](#drytorch.lib.gradient_ops.ParamNormalizer) and [`ZScoreNormalizer`](#drytorch.lib.gradient_ops.ZScoreNormalizer): rescale the gradient of each
  parameter to unit norm, or standardize it.
* [`HistClipper`](#drytorch.lib.gradient_ops.HistClipper) and [`ParamHistClipper`](#drytorch.lib.gradient_ops.ParamHistClipper): clip the gradient norm when it is an
  outlier compared to the previous ones, for all the parameters together or for
  each parameter.

```{code-cell} ipython3
from drytorch.lib import gradient_ops


parameter = torch.nn.Parameter(torch.zeros(3))
parameter.grad = torch.tensor([3.0, 4.0, 0.0])
gradient_ops.NormClipper(1.0)([parameter])
if not torch.isclose(parameter.grad.norm(), torch.tensor(1.0)):
    raise AssertionError('The gradient norm should be clipped to 1.')
```

### Adaptive clipping
[`HistClipper`](#drytorch.lib.gradient_ops.HistClipper) and [`ParamHistClipper`](#drytorch.lib.gradient_ops.ParamHistClipper) first collect the gradient norms for
`n_warmup_steps` steps, clipping with `warmup_clip_strategy` in the meantime.
Afterwards, a criterion decides when a norm is an outlier and how much to clip:

* [`ZStatCriterion`](#drytorch.lib.gradient_ops.ZStatCriterion) (the default) tracks the mean and the variance of the norms
  and clips when the Z-score is above `z_thresh`.
* [`EMACriterion`](#drytorch.lib.gradient_ops.EMACriterion) tracks the mean and clips when the norm is above `r_thresh`
  times the mean.

The criterion's `clipping_function` determines the clipped value:
[`max_clipping`](#drytorch.lib.gradient_ops.max_clipping) caps outliers at the threshold, [`mean_clipping`](#drytorch.lib.gradient_ops.mean_clipping) resets them
to the running mean, and [`reciprocal_clipping`](#drytorch.lib.gradient_ops.reciprocal_clipping) suppresses larger spikes
more aggressively by scaling inversely with their size.

```{code-cell} ipython3
adaptive_clipper = gradient_ops.HistClipper(
    criterion=gradient_ops.EMACriterion(
        r_thresh=1.5, clipping_function=gradient_ops.mean_clipping
    ),
    warmup_clip_strategy=gradient_ops.NormClipper(5.0),
    n_warmup_steps=10,
)
```

### Custom gradient operations
Any callable that takes the parameters works as a gradient operation.

```{code-cell} ipython3
from collections.abc import Iterable


def zero_nan_gradients(params: Iterable[torch.nn.Parameter]) -> None:
    """Replace missing gradient values with zeros.

    Args:
        params: model parameters to sanitize.
    """
    for param in params:
        if param.grad is not None:
            torch.nan_to_num_(param.grad, nan=0.0)

    return


parameter.grad = torch.tensor([float('nan'), 1.0, 2.0])
zero_nan_gradients([parameter])
if not parameter.grad[0] == 0.0:
    raise AssertionError('The missing value should be zero.')
```

## Stop the run

```{code-cell} ipython3
run.stop()
```
