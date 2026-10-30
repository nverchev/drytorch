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

# Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/models.ipynb)

A [`Model`](#drytorch.lib.models.Model) wraps a [`torch.nn.Module`](#torch.nn.Module) to manage device placement, epoch state,
and checkpoints.

## Design
A model satisfies [`ModelProtocol`](#drytorch.core.protocols.ModelProtocol), bundling an architecture (or module)
and training utilities (bound optimizer, checkpoint, device placement, etc.).


DRYTorch imposes no restrictions on the architecture: any [`torch.nn.Module`](#torch.nn.Module) is
supported, accepting arbitrary collatable inputs and returning arbitrary
outputs. Type annotations are recommended for static verification but not
required.

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Set up
The examples train a small linear model on samples of a line.

```{code-cell} ipython3
:tags: [hide-input]

import torch

from torch.utils import data
from typing_extensions import override

from drytorch import (
    DataLoader,
    Experiment,
    LearningSchema,
    Loss,
    Model,
    Trainer,
)


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


class Linear(torch.nn.Linear):
    """Linear layer with annotated forward."""

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return super().forward(input)


experiment = Experiment(None, name='Models', par_dir='experiments')
run = experiment.create_run()
run.start()
train_loader = DataLoader(LineDataset(32), batch_size=8)
val_loader = DataLoader(LineDataset(16), batch_size=8)
loss = Loss(squared_error, name='MSE')
learning_schema = LearningSchema.adam(base_lr=0.05)
```

## Creating a Model
Provide a [`torch.nn.Module`](#torch.nn.Module) and optional arguments to [`Model`](#drytorch.lib.models.Model) for configuration:

* `name`: the name in the metadata and in the logs. Defaults to the class name.
* `device`: where to store the weights. Defaults to an accelerator if available, CPU otherwise.
* `checkpoint`: object to save and load the state. Default persists the state locally.
* `mixed_precision`: the autocast precision, `torch.float16` or `torch.bfloat16`, or `None` without mixed precision.
* `torch_compile`: true to compile the module with [`torch.compile`](#torch.compile).
* `distribute`: true to wrap the module for [`DistributedDataParallel`](#torch.nn.parallel.DistributedDataParallel).

```{code-cell} ipython3
model = Model(
    Linear(1, 1),
    name='LineModel',
    device=torch.device('cpu'),
    mixed_precision=None,
)
if model.epoch != 0:
    raise AssertionError('A new model starts from epoch 0.')
```

## Updating model state
Override [`post_batch_update`](#drytorch.core.protocols.ModelProtocol.post_batch_update) or [`post_epoch_update`](#drytorch.core.protocols.ModelProtocol.post_epoch_update) to perform custom updates
after each batch or epoch (e.g. updating buffers or enforcing parameter constraints).

```{code-cell} ipython3
class NonNegativeModel(Model[torch.Tensor, torch.Tensor]):
    """Keep the weights of the module non-negative."""

    def post_batch_update(self) -> None:
        """Set the negative weights to zero."""
        with torch.no_grad():
            for param in self.module.parameters():
                param.clamp_(min=0)

        return


non_negative = NonNegativeModel(torch.nn.Linear(1, 1))
constrained_trainer = Trainer(
    non_negative,
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
)
constrained_trainer.train(2)
if any((param < 0).any() for param in non_negative.module.parameters()):
    raise AssertionError('The weights should be non-negative.')
```

## Averaged weights
Track running weight averages alongside trained weights with [`SWAModel`](#drytorch.lib.models.SWAModel) or
[`EMAModel`](#drytorch.lib.models.EMAModel). In inference mode (such as during evaluation), the model automatically
uses the averaged weights, and checkpoints save both.

* [`SWAModel`](#drytorch.lib.models.SWAModel): epoch-wise stochastic weight averaging starting from `start_epoch`.
* [`EMAModel`](#drytorch.lib.models.EMAModel): batch-wise exponential moving average smoothed by `decay`.

```{code-cell} ipython3
from drytorch.lib.models import EMAModel, SWAModel


swa_model = SWAModel(torch.nn.Linear(1, 1), start_epoch=2, name='SWA')
ema_model = EMAModel(torch.nn.Linear(1, 1), decay=0.9, name='EMA')
for averaged_model in (swa_model, ema_model):
    builder = Trainer.builder(
        averaged_model,
        loader=train_loader,
        loss=loss,
        learning_schema=learning_schema,
    )
    builder.add_validation(val_loader)
    builder.build().train(3)

trained = ema_model.module.state_dict()['weight']
averaged = ema_model.averaged_module.state_dict()['weight']
if torch.equal(trained, averaged):
    raise AssertionError('The average should differ from the last weights.')
```

## Saving and loading
Save and restore weights, optimizer state, and epoch with [`save_state`](#drytorch.core.protocols.ModelProtocol.save_state) and
[`load_state`](#drytorch.core.protocols.ModelProtocol.load_state) (defaults to the latest saved epoch). Trainers invoke them
automatically during training via [`SavingHook`](#drytorch.lib.hooks.SavingHook) (see the
[training](training.md)
tutorial).

```{code-cell} ipython3
model.save_state()
model.load_state()
if model.epoch != 0:
    raise AssertionError('The model should load the saved epoch.')
```

## Write your own checkpoint
[`LocalCheckpoint`](#drytorch.lib.checkpoints.LocalCheckpoint) stores states in the run directory. To persist states elsewhere
(e.g. cloud storage or a remote database), subclass [`AbstractCheckpoint`](#drytorch.lib.checkpoints.AbstractCheckpoint).

The following example keeps the states in memory.

```{code-cell} ipython3
import copy
import pathlib

from typing import Any

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.lib.checkpoints import AbstractCheckpoint


class MemoryCheckpoint(AbstractCheckpoint):
    """Keep the saved states in memory."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.states: dict[int, dict[str, Any]] = {}
        return

    @override
    def _check_location(self, epoch: int) -> None:
        if epoch not in self.states:
            raise exceptions.EpochNotFoundError(epoch, pathlib.Path('memory'))

        return

    @override
    def _get_last_saved_epoch(self) -> int:
        if not self.states:
            raise exceptions.ModelNotFoundError(pathlib.Path('memory'))

        return max(self.states)

    @override
    def _get_location(self, epoch: int) -> str:
        return f'memory, epoch {epoch}'

    @override
    def _load_module(
        self, name: str, module: torch.nn.Module, epoch: int
    ) -> None:
        module.load_state_dict(self.states[epoch][name])
        return

    @override
    def _load_optimizer(self, epoch: int) -> None:
        if self.optimizer is not None:
            self.optimizer.load_state_dict(self.states[epoch]['optimizer'])

        return

    @override
    def _save_module(
        self, name: str, module: torch.nn.Module, epoch: int
    ) -> None:
        state = copy.deepcopy(module.state_dict())
        self.states.setdefault(epoch, {})[name] = state
        return

    @override
    def _save_optimizer(self, epoch: int) -> None:
        if self.optimizer is not None:
            state = copy.deepcopy(self.optimizer.state_dict())
            self.states.setdefault(epoch, {})['optimizer'] = state

        return


memory_checkpoint = MemoryCheckpoint()
memory_model = Model(torch.nn.Linear(1, 1), checkpoint=memory_checkpoint)
memory_trainer = Trainer(
    memory_model,
    loader=train_loader,
    loss=loss,
    learning_schema=learning_schema,
)
memory_trainer.train(1)
memory_trainer.save_checkpoint()
memory_trainer.train(1)
memory_trainer.load_checkpoint()
if memory_model.epoch != 1:
    raise AssertionError('The model should return to the saved epoch.')
```

## Stop the run

```{code-cell} ipython3
run.stop()
```

For distributed model wrapping and checkpointing across processes, see the
[distributed_training](distributed_training.md)
tutorial.
