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

# Typing Support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/typing_support.ipynb)

This tutorial covers static type safety in DRYTorch across configuration,
data loaders, models, and objectives.

## Design
DRYTorch relies on structural subtyping via protocols (see the
[protocols](../protocols.md) overview) to define contracts between components
without enforcing rigid inheritance hierarchies.

### Core types
Three generic types parameterize the pipeline:

- `Input`: produced by the dataset or loader and consumed by the model.
- `Output`: produced by the model and consumed by objectives.
- `Target`: produced by the dataset or loader and consumed by objectives.

### Core protocols
Component contracts are defined by generic protocols parameterized by these
types:

- [`LoaderProtocol`](#drytorch.core.protocols.LoaderProtocol)`[Input, Target]`: defines data loading and batch iteration.
- [`ModelProtocol`](#drytorch.core.protocols.ModelProtocol)`[Input, Output]`: defines the forward call mapping inputs to
  predictions.
- [`LossProtocol`](#drytorch.core.protocols.LossProtocol)`[Output, Target]`: defines loss computation taking model outputs
  and targets.
- [`ObjectiveProtocol`](#drytorch.core.protocols.ObjectiveProtocol)`[Output, Target]`: defines metric accumulation and
  reduction.
- [`TrainerProtocol`](#drytorch.core.protocols.TrainerProtocol)`[Input, Target, Output]`: ties together loader, model, loss,
  and optimization.

When components are annotated with these types, static type checkers (such as
`basedpyright` or `mypy`) verify end-to-end compatibility across the pipeline:

1. The model's input type matches the loader's input type (`Input`).
2. The model's output type matches the objective's output type (`Output`).
3. The loader's target type matches the objective's target type (`Target`).

### Experiment specification
[`Experiment`](#drytorch.core.experimenting.Experiment)`[Specification]` is generic over its configuration specification,
providing full static type inference for hyperparameters throughout a run.

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Typed experiment configuration
Subclassing [`Experiment`](#drytorch.core.experimenting.Experiment)`[Specification]` with your configuration class enables
[`Experiment.get_config()`](#drytorch.core.experimenting.Experiment.get_config) to return the exact configuration type with complete
static type inference during an active run.

This provides IDE support across project modules without manually passing config
objects:

- **Autocompletion**: typing `cfg.` suggests defined hyperparameter fields.
- **Hover documentation**: hovering over settings displays their docstrings
  and type signatures.
- **Type safety**: static type checkers catch typos and invalid accesses before
  runtime.

Pydantic dataclasses (`from pydantic.dataclasses import dataclass`) are
recommended as specification classes to validate settings and enforce value
constraints.

```{code-cell} ipython3
import dataclasses

from drytorch import Experiment


@dataclasses.dataclass
class Config:
    """Experiment hyperparameters."""

    lr: float = 0.01
    """Learning rate for optimizer."""

    hidden_dim: int = 16
    """Hidden dimension of the model."""


class TypedExperiment(Experiment[Config]):
    """Experiment parameterized with Config specification."""


experiment = TypedExperiment(
    Config(lr=0.02, hidden_dim=32),
    name='Typing Support',
    par_dir='experiments',
)
run = experiment.create_run()
run.start()

# IDEs such as VS Code and PyCharm show docstrings and types for settings:
cfg = TypedExperiment.get_config()
(cfg.lr, cfg.hidden_dim)
```

## Structured inputs, targets, and outputs
DRYTorch supports structured data for multi-modal inputs, auxiliary targets,
and multi-task predictions:
- **`Input` and `Target`**: use `NamedTuple` so PyTorch's default `collate_fn`
  batches samples automatically (see the [data](data.md) tutorial).
- **`Output`**: can be any type; a `@dataclass` is ideal for multi-head models.

```{code-cell} ipython3
import dataclasses
import typing

import torch

from torch import nn
from torch.utils import data


class Inputs(typing.NamedTuple):
    """Structured model inputs."""

    features: torch.Tensor
    """Input feature tensor."""

    mask: torch.Tensor
    """Binary mask tensor."""


class Targets(typing.NamedTuple):
    """Structured supervision targets."""

    label: torch.Tensor
    """Ground truth class label."""


@dataclasses.dataclass(slots=True)
class Outputs:
    """Structured model predictions."""

    logits: torch.Tensor
    """Model prediction logits."""
```

### Custom components
Through structural subtyping, standard PyTorch datasets, modules, and loss
functions satisfy DRYTorch contracts directly via type annotations, without
having to inherit from library-specific base classes:

```{code-cell} ipython3
class DemoDataset(data.Dataset[tuple[Inputs, Targets]]):
    """Dataset producing structured inputs and targets."""

    def __init__(self, size: int = 16) -> None:
        """Initialize synthetic data.

        Args:
            size: number of samples.
        """
        super().__init__()
        torch.manual_seed(42)
        self.features: torch.Tensor = torch.randn(size, 8)
        self.mask: torch.Tensor = torch.ones(size, dtype=torch.bool)
        self.labels: torch.Tensor = torch.randint(0, 2, (size,))
        return

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        """Return a structured sample."""
        return (
            Inputs(self.features[index], self.mask[index]),
            Targets(self.labels[index]),
        )


class DemoModel(nn.Module):
    """Model consuming structured inputs and producing structured outputs."""

    def __init__(self, in_features: int = 8, num_classes: int = 2) -> None:
        """Initialize layers.

        Args:
            in_features: input feature dimension.
            num_classes: number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass.

        Args:
            inputs: structured inputs.

        Returns:
            Structured predictions.
        """
        return Outputs(logits=self.linear(inputs.features))


def classification_loss(outputs: Outputs, targets: Targets) -> torch.Tensor:
    """Per-sample cross entropy loss.

    Args:
        outputs: model predictions.
        targets: supervision targets.

    Returns:
        Loss tensor per sample.
    """
    return nn.functional.cross_entropy(
        outputs.logits, targets.label, reduction='none'
    )
```

## Static type safety in action
When assembling the pipeline, generic parameters guarantee type safety across
the loader, model, loss, and trainer. Static type checkers (such as
`basedpyright`) verify compatibility either in your IDE or by running
`uv run basedpyright`:

```{code-cell} ipython3
from drytorch import DataLoader, LearningSchema, Loss, Model, Trainer


loader = DataLoader(DemoDataset(), batch_size=4)
model = Model(DemoModel(), name='DemoModel')
loss = Loss(classification_loss, name='CrossEntropy')

# Instantiating Trainer passes static type checking:
trainer = Trainer(
    model,
    loader=loader,
    loss=loss,
    learning_schema=LearningSchema.adam(cfg.lr),
)
```

### Catching errors before runtime
If a component's types do not align, the type checker flags an error. For
example, consider a loss function expecting raw tensors instead of `Outputs`:

```python
def wrong_loss_fn(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
    return outputs.sum()


wrong_loss = Loss(wrong_loss_fn, name='WrongLoss')

trainer = Trainer(
    model,  # Model[Inputs, Outputs]
    loader=loader,  # DataLoader[tuple[Inputs, Targets]]
    loss=wrong_loss,  # Loss[torch.Tensor, Targets]
    learning_schema=LearningSchema.adam(0.01),
)
```

`basedpyright` flags this error immediately at edit time or in the terminal:
```text
Argument of type "Loss[Tensor, Targets]" cannot be assigned to parameter
"loss" of type "LossProtocol[Outputs, Targets]" in function "__init__"
  Type "Tensor" is incompatible with type "Outputs"
```

## Stop the run
Stopping the run ensures proper resource cleanup and logging finalization:

```{code-cell} ipython3
run.stop()
```
