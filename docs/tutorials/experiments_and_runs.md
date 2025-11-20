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

# Experiments and Runs

## Defining an Experiment

In the DRYTorch framework, an experiment is a fully reproducible execution of code defined entirely by a configuration file. For example, this design implies that:

- a result obtained by modifying the configuration file (e.g., changing the optimizer) constitutes a new experiment instance.

-  a parameter sweep (or grid search), when fully described within the configuration file, is considered a single experiment.

To define an experiment, you should subclass or annotate DRYTorch's Experiment class, specifying the required configuration type. The Experiment class needs a unique name for each instance and also accepts optional tags and a designated output directory for the run, which other framework components will utilize.

```{code-cell} ipython3
! uv pip install drytorch
```

```{code-cell} ipython3
import dataclasses

from drytorch import Experiment as GenericExperiment


@dataclasses.dataclass(frozen=True)
class SimpleConfig:
    """A simple configuration."""

    batch_size: int


class MyExperiment(GenericExperiment[SimpleConfig]):
    """Class for Simple Experiments."""


my_config = SimpleConfig(32)
first_experiment = MyExperiment(
    my_config,
    name='FirstExp',
    par_dir='experiments/',
    tags=[],
)
```

# Starting a Run
In the DRYTorch framework, a run is a single execution instance of an experiment's code. Multiple runs of the same experiment—for example, by varying the random seed—are used to replicate and validate results.

You initiate a run instance using the Experiment's create_run method. This instance serves as a context manager for the experiment's execution code.

The run's ID is a timestamp by default, but you can specify a unique, descriptive name.

You can resume a run by specifying its unique name in create_run. If a name is not provided, DRYTorch attempts to resume the last recorded run.

There can only be an active run at once.

Note: DRYTorch maintains a run registry on the local disk to track and manage all run IDs and states.

```{code-cell} ipython3
def implement_experiment() -> None:
    """Here should the code for the experiment."""


with first_experiment.create_run() as run:
    first_id = run.id
    implement_experiment()


with first_experiment.create_run(resume=True) as run:
    second_id = run.id
    implement_experiment()

if first_id != second_id:
    raise
```

For convenience, especially in interactive environments like notebooks, you can manually start and stop a run, avoiding the context manager.

To do this, use the Experiment's start_run() method and ensure you explicitly call run.stop() when finished.

Warning: If you forget to call run.stop(), the run may not be properly recorded or finalized. While DRYTorch uses weak references to attempt cleanup at the end of the session, this behavior is unreliable and should not be depended upon for correct run logging.

```{code-cell} ipython3
run = first_experiment.create_run()
run.start()
run.stop()
```

## Global configuration

It is possible to access the configuration file directly from the Experiment class when the a run is on. Otherwise, this operation will fail.

```{code-cell} ipython3
from drytorch.core import exceptions


def get_batch() -> int:
    """Retrieve the batch size setting."""
    return MyExperiment.get_config().batch_size


with first_experiment.create_run():
    get_batch()

try:
    get_batch()
except (exceptions.AccessOutsideScopeError, exceptions.NoActiveExperimentError):
    err_str = 'Configuration accessed when no run is on.'
else:
    err_str = ''


err_str
```

## Registration

### Register model

DRYTorch discourages information leakage between runs to ensure reproducibility.

The framework explicitly prevents the construction of a Model instance based on a module registered in a previous run. This isolation ensures that each run starts from a clean state defined solely by its configuration.

This happens because the Model class registers its module at instantiation between the one in use.

To use the same module, you must first `unregister` it.

```{code-cell} ipython3
from torch.nn import Linear

from drytorch import Model
from drytorch.core import exceptions


second_experiment = MyExperiment(
    my_config,
    name='SecondExp',
    par_dir='experiments/',
    tags=[],
)
with first_experiment.create_run():
    first_model = Model(Linear(1, 1))

with second_experiment.create_run():
    try:
        second_model = Model(first_model.module)
    except exceptions.ModuleAlreadyRegisteredError:
        exception_is_triggered = True
    else:
        exception_is_triggered = False

exception_is_triggered
```

```{code-cell} ipython3
from drytorch.core import register


with second_experiment.create_run():
    register.unregister_model(first_model)
    try:
        second_model = Model(first_model.module)
    except exceptions.ModuleAlreadyRegisteredError:
        exception_is_triggered = True
    else:
        exception_is_triggered = False

exception_is_triggered
```

## Register actor

An **actor** is an object, like a trainer or a test class, that acts upon a model or produces logging from it.


Registration checks that the model and the actor belong to the same experiment.Actors from the library implementation register themselves when called.

```{code-cell} ipython3
import torch

from torch.utils.data import Dataset, StackDataset
from torch.utils.data.dataset import TensorDataset
from typing_extensions import override

from drytorch.lib.load import DataLoader
from drytorch.lib.runners import ModelRunner


class MyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Example dataset containing tensor with value one."""

    def __len__(self) -> int:
        """Size of the dataset."""
        return 1

    @override
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(1), torch.ones(1)


with second_experiment.create_run(resume=True):  # correctly resuming run
    one_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = MyDataset()
    loader = DataLoader(one_dataset, batch_size=1)
    model_caller = ModelRunner(second_model, loader=loader)
    try:
        model_caller()
    except exceptions.ModuleNotRegisteredError:
        exception_is_triggered = True
    else:
        exception_is_triggered = False

exception_is_triggered
```

```{code-cell} ipython3
with second_experiment.create_run():  # new run
    one_dataset = TensorDataset(torch.ones(1, 1))
    tuple_dataset = StackDataset(one_dataset, one_dataset)
    loader = DataLoader(tuple_dataset, batch_size=1)
    model_caller = ModelRunner(second_model, loader=loader)
    try:
        model_caller()
    except exceptions.ModuleNotRegisteredError:
        exception_is_triggered = True
    else:
        exception_is_triggered = False

exception_is_triggered
```
