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

- a parameter sweep (or grid search), when fully described within the configuration file, is considered a single experiment.

To define an experiment, subclass the DRYTorch's `Experiment` class specifying the required specification.
The `Experiment` class needs a name unique for each instance and accepts tags and a directory for logging as optional arguments, which is communicated to trackers.

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
In the DRYTorch framework, a run is a single execution instance of an experiment's code.
Multiple runs of the same experiment are used to replicate and validate results, often using different seeds for the pseudo number generator.
There can only be an active run at once.

You initiate a run instance using the `Experiment.create_run` method. This instance serves as a context manager for the experiment's execution code.

The run's ID is a timestamp by default, but you can specify a unique, descriptive name.
You can resume a run by specifying its name in `create_run`. If a name is not provided, DRYTorch attempts to resume the last recorded run.

Note: DRYTorch maintains a run registry on the local disk to track and manage  all run IDs and states. It also attempts to record the last commit hash  when git is available.

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
    raise AssertionError('The resumed run should keep the id.')
```

### Notebooks

For convenience, especially in interactive environments like notebooks, you
can directly start and stop a run, avoiding the context manager.

Alternatively, you can use the `Run.start` and `Run.stop` methods directly.
To do this, use the `Run.start` and  method and ensure you explicitly call `Run.stop` when finished.

It is recommended to stop the run explicitly, otherwise DRYTorch will attempt to
clean up the metadata and exit gracefully when the Python session terminates.

```{code-cell} ipython3
run = first_experiment.create_run()
run.start()
run.stop()
```

## Global configuration

It is possible to access the configuration file directly from the `Experiment` class when a run is on. If no experiment is running, the operation will throw an exception.

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
    pass
else:
    raise AssertionError('Configuration accessed when no run is on.')
```

## Registration

### Register model

DRYTorch discourages information leakage between runs to ensure reproducibility.

The framework explicitly prevents the construction of a `Model` instance based on a module registered in a previous run.
This isolation ensures that each run starts from a clean state defined solely by its configuration.

The registration to the current run happens during the `Model` instantiation. If no experiment is running, the `Model` class will not be instantiated.

To use the same module, you must first `unregister` it.

```{code-cell} ipython3
from torch import nn

from drytorch import Model
from drytorch.core import exceptions


second_experiment = MyExperiment(
    my_config,
    name='SecondExp',
    par_dir='experiments/',
    tags=[],
)
module = nn.Linear(1, 1)

with first_experiment.create_run():
    first_model = Model(module)

try:
    second_model = Model(module)
except exceptions.NoActiveExperimentError:
    pass
else:
    raise AssertionError('Model instantiated when no experiment is running.')


with second_experiment.create_run():
    try:
        second_model = Model(module)
    except exceptions.ModuleAlreadyRegisteredError:
        pass
    else:
        raise AssertionError('Module registered through two Model instances.')
```

```{code-cell} ipython3
from drytorch.core import register


with second_experiment.create_run():
    register.unregister_model(first_model)
    second_model = Model(first_model.module)
```

## Register actor

An **actor** is an object, like a trainer or a test class, that acts upon a model or produces logging from it.

Registration checks that the model and the actor belong to the same experiment. Actors from the library implementation register themselves when called.

```{code-cell} ipython3
import torch

from torch.utils.data import Dataset
from typing_extensions import override

from drytorch.lib.load import DataLoader
from drytorch.lib.runners import ModelRunner


class MyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Example dataset containing tensor with value one."""

    def __init__(self) -> None:
        """Initialize some dummy attributes."""
        super().__init__()
        self.empty_container = []
        self.array = torch.ones(3, 3)
        self.long_string =  9 * '9'

    @property
    def two(self) -> int:
        """TA dummy property evaluating to two."""
        return 1 + 1

    def __len__(self) -> int:
        """Size of the dataset."""
        return 1

    @override
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(1), torch.ones(1)

one_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = MyDataset()

with second_experiment.create_run(resume=True):  # correctly resuming run
    loader = DataLoader(one_dataset, batch_size=1)
    model_caller = ModelRunner(second_model, loader=loader)
    model_caller()
```

```{code-cell} ipython3
with second_experiment.create_run():  # new run
    loader = DataLoader(one_dataset, batch_size=1)
    model_caller = ModelRunner(second_model, loader=loader)
    try:
        model_caller()
    except exceptions.ModuleNotRegisteredError:
        pass
    else:
        raise AssertionError('Model not registered in the current run')
```

## Metadata Extraction

DRYTorch automatically documents the models and actors during registration  by extracting a readable representation at runtime.
The metadata is then handled by the tracker. By default, when PyAML 6.0 or later is installed, metadata is dumped in YAML format.

To better visualize it, we create an ad-hoc tracker for this tutorial.

```{code-cell} ipython3
import functools
import pprint

from drytorch.core.track import Tracker
from drytorch.core import log_events


class MetadataVisualizer(Tracker):

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ModelRegistrationEvent) -> None:
       pprint.pprint(event.architecure_repr)

    @notify.register
    def _(self, event: log_events.ActorRegistrationEvent) -> None:
        pprint.pprint(event.metadata)

third_experiment = MyExperiment(
    my_config,
    name='ThirdExp',
    par_dir='experiments/',
    tags=[],
)

third_experiment.trackers.register(MetadataVisualizer())
```

#### Model metadata
The readable representation of a Model is simply the native
representation of the wrapped `nn.Module`.

```{code-cell} ipython3
with third_experiment.create_run():  # correctly resuming run
    third_model = Model(nn.Linear(1, 1))

    # loader = DataLoader(one_dataset, batch_size=1)
    # model_caller = ModelRunner(second_model, loader=loader)
    # model_caller()
```

#### Actor Metadata
The readable representation of an actor not only documents the actor object, but all the public attributes recursively.

Private attributes, that is, attributes that start with an underscore, are ignored and so are attributes that have not been initialized (evaluating to None, or an empty container).

```{code-cell} ipython3
with third_experiment.create_run(resume=True):  # correctly resuming run
    loader = DataLoader(one_dataset, batch_size=1)
    model_caller = ModelRunner(third_model, loader=loader)
    model_caller()
```
