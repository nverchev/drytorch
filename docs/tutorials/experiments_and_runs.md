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

# Experiments and Runs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/experiments_and_runs.ipynb)

This tutorial explains how to configure experiments, manage execution runs, and
track run metadata.

## Design
In DRYTorch, an **Experiment** represents a reproducible machine learning setup
defined by its configuration, while a **Run** represents a single execution of
that experiment.

Execution is coupled directly to configuration: during an active run, settings
are globally accessible through the experiment class, avoiding parameter
plumbing and preventing configuration drift.

### Terminology
* **Experiment:** A reproducible setup defined by its configuration. Any change to settings affecting results constitutes a new experiment.
* **Run:** A single execution of an experiment. Multiple runs of the same experiment (such as across random seeds) verify stability and should produce consistent results.
* **Registry:** Persistent storage tracking run history, status, and experiment metadata.

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Set up
The examples use a minimal dataset yielding pairs of tensors.

```{code-cell} ipython3
import torch

from torch.utils.data import Dataset
from typing_extensions import override


class MyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Example dataset containing tensor with value one."""

    def __init__(self, size: int = 1) -> None:
        """Initialize.

        Args:
            size: number of samples in the dataset.
        """
        super().__init__()
        self.size = size
        return

    def __len__(self) -> int:
        """Size of the dataset."""
        return self.size

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _not_used = index
        return torch.ones(1), torch.ones(1)


one_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = MyDataset()
```

## Defining an Experiment
Subclass [`Experiment`](#drytorch.core.experimenting.Experiment)`[Config]` with your configuration class (see the
[typing support](typing_support.md)
tutorial).

When creating an experiment instance, you specify:
- **`config`**: the configuration object holding your hyperparameters.
- **`name`**: a unique identifier for the experiment, used to organize output
  directories.
- **`par_dir`**: the root directory where trackers save metrics, checkpoints,
  and metadata.
- **`tags`**: optional labels describing the experiment (communicated to
  trackers for filtering and comparison).

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
    name='ExperimentsAndRunsFirst',
    par_dir='experiments/',
    tags=[],
)
```

## Starting a Run
Initiate a run session with [`experiment.create_run()`](#drytorch.core.experimenting.Experiment.create_run), which serves as a
context manager for the execution code. Only one run can be active at a time.

Pass `run_id` to assign a custom identifier (defaults to a timestamp). Setting
`resume=True` resumes the last run of the experiment, or a specific run if
`run_id` is provided:

```{code-cell} ipython3
def implement_experiment() -> None:
    """Here should be the code for the experiment."""
    return


with first_experiment.create_run() as run:
    first_id = run.id
    implement_experiment()

with first_experiment.create_run(resume=True) as run:
    second_id = run.id
    implement_experiment()

if first_id != second_id:
    raise AssertionError('The resumed run should keep the id.')
```

### Several runs of one experiment
An experiment can hold several runs of the same setting. In a k-fold
cross-validation, for example, each fold trains a new model on its own split
of the data, so each fold is a run of the experiment, named after the fold.
Run ids are unique within an experiment, so running this cell again requires
removing the experiment data first (see
[Managing experiment data](#managing-experiment-data)).

```{code-cell} ipython3
from drytorch.lib.loading import DataLoader


cv_loader = DataLoader(MyDataset(size=10), batch_size=2)
for fold, (_train_loader, _val_loader) in enumerate(cv_loader.fold(n_folds=5)):
    with first_experiment.create_run(run_id=f'fold_{fold}'):
        # train on _train_loader, validate on _val_loader
        implement_experiment()
```

### Unrecorded runs
Passing `record=False` creates an unrecorded run that does not record its
session on disk. The run also notifies subscribed trackers to forego persisting
information for the session. File dumpers write no files and database
connections store no records, while console loggers and progress bars still
report progress normally. This is useful for interactive debugging, smoke
tests, or tuning runs recorded separately.

```{code-cell} ipython3
with first_experiment.create_run(record=False) as debug_run:
    if debug_run.record:
        raise AssertionError('The run should not be recorded.')
```

### Pausing and continuing a run
Only one run is active at a time, so a run pauses to let another experiment
work, and continues in the same run afterwards.

```{code-cell} ipython3
side_experiment = MyExperiment(
    my_config, name='SideExp', par_dir='experiments/'
)

run = first_experiment.create_run()
run.start()
run_id = run.id
run.pause()

with side_experiment.create_run():
    pass

run.start()
run.stop()
if run.id != run_id:
    raise AssertionError('The same run should continue.')
```

### Notebooks

In interactive environments like notebooks, call [`run.start()`](#drytorch.core.experimenting.Run.start) and [`run.stop()`](#drytorch.core.experimenting.Run.stop)
directly instead of using a context manager. Remember to stop the run explicitly
when finished to release resources and finalize metadata:

```{code-cell} ipython3
run = first_experiment.create_run()
run.start()
run.stop()
```

## Global configuration

To ensure reproducibility across all components, retrieve hyperparameters
directly from the [`Experiment`](#drytorch.core.experimenting.Experiment) subclass via [`get_config()`](#drytorch.core.experimenting.Experiment.get_config) during an active
run. If no experiment is running, attempting to access configuration raises
an exception, ensuring that unmonitored code does not execute with untracked
settings.

```{code-cell} ipython3
from drytorch.core import exceptions


def get_batch() -> int:
    """Retrieve the batch size setting.

    Returns:
        Configured batch size.
    """
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

DRYTorch discourages information leakage between runs and supports metadata
extraction for experiment tracking.

### Register model


DRYTorch prevents registering a module that was registered in a previous run to
ensure run isolation. To intentionally reuse a module across runs, explicitly
unregister it first:

```{code-cell} ipython3
from torch import nn

from drytorch import Model
from drytorch.core import exceptions, registering


second_experiment = MyExperiment(
    my_config,
    name='ExperimentsAndRunsSecond',
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
with second_experiment.create_run():
    registering.unregister_model(first_model)
    second_model = Model(first_model.module)
```

## Register runner

A **runner** (such as a trainer or evaluation source) manages batching and
iteration over a dataset for a model. Upon instantiation, [`Runner`](#drytorch.lib.runners.Runner) registers
with the active run and verifies that its model belongs to the same run.

```{code-cell} ipython3
from drytorch.lib.loading import DataLoader
from drytorch.lib.runners import Runner


with second_experiment.create_run(resume=True):  # correctly resuming run
    loader = DataLoader(one_dataset, batch_size=1)
    runner = Runner(second_model, loader=loader)
    runner()
```

```{code-cell} ipython3
with second_experiment.create_run():  # new run
    loader = DataLoader(one_dataset, batch_size=1)
    try:
        Runner(second_model, loader=loader)
    except exceptions.ModuleFromAnotherRunError:
        pass
    else:
        raise AssertionError('Model registered in another run')
```

### Metadata extraction
DRYTorch automatically documents models and runners during registration by
extracting a readable representation at runtime. These representations are
forwarded to subscribed trackers, such as [`MetadataDumper`](#drytorch.trackers.yaml.MetadataDumper) or custom trackers
(see the [trackers](trackers.md)
tutorial).


## Managing experiment data

DRYTorch organizes tracker directories and checkpoints by experiment name under
`par_dir`. The module `drytorch.utils.local_ops` provides utilities to manage
these folders:
- [`clone_experiment_data`](#drytorch.utils.local_ops.clone_experiment_data)`(par_dir, exp_name, new_exp_name)`: duplicates an
  experiment's folders under a new name.
- [`rename_experiment_data`](#drytorch.utils.local_ops.rename_experiment_data)`(par_dir, exp_name, new_exp_name)`: renames the
  experiment folders across tracker directories.
- [`delete_experiment_data`](#drytorch.utils.local_ops.delete_experiment_data)`(par_dir, exp_name)`: removes all folders associated
  with the experiment.

```{code-cell} ipython3
import pathlib

from drytorch.utils.local_ops import (
    clone_experiment_data,
    delete_experiment_data,
    rename_experiment_data,
)


demo_dir = pathlib.Path('experiments', 'yaml')
(demo_dir / 'DemoExp').mkdir(parents=True, exist_ok=True)
clone_experiment_data('experiments', 'DemoExp', 'ClonedExp')
if not (demo_dir / 'ClonedExp').is_dir():
    raise AssertionError('Cloned experiment folder should exist.')

rename_experiment_data('experiments', 'ClonedExp', 'RenamedExp')
if not (demo_dir / 'RenamedExp').is_dir():
    raise AssertionError('Renamed experiment folder should exist.')

delete_experiment_data('experiments', 'DemoExp')
delete_experiment_data('experiments', 'RenamedExp')
if (demo_dir / 'DemoExp').is_dir() or (demo_dir / 'RenamedExp').is_dir():
    raise AssertionError('Deleted experiment folders should no longer exist.')
```
