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

# Trackers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/trackers.ipynb)

This tutorial explains how trackers work and how to write your own.

## Design
DRYTorch uses a subscription-based event system (see the [events](events.md)
tutorial) that delegates logging, plotting, and artifact export to **trackers**.
This decouples output handling from core experiment logic. Trackers allow you to:

* Reproduce an experiment while preserving your preferred logging and export
  formats.
* Reuse custom tracking logic across projects with minimal code changes.
* Log, plot, and structure run data without modifying model or training code.
* Finalize and flush records safely even if a run fails.

### Terminology
A **tracker** handles side effects such as logging, plotting, and file export.
It:

* Receives notifications about lifecycle events emitted during an experiment.
* Does **not** intervene in training, evaluation, or inference logic.
* Operates independently without altering the behavior of other trackers.

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Available trackers
DRYTorch provides built-in trackers and integrations with external libraries:

* [`BuiltinLogger`](#drytorch.trackers.logging.BuiltinLogger) (**default**): logs experiment lifecycle,
  training progress, and metrics using Python's standard `logging` library.
* [`TqdmLogger`](#drytorch.trackers.tqdm.TqdmLogger) (**default**): renders dynamic progress bars in the
  terminal (used by default when `tqdm` is installed).
* [`MetadataDumper`](#drytorch.trackers.yaml.MetadataDumper) (**default**): dumps experiment configuration,
  environment metadata, and tags to YAML files (used by default when `PyYAML` is
  installed).
* [`CSVDumper`](#drytorch.trackers.csv.CSVDumper): exports training and
  validation metrics into separate CSV files for each runner, and tests into a
  `test_metrics/` subdirectory.
* [`HydraLink`](#drytorch.trackers.hydra.HydraLink): links and optionally copies Hydra output folders
  into the run directory.
* [`MatPlotter`](#drytorch.trackers.matplotlib.MatPlotter): plots the metrics of a model with
  Matplotlib.
* [`PlotlyPlotter`](#drytorch.trackers.plotly.PlotlyPlotter): plots the metrics of a model with Plotly.
* [`SQLConnection`](#drytorch.trackers.sqlalchemy.SQLConnection): records experiment metadata, runs,
  models, and metric history in a SQL database. One database holds the experiments of one base
  configuration.
* [`TensorBoard`](#drytorch.trackers.tensorboard.TensorBoard): logs scalars and metrics to TensorBoard
  event files, launching inline in notebooks or logging the command to launch a
  server otherwise.
* [`Wandb`](#drytorch.trackers.wandb.Wandb): logs metrics, hyperparameters, and artifacts to
  Weights & Biases (W&B).

Many of these trackers offer additional customization parameters. For example,
`trackers.wandb.Wandb` accepts a `wandb.sdk.wandb_settings.Settings` object
containing options passed to `wandb.init`.

Trackers that write files to disk (such as [`CSVDumper`](#drytorch.trackers.csv.CSVDumper) and [`MetadataDumper`](#drytorch.trackers.yaml.MetadataDumper))
inherit from [`Dumper`](#drytorch.trackers.base_classes.Dumper). Dumpers accept a `par_dir` argument to specify the root
output directory (defaulting to the experiment folder).

Plotters ([`PlotlyPlotter`](#drytorch.trackers.plotly.PlotlyPlotter) and [`MatPlotter`](#drytorch.trackers.matplotlib.MatPlotter)) collect metrics during the session
or load stored metrics from a [`MetricLoader`](#drytorch.trackers.base_classes.MetricLoader) (such as [`CSVDumper`](#drytorch.trackers.csv.CSVDumper) or
[`SQLConnection`](#drytorch.trackers.sqlalchemy.SQLConnection)). They save the training plots when training ends and the test plot after each test.
With a metric loader, the test plot compares the last run of every experiment. It is saved in the `test_metrics` folder of the plotter, one file for
each model, so that the variations of a configuration appear together.
The `plot` and `save_plots` methods do the same on demand during a run.

## Initialization modes
DRYTorch provides initialization modes to configure which default trackers
subscribe upon experiment creation:

* **standard**: logs to `stderr`, using `tqdm` for dynamic progress bars when
  installed.
* **hydra**: logs to `stdout` and accommodates Hydra run directory structures.
* **minimal**: reduces console output and disables metadata dumping.
* **none**: disables all default trackers.

Select a mode using the `DRYTORCH_INIT_MODE` environment variable or by calling
`init_trackers`:

```{code-cell} ipython3
from drytorch import Experiment, init_trackers


init_trackers(mode='minimal')
tuning_experiment = Experiment(
    None, name='TrackersMinimal', par_dir='experiments'
)
with tuning_experiment.create_run():
    pass  # no logging output

init_trackers()
standard_experiment = Experiment(
    None, name='TrackersStandard', par_dir='experiments'
)
with standard_experiment.create_run():
    pass  # log experiment's name
```

## Subscribing and accessing trackers

### Subscribing to an experiment
To activate a tracker, subscribe it to an experiment before starting a run:

```python
my_experiment.trackers.subscribe(my_tracker)
```

Alternatively, call [`extend_default_trackers`](#drytorch.core.tracking.extend_default_trackers) to attach a tracker to
every subsequently created experiment:

```{code-cell} ipython3
import pathlib

from drytorch import extend_default_trackers
from drytorch.trackers.csv import CSVDumper


csv_dumper = CSVDumper(par_dir=pathlib.Path('experiments', 'my_directory'))
extend_default_trackers([csv_dumper])
```

### Accessing the current tracker
Because trackers remain decoupled from model and training code, retrieve an
active tracker instance from anywhere during a run using its [`get_current`](#drytorch.core.tracking.Tracker.get_current) class
method:

```{code-cell} ipython3
csv_dumper_experiment = Experiment(
    None, name='TrackersCSVDumper', par_dir='experiments'
)
with csv_dumper_experiment.create_run():
    if CSVDumper.get_current() is not csv_dumper:
        raise AssertionError('Trackers should coincide.')
```

## Write your own tracker

### Event system
DRYTorch decouples execution from observation using an event-driven
architecture:

1. Core components and runners emit **events** at lifecycle milestones (e.g.,
   epoch start, metric evaluation, model checkpointing).
2. All trackers **subscribed** to the active experiment receive the event.
3. Each tracker handles the event independently (e.g., logging to console,
   streaming to external dashboards, or writing to disk).

For built-in event definitions, custom events, and additional examples of
custom trackers, see the [events](events.md) tutorial.

### The Tracker class
To write a custom tracker, subclass [`Tracker`](#drytorch.core.tracking.Tracker) and implement event
handlers by decorating [`notify`](#drytorch.core.tracking.Tracker.notify) with `@functools.singledispatchmethod`. Each
handler registers a specific event type and delegates to
[`super().notify(event)`](#drytorch.core.tracking.Tracker.notify).

Calling [`super().notify(event)`](#drytorch.core.tracking.Tracker.notify) allows parent classes to manage tracker state:
[`Tracker`](#drytorch.core.tracking.Tracker) marks the tracker as active, while [`Dumper`](#drytorch.trackers.base_classes.Dumper)
stores the run directory and recording flag. In a [`StartExperimentEvent`](#drytorch.core.log_events.StartExperimentEvent)
handler, call `super().notify(event)` first so this state is available; in other handlers, call it last so the state
remains intact while the handler runs.

Trackers can also override [`clean_up`](#drytorch.core.tracking.Tracker.clean_up) when an experiment ends. Trackers should
check for unrecorded runs (`create_run(record=False)`) via
[`StartExperimentEvent`](#drytorch.core.log_events.StartExperimentEvent)`.record` to skip persistence.

Always delegate unhandled events to [`super().notify`](#drytorch.core.tracking.Tracker.notify):

```python
@functools.singledispatchmethod
@override
def notify(self, event: log_events.Event) -> None:
    return super().notify(event)


@notify.register
def _(self, event: log_events.StopExperimentEvent) -> None:
    ...
    return super().notify(event)
```

### Metadata extraction
During registration, DRYTorch automatically extracts readable representations
of models and runners, broadcasting their architecture and configuration
metadata through [`ModelRegistrationEvent`](#drytorch.core.log_events.ModelRegistrationEvent) and [`RunnerRegistrationEvent`](#drytorch.core.log_events.RunnerRegistrationEvent).

As an example, the following custom tracker captures and prints this metadata:

```{code-cell} ipython3
import datetime
import functools
import pprint

from collections.abc import Mapping
from typing import Any

from torch import nn
from typing_extensions import override

from drytorch import Experiment, Model
from drytorch.core import log_events
from drytorch.core.tracking import Tracker


class MetadataVisualizer(Tracker):
    """Tracker that prints and records model and runner metadata."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.recorded_models: list[str] = []
        self.recorded_runners: list[Mapping[str, Any] | str] = []
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ModelRegistrationEvent) -> None:
        pprint.pp(event.architecture_repr)
        self.recorded_models.append(event.architecture_repr)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.RunnerRegistrationEvent) -> None:
        pprint.pp(event.metadata)
        self.recorded_runners.append(event.metadata)
        return super().notify(event)
```

```{code-cell} ipython3
visualizer = MetadataVisualizer()
custom_tracker_exp = Experiment(
    None, name='TrackersCustom', par_dir='experiments'
)
custom_tracker_exp.trackers.subscribe(visualizer)

with custom_tracker_exp.create_run():
    model = Model(nn.Linear(1, 1))
    log_events.RunnerRegistrationEvent(
        runner_name='DemoRunner',
        runner_ts=datetime.datetime.now(),
        model_name=model.name,
        model_ts=datetime.datetime.now(),
        metadata={'batch_size': 32, 'device': 'cpu'},
    )

if not visualizer.recorded_models:
    raise AssertionError('Model registration event should be captured.')

if not visualizer.recorded_runners:
    raise AssertionError('Runner registration event should be captured.')
```

## Logging settings
The built-in logger (`trackers.logging.BuiltinLogger`) emits messages to the
`'drytorch'` logger using custom levels defined in `INFO_LEVELS`:
- `internal` (19): component registration and setup messages.
- `metrics` (21): computed metric reports.
- `epoch` (23): epoch progress messages.
- `model_state` (25): checkpoint save and load messages.
- `experiment` (27): experiment start, pause, continue, and stop messages.
- `training` (28): training cycle start and completion messages.
- `test` (29): evaluation and test cycle messages.

Use [`get_verbosity`](#drytorch.trackers.logging.get_verbosity) and [`set_verbosity`](#drytorch.trackers.logging.set_verbosity) to inspect and configure the threshold;
messages with a level below the threshold are filtered out.

The output formatter controls message layout on the stream:
- [`set_formatter`](#drytorch.trackers.logging.set_formatter)`('drytorch')` ([`DryTorchFormatter`](#drytorch.trackers.logging.DryTorchFormatter)): prints each message on
  a new line.
- `set_formatter('progress')` ([`ProgressFormatter`](#drytorch.trackers.logging.ProgressFormatter)): overwrites epoch and metric
  progress in place using carriage returns (`\r`).

```{code-cell} ipython3
from drytorch.trackers.logging import (
    INFO_LEVELS,
    get_verbosity,
    set_formatter,
    set_verbosity,
)


current_level = get_verbosity()
set_verbosity(INFO_LEVELS.training)
if get_verbosity() != INFO_LEVELS.training:
    raise AssertionError('Verbosity level should be set to training.')

set_formatter('progress')
set_formatter('drytorch')
set_verbosity(current_level)
```
