---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Trackers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/trackers.ipynb)
DRYTorch uses a subscription-based event system for logging and
visualization, delegating implementation to external libraries via **trackers**.

This tutorial explains how trackers work and how to write your own.

## Design Principles
DRYTorch's philosophy of replicability and reusability is embodied in its **tracker system**. Trackers allow you to:

* Reproduce someone else's machine learning workflow while keeping logs in your preferred format.
* Reuse custom trackers across multiple projects with minimal code changes.
* Log, plot, and structure information without affecting the experiment logic.
* Ensure your experiment's completion even in case of a failure.

### Terminology
A **tracker** is an object responsible for **logging and plotting**. It:

* Receives notifications about events happening during an experiment.
* Does **not** intervene in the training, evaluation, or inference logic.
* Does **not** alter the behavior of other trackers. Trackers are completely independent.

```{code-cell} ipython2
! uv pip install drytorch
```

### Default Trackers

DRYTorch default trackers are subscribed to every experiment upon instantiation.
In standard mode, these are:
- `trackers.builtin_logging.BuiltinLogger` - handles the "drytorch" logger from the standard library
- `trackers.tqdm.TqdmLogger` - for progress bars (when `tqdm` is available)
- `trackers.yaml.YamlDumper` - to dump metadata (when `PyAML` is available)

#### Modes
DRYTorch offers carefully designed configurations, or initialization modes,
for the default trackers.

Available initialization modes:
- **standard**: log to stderr, preferring `tqdm` over the built-in logger.
- **hydra**: log to stdout and accommodate default Hydra settings.
- **tuning**: reduce output and avoid dumping metadata.
- **none**: no default trackers.

You can change the mode by assigning a mode to a `DRYTORCH_INIT_MODE` environment variable or reinitializing the trackers with `init_trackers`.

```{code-cell} ipython2
from drytorch import Experiment, init_trackers


init_trackers(mode='tuning')

tuning_experiment = Experiment(
    None, name='TuningExperiment', par_dir='experiments'
)
with tuning_experiment.create_run():
    pass  # no logging output

init_trackers()
standard_experiment = Experiment(
    None, name='StandardExperiment', par_dir='experiments'
)
with standard_experiment.create_run():
    pass  # log experiment's name
```

### Optional Trackers

DRYTorch also provides a variety of **optional trackers**, many of which rely on external libraries.

Available optional trackers include:

* `trackers.csv.CSVDumper` — creates a separate CSV file for each logging actor.
* `trackers.hydra.HydraLink` — links (and optionally copies) the Hydra output folder.
* `trackers.matplotlib.MatPlotter` — plots metrics and returns a `matplotlib` figure.
* `trackers.plotly.PlotlyPlotter` — plots metrics and returns a `plotly` figure.
* `trackers.sqlalchemy.SQLConnection` — creates a full SQL database for structured logging.
* `trackers.tensorboard.TensorBoard` — launches a TensorBoard server and logs metrics.
* `trackers.visdom.VisdomPlotter` — launches a Visdom server and logs metrics.
* `trackers.wandb.Wandb` — logs metrics and configuration to Weights & Biases.

Many of these trackers offer additional customization parameters. For example,
`trackers.wandb.Wandb` accepts a `wandb.sdk.wandb_settings.Settings` object containing all the options are normally passed to `wandb.init`.

Several optional trackers act as **dumpers**, meaning they store metadata directly on disk. Dumpers typically accept a `par_dir` argument to control the output directory. If no directory is provided, DRYTorch falls back to a standardized, time‑stamped folder structure based on the experiment’s name and start time.

`trackers.plotly.PlotlyPlotter`, `trackers.matplotlib.MatPlotter` and `trackers.visdom.VisdomPlotter` are instead **plotters**.
Plotters implement the `plot_metric` method, keep track of the metrics during the current session and can also load metrics from a previous session when initialized with a `MetricLoader`.
DryTorch offers two `MetricLoader`s out of the box: `trackers.sqlalchemy.SQLConnection` and `trackers.csv.CSVDumper` .

### Use a Tracker.

To use a tracker, subscribe it to an experiment before running it:
```python
my_experiment.trackers.subscribe(my_tracker)
```

Otherwise, use `drytorch.extend_default_trackers` to add a tracker to every experiment by default.

```{code-cell} ipython2
import pathlib

from drytorch import extend_default_trackers
from drytorch.trackers.csv import CSVDumper


csv_dumper = CSVDumper(par_dir=pathlib.Path('experiments', 'my_directory'))
extend_default_trackers([csv_dumper])
```

### Call an Existing Tracker
Trackers are designed to not be passed directly to the implementation of the experiment itself.
To allow more flexibility, DRYTorch allows you to call the subscribed tracker from everywhere during the experimental run using the tracker class  `get_current` method.

```{code-cell} ipython2
csv_dumper_experiment = Experiment(
    None, name='ExperimentWithCSVDumper', par_dir='experiments'
)

with csv_dumper_experiment.create_run():
    if CSVDumper.get_current() is not csv_dumper:
        raise AssertionError('Trackers should coincide.')
```

## Write a Custom Tracker
To write a custom tracker, it is important to first know the Event System
that calls it.

### Event System

To decouple experiment logic from logging, DRYTorch uses an **event-based architecture**.

#### How it works

1. During an experiment, different parts of the code emit **events**.
2. All trackers **subscribed** to the running experiment receive the event.
3. Each tracker decides **how to log or visualize** the event.

#### Types of Events

DRYTorch comes already with various **event classes** representing specific moments of the experiment's lifecycle.

These events are created by the core `Experiment` class, as well as by many library implementations.

Below is an overview of all built‑in events.

##### Experiment Lifecycle Events

* **StartExperimentEvent** — emitted when an experiment begins. Includes configuration, run ID, timestamp, tags, and resume state.
* **StopExperimentEvent** — emitted when an experiment finishes.

##### Model‑Related Events

* **ModelRegistrationEvent** — emitted when a model is instantiated. Contains a timestamp and a string representation of the architecture.
* **SaveModelEvent** — emitted when a checkpoint is saved (name, definition, location, epoch).
* **LoadModelEvent** — emitted when a model is loaded.

##### Actor / Source Registration

* **ActorRegistrationEvent** — records when an object (trainer, evaluator, etc.) is registered as a source of events related to a specific model.

##### Training Workflow Events

* **StartTrainingEvent** — emitted at the start of training, including start and end epoch.
* **StartEpochEvent** — marks the beginning of an epoch.
* **EndEpochEvent** — marks the end of an epoch.
* **IterateBatchEvent** — emitted during batch iteration. Supports `update()` callbacks for loggers that require push updates.
* **TerminatedTrainingEvent** — emitted when training is prematurely terminated, including the reason.
* **EndTrainingEvent** — emitted when training completes normally.

##### Testing Workflow Events

* **StartTestEvent** — emitted when testing starts.
* **EndTestEvent** — emitted when testing ends.

##### Metric‑Related Events

* **MetricEvent** — emitted when aggregated metrics are computed.
* **LearningRateEvent** — emitted when learning rates are updated by a scheduler.

These events form the backbone of the DRYTorch logging and tracker ecosystem and are automatically published upon creation.

### The Tracker class
To write custom tracker it is necessary to subclass the `core.track.Tracker` class and override its `notify` method.
You can also overwrite the `clean_up` method, which is called when
the experiment stops running, provided that:
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
calls the `super().notify` method, which is also always recommended.

```{code-cell} ipython2
import functools
import pprint

from typing_extensions import override

from drytorch.core import log_events
from drytorch.core.track import Tracker


class MetadataVisualizer(Tracker):
    """Tracker that prints the metadata on the console."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ModelRegistrationEvent) -> None:
        pprint.pp(event.architecture_repr)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ActorRegistrationEvent) -> None:
        pprint.pp(event.metadata)
        return super().notify(event)
```
