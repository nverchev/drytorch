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

# Events

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/events.ipynb)

This tutorial covers built-in events and explains how to define, emit, and
handle your own events.

## Design
Events represent lifecycle milestones, metric updates, and state changes emitted
during an experiment. Subscribed trackers (see the [trackers](trackers.md)
tutorial) process events independently and ignore those they do not handle,
decoupling state emission from tracking logic so new events never interfere with
existing trackers.


## Built-in events
### Experiment lifecycle
| Name | Notifies | Contains |
| :--- | :--- | :--- |
| [`StartExperimentEvent`](#drytorch.core.log_events.StartExperimentEvent) | Run start | Run configuration and metadata |
| [`PauseExperimentEvent`](#drytorch.core.log_events.PauseExperimentEvent) | Run pause | Run metadata |
| [`ContinueExperimentEvent`](#drytorch.core.log_events.ContinueExperimentEvent) | Run resumption | Run metadata |
| [`StopExperimentEvent`](#drytorch.core.log_events.StopExperimentEvent) | Run completion | Run metadata |

### Models and runners
| Name | Notifies | Contains |
| :--- | :--- | :--- |
| [`ModelRegistrationEvent`](#drytorch.core.log_events.ModelRegistrationEvent) | Model registration to the run | Architecture and metadata |
| [`RunnerRegistrationEvent`](#drytorch.core.log_events.RunnerRegistrationEvent) | Runner registration to the run | Runner metadata |
| [`SaveModelEvent`](#drytorch.core.log_events.SaveModelEvent) | Checkpoint saved to disk | Checkpoint metadata |
| [`LoadModelEvent`](#drytorch.core.log_events.LoadModelEvent) | Checkpoint restored from disk | Checkpoint metadata |

### Training and evaluation
| Name | Notifies | Contains |
| :--- | :--- | :--- |
| [`StartTrainingEvent`](#drytorch.core.log_events.StartTrainingEvent) | Training cycle start | Training bounds and target epochs |
| [`StartEpochEvent`](#drytorch.core.log_events.StartEpochEvent) | Training epoch start | Epoch progress |
| [`IterateBatchEvent`](#drytorch.core.log_events.IterateBatchEvent) | Batch iteration start | Iteration progress and update |
| [`MetricEvent`](#drytorch.core.log_events.MetricEvent) | Metric calculation and aggregation | Evaluated metrics |
| [`LearningRateEvent`](#drytorch.core.log_events.LearningRateEvent) | Learning rate update | Updated learning rates |
| [`IncrementEpochEvent`](#drytorch.core.log_events.IncrementEpochEvent) | Model epoch counter advance | Epoch progress |
| [`EndEpochEvent`](#drytorch.core.log_events.EndEpochEvent) | Training epoch completion | Epoch progress |
| [`TerminatedTrainingEvent`](#drytorch.core.log_events.TerminatedTrainingEvent) | Early training termination | Termination cause and progress |
| [`EndTrainingEvent`](#drytorch.core.log_events.EndTrainingEvent) | Training cycle completion | Runner metadata |
| [`StartTestEvent`](#drytorch.core.log_events.StartTestEvent) | Evaluation pass start | Evaluation metadata |
| [`EndTestEvent`](#drytorch.core.log_events.EndTestEvent) | Evaluation pass completion | Evaluation metadata |

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch matplotlib
```

## Set up
The example trains a small autoencoder on synthetic 4x4 gradient images. During
training, a hook reconstructs fixed samples and emits a custom `ImageEvent`,
which a custom tracker renders.

```{code-cell} ipython3
:tags: [hide-input]

import torch

from torch.utils import data

from drytorch import (
    DataLoader,
    Experiment,
    LearningSchema,
    Loss,
    Model,
    Trainer,
)


class GradientImages(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Images generated from a 2D latent space of gradients."""

    def __init__(self, n_samples: int) -> None:
        """Initialize.

        Args:
            n_samples: the number of samples.
        """
        grid = torch.linspace(0.1, 0.9, 4)
        weights = torch.rand(n_samples, 2, 1, 1)
        self.images: torch.Tensor = (
            weights[:, 0:1] * grid.view(1, 1, 1, 4)
            + weights[:, 1:2] * grid.view(1, 1, 4, 1)
        ) / 2
        return

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an image as input and as target."""
        image = self.images[index]
        return image, image

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.images)


class AutoEncoder(torch.nn.Module):
    """Compress an image to a low-dimensional space and reconstruct it."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.encoder: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(16, 4)
        )
        self.decoder: torch.nn.Linear = torch.nn.Linear(4, 16)
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode and decode the images.

        Args:
            inputs: input tensor.

        Returns:
            Reconstructed image tensor.
        """
        return self.decoder(self.encoder(inputs)).view(-1, 1, 4, 4)


def squared_error(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the squared error of each sample.

    Args:
        outputs: predicted model outputs.
        targets: ground truth targets.

    Returns:
        Squared error per sample.
    """
    return (outputs - targets).pow(2).flatten(1).mean(1)


train_dataset = GradientImages(64)
train_loader = DataLoader(train_dataset, batch_size=8)
loss = Loss(squared_error, name='MSE')
```

## Defining an event
Events are frozen dataclasses that inherit from [`Event`](#drytorch.core.log_events.Event). Instantiating
an event during an active run automatically publishes it to all subscribed
trackers:

```{code-cell} ipython3
import dataclasses

from drytorch.core import log_events


@dataclasses.dataclass(frozen=True)
class ImageEvent(log_events.Event):
    """Event logged when a hook produces images."""

    model_name: str
    epoch: int
    reconstruction: torch.Tensor
    ref_image: torch.Tensor
```

## Emitting an event from a hook
The hook caches a fixed batch of samples using [`take_from_dataset`](#drytorch.lib.loading.take_from_dataset) (see the
[data](data.md) tutorial) to evaluate every epoch. At each invocation, it
reconstructs these samples and publishes an `ImageEvent` carrying both the
targets and the reconstructions:

```{code-cell} ipython3
from drytorch.lib import hooks, monitoring
from drytorch.lib.loading import take_from_dataset


class ReconstructionHook(hooks.TrainerHook[monitoring.Condition]):
    """Emit the reconstructions of fixed samples."""

    def __init__(
        self,
        dataset: data.Dataset[tuple[torch.Tensor, torch.Tensor]],
        *,
        n_samples: int = 4,
        when: monitoring.Condition = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            dataset: the dataset providing the samples.
            n_samples: the number of samples to reconstruct.
            when: decides whether the hook runs.
        """
        super().__init__(when=when)
        _, (inputs, targets) = take_from_dataset(
            dataset, n_samples=n_samples, device=torch.device('cpu')
        )
        self.inputs: torch.Tensor = inputs
        self.targets: torch.Tensor = targets
        return

    def run(self) -> None:
        """Reconstruct the samples and emit them next to the targets."""
        model = self.trainer.model
        with torch.inference_mode():
            reconstructed = model(self.inputs.to(model.device))

        ImageEvent(
            model_name=model.name,
            epoch=model.epoch,
            reconstruction=reconstructed.clamp(0, 1).cpu(),
            ref_image=self.targets,
        )
        return
```

## Handling events in a custom tracker
Trackers process incoming events through [`notify`](#drytorch.core.tracking.Tracker.notify), routing them by event type
using [`functools.singledispatchmethod`](#functools.singledispatchmethod) (see the [trackers](trackers.md) tutorial for
details on tracker state setup and lifecycle hooks).

For example, a tracker can subclass [`TensorBoard`](#drytorch.trackers.tensorboard.TensorBoard) to log images to TensorBoard
event files:

```python
import functools
import pathlib

import torch

from typing_extensions import override

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard


class ImageTensorBoard(TensorBoard):
    """TensorBoard tracker that also writes images."""

    def __init__(
        self,
        *,
        flush_secs: int = 120,
        max_queue_size: int = 10,
        par_dir: pathlib.Path | None = None,
        start_server: bool = True,
    ) -> None:
        """Initialize.

        Args:
            flush_secs: tensorboard.SummaryWriter docs.
            max_queue_size: see tensorboard.SummaryWriter docs.
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            start_server: if ``True``, start a local TensorBoard server.
        """
        super().__init__(
            flush_secs=flush_secs,
            max_queue_size=max_queue_size,
            par_dir=par_dir,
            start_server=start_server,
        )
        self.n_written: int = 0
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ImageEvent) -> None:
        self.writer.add_images(
            f'{event.model_name}/targets and reconstructions',
            torch.cat([event.ref_image, event.reconstruction]),
            global_step=event.epoch,
        )
        self.writer.flush()
        self.n_written += 1
        return super().notify(event)
```

The example below implements an `ImagePlotter` using Matplotlib to render
reconstructions:

```{code-cell} ipython3
import functools

import matplotlib.pyplot as plt

from typing_extensions import override

from drytorch.core import log_events, tracking


class ImagePlotter(tracking.Tracker):
    """Plot reconstructed images with matplotlib."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.n_written: int = 0
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ImageEvent) -> None:
        """Plot the target and reconstructed images.

        Args:
            event: the image event to plot.
        """
        n_samples = len(event.ref_image)
        fig, axes = plt.subplots(
            2, n_samples, figsize=(n_samples * 2, 4), squeeze=False
        )
        for i in range(n_samples):
            axes[0, i].imshow(event.ref_image[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(event.reconstruction[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Target')
                axes[1, i].set_title('Reconstruction')

        fig.suptitle(f'{event.model_name} (Epoch {event.epoch})')
        fig.tight_layout()
        self.n_written += 1
        return super().notify(event)
```

## Running the experiment
Subscribe the tracker to the experiment before creating the run. The hook runs
every two epochs using `monitoring.Every(2)` (see the [training](training.md)
tutorial):

```{code-cell} ipython3
image_plotter = ImagePlotter()
experiment = Experiment(None, name='CustomEvents', par_dir='experiments')
experiment.trackers.subscribe(image_plotter)
run = experiment.create_run()
run.start()
trainer = Trainer(
    Model(AutoEncoder(), name='AutoEncoder'),
    loader=train_loader,
    loss=loss,
    learning_schema=LearningSchema.adam(base_lr=0.04),
    post_epoch_hooks=[
        ReconstructionHook(train_dataset, when=monitoring.Every(2))
    ],
)
trainer.train(4)
if image_plotter.n_written != 2:
    raise AssertionError('The images should be written at epochs 2 and 4.')
```

## Stop the run

```{code-cell} ipython3
run.stop()
```
