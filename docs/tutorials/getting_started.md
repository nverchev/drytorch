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

# Getting Started with DRYTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/getting_started.ipynb)

Define an experiment and train a simple neural network using the `drytorch`
library.

## Tutorial overview

- [experiments_and_runs](experiments_and_runs.md):
  defining experiments, managing runs, and registry tracking.
- [data](data.md): data
  loaders and dataset splitting.
- [models](models.md):
  wrapping neural network modules and checkpointing.
- [objectives](objectives.md):
  losses, metrics, and accumulators.
- [learning](learning.md):
  optimizers, learning rate schemas, and schedulers.
- [training](training.md):
  trainer configuration, validation, hooks, and checkpointing.
- [trackers](trackers.md):
  logging metrics, plots, and run states across backends.
- [events](events.md):
  writing and tracking your own events.
- [distributed_training](distributed_training.md):
  data-parallel training across multiple processes.
- [typing_support](typing_support.md):
  multi-modal inputs, custom model outputs, and type safety.

## Requirements
DRYTorch’s only mandatory dependencies are `numpy` and `torch`. `PyYAML`
and `tqdm` are optional but recommended for enhanced functionality and a
smoother experience. The library requires Python 3.11 or newer. This tutorial
does not rely on any additional optional dependencies.

The cell below uses [`uv`](https://docs.astral.sh/uv/) as package manager.

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Deep Prior Upsampling

Given a low-resolution image, super-resolution reconstructs a high-resolution
counterpart. This tutorial performs super-resolution using Deep Image Prior
(DIP):
1. **Fixed input**: A neural network takes a fixed input tensor to generate a
   high-resolution image.
2. **Training objective**: Optimize the network weights to minimize the
   difference between a downsampled version of its output and the
   low-resolution image.
3. **Inductive prior**: Even though the full-resolution output is not directly
   supervised, the convolutional architecture biases reconstruction toward
   natural images.
4. **Evaluation**: Compare the network's full-resolution output against
   bilinear interpolation using Peak Signal-to-Noise Ratio (PSNR) on the
   ground-truth high-resolution image.

### Download the data

Download the ground-truth high-resolution image from the GitHub repository.

```{code-cell} ipython3
:tags: [hide-input]

import urllib.request

from io import BytesIO

import numpy as np


with urllib.request.urlopen(
    'https://raw.githubusercontent.com/nverchev/drytorch/main/docs/tutorials/data/flower.npy'
) as response:
    data_bytes = response.read()

flower_np = np.load(BytesIO(data_bytes))
```

### Create a low-resolution version of the image

Downsample the ground-truth image using bilinear interpolation and use it
again to upsample back to the original resolution to obtain a baseline.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import torch


TARGET_SIZE = 128
DOWNSAMPLED_SIZE = 64


def interpolate(image_torch: torch.Tensor, target_size: int) -> torch.Tensor:
    """Perform an interpolation on an image array.

    Args:
        image_torch: a tensor of an image.
        target_size: resulting height and width.

    Returns:
        A torch tensor of shape (batch, target_size, target_size).
    """
    return torch.nn.functional.interpolate(
        image_torch,
        size=(target_size, target_size),
        mode='bilinear',
    )


flower_torch = torch.from_numpy(flower_np).permute(2, 0, 1).float()
flower_torch /= 255
flower_torch = interpolate(flower_torch.unsqueeze(0), TARGET_SIZE).squeeze()
flower_torch_downsampled = interpolate(
    flower_torch.unsqueeze(0), DOWNSAMPLED_SIZE
).squeeze()
baseline_reconstruction = interpolate(
    flower_torch_downsampled.unsqueeze(0), TARGET_SIZE
).squeeze()
```

Visualize the ground-truth image, the low-resolution input, and the baseline
reconstruction.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
tags: [hide-input]
---
from IPython.display import HTML
from IPython.display import display as ip_display
from PIL import Image as Image


def display_tensor_images(named_images: dict[str, torch.Tensor]) -> None:
    """Display PyTorch image tensors side by side.

    Args:
        named_images: mapping of titles to image tensors of shape (C, H, W).
    """
    title_html = ''
    for image_name in named_images:
        title_html += f"""
        <div style="text-align: center; width: 256px; font-size: 2em;">
        {image_name}
        </div>"""

    ip_display(HTML(f"""<div style=" display: flex;">{title_html}</div>"""))
    resized_images = [
        torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
        ).squeeze(0)
        for img in named_images.values()
    ]
    image = torch.cat(resized_images, dim=2)
    np_array = image.detach().cpu().permute(1, 2, 0).numpy().copy()
    np_array_int = (np_array * 255).astype(np.uint8)
    ip_display(Image.fromarray(np_array_int))
    return


low_resolution = torch.nn.functional.interpolate(
    flower_torch_downsampled.unsqueeze(0),
    size=(256, 256),
    mode='nearest',
).squeeze()
display_tensor_images(
    {
        'Ground Truth': flower_torch,
        'Low Resolution': low_resolution,
        'Baseline': baseline_reconstruction,
    }
)
```

## Set up the experiment

In DRYTorch, an **experiment** represents a reproducible setup defined by its
configuration, while a **run** represents a single execution session. The
configuration is globally accessible through the experiment class during an
active run, avoiding parameter passing and ensuring reproducibility.

### Define the configuration and experiment
The configuration contains the experiment hyperparameters:
the learning rate, the number of epochs, and the model capacity. Subclassing
[`Experiment`](#drytorch.core.experimenting.Experiment)`[Config]` provides type-checked settings throughout the pipeline.

```{code-cell} ipython3
import dataclasses
import os
import pathlib

from drytorch import Experiment


@dataclasses.dataclass(frozen=True)
class UpsamplingConfig:
    """Experiment configuration."""

    lr: float = 0.001
    n_epochs: int = int(os.getenv('EPOCHS', '30'))
    max_resnet_width: int = 32


class UpsamplingExperiment(Experiment[UpsamplingConfig]):
    """Class for upsampling experiments."""


settings = UpsamplingConfig()
experiment = UpsamplingExperiment(
    settings, name='Getting Started', par_dir=pathlib.Path('experiments')
)
```

### Select the trackers
Trackers handle logging, metrics export, and plotting without affecting the
experiment logic. By default, DRYTorch logs to the terminal, renders a `tqdm` progress
bar, and dumps metadata to YAML files. Subscribe [`CSVDumper`](#drytorch.trackers.csv.CSVDumper) to export
evaluation metrics into CSV files, and [`MatPlotter`](#drytorch.trackers.matplotlib.MatPlotter) to plot them on request.

```{code-cell} ipython3
from drytorch.trackers.csv import CSVDumper
from drytorch.trackers.matplotlib import MatPlotter


plotter = MatPlotter()
experiment.trackers.subscribe(CSVDumper())
experiment.trackers.subscribe(plotter)
```

### Start the run
Create a run session with `experiment.create_run()`. The session is active
between `run.start()` and `run.stop()`, or within a `with run:` block.

```{code-cell} ipython3
run = experiment.create_run()
run.start()
```

## Datasets and data loading

DRYTorch relies on standard `torch.utils.data` datasets. In Deep Image Prior,
data is only fed as a target for the loss function and for evaluation.

```{code-cell} ipython3
:tags: [hide-input]

import torch

from torch.utils import data


class SingleImageDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset repeating a single target image and an empty input tensor."""

    def __init__(self, target_image: torch.Tensor, length: int = 1) -> None:
        """Initialize with target image and number of samples.

        Args:
            target_image: image tensor to fit or evaluate.
            length: number of samples per epoch.
        """
        super().__init__()
        self.inputs = torch.empty(length, 0)
        self.targets = target_image.unsqueeze(0).expand(length, -1, -1, -1)
        return

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input and target for a sample."""
        return self.inputs[index], self.targets[index]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.inputs)


training_dataset = SingleImageDataset(flower_torch_downsampled, length=32)
test_dataset = SingleImageDataset(flower_torch, length=1)
```

## Define the neural network

DRYTorch supports any [`torch.nn.Module`](#torch.nn.Module). The residual convolutional network
below retrieves `max_resnet_width` directly from the active experiment
configuration via [`UpsamplingExperiment.get_config()`](#drytorch.core.experimenting.Experiment.get_config).

```{code-cell} ipython3
:tags: [hide-input]

from torch import nn


class ResNetBlock(nn.Module):
    """Fundamental residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolutions and skip connection.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.shortcut: nn.Module = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_activation: nn.ReLU = nn.ReLU(inplace=True)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through net, adding the residual connection."""
        out = self.net(x) + self.shortcut(x)
        return self.final_activation(out)


class ResNetStage(nn.Sequential):
    """A single stage of the ResNet architecture."""

    def __init__(
        self, in_channels: int, out_channels: int, n_blocks: int = 2
    ) -> None:
        """Initialize a stage with the required number of blocks."""
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(n_blocks):
            layers.append(ResNetBlock(current_channels, out_channels))
            current_channels = out_channels

        super().__init__(*layers)
        return


class ResNet(nn.Sequential):
    """A ResNet model for image reconstruction."""

    def __init__(self) -> None:
        """Initialize the entire ResNet structure."""
        cfg = UpsamplingExperiment.get_config()
        max_resnet_width = cfg.max_resnet_width
        layers: list[nn.Module] = [
            nn.Conv2d(
                16, max_resnet_width // 2, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            ResNetStage(max_resnet_width // 2, max_resnet_width, n_blocks=2),
            ResNetStage(max_resnet_width, max_resnet_width, n_blocks=2),
            ResNetStage(max_resnet_width, max_resnet_width // 2, n_blocks=2),
            nn.Conv2d(max_resnet_width // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]
        super().__init__(*layers)
        return


class DeepPriorNet(nn.Module):
    """Network with a fixed input parameter for Deep Image Prior."""

    def __init__(self) -> None:
        """Instantiate the network and the fixed input."""
        super().__init__()
        self.fixed_input: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(1, 16, TARGET_SIZE, TARGET_SIZE)
        )
        self.net: ResNet = ResNet()
        return

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        """Pass the fixed input through the network."""
        return self.net(self.fixed_input)


network = DeepPriorNet()
```

## Define loss and metric functions

The loss function is the Mean Squared Error (MSE) between the low-resolution
image and the downsampled reconstruction. The metric for evaluation is the Peak
Signal-to-Noise Ratio (PSNR) with respect to the high-resolution image
(baseline: PSNR=25.90 dB).

```{code-cell} ipython3
:tags: [hide-input]

def psnr(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio between two images ([0, 1] range).

    Args:
        recon: reconstructed image.
        target: target ground-truth image.

    Returns:
        PSNR value.
    """
    mse = nn.functional.mse_loss(recon, target)
    if mse == 0:
        return torch.tensor(torch.inf)

    return -10 * torch.log10(mse)


def downsampled_mse(
    outputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Mean Squared Error in the downsampled space.

    Args:
        outputs: model outputs containing reconstructed image.
        targets: training targets containing downsampled image.

    Returns:
        MSE value in downsampled space.
    """
    reconstruction_downsampled = interpolate(outputs, DOWNSAMPLED_SIZE)
    return nn.functional.mse_loss(reconstruction_downsampled, targets)


baseline_psnr = psnr(baseline_reconstruction, flower_torch).item()
if not np.isclose(baseline_psnr, 25.90, atol=1e-2):
    raise AssertionError('Baseline value should be about 25.90.')
```

## Training and evaluation

DRYTorch provides modular components to assemble and run the workflow:
- **[`DataLoader`](#drytorch.lib.loading.DataLoader)**: wraps a dataset to handle batching and iteration across
  training and evaluation.
- **[`Model`](#drytorch.lib.models.Model)**: wraps a neural network, managing weights, device placement,
  and checkpointing.
- **[`Loss`](#drytorch.lib.objectives.Loss)** and **[`Metric`](#drytorch.lib.objectives.Metric)**: wrap functions into composable, self-documenting
  objectives for training and evaluation.
- **[`LearningSchema`](#drytorch.lib.learning.LearningSchema)**: configures the optimizer, learning rate schedule, and
  gradient operations.
- **[`Trainer`](#drytorch.lib.training.Trainer)** and **[`Test`](#drytorch.lib.evaluations.Test)**: execute training and evaluation loops, reporting
  metrics to trackers and managing validation and checkpoints.

### Train the model
Assemble the training pipeline and run optimization.

```{code-cell} ipython3
from drytorch import DataLoader, LearningSchema, Loss, Model, Trainer


cfg = UpsamplingExperiment.get_config()
train_loader = DataLoader(training_dataset, batch_size=1)
model = Model(network, name='DeepPriorNet')
learning_schema = LearningSchema.adam(cfg.lr)
loss = Loss(downsampled_mse, name='MSE')
trainer = Trainer(
    model, loader=train_loader, loss=loss, learning_schema=learning_schema
)
trainer.train(cfg.n_epochs)
```

Plot the learning curve:

```{code-cell} ipython3
plotter.plot('DeepPriorNet')
```

### Evaluate the model
Evaluate the reconstructed image against the full-resolution target using
[`Test`](#drytorch.lib.evaluations.Test) with `store_outputs=True` to retain the reconstructed image for
inspection:

```{code-cell} ipython3
from drytorch import Metric, Test


test_loader = DataLoader(test_dataset, batch_size=1)
metric = Metric(psnr, name='PSNR', higher_is_better=True)
test = Test(model, loader=test_loader, metric=metric)
test(store_outputs=True)
```

### Qualitative results
Inspect the output image stored by [`Test`](#drytorch.lib.evaluations.Test) and display it alongside the ground-truth image and the
baseline reconstruction:

```{code-cell} ipython3
:tags: [hide-input]

reconstructed = test.outputs_list[0].squeeze()
display_tensor_images(
    {
        'Ground Truth': flower_torch,
        'Baseline': baseline_reconstruction,
        'Deep Prior': reconstructed,
    }
)
```

## Stop the run

Stopping the run notifies the subscribed trackers that the run has concluded.

Experiment documentation is stored in the local files:
- Run metadata is saved in the `.drytorch` folder.
- Model architecture and runner attributes are recorded in the `metadata`
  directory (requires PyYAML).
- Metric values and history are dumped in the `csv` directory by [`CSVDumper`](#drytorch.trackers.csv.CSVDumper).

```{code-cell} ipython3
run.stop()
```
