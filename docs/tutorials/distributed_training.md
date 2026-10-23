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

# Distributed Data Parallel Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/distributed_training.ipynb)

This tutorial covers distributed data parallel (DDP) training across multiple
processes using PyTorch's distributed package ([`torch.distributed`](#torch.distributed)).

## Design
DRYTorch integrates with PyTorch's Distributed Data Parallel (DDP) package.
Rather than enforcing a rigid cluster setup or launcher, DRYTorch lets you
initialize the process group and device allocation, then automates the rest of
the distributed workflow:

- **Distributed data loading**: [`DataLoader`](#drytorch.lib.loading.DataLoader) detects the active process group and
  partitions the dataset across ranks using [`DistributedSampler`](#torch.utils.data.distributed.DistributedSampler) (see
  [Distributed data loading](#distributed-data-loading) below).
- **Model distribution**: When `distribute=True` (the default), [`Model`](#drytorch.lib.models.Model) moves
  the module to the process device, synchronizes batch normalization statistics
  across ranks on CUDA, and wraps the network in
  [`torch.nn.parallel.DistributedDataParallel`](#torch.nn.parallel.DistributedDataParallel) (see the [models](models.md)
  tutorial).
- **Synchronized metrics**: [`Metric`](#drytorch.lib.objectives.Metric) and [`Loss`](#drytorch.lib.objectives.Loss) instances synchronize their
  accumulated values across ranks at evaluation time (see the
  [objectives](objectives.md) tutorial).
- **Deduplicated tracking**: Only the main process (rank 0) dispatches tracking
  events, preventing duplicate logs, plots, and saved artifacts across ranks
  (see the [trackers](trackers.md) tutorial).

## Distributed data loading
When a process group is active, [`DataLoader`](#drytorch.lib.loading.DataLoader) automatically partitions the
dataset across ranks using [`DistributedSampler`](#torch.utils.data.distributed.DistributedSampler):

- **Epoch synchronization**: Runners automatically call
  `sampler.set_epoch(epoch)` each epoch so distributed shuffling stays in sync
  without manual boilerplate.
- **Evaluation padding**: If the dataset size is not evenly divisible across
  processes, samples are padded so all ranks evaluate the same batch count.
- **Global length**: Calling `len(loader)` returns the total batch count across
  all processes combined.

## Model handling
When `distribute=True` (the default), [`Model`](#drytorch.lib.models.Model) manages DDP integration and device
placement:

- **DDP wrapping**: Moves the module to the target process device and wraps it
  in [`torch.nn.parallel.DistributedDataParallel`](#torch.nn.parallel.DistributedDataParallel).
- **Batch normalization**: Synchronizes normalization statistics across all
  processes on CUDA devices using [`SyncBatchNorm`](#torch.nn.SyncBatchNorm).
- **Evaluation bypass**: Runs evaluation and inference directly on the unwrapped
  module to avoid DDP communication overhead.
- **Checkpointing**: Only the main process (rank 0) saves checkpoints to disk,
  and all ranks automatically synchronize upon loading them.

## Metric synchronization
Each rank computes metrics on its local data shard. At the end of an evaluation
pass, runners query DRYTorch's [`SyncProtocol`](#drytorch.core.protocols.SyncProtocol) to synchronize objectives across
ranks before reporting values:

- **Built-in and wrapped objectives**: [`Metric`](#drytorch.lib.objectives.Metric), [`Loss`](#drytorch.lib.objectives.Loss), and wrappers from
  [`from_torchmetrics`](#drytorch.contrib.torchmetrics.from_torchmetrics) or [`from_torcheval`](#drytorch.contrib.torcheval.from_torcheval) implement [`SyncProtocol`](#drytorch.core.protocols.SyncProtocol)
  automatically.
- **Accumulators**: The [`Objective`](#drytorch.lib.objectives.Objective) base class delegates to accumulators that
  all-reduce sums or all-gather tensors across ranks.
- **Custom objectives**: All-reduce or all-gather their state across ranks to
  satisfy [`SyncProtocol`](#drytorch.core.protocols.SyncProtocol).

## Distributed worker example
Distributed training runs as a standalone script rather than inside
interactive notebook cells, so the example below is provided as a Python code
block. It demonstrates a complete worker setup using `gloo` on CPU or `nccl`
on multi-GPU Linux systems (see the
[PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
for environment setup):

```python
import dataclasses
import os
import pathlib

import torch
from torch import distributed as dist, nn
from torch.utils import data

from drytorch import (
    DataLoader,
    Experiment,
    LearningSchema,
    Loss,
    Metric,
    Model,
    Test,
    Trainer,
)
from drytorch.trackers import CSVDumper


@dataclasses.dataclass(frozen=True)
class DistributedConfig:
    """Experiment configuration."""

    lr: float = 1e-3
    n_epochs: int = 5
    batch_size: int = 16


class DistributedExperiment(Experiment[DistributedConfig]):
    """Experiment class for distributed training."""


class ToyDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Toy dataset generating simple regression pairs."""

    def __init__(self, size: int = 128) -> None:
        """Initialize dataset with random samples."""
        super().__init__()
        self.x = torch.randn(size, 8)
        self.y = torch.randn(size, 1)
        return

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single sample."""
        return self.x[index], self.y[index]


def setup(rank: int, world_size: int) -> None:
    """Initialize the process group.

    Args:
        rank: rank of current process.
        world_size: total number of processes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    return


def cleanup() -> None:
    """Destroy process group."""
    dist.destroy_process_group()
    return


def train_and_eval(device: torch.device) -> None:
    """Train and evaluate the model in distributed mode.

    Args:
        device: target device for the current process.
    """
    cfg = DistributedExperiment.get_config()
    train_loader = DataLoader(ToyDataset(), batch_size=cfg.batch_size)
    val_loader = DataLoader(ToyDataset(32), batch_size=cfg.batch_size)

    network = nn.Linear(8, 1)
    model = Model(network, name='LinearModel', device=device)
    loss = Loss(nn.functional.mse_loss, name='MSE')
    metric = Metric(nn.functional.l1_loss, name='MAE', higher_is_better=False)

    trainer = Trainer(
        model,
        loader=train_loader,
        loss=loss,
        learning_schema=LearningSchema.adam(cfg.lr),
    )
    trainer.train(cfg.n_epochs)

    test = Test(model, loader=val_loader, metric=metric)
    test()
    return


def worker(rank: int, world_size: int) -> None:
    """Process worker function.

    Args:
        rank: rank of current process.
        world_size: total number of processes.
    """
    setup(rank=rank, world_size=world_size)
    device = (
        torch.device('cuda', rank)
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    experiment = DistributedExperiment(
        DistributedConfig(),
        name='DistributedTraining',
        par_dir=pathlib.Path('experiments'),
    )
    experiment.trackers.subscribe(CSVDumper())
    try:
        with experiment.create_run():
            train_and_eval(device)
    finally:
        cleanup()

    return


if __name__ == '__main__':
    world_size = 2
    torch.multiprocessing.spawn(
        worker, args=(world_size,), nprocs=world_size, join=True
    )
```
