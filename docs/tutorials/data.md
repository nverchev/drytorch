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

# Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nverchev/drytorch/blob/main/docs/tutorials/data.ipynb)

DRYTorch supports structured data and custom classes for multi-modal and
nested data.

## Design
Data is always defined as tuple pairs of (input, target).
The first is given to the model; the second is used by the objective function.
For example, an autoencoder passes the same object as input and target.

### Collation compatibility
Inputs and targets must be compatible with PyTorch's default `collate_fn`.
PyTorch natively supports:
- [`torch.Tensor`](#torch.Tensor)
- standard `tuple` or `list` of tensors
- `typing.NamedTuple` containing tensors or nested collatable structures

```{code-cell} ipython3
:tags: [skip-execution]

! uv pip install drytorch
```

## Set up
Start a run to track data loading operations:

```{code-cell} ipython3
from drytorch import Experiment


experiment = Experiment(None, name='Data', par_dir='experiments')
run = experiment.create_run()
run.start()
```

## The dataset
Annotate the custom dataset's `__getitem__` with the pair as a tuple.

```{code-cell} ipython3
import torch

from torch.utils import data


class LineDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Samples of the line y = 2x."""

    def __init__(self, *, n_samples: int) -> None:
        """Initialize.

        Args:
            n_samples: number of samples.
        """
        self.inputs: torch.Tensor = torch.arange(n_samples).float().unsqueeze(1)
        self.targets: torch.Tensor = 2 * self.inputs
        return

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input and the target of a sample."""
        return self.inputs[index], self.targets[index]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.inputs)


dataset = LineDataset(n_samples=100)
```

## Taking a few samples
[`take_from_dataset`](#drytorch.lib.loading.take_from_dataset) extracts a batch of samples (in order by default) and places them on the specified device for quick inspection or consistent checks.
It returns the indices of the samples in the dataset, then the batch: with
`preserve_order=False`, the samples are random, and the indices tell which they are.

```{code-cell} ipython3
import matplotlib.pyplot as plt

from drytorch.lib.loading import take_from_dataset


indices, (inputs, targets) = take_from_dataset(
    dataset, n_samples=4, preserve_order=False, device=torch.device('cpu')
)
plt.scatter(inputs, targets)
plt.xlabel('input')
plt.ylabel('target')
plt.show()
indices
```

With `indices`, it takes the given samples in the given order, for example the
same few samples to inspect after every epoch.

```{code-cell} ipython3
_, (inputs, targets) = take_from_dataset(
    dataset, indices=[90, 3, 42], device=torch.device('cpu')
)
inputs.squeeze(1)
```

## The DataLoader class
[`DataLoader`](#drytorch.lib.loading.DataLoader) wraps the dataset. It manages batching, sampler selection, and
memory pinning.

```{code-cell} ipython3
from drytorch.lib.loading import DataLoader


loader = DataLoader(dataset, batch_size=16)
if loader.batch_size != 16:
    raise AssertionError('The batch size should be 16.')

if loader.dataset_len != 100:
    raise AssertionError('The dataset length should be 100.')
```

### Training and inference loaders
[`DataLoader`](#drytorch.lib.loading.DataLoader) adapts automatically based on whether PyTorch's inference mode
(`torch.inference_mode()`) is active:
- In training, it draws samples randomly and drops incomplete batches to avoid
  biased gradients.
- In inference, it preserves sample order and retains all batches to evaluate
  the full dataset.

```{code-cell} ipython3
train_torch_loader = loader.get_loader(inference=False)
eval_torch_loader = loader.get_loader(inference=True)
if not train_torch_loader.drop_last:
    raise AssertionError('Training loader should drop incomplete batches.')

if eval_torch_loader.drop_last:
    raise AssertionError('Inference loader should retain the last batch.')

if len(train_torch_loader) != 6:
    raise AssertionError('Drop-last loader should yield 6 batches.')

if len(eval_torch_loader) != 7:
    raise AssertionError('Inference loader should yield 7 batches.')
```

## Splitting a loader
The method [`split`](#drytorch.lib.loading.DataLoader.split) divides the dataset into two and creates two loaders,
for example a training loader and a validation loader.

[`split`](#drytorch.lib.loading.DataLoader.split) takes:
- `ratio`: the fraction assigned to the second loader (default `0.2`).
- `shuffle`: whether to shuffle before splitting (default `True`).
- `seed`: the random seed for shuffling (default `42`).

```{code-cell} ipython3
train_loader, val_loader = loader.split(ratio=0.2, shuffle=True, seed=42)
if train_loader.dataset_len != 80:
    raise AssertionError('Train loader should have 80 samples.')

if val_loader.dataset_len != 20:
    raise AssertionError('Validation loader should have 20 samples.')
```

The method [`fold`](#drytorch.lib.loading.DataLoader.fold) prepares a k-fold cross-validation: it divides the dataset
into `n_folds` parts and, for each part, returns a training loader on the
other parts and a validation loader on that part. It takes `shuffle` and
`seed` as [`split`](#drytorch.lib.loading.DataLoader.split) does.

```{code-cell} ipython3
folds = loader.fold(n_folds=5)
for train_loader, val_loader in folds:
    if (train_loader.dataset_len, val_loader.dataset_len) != (80, 20):
        raise AssertionError('Each fold should train on 80 and validate on 20.')
```

## Custom samplers
Pass `sampler` to [`DataLoader`](#drytorch.lib.loading.DataLoader) to apply a custom sampling strategy. When
provided, [`DataLoader`](#drytorch.lib.loading.DataLoader) uses it directly for both training and inference:

```{code-cell} ipython3
from torch.utils.data import SequentialSampler


custom_sampler = SequentialSampler(range(len(dataset)))
custom_loader = DataLoader(dataset, batch_size=10, sampler=custom_sampler)
if custom_loader.sampler is not custom_sampler:
    raise AssertionError('The loader should keep the custom sampler.')
```

## Custom loaders
Any loader satisfying [`LoaderProtocol`](#drytorch.core.protocols.LoaderProtocol) works, such as PyTorch's native
[`DataLoader`](#torch.utils.data.DataLoader):

```{code-cell} ipython3
from torch.utils import data


torch_loader = data.DataLoader(dataset, batch_size=16)
if len(torch_loader) != 7:
    raise AssertionError('The loader should yield 7 batches.')
```

## Stop the run

```{code-cell} ipython3
run.stop()
```

For multi-process distributed data loading and partition details, see the
[distributed_training](distributed_training.md)
tutorial.
