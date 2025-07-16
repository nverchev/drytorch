"""Module containing classes to save the model state and its optimizer state."""

import abc
import pathlib
from typing import Optional
import warnings

import numpy
import torch

from drytorch import log_events
from drytorch import exceptions
from drytorch import experiments
from drytorch import protocols as p

SAFE_GLOBALS = [getattr(numpy.dtypes, name) for name in numpy.dtypes.__all__]
SAFE_GLOBALS.extend([numpy.core.multiarray.scalar, numpy.dtype])  # type: ignore
torch.serialization.add_safe_globals(SAFE_GLOBALS)


class CheckpointPathManager:
    """
    Manage paths for the experiment.

    Attributes:
        model: the model whose paths are to be managed.
    """

    def __init__(self,
                 model: p.ModelProtocol,
                 root_dir: Optional[pathlib.Path] = None) -> None:
        """Constructor.

        Args:
            model: the model whose paths are to be managed.
            root_dir: parent directory for experiment data.
        """
        self.model = model
        self._root_dir = root_dir

    @property
    def root_dir(self) -> pathlib.Path:
        """Root directory."""
        if self._root_dir is None:
            try:
                return experiments.Experiment.current().dir
            except exceptions.NoActiveExperimentError:
                raise exceptions.AccessOutsideScopeError

        return self._root_dir

    @property
    def model_dir(self) -> pathlib.Path:
        """Directory for the model."""
        model_dir = self.root_dir / self.model.name
        model_dir.mkdir(exist_ok=True, parents=True)
        return model_dir

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Directory for a checkpoint at the current epoch."""
        epoch_directory = self.model_dir / 'checkpoints'
        epoch_directory.mkdir(exist_ok=True)
        return epoch_directory

    @property
    def epoch_dir(self) -> pathlib.Path:
        """Directory for a checkpoint at the current epoch."""
        epoch_directory = self.checkpoint_dir / f'epoch_{self.model.epoch}'
        epoch_directory.mkdir(exist_ok=True)
        return epoch_directory

    @property
    def state_path(self) -> pathlib.Path:
        """Name of the file with the state."""
        epoch_directory = self.epoch_dir
        return epoch_directory / 'state.pt'

    @property
    def optimizer_path(self) -> pathlib.Path:
        """Name of the file with the optimizer state."""
        epoch_directory = self.epoch_dir
        return epoch_directory / 'optimizer.pt'


class AbstractCheckpoint(p.CheckpointProtocol, abc.ABC):
    """Abstract class that stores and loads weight for a ModelProtocol class."""

    def __init__(self) -> None:
        self._model: p.ModelProtocol | None = None
        self._optimizer: torch.optim.Optimizer | None = None

    @property
    def model(self):
        """The registered model to be saved and loaded."""
        if self._model is None:
            raise exceptions.CheckpointNotInitializedError()
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        """The registered optimizer for the model."""
        return self._optimizer

    def load(self, epoch: int = -1) -> None:
        """Load the model and optimizer state dictionaries."""
        self._update_epoch(epoch)
        log_events.LoadModel(model_name=self.model.name,
                             definition=self._get_definition(),
                             location=self._get_location(),
                             epoch=self.model.epoch)

    def remove_model(self):
        """Remove registered model."""
        self._model = None
        self._optimizer = None

    def register_model(self, model: p.ModelProtocol):
        """Register the model to manage."""
        self._model = model

    def register_optimizer(self, optimizer: torch.optim.Optimizer):
        """Register the optimizer connected to the model."""
        self._optimizer = optimizer

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""
        log_events.SaveModel(model_name=self.model.name,
                             definition=self._get_definition(),
                             location=self._get_location(),
                             epoch=self.model.epoch)

    def _get_definition(self) -> str:
        return 'model_state' if self.optimizer is None else 'checkpoint'

    @abc.abstractmethod
    def _get_last_saved_epoch(self) -> int:
        ...

    @abc.abstractmethod
    def _get_location(self) -> str:
        ...

    def _update_epoch(self, epoch: int):
        if epoch < -1:
            ValueError('Epoch must be larger than -1.')
        epoch = epoch if epoch >= 0 else self._get_last_saved_epoch()
        self.model.epoch = epoch


class LocalCheckpoint(AbstractCheckpoint):
    """Manage locally saving and loading the model state and optimizer."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def paths(self) -> CheckpointPathManager:
        """Path manager for directories and checkpoints."""
        return CheckpointPathManager(self.model)

    def load(self, epoch: int = -1) -> None:
        super().load(epoch)
        self.model.module.load_state_dict(
            torch.load(self.paths.state_path,
                       map_location=self.model.device,
                       weights_only=True))
        if self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(
                    torch.load(self.paths.optimizer_path,
                               map_location=self.model.device,
                               weights_only=True),
                )
            except ValueError as ve:
                warnings.warn(exceptions.OptimizerNotLoadedWarning(ve))

        return

    def save(self) -> None:
        super().save()
        torch.save(self.model.module.state_dict(), self.paths.state_path)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), self.paths.optimizer_path)

        return

    def _get_last_saved_epoch(self) -> int:
        checkpoint_directory = self.paths.checkpoint_dir
        all_epochs = [d for d in checkpoint_directory.iterdir() if d.is_dir()]
        if not all_epochs:
            raise exceptions.ModelNotFoundError(checkpoint_directory)

        last_epoch_dir = max(all_epochs, key=self._creation_time)
        return self._get_epoch(last_epoch_dir)

    def _get_location(self) -> str:
        return str(self.paths.epoch_dir)

    @staticmethod
    def _creation_time(directory: pathlib.Path) -> float:
        creation_time = 0.
        for file in directory.iterdir():
            creation_time = max(creation_time, file.stat().st_ctime)

        return creation_time

    @staticmethod
    def _get_epoch(directory: pathlib.Path) -> int:
        return int(directory.stem.rsplit('_', 1)[-1])
