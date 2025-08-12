"""Module containing classes to save the model state and its optimizer state."""

import abc
import pathlib
import warnings

from typing import Any

import numpy as np
import numpy._core.multiarray  # pyright: ignore[reportPrivateImportUsage]
import torch

from typing_extensions import override

from drytorch.core import exceptions, experiments, log_events
from drytorch.core import protocols as p


SAFE_GLOBALS: list[Any] = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.dtype,
    np._core.multiarray.scalar  # type: ignore
]
SAFE_GLOBALS.extend([getattr(np.dtypes, name) for name in np.dtypes.__all__])
torch.serialization.add_safe_globals(SAFE_GLOBALS)


class CheckpointPathManager:
    """Manage paths for the experiment.

    Class Attributes:
        folder_name: name of the folder where the checkpoints are stored.

    Attributes:
        model: the model whose paths are to be managed.
    """
    folder_name = 'checkpoints'

    def __init__(
        self,
        model: p.ModelProtocol[Any, Any],
        exp_dir: pathlib.Path | None = None,
    ) -> None:
        """Constructor.

        Args:
            model: the model whose paths are to be managed.
            exp_dir: the directory for experiment data.
        """
        self.model = model
        self._exp_dir = exp_dir

    @property
    def exp_dir(self) -> pathlib.Path:
        """Parent directory for the checkpoints."""
        if self._exp_dir is None:
            try:
                exp = experiments.Experiment[Any].get_current()
            except exceptions.NoActiveExperimentError as naee:
                raise exceptions.AccessOutsideScopeError from naee
            else:
                return exp.par_dir / self.folder_name / exp.name

        return self._exp_dir

    @property
    def model_dir(self) -> pathlib.Path:
        """Directory for the model."""
        model_dir = self.exp_dir / self.model.name
        return model_dir

    @property
    def epoch_dir(self) -> pathlib.Path:
        """Directory for a checkpoint at the current epoch."""
        epoch_directory = self.model_dir / f'epoch_{self.model.epoch}'
        return epoch_directory

    @property
    def model_state_path(self) -> pathlib.Path:
        """Name of the file with the state."""
        epoch_directory = self.epoch_dir
        return epoch_directory / 'model_state.pt'

    @property
    def optimizer_state_path(self) -> pathlib.Path:
        """Name of the file with the optimizer state."""
        epoch_directory = self.epoch_dir
        return epoch_directory / 'optimizer_state.pt'


class AbstractCheckpoint(p.CheckpointProtocol, abc.ABC):
    """Abstract class that stores and loads weight for a ModelProtocol class."""

    def __init__(self) -> None:
        """Constructor."""
        self._model: p.ModelProtocol[Any, Any] | None = None
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
        log_events.LoadModelEvent(
            model_name=self.model.name,
            definition=self._get_definition(),
            location=self._get_location(),
            epoch=self.model.epoch,
        )

    def remove_model(self):
        """Remove registered model."""
        self._model = None
        self._optimizer = None

    def register_model(self, model: p.ModelProtocol[Any, Any]):
        """Register the model to manage."""
        self._model = model

    def register_optimizer(self, optimizer: torch.optim.Optimizer):
        """Register the optimizer connected to the model."""
        self._optimizer = optimizer

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""
        log_events.SaveModelEvent(
            model_name=self.model.name,
            definition=self._get_definition(),
            location=self._get_location(),
            epoch=self.model.epoch,
        )

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
            raise ValueError('Epoch must be larger than -1.')
        epoch = epoch if epoch >= 0 else self._get_last_saved_epoch()
        self.model.epoch = epoch


class LocalCheckpoint(AbstractCheckpoint):
    """Manage locally saving and loading the model state and optimizer."""

    def __init__(self, par_dir: pathlib.Path | None = None) -> None:
        """Constructor.

        Args:
            par_dir: parent directory for experiment data.
        """
        super().__init__()
        self._par_dir = par_dir
        return

    @property
    def paths(self) -> CheckpointPathManager:
        """Path manager for directories and checkpoints."""
        return CheckpointPathManager(self.model, self._par_dir)

    @override
    def load(self, epoch: int = -1) -> None:
        super().load(epoch)
        if not self.paths.model_dir.exists():
            raise exceptions.ModelNotFoundError(self.paths.exp_dir)

        if not self.paths.epoch_dir.exists():
            raise exceptions.EpochNotFoundError(epoch, self.paths.model_dir)

        self.model.module.load_state_dict(
            torch.load(
                self.paths.model_state_path,
                map_location=self.model.device,
                weights_only=True,
            )
        )
        if self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(
                    torch.load(
                        self.paths.optimizer_state_path,
                        map_location=self.model.device,
                        weights_only=True,
                    ),
                )
            except ValueError as ve:
                # noinspection PyArgumentEqualDefault
                warnings.warn(
                    exceptions.OptimizerNotLoadedWarning(ve), stacklevel=1
                )

        return

    @override
    def save(self) -> None:
        super().save()
        self.paths.epoch_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.module.state_dict(), self.paths.model_state_path)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(),
                       self.paths.optimizer_state_path)

        return

    def _get_last_saved_epoch(self) -> int:
        model_directory = self.paths.model_dir
        if model_directory.exists():
            all_epochs = [d for d in model_directory.iterdir() if d.is_dir()]
        else:
            all_epochs = []

        if not all_epochs:
            raise exceptions.ModelNotFoundError(model_directory)

        last_epoch_dir = max(all_epochs, key=self._creation_time)
        return self._get_epoch(last_epoch_dir)

    def _get_location(self) -> str:
        return str(self.paths.epoch_dir)

    @staticmethod
    def _creation_time(directory: pathlib.Path) -> float:
        creation_time = 0.0
        for file in directory.iterdir():
            creation_time = max(creation_time, file.stat().st_ctime)

        return creation_time

    @staticmethod
    def _get_epoch(directory: pathlib.Path) -> int:
        return int(directory.stem.rsplit('_', 1)[-1])
