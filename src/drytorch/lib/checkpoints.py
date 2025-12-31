"""Module containing classes to save the model state and its optimizer state."""

import abc
import codecs
import pathlib
import warnings

from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import torch

from torch import distributed as dist
from typing_extensions import override

from drytorch.core import exceptions, experiment, log_events
from drytorch.core import protocols as p


__all__ = [
    'AbstractCheckpoint',
    'LocalCheckpoint',
]


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
    codecs.encode,
]
try:
    from numpy._core.multiarray import scalar  # type: ignore # pyright: ignore
except ImportError:
    pass
else:
    SAFE_GLOBALS.append(scalar)

SAFE_GLOBALS.extend([getattr(np.dtypes, name) for name in np.dtypes.__all__])
torch.serialization.add_safe_globals(SAFE_GLOBALS)


class CheckpointPathManager:
    """Manage paths for the experiment.

    Class Attributes:
        folder_name: name of the folder where the checkpoints are stored.
    """

    folder_name: ClassVar[str] = 'checkpoints'
    _model: p.ModelProtocol[Any, Any]
    _run_dir: Path | None

    def __init__(
        self,
        model: p.ModelProtocol[Any, Any],
        run_dir: pathlib.Path | None = None,
    ) -> None:
        """Initialize.

        Args:
            model: the model whose paths are to be managed.
            run_dir: the directory for experiment data.
        """
        self._model: Final = model
        self._run_dir = run_dir

    @property
    def run_dir(self) -> pathlib.Path:
        """Parent directory for the checkpoints."""
        if self._run_dir is None:
            try:
                exp = experiment.Experiment[Any].get_current()
            except exceptions.NoActiveExperimentError as naee:
                raise exceptions.AccessOutsideScopeError from naee
            else:
                exp_dir = exp.par_dir / self.folder_name / exp.name
                if '@' in exp.run.id:
                    day, time = exp.run.id.split('@')
                    return exp_dir / day / time
                else:
                    return exp_dir / exp.run.id

        return self._run_dir

    @property
    def model_dir(self) -> pathlib.Path:
        """Directory for the model."""
        model_dir = self.run_dir / self._model.name
        return model_dir

    @property
    def epoch_dir(self) -> pathlib.Path:
        """Directory for a checkpoint at the current epoch."""
        epoch_directory = self.model_dir / f'epoch_{self._model.epoch}'
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

    _model: p.ModelProtocol[Any, Any] | None
    _optimizer: torch.optim.Optimizer | None

    def __init__(self) -> None:
        """Initialize."""
        self._model = None
        self._optimizer = None

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
        if dist.is_available and dist.is_initialized():
            dist.barrier()

        self._update_epoch(epoch)
        log_events.LoadModelEvent(
            model_name=self.model.name,
            definition=self._get_definition(),
            location=self._get_location(),
            epoch=self.model.epoch,
        )
        self._load()
        return

    def remove_model(self) -> None:
        """Remove registered model."""
        self._model = None
        self._optimizer = None
        return

    def bind_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Bind the model to manage."""
        self._model = model
        return

    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Bind the optimizer connected to the model."""
        self._optimizer = optimizer
        return

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""
        log_events.SaveModelEvent(
            model_name=self.model.name,
            definition=self._get_definition(),
            location=self._get_location(),
            epoch=self.model.epoch,
        )
        if dist.is_available and dist.is_initialized() and dist.get_rank():
            return

        self._save()
        return

    def _get_definition(self) -> str:
        return 'state' if self.optimizer is None else 'checkpoint'

    @abc.abstractmethod
    def _load(self) -> None: ...

    @abc.abstractmethod
    def _get_last_saved_epoch(self) -> int: ...

    @abc.abstractmethod
    def _get_location(self) -> str: ...

    @abc.abstractmethod
    def _save(self) -> None: ...

    def _update_epoch(self, epoch: int):
        if epoch < -1:
            raise ValueError('Epoch must be larger than -1.')

        epoch = epoch if epoch >= 0 else self._get_last_saved_epoch()
        self.model.epoch = epoch


class LocalCheckpoint(AbstractCheckpoint):
    """Manage locally saving and loading the model state and optimizer."""

    def __init__(self, par_dir: pathlib.Path | None = None) -> None:
        """Initialize.

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
    def _load(self, epoch: int = -1) -> None:
        if not self.paths.model_dir.exists():
            raise exceptions.ModelNotFoundError(self.paths.run_dir)

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
                warnings.warn(
                    exceptions.OptimizerNotLoadedWarning(ve), stacklevel=1
                )

        return

    @override
    def _save(self) -> None:
        self.paths.epoch_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.module.state_dict(), self.paths.model_state_path)
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(), self.paths.optimizer_state_path
            )

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
