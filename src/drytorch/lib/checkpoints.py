"""Classes to save the model state and its optimizer state.

Attributes:
    SAFE_GLOBALS: types registered as safe for torch serialization.
"""

import abc
import codecs
import pathlib
import warnings

from typing import Any, ClassVar, Final

import numpy as np
import torch

from torch import distributed as dist
from typing_extensions import override

from drytorch.core import exceptions, experimenting, log_events, registering
from drytorch.core import protocols as p


__all__ = [
    'AbstractCheckpoint',
    'LocalCheckpoint',
]

_np_scalars: list[Any] = []
try:
    from numpy._core import multiarray
except ImportError:
    pass
else:
    _np_scalars.append(multiarray.scalar)

SAFE_GLOBALS: Final[tuple[Any, ...]] = (
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
    *_np_scalars,
    *(getattr(np.dtypes, name) for name in np.dtypes.__all__),
)
torch.serialization.add_safe_globals(list(SAFE_GLOBALS))


class CheckpointPathManager:
    """Manage paths for the experiment.

    Attributes:
        folder_name: name of the folder where the checkpoints are stored.
        model_dir: the directory for model checkpoints.
        run_dir: the directory for experiment data.
    """

    folder_name: ClassVar[str] = 'checkpoints'
    model_dir: Final[pathlib.Path]
    run_dir: Final[pathlib.Path]
    _model: Final[p.ModelProtocol[Any, Any]]

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
        run_dir_path = run_dir if run_dir is not None else self._get_run_dir()
        self.model_dir = run_dir_path / model.name
        self.run_dir = run_dir_path
        self._model = model
        return

    def get_epoch_dir(self, epoch: int) -> pathlib.Path:
        """Get the directory for a checkpoint at an epoch.

        Args:
            epoch: the epoch of the checkpoint.

        Returns:
            The directory of the checkpoint.
        """
        return self.model_dir / f'epoch_{epoch}'

    def get_model_state_path(
        self, module_name: str, epoch: int
    ) -> pathlib.Path:
        """Get the name of the file with the model state.

        Args:
            module_name: the name of the module.
            epoch: the epoch of the checkpoint.

        Returns:
            The path of the file.
        """
        return self.get_epoch_dir(epoch) / f'{module_name}_state.pt'

    def get_optimizer_state_path(self, epoch: int) -> pathlib.Path:
        """Get the name of the file with the optimizer state.

        Args:
            epoch: the epoch of the checkpoint.

        Returns:
            The path of the file.
        """
        return self.get_epoch_dir(epoch) / 'optimizer_state.pt'

    @classmethod
    def _get_run_dir(cls) -> pathlib.Path:
        try:
            exp = experimenting.Experiment[Any].get_current()
        except exceptions.NoActiveExperimentError as naee:
            raise exceptions.AccessOutsideScopeError from naee
        else:
            exp_dir = exp.par_dir / cls.folder_name / exp.name
            day, sep, time = exp.run.id.partition('@')
            return exp_dir / day / time if sep else exp_dir / exp.run.id


class AbstractCheckpoint(abc.ABC):
    """Abstract class that stores and loads weight for a ModelProtocol class."""

    _model: p.ModelProtocol[Any, Any] | None
    _modules: dict[str, Any]
    _optimizer: torch.optim.Optimizer | None

    def __init__(self) -> None:
        """Initialize."""
        self._model = None
        self._modules = {}
        self._optimizer = None
        return

    @property
    def model(self):
        """The registered model to be saved and loaded.

        Raises:
            CheckpointNotInitializedError: if no model has been bound.
        """
        if self._model is None:
            raise exceptions.CheckpointNotInitializedError()

        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        """The registered optimizer for the model."""
        return self._optimizer

    def bind_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Bind the model to manage."""
        if self._model is not None and self._model is not model:
            raise exceptions.ModelAlreadyBoundError(self._model.name)

        self._model = model
        self.bind_module('model', model.module)
        return

    def bind_module(self, name: str, module: torch.nn.Module) -> None:
        """Bind a module connected to the model."""
        self._modules[name] = module
        return

    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Bind the optimizer connected to the model."""
        if self._optimizer is not None and self._optimizer is not optimizer:
            raise exceptions.OptimizerAlreadyBoundError(self.model.name)

        self._optimizer = optimizer
        return

    def load(self, epoch: int = -1) -> int:
        """Load the model and optimizer state dictionaries.

        Args:
            epoch: epoch to load.

        Returns:
            The loaded epoch.

        Raises:
            ModelNotFoundError: if the model location does not exist.
            EpochNotFoundError: if the epoch location does not exist.
        """
        registering.check_current_run(self.model)
        if dist.is_available() and dist.is_initialized():
            device_idx = self.model.device.index
            if device_idx is not None:
                dist.barrier(device_ids=[device_idx])
            else:
                dist.barrier()

        epoch = self._resolve_epoch(epoch)
        self._check_location(epoch)
        for name, module in self._modules.items():
            self._load_module(name, module, epoch)

        self._load_optimizer(epoch)
        log_events.LoadModelEvent(
            model_name=self.model.name,
            definition=self._get_definition(),
            location=self._get_location(epoch),
            epoch=epoch,
        )
        return epoch

    def remove_model(self) -> None:
        """Remove registered model."""
        self._model = None
        self._optimizer = None
        self._modules.clear()
        return

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""
        registering.check_current_run(self.model)
        epoch = self.model.epoch
        is_main_process = not (
            dist.is_available() and dist.is_initialized() and dist.get_rank()
        )
        if is_main_process:
            for name, module in self._modules.items():
                self._save_module(name, module, epoch)

            self._save_optimizer(epoch)

        log_events.SaveModelEvent(
            model_name=self.model.name,
            definition=self._get_definition(),
            location=self._get_location(epoch),
            epoch=epoch,
        )
        return

    @abc.abstractmethod
    def _check_location(self, epoch: int) -> None: ...

    def _get_definition(self) -> str:
        return 'state' if self.optimizer is None else 'checkpoint'

    @abc.abstractmethod
    def _get_last_saved_epoch(self) -> int: ...

    @abc.abstractmethod
    def _get_location(self, epoch: int) -> str: ...

    @abc.abstractmethod
    def _load_module(
        self, name: str, module: torch.nn.Module, epoch: int
    ) -> None: ...

    @abc.abstractmethod
    def _load_optimizer(self, epoch: int) -> None: ...

    @abc.abstractmethod
    def _save_module(
        self, name: str, module: torch.nn.Module, epoch: int
    ) -> None: ...

    @abc.abstractmethod
    def _save_optimizer(self, epoch: int) -> None: ...

    def _resolve_epoch(self, epoch: int) -> int:
        if epoch < -1:
            raise ValueError('Epoch must be larger than -1.')

        return epoch if epoch >= 0 else self._get_last_saved_epoch()


class LocalCheckpoint(AbstractCheckpoint):
    """Manage locally saving and loading the model state and optimizer.

    Attributes:
        par_dir: parent directory for experiment data.
    """

    par_dir: Final[pathlib.Path | None]

    def __init__(self, par_dir: pathlib.Path | None = None) -> None:
        """Initialize.

        Args:
            par_dir: parent directory for experiment data.
        """
        super().__init__()
        self.par_dir = par_dir
        self._paths: CheckpointPathManager | None = None
        return

    @property
    def paths(self) -> CheckpointPathManager:
        """Path manager for directories and checkpoints."""
        if self._paths is None:
            raise exceptions.CheckpointNotInitializedError()

        return self._paths

    @override
    def bind_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        super().bind_model(model)
        self._paths = CheckpointPathManager(model, self.par_dir)
        return

    @override
    def save(self) -> None:
        self.paths.get_epoch_dir(self.model.epoch).mkdir(
            exist_ok=True, parents=True
        )
        super().save()
        return

    @override
    def _check_location(self, epoch: int) -> None:
        if not self.paths.model_dir.exists():
            raise exceptions.ModelNotFoundError(self.paths.run_dir)

        if not self.paths.get_epoch_dir(epoch).exists():
            raise exceptions.EpochNotFoundError(epoch, self.paths.model_dir)

        return

    def _get_last_saved_epoch(self) -> int:
        model_directory = self.paths.model_dir
        if model_directory.exists():
            all_epochs: list[pathlib.Path] = [
                d for d in model_directory.iterdir() if d.is_dir()
            ]
        else:
            all_epochs = []

        if not all_epochs:
            raise exceptions.ModelNotFoundError(model_directory)

        last_epoch_dir = max(all_epochs, key=self._get_epoch)
        return self._get_epoch(last_epoch_dir)

    def _get_location(self, epoch: int) -> str:
        return str(self.paths.get_epoch_dir(epoch))

    @override
    def _load_module(
        self, name: str, module: torch.nn.Module, epoch: int = -1
    ) -> None:
        state_dict = torch.load(
            self.paths.get_model_state_path(name, epoch),
            map_location=self.model.device,
            weights_only=True,
        )
        module.load_state_dict(state_dict)
        return

    @override
    def _load_optimizer(self, epoch: int = -1) -> None:
        if not self.paths.model_dir.exists():
            raise exceptions.ModelNotFoundError(self.paths.run_dir)

        if not self.paths.get_epoch_dir(epoch).exists():
            raise exceptions.EpochNotFoundError(epoch, self.paths.model_dir)
        if self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(
                    torch.load(
                        self.paths.get_optimizer_state_path(epoch),
                        map_location=self.model.device,
                        weights_only=True,
                    ),
                )
            except (ValueError, FileNotFoundError) as err:
                warnings.warn(
                    exceptions.OptimizerNotLoadedWarning(err), stacklevel=1
                )
        return

    @override
    def _save_module(
        self, name: str, module: torch.nn.Module, epoch: int
    ) -> None:
        torch.save(
            module.state_dict(), self.paths.get_model_state_path(name, epoch)
        )
        return

    @override
    def _save_optimizer(self, epoch: int) -> None:
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                self.paths.get_optimizer_state_path(epoch),
            )
        return

    @staticmethod
    def _get_epoch(directory: pathlib.Path) -> int:
        return int(directory.stem.rsplit('_', 1)[-1])
