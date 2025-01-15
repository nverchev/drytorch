import warnings
import pathlib

import torch

from src.dry_torch import log_events
from src.dry_torch import exceptions
from src.dry_torch import tracking
from src.dry_torch import protocols as p


class PathManager:
    """
    Manages paths for the experiment.

    Args:
        model: The model whose paths are to be managed.

    Attributes:
        model: The model instance for which paths are managed.
        model_dir: Directory for the model's data.

    Properties:
        checkpoint_dir: Directory for storing checkpoints.
        epoch_dir: Directory for storing epoch-specific checkpoints.
        metadata_dir: Directory for model metadata storage.
    """

    def __init__(self, model: p.ModelProtocol) -> None:
        self.model = model
        experiment = tracking.Experiment.current()
        self.model_dir = experiment.dir / model.name

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        checkpoint_directory = self.model_dir / 'checkpoints'
        checkpoint_directory.mkdir(exist_ok=True, parents=True)
        return checkpoint_directory

    @property
    def epoch_dir(self) -> pathlib.Path:
        epoch_directory = self.checkpoint_dir / f'epoch_{self.model.epoch}'
        epoch_directory.mkdir(exist_ok=True)
        return epoch_directory

    @property
    def metadata_dir(self) -> pathlib.Path:
        metadata_directory = self.model_dir / 'metadata'
        metadata_directory.mkdir(exist_ok=True)
        return metadata_directory


class ModelStateIO:
    """
    Manages saving and loading the model state, handling checkpoints and epochs.

    Args:
        model: The model instance.

    Attributes:
        model: The model instance.
        paths: Path manager for directories and checkpoints.

    """
    definition = 'state'

    def __init__(self, model: p.ModelProtocol) -> None:
        self.model = model
        self.paths = PathManager(model)

    @property
    def state_path(self) -> pathlib.Path:
        epoch_directory = self.paths.epoch_dir
        return epoch_directory / 'state.pt'

    def _update_epoch(self, epoch: int):
        epoch = epoch if epoch >= 0 else self._get_last_saved_epoch()
        self.model.epoch = epoch

    def save(self) -> None:
        """Saves the model's state dictionary."""
        torch.save(self.model.module.state_dict(), self.state_path)
        log_events.SaveCheckpoint(model_name=self.model.name,
                                  definition=self.definition,
                                  location=str(self.paths.epoch_dir),
                                  epoch=self.model.epoch)

    def load(self, epoch: int = -1) -> None:
        """Loads the model's state dictionary."""
        self._update_epoch(epoch)
        self.model.module.load_state_dict(
            torch.load(self.state_path, map_location=self.model.device))
        log_events.LoadCheckpoint(model_name=self.model.name,
                                  definition=self.definition,
                                  location=str(self.paths.epoch_dir),
                                  epoch=self.model.epoch)

    def _get_last_saved_epoch(self) -> int:
        checkpoint_directory = self.paths.checkpoint_dir
        past_epochs = [
            int(path.stem.rsplit('_', 1)[-1])
            for path in checkpoint_directory.iterdir() if path.is_dir()
        ]
        if not past_epochs:
            raise exceptions.ModelNotFoundError(checkpoint_directory)
        return max(past_epochs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module={self.model.name})"


class CheckpointIO(ModelStateIO):
    """
    Manages saving and loading both model and optimizer states for checkpoints.

    Args:
        model: The model instance.
        optimizer: The optimizer instance.

    Attributes:
        optimizer: The optimizer instance.
    """
    definition = 'checkpoint'

    def __init__(self, model: p.ModelProtocol,
                 optimizer: torch.optim.Optimizer) -> None:
        super().__init__(model)
        self.optimizer = optimizer

    @property
    def optimizer_path(self) -> pathlib.Path:
        epoch_directory = self.paths.epoch_dir
        return epoch_directory / 'optimizer.pt'

    def save(self) -> None:
        """Saves the model and optimizer state dictionaries."""
        super().save()
        torch.save(self.optimizer.state_dict(), self.optimizer_path)

    def load(self, epoch: int = -1) -> None:
        """Loads the model and optimizer state dictionaries."""
        super().load(epoch)
        try:
            self.optimizer.load_state_dict(
                torch.load(self.optimizer_path, map_location=self.model.device),
            )
        except ValueError as ve:
            warnings.warn(exceptions.OptimizerNotLoadedWarning(ve))
