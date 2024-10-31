import warnings
import pathlib

import yaml  # type: ignore
import torch

from src.dry_torch import events
from src.dry_torch import descriptors
from src.dry_torch import exceptions
from src.dry_torch import tracking
from src.dry_torch import protocols as p


class PathManager:
    """
    Manages the paths for the experiment.

    Args:
        model: The model.

    Attributes:
        tracker  The model_name of the module.
        checkpoints.
    Properties:
        exp (Experiment): The active environment of the experiment.
        model_tracking (ModelTracker): The tracker information of the module.
        directory (Path): The directory for the experiment.
        config (Path): The configuration file path.
        model_directory (Path): The directory for the module.
        checkpoint_directory (Path): The directory for the checkpoints.
        logs_directory (Path): The directory for the logs.
        metadata (Path): The metadata file path.
        log (dict): A dictionary containing the paths for the logs.
        checkpoint (StatePath): The path for the checkpoint.

    Methods:
        get_last_saved_epoch(self) -> int: Get the last saved epoch.
    """

    def __init__(self, model: p.ModelProtocol) -> None:
        self.model = model

    @property
    def exp(self) -> tracking.Experiment:
        return tracking.Experiment.current()

    @property
    def exp_dir(self) -> pathlib.Path:
        exp = tracking.Experiment.current()
        return exp.dir

    @property
    def model_dir(self) -> pathlib.Path:
        directory = self.exp_dir / self.model.name
        directory.mkdir(exist_ok=True)
        return directory

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        checkpoint_directory = self.model_dir / 'checkpoints'
        checkpoint_directory.mkdir(exist_ok=True)
        return checkpoint_directory

    @property
    def epoch_dir(self) -> pathlib.Path:
        epoch_directory = self.checkpoint_dir / f'epoch_{self.model.epoch}'
        epoch_directory.mkdir(exist_ok=True)
        return epoch_directory

    @property
    def checkpoint(self) -> descriptors.StatePath:
        epoch_directory = self.epoch_dir
        return dict(
            state=epoch_directory / 'state.pt',
            optimizer=epoch_directory / 'optimizer.pt',
        )

    @property
    def metadata_dir(self) -> pathlib.Path:
        metadata_directory = self.model_dir / 'metadata'
        metadata_directory.mkdir(exist_ok=True)
        return metadata_directory

    def get_last_saved_epoch(self) -> int:
        """
        Get the last saved epoch.

        Returns:
            int: The last saved epoch.
        """
        checkpoint_directory = self.checkpoint_dir
        past_epochs: list[int] = []
        for path in checkpoint_directory.iterdir():
            checkpoint_desc, epoch_str = path.stem.rsplit("_", 1)
            past_epochs.append(int(epoch_str))
        if not past_epochs:
            raise exceptions.ModelNotFoundError(checkpoint_directory)
        return max(past_epochs)


class ModelStateIO:
    """
    Save and load checkpoints. The folder with the savings has the address
    of the form: pardir/name.

    Args:
        model: contain the module and the optimizing strategy.
        . Defaults to module.


    Methods:
        save: save a checkpoint.
        load: load a checkpoint.
        name: property with the model_name of the experiment.
        epoch: property with the current epoch.
    """
    definition = 'state'

    def __init__(self,
                 model: p.ModelProtocol) -> None:
        self.model = model
        self.paths = PathManager(model)

    def _update_epoch(self, epoch: int):
        epoch = epoch if epoch >= 0 else self.paths.get_last_saved_epoch()
        self.model.epoch = epoch

    def save(self) -> None:
        """
        Save a checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        torch.save(self.model.module.state_dict(),
                   self.paths.checkpoint['state'])
        events.SaveCheckpoint(definition=self.definition,
                              location=str(self.paths.model_dir))
        return

    def load(self, epoch: int = -1) -> None:
        """
        Load a checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the checkpoint to load, the last checkpoint is
            loaded if a negative value is given.
        """
        self._update_epoch(epoch)
        self.model.module.load_state_dict(
            torch.load(self.paths.checkpoint['state'],
                       map_location=self.model.device),
        )
        events.LoadCheckpoint(definition=self.definition,
                              location=str(self.paths.model_dir),
                              epoch=self.model.epoch)
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(module={self.model.name})'


class CheckpointIO(ModelStateIO):
    """
    Save and load checkpoints. The folder with the savings has the address
    of the form: pardir/name.

    Args:
        model: contain the module and the optimizing strategy.
        . Defaults to module.


    Methods:
        save: save a checkpoint.
        load: load a checkpoint.
        name: property with the model_name of the experiment.
        epoch: property with the current epoch.
    """

    definition = 'checkpoint'

    def __init__(self,
                 model: p.ModelProtocol,
                 optimizer: torch.optim.Optimizer) -> None:
        super().__init__(model)
        self.optimizer = optimizer

    def save(self) -> None:
        """
        Save a checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        super().save()
        torch.save(self.optimizer.state_dict(),
                   self.paths.checkpoint['optimizer'])
        return

    def load(self, epoch: int = -1) -> None:
        """
        Load a checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the checkpoint to load, the last checkpoint is
            loaded if a negative value is given.
        """
        super().load(epoch)
        try:
            self.optimizer.load_state_dict(
                torch.load(self.paths.checkpoint['optimizer'],
                           map_location=self.model.device),
            )
        except ValueError as ve:
            warnings.warn(exceptions.OptimizerNotLoadedWarning(ve))
        return
