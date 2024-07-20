import warnings
import pathlib
import datetime
from typing import Any
import dry_torch.protocols
import yaml  # type: ignore
import logging
import pandas as pd
import torch
from dry_torch import default_logging, exceptions
from dry_torch import tracking
from dry_torch import protocols as p

logger = logging.getLogger('dry_torch')


class PathManager:
    """
    Manages the paths for the experiment.

    Args:
        model_name: The model_name of the module.

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
        _checkpoint (StatePath): The path for the _checkpoint.

    Methods:
        get_last_saved_epoch(self) -> int: Get the last saved epoch.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def exp(self) -> tracking.Experiment:
        return tracking.Experiment.current()

    @property
    def exp_dir(self) -> pathlib.Path:
        exp = tracking.Experiment.current()
        return exp.exp_dir

    @property
    def model_tracking(self) -> tracking.ModelTracker:
        return self.exp.tracker[self.model_name]

    @property
    def config(self) -> pathlib.Path:
        return self.exp.config_path

    @property
    def model_dir(self) -> pathlib.Path:
        directory = self.exp_dir / self.model_tracking.name
        directory.mkdir(exist_ok=True)
        return directory

    @property
    def logs_dir(self) -> pathlib.Path:
        checkpoint_directory = self.model_dir / 'logs'
        checkpoint_directory.mkdir(exist_ok=True)
        return checkpoint_directory

    @property
    def metadata(self) -> pathlib.Path:
        return self.model_dir / 'metadata.yml'

    @property
    def log(self) -> p.PathDict:
        logs_directory = self.logs_dir
        return {split: logs_directory / f'{split.name.lower()}_log.csv'
                for split in dry_torch.protocols.Split}

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        checkpoint_directory = self.model_dir / 'checkpoints'
        checkpoint_directory.mkdir(exist_ok=True)
        return checkpoint_directory

    @property
    def epoch_dir(self) -> pathlib.Path:
        epoch = self.model_tracking.epoch
        epoch_directory = self.checkpoint_dir / f'epoch_{epoch}'
        epoch_directory.mkdir(exist_ok=True)
        return epoch_directory

    @property
    def checkpoint(self) -> p.StatePath:
        epoch_directory = self.epoch_dir
        return dict(
            state=epoch_directory / 'state.pt',
            optimizer=epoch_directory / 'optimizer.pt',
        )

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


class LogIO:
    definition = 'logs'
    """
    Save and load checkpoints. The folder with the savings has the address
    of the form: exp_pardir/exp_name.

    Args:
        model: contain the module and the optimizing strategy.
        . Defaults to module.


    Methods:
        save: save a _checkpoint.
        load: load a _checkpoint.
        exp_name: property with the model_name of the experiment.
        epoch: property with the current epoch.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.paths = PathManager(model_name)

    @property
    def model_tracker(self) -> tracking.ModelTracker:
        return tracking.Experiment.current().tracker[self.model_name]

    def _update_epoch(self, epoch: int):
        epoch = epoch if epoch >= 0 else self.paths.get_last_saved_epoch()
        self.model_tracker.epoch = epoch

    def save(self) -> None:
        """
        Save a _checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        for split, path in self.paths.log.items():
            # write instead of append to be safe from bugs
            self.model_tracker.log[split].to_csv(path)
        logger.log(default_logging.INFO_LEVELS.checkpoint,
                   f"%(definition)s saved in: %(model_dir)s.",
                   {'definition': self.definition.capitalize(),
                    'model_dir': self.paths.model_dir}
                   )
        return

    def load(self, epoch: int = -1) -> None:
        """
        Load a _checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the _checkpoint to load, the last _checkpoint is
            loaded if a negative value is given.
        """
        self._update_epoch(epoch)
        for split, path in self.paths.log.items():
            try:
                df = pd.read_csv(path, index_col=0)
            except FileNotFoundError:
                df = pd.DataFrame()
            if split is not p.Split.TEST:
                # filter out future epochs from logs
                df = df[df.index <= self.model_tracker.epoch]
            self.model_tracker.log[split] = df
        logger.log(default_logging.INFO_LEVELS.checkpoint,
                   f"Loaded %(definition)s at epoch %(epoch)d.",
                   {'definition': self.definition,
                    'epoch': self.model_tracker.epoch}
                   )
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(module={self.model_tracker.name})'


class ModelStateIO(LogIO):
    """
    Save and load checkpoints. The folder with the savings has the address
    of the form: exp_pardir/exp_name.

    Args:
        model: contain the module and the optimizing strategy.
        . Defaults to module.


    Methods:
        save: save a _checkpoint.
        load: load a _checkpoint.
        exp_name: property with the model_name of the experiment.
        epoch: property with the current epoch.
    """
    definition = 'state'

    def __init__(self,
                 model: p.ModelProtocol) -> None:
        super().__init__(model.name)
        self.model = model

    def save(self) -> None:
        """
        Save a _checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        torch.save(self.model.module.state_dict(),
                   self.paths.checkpoint['state'])
        return super().save()

    def load(self, epoch: int = -1) -> None:
        """
        Load a _checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the _checkpoint to load, the last _checkpoint is
            loaded if a negative value is given.
        """
        super().load(epoch)
        self.model.module.load_state_dict(
            torch.load(self.paths.checkpoint['state'],
                       map_location=self.model.device),
        )
        return


class CheckpointIO(ModelStateIO):
    """
    Save and load checkpoints. The folder with the savings has the address
    of the form: exp_pardir/exp_name.

    Args:
        model: contain the module and the optimizing strategy.
        . Defaults to module.


    Methods:
        save: save a _checkpoint.
        load: load a _checkpoint.
        exp_name: property with the model_name of the experiment.
        epoch: property with the current epoch.
    """

    definition = 'checkpoint'

    def __init__(self,
                 model: p.ModelProtocol,
                 optimizer: torch.optim.Optimizer) -> None:
        super().__init__(model)
        self.optimizer = optimizer

    def save(self, replace_previous: bool = False) -> None:
        """
        Save a _checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        if replace_previous:
            epoch_dirs = list(self.paths.checkpoint_dir.iterdir())
            if epoch_dirs:
                last_epoch_dir = sorted(epoch_dirs)[-1]
                pathlib.Path(last_epoch_dir).rename(self.paths.epoch_dir)
        super().save()
        torch.save(self.optimizer.state_dict(),
                   self.paths.checkpoint['optimizer'])
        return

    def load(self, epoch: int = -1) -> None:
        """
        Load a _checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the _checkpoint to load, the last _checkpoint is
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


def dump_metadata(model_name: str, metadata: dict[str, Any]) -> None:
    with PathManager(model_name).metadata.open('w') as metadata_file:
        now = datetime.datetime.now().replace(microsecond=0)
        metadata = {'timestamp': now} | metadata
        yaml.dump(metadata, metadata_file, sort_keys=False,
                  default_flow_style=False)
    return
