import warnings
import pathlib
import datetime

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
        tracking  The model_name of the module.
        checkpoints.
    Properties:
        exp (Experiment): The active environment of the experiment.
        model_tracking (ModelTracker): The tracking information of the module.
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
    def model_tracking(self) -> tracking.ModelTracker:
        return self.exp.tracking[self.model_name]

    @property
    def exp_dir(self) -> pathlib.Path:
        directory = self.exp.exp_pardir / self.exp.exp_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def config(self) -> pathlib.Path:
        return self.exp_dir / 'config.yml'

    @property
    def model_dir(self) -> pathlib.Path:
        directory = self.exp_dir / self.model_tracking.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def logs_dir(self) -> pathlib.Path:
        checkpoint_directory = self.model_dir / 'logs'
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
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
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        return checkpoint_directory

    @property
    def epoch_dir(self) -> pathlib.Path:
        epoch = self.model_tracking.epoch
        epoch_directory = self.checkpoint_dir / f'epoch_{epoch}'
        epoch_directory.mkdir(parents=True, exist_ok=True)
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


class TrackingIO:
    definition = 'metadata'
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
        self.paths = PathManager(model_name=model_name)

    @property
    def model_tracking(self) -> tracking.ModelTracker:
        exp = tracking.Experiment.current()
        return exp.tracking[self.model_name]

    def _update_epoch(self, epoch: int):
        epoch = epoch if epoch >= 0 else self.paths.get_last_saved_epoch()
        self.model_tracking.epoch = epoch

    def save(self) -> None:
        """
        Save a _checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        exp = tracking.Experiment.current()
        config = exp.config
        if config is not None:
            with self.paths.config.open('w') as config_file:
                yaml.dump(config, config_file, sort_keys=False)
        with self.paths.metadata.open('w') as metadata_file:
            now: str = datetime.datetime.now().isoformat(' ', 'seconds')
            metadata = {'timestamp': now} | self.model_tracking.metadata
            yaml.dump(metadata, metadata_file, sort_keys=False)
        for split, path in self.paths.log.items():
            # write instead of append to be safe from bugs
            self.model_tracking.log[split].to_csv(path)
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
                df = df[df.index <= self.model_tracking.epoch]
            self.model_tracking.log[split] = df
        logger.log(default_logging.INFO_LEVELS.checkpoint,
                   f"Loaded %(definition)s at epoch %(epoch)d.",
                   {'definition': self.definition,
                    'epoch': self.model_tracking.epoch}
                   )
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(module={self.model_name})'


class ModelStateIO(TrackingIO):
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

    def __init__(self, model: p.ModelProtocol) -> None:
        self.model = model
        super().__init__(model.name)

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


def save_all_metadata() -> None:
    for model_name in tracking.Experiment.current().tracking:
        tracking_io = TrackingIO(model_name)
        tracking_io.save()
