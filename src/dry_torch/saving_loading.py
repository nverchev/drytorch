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
        model_tracking (ModelTracking): The tracking information of the module.
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
    def model_tracking(self) -> tracking.ModelTracking:
        return self.exp.tracking[self.model_name]

    @property
    def directory(self) -> pathlib.Path:
        directory = self.exp.exp_pardir / self.exp.exp_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def config(self) -> pathlib.Path:
        return self.directory / 'config.yml'

    @property
    def model_directory(self) -> pathlib.Path:
        directory = self.directory / self.model_tracking.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def checkpoint_directory(self) -> pathlib.Path:
        checkpoint_directory = self.model_directory / 'checkpoints'
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        return checkpoint_directory

    @property
    def logs_directory(self) -> pathlib.Path:
        checkpoint_directory = self.model_directory / 'logs'
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        return checkpoint_directory

    @property
    def metadata(self) -> pathlib.Path:
        return self.model_directory / 'metadata.yml'

    @property
    def log(self) -> p.PathDict:
        logs_directory = self.logs_directory
        return {split: logs_directory / f'{split.name.lower()}_log.csv'
                for split in dry_torch.protocols.Split}

    @property
    def checkpoint(self) -> p.StatePath:
        epoch = self.model_tracking.epoch
        epoch_directory = self.checkpoint_directory / f'epoch_{epoch}'
        epoch_directory.mkdir(parents=True, exist_ok=True)
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
        checkpoint_directory = self.checkpoint_directory
        past_epochs: list[int] = []
        for path in checkpoint_directory.iterdir():
            checkpoint_desc, epoch_str = path.stem.rsplit("_", 1)
            past_epochs.append(int(epoch_str))
        if not past_epochs:
            raise exceptions.ModelNotFoundError(checkpoint_directory)
        return max(past_epochs)


class MetadataIO:
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
    def model_tracking(self) -> tracking.ModelTracking:
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
                    'model_dir': self.paths.model_directory}
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
            df = df[df.index <= epoch]  # filter out future epochs from logs
            self.model_tracking.log[split] = df
        logger.log(default_logging.INFO_LEVELS.checkpoint,
                   f"Loaded %(definition)s at epoch %(epoch)d.",
                   {'definition': self.definition,
                    'epoch': self.model_tracking.epoch}
                   )
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(module={self.model_name})'


class ModelStateIO(MetadataIO):
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
        super().save()
        if self.model is not None:
            self.model.module.eval()
            torch.save(self.model.module.state_dict(),
                       self.paths.checkpoint['state'])
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
        if self.model is not None:
            self.model.module.load_state_dict(
                torch.load(self.paths.checkpoint['state'],
                           map_location=self.model.device),
            )
        return super().load(epoch)


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

    def save(self) -> None:
        """
        Save a _checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """

        torch.save(self.optimizer.state_dict(),
                   self.paths.checkpoint['optimizer'])
        return super().save()

    def load(self, epoch: int = -1) -> None:
        """
        Load a _checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the _checkpoint to load, the last _checkpoint is
            loaded if a negative value is given.
        """
        self._update_epoch(epoch)
        try:
            self.optimizer.load_state_dict(
                torch.load(self.paths.checkpoint['optimizer'],
                           map_location=self.model.device),
            )
        except ValueError as ve:
            warnings.warn(exceptions.OptimizerNotLoadedWarning(ve))
        return super().load(epoch)


def save_all_metadata() -> None:
    for model_name in tracking.Experiment.current().tracking:
        checkpoint = MetadataIO(model_name)
        checkpoint.save()
