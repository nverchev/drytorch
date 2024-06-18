import warnings
import pathlib
import datetime
import yaml  # type: ignore
from typing import Optional
import logging
import pandas as pd
import torch
from dry_torch import default_logging
from dry_torch import tracking
from dry_torch import protocols
from dry_torch import data_types

logger = logging.getLogger('dry_torch')


class PathManager:
    """
    Manages the paths for the experiment.

    Args:
        network: The name of the module.

    Attributes:
        network  The name of the module.
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
        checkpoint (CheckpointPath): The path for the checkpoint.

    Methods:
        get_last_saved_epoch(self) -> int: Get the last saved epoch.
    """

    def __init__(self, network: protocols.NetworkProtocol) -> None:
        self.network = network

    @property
    def exp(self) -> tracking.Experiment:
        return tracking.Experiment.current()

    @property
    def model_tracking(self) -> tracking.ModelTracking:
        return self.exp.model[self.network.name]

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
    def log(self) -> data_types.PathDict:
        logs_directory = self.logs_directory
        return {split: logs_directory / f'{split.name.lower()}_log.csv'
                for split in data_types.Split}

    @property
    def checkpoint(self) -> protocols.CheckpointPath:
        checkpoint_directory = self.checkpoint_directory
        epoch = self.network.epoch
        return dict(
            module=checkpoint_directory / f'module_epoch_{epoch}.pt',
            optimizer=checkpoint_directory / f'optimizer_epoch_{epoch}.pt',
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
            if checkpoint_desc.startswith('module'):
                past_epochs.append(int(epoch_str))
        if not past_epochs:
            raise FileNotFoundError(
                f'No saved module found in {checkpoint_directory}.')
        return max(past_epochs)


class CheckpointIO:
    """
    Save and load checkpoints. The folder with the savings has the address
    of the form: exp_pardir/exp_name.

    Args:
        network: contain the module and the optimizing strategy.
        . Defaults to module.


    Methods:
        save: save a checkpoint.
        load: load a checkpoint.
        exp_name: property with the name of the experiment.
        epoch: property with the current epoch.
    """

    def __init__(
            self,
            network: protocols.NetworkProtocol,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:

        self.network = network
        self.optimizer = optimizer
        self.paths = PathManager(network=network)

    @property
    def model_info(self) -> tracking.ModelTracking:
        exp = tracking.Experiment.current()
        return exp.model[self.network.name]

    def save(self) -> None:
        """
        Save a checkpoint for the module and the optimizer with the metadata of
        the experiments, the test results,
        and the training and validation learning curves.
        """
        self.network.module.eval()
        torch.save(self.network.module.state_dict(),
                   self.paths.checkpoint['module'])
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(),
                       self.paths.checkpoint['optimizer'])
        for df_name, path in self.paths.log.items():
            # write instead of append to be safe from bugs
            self.network.log[df_name].to_csv(path)
        exp = tracking.Experiment.current()
        config = exp.config
        if config:
            with self.paths.config.open('w') as config_file:
                yaml.dump(config, config_file, sort_keys=False)
        with self.paths.metadata.open('w') as metadata_file:
            now: str = datetime.datetime.now().strftime('%H:%M:%S %d/%m/%Y')
            metadata = {'timestamp': now} | self.model_info.metadata
            yaml.dump(metadata, metadata_file, sort_keys=False)
        logger.log(default_logging.INFO_LEVELS.checkpoint,
                   f"Model saved in: %(model_path)s",
                   {'model_path': self.paths.checkpoint['module']})
        return

    def load(self, epoch: int = -1) -> None:
        """
        Load a checkpoint for the module and the optimizer, the training and
        validation metrics and the test results.

        Args:
            epoch: the epoch of the checkpoint to load, the last checkpoint is
            loaded if a negative value is given.
        """
        epoch = epoch if epoch >= 0 else self.paths.get_last_saved_epoch()
        self.network.epoch = epoch
        self.network.module.load_state_dict(
            torch.load(self.paths.checkpoint['module'],
                       map_location=self.network.device),
        )
        if self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(
                    torch.load(self.paths.checkpoint['optimizer'],
                               map_location=self.network.device),
                )
            except ValueError as ve:
                message = f'The optimizer has not been correctly loaded:\n{ve}'
                warnings.warn(message, RuntimeWarning)
        for df_name, path in self.paths.log.items():
            try:
                df = pd.read_csv(path, index_col=0)
            except FileNotFoundError:
                df = pd.DataFrame()
            df = df[df.index <= epoch]  # filter out future epochs from logs
            self.network.log[df_name] = df
        logger.log(default_logging.INFO_LEVELS.checkpoint,
                   f"Loaded: %(model_path)s",
                   {'model_path': self.paths.checkpoint['module']})
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(module={self.network})'
