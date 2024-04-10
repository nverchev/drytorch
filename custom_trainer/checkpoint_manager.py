import json
import os
import warnings
from typing import Any, Optional

import pandas as pd
import torch

from custom_trainer.experiment_tracker import ExperimentTracker
from custom_trainer.model_handler import ModelHandler


class CheckpointManager:
    """
            model_pardir: parent directory for the folders with the model checkpoints

    """

    def __init__(self, model_handler: ModelHandler,
                 exp_tracker: ExperimentTracker,
                 model_pardir: str = 'models') -> None:

        self.model_handler: ModelHandler = model_handler
        self.exp_tracker: ExperimentTracker = exp_tracker
        self.model_pardir: str = model_pardir
        self.settings: dict[str, Any] = {}

    @property
    def exp_name(self) -> str:
        return self.exp_tracker.exp_name

    @property
    def epoch(self) -> int:
        return self.model_handler.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.model_handler.epoch = value

    def save(self, new_exp_name: Optional[str] = None) -> None:
        """
        It saves a checkpoint for the model and the optimizer with the settings of the experiments, the results,
        and the training and validation learning curves. The folder has the name of the experiment and is located in
        {attribute model_pardir}/models

        Args:
            new_exp_name: optionally save the model in a folder with this name to branch the model.
             The Trainer object won't modify self.exp_name
        """
        self.model_handler.model.eval()
        paths = self.paths(model_pardir=self.model_pardir, exp_name=new_exp_name or self.exp_name, epoch=self.epoch)
        torch.save(self.model_handler.model.state_dict(), paths['model'])
        torch.save(self.model_handler.optimizer.state_dict(), paths['optim'])
        for df_name, df in self.exp_tracker.log.items():
            # write instead of append to be safe from bugs
            df.to_csv(paths[df_name])
        with open(paths['settings'], 'w') as json_file:
            json.dump(self.settings, json_file, default=str, indent=4)
        print('\rModel saved at: ', paths['model'])
        return

    def load(self, epoch: int = -1) -> None:
        """
        It loads a checkpoint for the model and the optimizer, and the training and validation learning curves.
        Args:
            epoch: the epoch of the checkpoint to load, -1 if last
        """
        if epoch >= 0:
            self.epoch = epoch
        else:
            past_epochs = []  # here it looks for the most recent model
            local_path = os.path.join(self.model_pardir, self.exp_name)
            if os.path.exists(local_path):
                for file in os.listdir(local_path):
                    if file[:5] == 'model':
                        past_epochs.append(int(''.join(filter(lambda c: c.isdigit(), file))))  # available epochs
            if not past_epochs:
                warnings.warn(f'No saved models found in {local_path}. Training from scratch.', UserWarning)
                return
            self.epoch = max(past_epochs)
        paths = self.paths(model_pardir=self.model_pardir, exp_name=self.exp_name, epoch=self.epoch)
        device = self.model_handler.device
        self.model_handler.model.load_state_dict(torch.load(paths['model'], map_location=device))
        try:
            self.model_handler.optimizer.load_state_dict(torch.load(paths['optim'], map_location=device))
        except ValueError as err:
            warnings.warn('Optimizer has not been correctly loaded:')
            print(err)
        for df_name in self.exp_tracker.log.keys():
            try:
                df = pd.read_csv(paths[df_name], index_col=0)
            except FileNotFoundError:
                df = pd.DataFrame()
            # filter out future epochs from the logs
            df = df[df.index <= self.epoch]
            self.exp_tracker.log[df_name] = df
        print('Loaded: ', paths['model'])
        return

    @classmethod
    def paths(cls, exp_name: str, epoch: int, model_pardir: str = 'models') -> dict[str, str]:
        """
        It gets the paths for saving and loading the experiment.

        Args:
            model_pardir: the path to the folder where to save experiments
            exp_name: the name of the folder of the experiment
            epoch: the epoch of the checkpoint for the model and optimizer
        """
        if not os.path.exists(model_pardir):
            os.mkdir(model_pardir)
        directory = os.path.join(model_pardir, exp_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths: dict[str, str] = {'settings': os.path.join(directory, 'settings.json')}
        # paths for results and learning curves
        for df_name in ['train_log_metric', 'val_log', 'saved_test_metrics']:
            paths[df_name] = os.path.join(directory, f'{df_name}.csv')
        # paths for the model and the optimizer checkpoints
        for pt_file in ['model', 'optim']:
            paths[pt_file] = os.path.join(directory, f'{pt_file}_epoch{epoch}.pt')
        return paths
