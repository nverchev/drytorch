import json
import os
import warnings
from typing import Optional

import pandas as pd
import torch

from dry_torch.experiment_tracker import ExperimentTracker
from dry_torch.model_handler import ModelHandler


class CheckpointManager:
    """
    Save and load checkpoints. The folder with the savings has the address of the form: model_pardir/exp_name.

    Args:
        model_handler: contain the model and the optimizing strategy.
        exp_tracker: track of the metrics and setting of the experiment.
        model_pardir: parent directory for the folders with the model checkpoints

    Methods:
        save: save a checkpoint.
        load: load a checkpoint.
        exp_name: property with the name of the experiment.
        epoch: property with the current epoch.
    """

    def __init__(self, model_handler: ModelHandler,
                 exp_tracker: ExperimentTracker,
                 model_pardir: str = 'models') -> None:

        self.model_handler: ModelHandler = model_handler
        self.exp_tracker: ExperimentTracker = exp_tracker
        self.model_pardir: str = model_pardir

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
        Save a checkpoint for the model and the optimizer with the metadata of the experiments, the test results,
        and the training and validation learning curves.

        Args:
            new_exp_name: save the model in a folder with this name to create a new branch for the experiment, optional.
        """
        self.model_handler.model.eval()
        paths = self._paths(exp_name=new_exp_name or self.exp_name)

        torch.save(self.model_handler.model.state_dict(), paths['model'])
        torch.save(self.model_handler.optimizer.state_dict(), paths['optim'])
        for df_name, df in self.exp_tracker.log.items():
            # write instead of append to be safe from bugs
            df.to_csv(paths[df_name])
        with open(paths['metadata'], 'w') as json_file:
            json.dump(self.exp_tracker.metadata, json_file, default=str, indent=4)
        print('\rModel saved at: ', paths['model'])
        return

    def load(self, epoch: int = -1) -> None:
        """
        Load a checkpoint for the model and the optimizer, the training and validation metrics and the test results.

        Args:
            epoch: the epoch of the checkpoint to load, -1 if last.
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
        paths = self._paths(self.exp_name)
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

    def _paths(self, exp_name) -> dict[str, str]:
        if not os.path.exists(self.model_pardir):
            os.mkdir(self.model_pardir)
        directory = os.path.join(self.model_pardir, exp_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths: dict[str, str] = {'metadata': os.path.join(directory, 'metadata.json')}
        # paths for results and learning curves
        for df_name in ['train_log_metric', 'val_log', 'saved_test_metrics']:
            paths[df_name] = os.path.join(directory, f'{df_name}.csv')
        # paths for the model and the optimizer checkpoints
        for pt_file in ['model', 'optim']:
            paths[pt_file] = os.path.join(directory, f'{pt_file}_epoch{self.epoch}.pt')
        return paths
