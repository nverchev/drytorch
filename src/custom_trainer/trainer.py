import os
import warnings
import json
import torch
from torch.cuda.amp import GradScaler
from torch import autocast, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import visdom
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import sys
from typing import Any, Literal
from custom_trainer.schedulers import Scheduler, ConstantScheduler
from custom_trainer.utils import DictList, apply, dict_repr, C


class Trainer(metaclass=ABCMeta):
    """
    This abstract class manages training and general utilities for a Pytorch model.
    You need to subclass it and override the loss method.

    The motivation for using this class is the following:
        I) Inheritance allows user defined class for complex models and / or training by only adding minimal code
            1) Compartmentalization - the methods can be easily extended and overriden
            2) Flexible containers - the trainer uses dictionaries that can handle complex inputs / outputs
            3) Hooks that grant further customizability
        II) Already implemented complex functionalities during training:
            1) Scheduler for the learning rate decay and different learning rates for different parameters' groups
            2) Mixed precision training using torch.cuda.amp
            3) Visualization of the learning curves using the visdom library
        III) Utilities for logging, metrics and investigation of the model's outputs
            1) Simplified pipeline for logging, saving and loading a model
            2) The class attempts full model documentation
            3) DictList allows indexing of complex outputs for output exploration

    """
    quiet_mode: bool = False  # less output
    max_stored_output: int = float('inf')  # maximum amount of stored evaluated test samples
    max_length_config: int = 20
    tqdm_update_frequency = 10

    def __new__(cls, model, **kwargs):
        """
        Overridden to register init keyword arguments. It tries to document the settings as much as possible.
        Settings will be dumped in a JSON file when saving the model.
        Warning:
        If you load and save a model, you will overwrite the previous settings.

        Args:
            model: Pytorch nn.Module with an optional attribute (settings: dict) for extra documentation
        """
        obj = super().__new__(cls)

        # tries to get the most informative representation of the settings.
        kwargs = {k: dict_repr(v, max_length=cls.max_length_config) for k, v in kwargs.items()}
        model_architecture = {'model': model}  # JSON files do not allow strings on multiple lines
        model_settings: dict = getattr(model, 'settings', {})
        obj.settings = model_architecture | model_settings | kwargs
        return obj

    def __init__(self, model: torch.nn.Module, *, exp_name: str, device: torch.device,
                 optimizer_cls: type[torch.optim.Optimizer], optim_args: dict,
                 train_loader: DataLoader, val_loader: DataLoader = None, test_loader: DataLoader = None,
                 model_pardir: str = './models', amp: bool = False, scheduler: Scheduler = ConstantScheduler(),
                 **extra_config: dict) -> None:
        """
        Args:
            model: Pytorch model with a settings attribute which is a dictionary for extra details
            exp_name: name used for the folder containing the checkpoints and the metadata
            device: only gpu and cpu are supported
            optimizer_cls: a Pytorch optimizer class
            optim_args: arguments for the optimizer (see the optimizer_settings setter for more details)
            train_loader: loader for the dataset for training used in the train method
            val_loader: loader for dataset for validation used both in the train method and in the test method
            test_loader: loader for the dataset for testing only used in the test method
            scheduler: instance of a Scheduler class that modifies the learning rate depending on the epoch
            model_pardir: parent directory for the folders with the model checkpoints
            amp: mixed precision computing from torch.cuda.amp (works only with cuda)
            extra_config: extraneous settings for logging (see __new__)

            Important!
            The loaders should be compatible with the torch.utils.data.dataloader.DataLoader class.
            They must yield (inputs, targets, indices) where:
            inputs: inputs for the model. Either a tensor, or a structured container of tensors that uses list and dict
            targets: arguments only used by the loss function. It can be structured as wll
            indices: indices of the sampled rows of the dataset (set to 0 if it does not apply)

            The handling of the inputs and outputs can be specified overriding the helper_inputs and loss methods

        """
        self.device = device
        self.model = model.to(device)
        self.exp_name = exp_name
        self.scheduler = scheduler
        self.epoch = 0
        # this is a property that updates the learning rate according to the epoch and the scheduler
        self.optimizer_settings = optim_args.copy()
        self.optimizer = optimizer_cls(**self.optimizer_settings)
        self.scaler = GradScaler(enabled=amp and self.device.type == 'cuda')
        self.amp = amp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_log: dict[int: dict[str, float]] = defaultdict(dict)
        self.val_log: dict[int: dict[str, float]] = defaultdict(dict)
        self.test_indices: list[int]
        self.test_metadata: dict[str: int | str]
        self.test_outputs: DictList[torch.Tensor]  # store last test evaluation
        self.saved_test_metrics: dict[str, float] = {}  # saves metrics of last evaluation
        self.model_pardir = model_pardir
        self.vis = None
        return

    @property
    def optimizer_settings(self) -> dict:  # settings depend on the epoch
        """
        It implements the scheduler and separate learning rates for the parameter groups.
        If the optimizer is correctly updated, it should be a copy of its params groups

        Returns:
            list of dictionaries with parameters and their updated learning rates plus the other fixed settings
        """
        optim_groups = self._optimizer_settings[0]
        params = [{'params': group['params'], 'lr': self.scheduler(group['lr'], self.epoch)} for group in optim_groups]
        return {'params': params, **self._optimizer_settings[1]}

    @optimizer_settings.setter
    def optimizer_settings(self, optim_args: dict) -> None:
        """
        It pairs the learning rates to the parameter groups.
        If only one number for the learning rate is given, it uses its value for each parameter

        Args:
            optim_args are similar to the default for the optimizer, a dictionary with a required lr key
            The only difference is that lr can be a dict of the form {parameter_name: learning_rate}
        """
        lr: float | dict[str, float] = optim_args.pop('lr')  # removes 'lr' as setting, we move it inside 'params'
        if isinstance(lr, dict):  # support individual lr for each parameter (for fine-tuning for example)
            self._optimizer_settings = \
                [{'params': getattr(self.model, k).parameters(), 'lr': v} for k, v in lr.items()], optim_args
        else:
            self._optimizer_settings = [{'params': self.model.parameters(), 'lr': lr}], optim_args
        return

    def update_learning_rate(self, new_lr: float | list[dict[str, float]]) -> None:
        """
        It updates the learning rates of the optimizer.

        Args:
            new_lr: a global learning rate for all the parameters or a list of dict with a 'lr' as a key
            If you call it externally, make sure the list has the same order as in self.optimizer_settings
        """
        if isinstance(new_lr, float):  # transforms to list
            new_lr = [{'lr': new_lr} for _ in self.optimizer.param_groups]
        for g, up_g in zip(self.optimizer.param_groups, new_lr):
            g['lr'] = up_g['lr']
        return

    def train(self, num_epoch: int, val_after_train: bool = False) -> None:
        """
        It trains the model for the given epochs

        Args:
            num_epoch: a global learning rate for all the parameters or a list of dict with a 'lr' as a key
            val_after_train: run inference on the validation dataset after each epoch and calculates the loss
        """
        if not self.quiet_mode:
            print('Training {}'.format(self.exp_name), end='')
        for _ in range(num_epoch):
            self.update_learning_rate(self.optimizer_settings['params'])
            self.epoch += 1
            if self.quiet_mode:
                print('\r====> Epoch:{:4d}'.format(self.epoch), end=' ... ')
            else:
                print('\n====> Epoch:{:4d}'.format(self.epoch))
            self.model.train()
            self.hook_before_training_epoch()
            self._run_session(partition='train')
            self.hook_after_training_epoch()
            if self.val_loader and val_after_train:  # check losses on val
                print(end=('' if self.quiet_mode else '\n'))
                self.model.eval()
                with torch.inference_mode():
                    self._run_session(partition='val', use_metrics=False)
        print()
        return

    @torch.inference_mode()
    def test(self, partition: Literal['train', 'val', 'test'], save_outputs: bool = False, **kwargs) -> None:
        """
        It tests the current model in inference. It saves the metrics in saved_test_metrics

        Args:
            partition: the dataset used for testing
            save_outputs: if True, it stores the outputs of the model
        """
        self.model.eval()
        self._run_session(partition=partition, save_outputs=save_outputs, use_metrics=True)
        print()
        return

    def _run_session(self, partition: Literal['train', 'val', 'test'],
                     save_outputs: bool = False,
                     use_metrics: bool = False) -> None:
        """
        It implements batching, backpropagation, printing and storage of the outputs

        Args:
            partition: the dataset used for the session
            save_outputs: if True, it stores the outputs of the model
            use_metrics: whether to use self.loss or self.metrics
        """
        match partition:
            case 'train':
                loader = self.train_loader
                dict_log = self.train_log[self.epoch]
            case 'val':
                loader = self.val_loader
                dict_log = self.val_log[self.epoch]
            case 'test':
                loader = self.test_loader
                dict_log = self.saved_test_metrics
            case _:
                raise ValueError('partition should be "train", "val" or "test"')

        if save_outputs:
            self.test_indices, self.test_outputs = [], DictList()
            self.test_metadata = dict(partition=partition, max_ouputs=self.max_stored_output)

        epoch_log = defaultdict(float)
        num_batch = len(loader)
        with tqdm(enumerate(loader), total=num_batch, disable=self.quiet_mode, file=sys.stderr) as tqdm_loader:
            epoch_seen = 0
            for batch_idx, (inputs, targets, indices) in tqdm_loader:
                epoch_seen += indices.shape[0]
                inputs, targets = self.recursive_to([inputs, targets], self.device)
                inputs_aux = self.helper_inputs(inputs)
                with autocast(device_type=self.device.type, enabled=self.amp):
                    outputs = self.model(**inputs_aux)
                    if use_metrics:
                        batch_log = self.metrics(outputs, inputs, targets)
                    else:
                        batch_log = self.loss(outputs, inputs, targets)
                        criterion = batch_log['Criterion'].mean(0)
                    for loss_or_metric, value in batch_log.items():
                        epoch_log[loss_or_metric] += value.sum(0).item()
                if not torch.is_inference_mode_enabled():
                    if torch.isnan(criterion):
                        raise ValueError('Criterion is nan')
                    if torch.isinf(criterion):
                        raise ValueError('Criterion is inf')
                    self.scaler.scale(criterion).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if batch_idx % (num_batch // self.tqdm_update_frequency or 1) == 0:  # or 1 prevents division by 0
                        tqdm_loader.set_postfix({'Seen': epoch_seen,
                                                 'Loss': criterion.item()})
                # if you get memory error, limit max_stored_output
                if save_outputs and self.max_stored_output >= epoch_seen:
                    self.test_outputs.extend_dict(self.recursive_to(outputs, 'detach_cpu'))
                    self.test_indices.extend(map(int, indices))

        print(f'Average {partition} {"metric(s)" if torch.is_inference_mode_enabled() else "loss(es)"}:', end=' ')
        for loss_or_metric, value in epoch_log.items():
            value /= epoch_seen
            if torch.is_inference_mode_enabled() ^ (partition == 'train'):  # Evaluating train does not overwrite log
                dict_log[loss_or_metric] = value
            print('{}: {:.4e}'.format(loss_or_metric, value), end='\t')
        return

    @abstractmethod
    def loss(self, outputs: C, inputs: C, targets: C) -> dict[str, Tensor]:
        """
        You must override this method and return a dictionary "dict" with dict['Criterion'] = loss to backprop.

        Args:
            outputs: the outputs of the model
            inputs: inputs of the model (in self learning the inputs replace the targets)
            targets: the targets of the model

        Returns:
            loss: dictionary whose values must be tensors of shape (batch_size, 1)

        """
        return {'Criterion': torch.FloatTensor([[0], ])}

    def metrics(self, outputs: C, inputs: C, targets: C) -> dict[str, Tensor]:
        """
        This method works similarly to self.loss and defaults to it. Override for complex metrics during testing.

        Args:
            outputs: the outputs of the model
            inputs: inputs of the model (in self learning the inputs replace the targets)
            targets: the targets of the model

        Returns:
            metrics: dictionary whose values must be tensors of shape (batch_size, 1)

        """
        return self.loss(outputs, inputs, targets)

    def helper_inputs(self, inputs: C) -> dict[str, Any]:
        """
        This method is a hook that rearranges the inputs following the model's named arguments.

        Args:
            inputs: inputs of the model

        Returns:
            dictionary of the form {named_argument: input}
        """
        return {'x': inputs}

    def check_visdom_connection(self) -> bool:
        """
        It activates and check the connection to the visdom server https://github.com/fossasia/visdom

        Returns:
            True if there is an active connection, False otherwise
        """
        if self.vis is None:
            self.vis = visdom.Visdom(env=self.exp_name)
        return self.vis.check_connection()

    def plot_learning_curves(self, loss_or_metric: str = 'Criterion', start: int = 0,
                             win: str = 'Learning Curves') -> None:
        """
        This method is a hook that rearranges the inputs following the model's named arguments.

        Args:
            loss_or_metric: the loss or the metric to visualize
            start: the epoch from where you want to display the curve
            win: the name of the window (and title) of the plot in the visdom interface

        """
        if not self.check_visdom_connection():
            warnings.warn('Impossible to display the learning curves on the server. Check the connection.')
            return
        epochs_train = sorted(self.train_log.keys())[start:]
        values_train = [self.train_log[epoch][loss_or_metric] for epoch in self.train_log.keys()][start:]
        layout = dict(xlabel='Epoch', ylabel=loss_or_metric, title=win, update='replace', showlegend=True)
        self.vis.line(X=epochs_train, Y=values_train, win=win, opts=layout, name='Training')
        if self.val_log:
            epochs_val = []
            values_val = []
            for str_epoch in self.val_log.keys():
                int_epoch = int(str_epoch)
                if int_epoch >= start:
                    epochs_val.append(int_epoch)
                    values_val.append(self.val_log[str_epoch][loss_or_metric])
            self.vis.line(X=epochs_val, Y=values_val, win=win, opts=layout, update='append', name='Validation')
        return

    def save(self, new_exp_name: str = None) -> None:
        """
        It saves a checkpoint for the model and the optimizer with the settings of the experiments, the results,
        and the training and validation learning curves. The folder has the name of the experiment and is located in
        {self.model_pardir}/models

        Args:
            new_exp_name: optionally save the model in a folder with this name to branch the model.
             The Trainer object won't modify self.exp_name
        """
        self.model.eval()
        paths = self._paths(new_exp_name)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        # Write instead of append to be but safe from bugs
        for json_file_name in ['train_log', 'val_log', 'saved_test_metrics', 'settings']:
            with open(paths[json_file_name], 'w') as json_file:
                file_content = self.__getattribute__(json_file_name)
                if file_content:
                    json.dump(file_content, json_file, default=str, indent=4)
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
                        past_epochs.append(int(''.join(filter(str.isdigit, file))))  # available epochs
            if not past_epochs:
                warnings.warn('No saved models found. Training from scratch.', UserWarning)
                return
            self.epoch = max(past_epochs)
        paths = self._paths()
        self.model.load_state_dict(torch.load(paths['model'], map_location=torch.device(self.device)))
        try:
            self.optimizer.load_state_dict(torch.load(paths['optim'], map_location=torch.device(self.device)))
        except ValueError as err:
            warnings.warn('Optimizer has not been correctly loaded:')
            print(err)
        for json_file_name in ['train_log', 'val_log']:
            with open(paths[json_file_name], 'r') as log_file:
                file_str = log_file.read()
                json_file = json.loads(file_str) if file_str else {}
            # converts keys back to int and filter out future epochs from the logs
            json_file = {int_key: value for key, value in json_file.items() if (int_key := int(key)) <= self.epoch}
            self.__setattr__(json_file_name, defaultdict(dict, json_file))
        print('Loaded: ', paths['model'])
        return

    def _paths(self, new_exp_name: str = None) -> dict[str, str]:
        """
        It gets the paths for saving and loading the experiment.

        Args:
            new_exp_name: the alternative name for the folder (see self.save)
        """
        if not os.path.exists(self.model_pardir):
            os.mkdir(self.model_pardir)
        directory = os.path.join(self.model_pardir, new_exp_name or self.exp_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths: dict[str, str] = {}
        # paths for settings, results and learning curves
        for json_file in ['settings', 'train_log', 'val_log', 'saved_test_metrics']:
            paths[json_file] = os.path.join(directory, f'{json_file}.json')
        # paths for the model and the optimizer checkpoints
        for pt_file in ['model', 'optim']:
            paths[pt_file] = os.path.join(directory, f'{pt_file}_epoch{self.epoch}.pt')
        return paths

    def hook_before_training_epoch(self) -> None:
        """
        This hook is called before running the training session.
        """
        ...

    def hook_after_training_epoch(self) -> None:
        """
        This hook is called after running the training session.
        """
        ...

    def __repr__(self) -> str:
        return 'Trainer for experiment: ' + self.exp_name

    @staticmethod
    def recursive_to(obj: C[torch.Tensor], device: Literal['detach_cpu'] | torch.device) -> C[torch.Tensor]:
        """
        It changes device recursively to tensors inside a container

        Args:
            obj: a container (a combination of dict and list) of Tensors
            device: the target device. Alternatively detach_cpu for including detaching
        """

        def to_device(x: Tensor) -> Tensor:
            return x.detach().cpu() if device == 'detach_cpu' else x.to(device)

        return apply(obj, expected_type=Tensor, func=to_device)
