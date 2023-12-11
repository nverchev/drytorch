import json
import os
import sys
import warnings
from collections import defaultdict
from typing import Any, Literal, Callable, Iterable

import pandas as pd
import torch
from custom_trainer.schedulers import Scheduler, ConstantScheduler
from custom_trainer.utils import DictList, UsuallyFalse, apply, dict_repr, C
from torch import autocast, Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, StackDataset, TensorDataset
from tqdm.auto import tqdm

try:
    import visdom
except ImportError:
    print("Plotting on visdom is disabled")
    visdom = None
try:
    import plotly.express as px
except ImportError:
    print("Plotting on plotly is disabled")
    px = None


class Trainer:
    """
    This class manages training and general utilities for a Pytorch model.

    The motivation for using this class is the following:
        I) Inheritance allows user defined class for complex models and / or training by only adding minimal code
            1) Compartmentalization - the methods can be easily extended and overriden
            2) Flexible containers - the trainer uses dictionaries that can handle complex inputs / outputs
            3) Hooks that grant further possibilities of customization
        II) Already implemented complex functionalities during training:
            1) Scheduler for the learning rate decay and different learning rates for different parameters' groups
            2) Mixed precision training using torch.cuda.amp
            3) Visualization of the learning curves using the visdom library
        III) Utilities for logging, metrics and investigation of the model's outputs
            1) Simplified pipeline for logging, saving and loading a model
            2) The class attempts full model documentation
            3) DictList allows indexing of complex outputs for output exploration

    """
    quiet = UsuallyFalse()  # less output
    max_stored_output: int = float('inf')  # maximum amount of stored evaluated test samples
    max_length_config: int = 20
    tqdm_update_frequency = 10
    vis = None

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

    def __init__(self, model: torch.nn.Module, *, exp_name: str, model_pardir: str = 'models', device: torch.device,
                 batch_size: int, train_dataset: Dataset, val_dataset: Dataset = None, test_dataset: Dataset = None,
                 optimizer_cls: type, optim_args: dict, scheduler: Scheduler = ConstantScheduler(),
                 loss: Callable, amp: bool = False, **extra_config: dict) -> None:
        """
        Args:
            model: Pytorch model with a settings attribute which is a dictionary for extra details
            exp_name: name used for the folder containing the checkpoints and the metadata
            device: only gpu and cpu are supported
            train_dataset: dataset for training used in the train method
            val_dataset: dataset for validation used both in the train method and in the test method
            test_dataset: dataset for testing only used in the test method
            optimizer_cls: a Pytorch optimizer class
            optim_args: arguments for the optimizer (see the optimizer_settings setter for more details)
            model_pardir: parent directory for the folders with the model checkpoints
            amp: mixed precision computing from torch.cuda.amp (works only with cuda)
                        scheduler: instance of a Scheduler class that modifies the learning rate depending on the epoch
            loss: this function may be a callable torch.nn.Module object (with reduction='none') or any function
                    that returns a tensor of shape (batch_size, ) with the loss evaluation of individual samples.
                    By default, its arguments are the model outputs and the targets. For more complex losses, and to
                    compute metrics during training, read the documentation on the loss method and override it
            extra_config: extraneous settings for logging (see the __new__ method)

            Important!
            Indexing the dataset must return a tuple (inputs, targets) where:
            inputs: inputs for the model. Either a tensor, or a structured container of tensors that uses list and dict
            targets: arguments only used by the loss function. It can be structured as well or can be None
            The handling of the inputs and outputs can be specified overriding the helper_inputs and loss methods

        """
        self.device = device
        self.model = model.to(device)
        self.exp_name = exp_name
        self.model_pardir = model_pardir

        self.train_loader = self.get_loader(train_dataset, batch_size, partition='train')
        self.val_loader = self.get_loader(val_dataset, batch_size, partition='val')
        self.test_loader = self.get_loader(test_dataset, batch_size, partition='test')

        self.epoch = 0
        self.scheduler = scheduler
        self.optimizer_settings = optim_args.copy()  # property that updates the learning rate according to the epoch
        self.optimizer = optimizer_cls(**self.optimizer_settings)
        self.scaler = GradScaler(enabled=amp and self.device.type == 'cuda')
        self.amp = amp
        self.loss = loss

        self.train_log = pd.DataFrame()
        self.val_log = pd.DataFrame()
        self.saved_test_metrics = pd.DataFrame()  # saves metrics of last evaluation
        self.test_indices: list[int]
        self.test_metadata: dict[str, str]
        self.test_outputs: DictList[torch.Tensor]  # store last test evaluation

        _not_used = extra_config
        return

    @staticmethod
    def get_loader(dataset: Dataset, batch_size: int, partition: Literal['train', 'val', 'test'] = 'train', **_):
        if dataset is None:
            return None
        # noinspection PyTypeChecker
        len_dataset: int = len(dataset)  # Dataset object must have a __len__ method
        dataset = StackDataset(dataset, TensorDataset(torch.arange(len_dataset)))  # adds indices
        pin_memory = torch.cuda.is_available()
        drop_last = True if partition == 'train' else False
        return DataLoader(dataset, drop_last=drop_last, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

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
        if not self.quiet:
            print('\rTraining {}'.format(self.exp_name))  # \r corrects asynchronous stderr/stdout printouts
        for _ in range(num_epoch):
            self.update_learning_rate(self.optimizer_settings['params'])
            self.epoch += 1
            print('\r====> Epoch:{:4d}'.format(self.epoch), end=' ...  :' if self.quiet else '\n')
            self.model.train()
            self.hook_before_training_epoch()
            self._run_session(partition='train')
            self.hook_after_training_epoch()
            if not self.quiet:
                print()
            if val_after_train:  # check losses on val
                self.model.eval()
                with torch.inference_mode():
                    self._run_session(partition='val', use_test_metrics=False)
                if not self.quiet:
                    print()
        if self.quiet:
            print()
        return

    @torch.inference_mode()
    def test(self, partition: Literal['train', 'val', 'test'] = 'val', save_outputs: bool = False) -> None:
        """
        It tests the current model in inference. It saves the metrics in saved_test_metrics

        Args:
            partition: the dataset used for testing
            save_outputs: if True, it stores the outputs of the model
        """
        self.model.eval()
        self._run_session(partition=partition, save_outputs=save_outputs, use_test_metrics=True)
        print()
        return

    def _run_session(self, partition: Literal['train', 'val', 'test'],
                     save_outputs: bool = False,
                     use_test_metrics: bool = False) -> None:
        """
        It implements batching, backpropagation, printing and storage of the outputs

        Args:
            partition: the dataset used for the session
            save_outputs: if True, it stores the outputs of the model
            use_test_metrics: whether to use possibly computationally expensive test_metrics
        """
        match partition:
            case 'train':
                loader = self.train_loader
                log = self.train_log
            case 'val':
                loader = self.val_loader
                log = self.val_log
            case 'test':
                loader = self.test_loader
                log = self.saved_test_metrics
            case _:
                raise ValueError('partition should be "train", "val" or "test"')

        if loader is None:
            warnings.warn(f' Impossible to run session on the {partition}_dataset: dataset not found.')
            return

        if save_outputs:
            self.test_indices: list[int] = []
            self.test_outputs: DictList[torch.Tensor] = DictList()
            self.test_metadata: dict[str, str] = dict(partition=partition)

        epoch_log: defaultdict[str, float] = defaultdict(float)
        num_batch = len(loader)
        with tqdm(enumerate(loader), total=num_batch, disable=bool(self.quiet), file=sys.stderr) as tqdm_loader:
            epoch_seen = 0
            for batch_idx, ((inputs, targets), (indices,)) in tqdm_loader:
                epoch_seen += indices.shape[0]
                inputs, targets = self.recursive_to([inputs, targets], self.device)
                iter_inputs, named_inputs = self.helper_inputs(inputs)
                with autocast(device_type=self.device.type, enabled=self.amp):
                    outputs = self.model(*iter_inputs, **named_inputs)
                    if use_test_metrics:
                        batch_log = self.test_metrics(outputs, inputs, targets)
                    else:
                        batch_log = self.loss_and_metrics(outputs, inputs, targets)
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
        print(f'\rAverage {partition} {"metric(s)" if torch.is_inference_mode_enabled() else "loss(es)"}:', end=' ')
        for loss_or_metric, value in epoch_log.items():
            value /= epoch_seen
            if torch.is_inference_mode_enabled() ^ (partition == 'train'):  # Evaluating train does not overwrite log
                log.loc[self.epoch, loss_or_metric] = value
            print('{}: {:.4e}'.format(loss_or_metric, value), end='\t')

        return

    def loss_and_metrics(self, outputs: C, inputs: C, targets: C, **_) -> dict[str, Tensor]:
        """
        It must return a dictionary "dict" with dict['Criterion'] = loss to backprop and other optional metrics.

        Args:
            outputs: the outputs of the model
            inputs: inputs of the model (in self learning the inputs replace the targets)
            targets: the targets of the model

        Returns:
            loss: dictionary whose values must be tensors of shape (batch_size, 1)

        """
        _not_used = inputs
        return {'Criterion': self.loss(outputs, targets)}

    def test_metrics(self, outputs: C, inputs: C, targets: C, **_) -> dict[str, Tensor]:
        """
        This method works similarly to the loss_and_metrics method and defaults to it.
        Override for complex metrics during testing.

        Args:
            outputs: the outputs of the model
            inputs: inputs of the model (in self learning the inputs replace the targets)
            targets: the targets of the model

        Returns:
            metrics: dictionary whose values must be tensors of shape (batch_size, 1)

        """
        return self.loss_and_metrics(outputs, inputs, targets)

    def helper_inputs(self, inputs: C) -> tuple[Iterable, dict[str, Any]]:
        """
        This method is a hook that rearranges the inputs following the model's named arguments.

        Args:
            inputs: inputs of the model

        Returns:
            dictionary of the form {named_argument: input}
        """
        _not_used = self
        return (inputs,), {}

    @classmethod
    def set_up_visdom_connection(cls, env) -> bool:
        """
        It initializes a visdom environment with the name of the experiment (if it does not exist already)

        Returns:
            True visdom is available, False otherwise
        """
        if visdom is None:
            return False
        if cls.vis is None:
            print('visdom: ', end='', file=sys.stderr)
            cls.vis = visdom.Visdom(env=env)
        return True

    @classmethod
    def check_visdom_connection(cls, env) -> bool:
        """
        It activates and check the connection to the visdom server https://github.com/fossasia/visdom

        Returns:
            True if there is an active connection, False otherwise
        """
        return cls.set_up_visdom_connection(env) and cls.vis.check_connection()

    def plot_learning_curves(self, loss_or_metric: str = 'Criterion', start: int = 0,
                             title: str = 'Learning Curves', lib: Literal['visdom', 'plotly', 'auto'] = 'auto') -> None:
        """
        This method plots the learning curves using either plotly or visdom as backends

        Args:
            loss_or_metric: the loss or the metric to visualize
            start: the epoch from where you want to display the curve
            title: the name of the window (and title) of the plot in the visdom interface
            lib: which library to use between visdom and plotly. 'auto' selects plotly if the visdom connection failed.
        """
        plot_args = (self.train_log, self.val_log, loss_or_metric, start, title)
        jupyter = os.path.basename(os.environ['_']) == 'jupyter'  # True when calling from a jupyter notebook
        if lib == 'plotly' or jupyter:
            self.plot_learning_curves_plotly(*plot_args)
        elif lib == 'auto':
            if self.check_visdom_connection(self.exp_name):
                self.plot_learning_curves_visdom(*plot_args)
            else:
                self.plot_learning_curves_plotly(*plot_args)
        elif lib == 'visdom':
            if self.check_visdom_connection(self.exp_name):
                self.plot_learning_curves_visdom(*plot_args)
            else:
                warnings.warn('Impossible to display the learning curves on the server. Check the connection.')
        else:
            raise ValueError(f'Library {lib} not supported.')

    @classmethod
    def plot_learning_curves_visdom(cls, train_log: pd.DataFrame,  val_log: pd.DataFrame,
                                    loss_or_metric: str = 'Criterion', start: int = 0,
                                    title: str = 'Learning Curves') -> None:
        """
        This class method plots the learning curves using visdom as backend. You need first to initialize an environment
        within the class attribute vis (see check_visdom_connection)

        Args:
            train_log: pandas Dataframe with the loss and metrics calculated during training on the training dataset
            val_log: pandas Dataframe with the loss and metrics calculated during training on the validation dataset
            loss_or_metric: the loss or the metric to visualize
            start: the epoch from where you want to display the curve
            title: the name of the window (and title) of the plot in the visdom interface
        """

        train_log = train_log[train_log.index > start][loss_or_metric]
        val_log = val_log[val_log.index > start][loss_or_metric]
        layout = dict(xlabel='Epoch', ylabel=loss_or_metric, title=title, update='replace', showlegend=True)
        cls.vis.line(X=train_log.index, Y=train_log, win=title, opts=layout, name='Training')
        if not val_log.empty:
            cls.vis.line(X=val_log.index, Y=val_log, win=title, opts=layout, update='append', name='Validation')
        return

    @classmethod
    def plot_learning_curves_plotly(cls, train_log: pd.DataFrame,  val_log: pd.DataFrame,
                                    loss_or_metric: str = 'Criterion', start: int = 0,
                                    title: str = 'Learning Curves') -> None:
        """
        This class method plots the learning curves using plotly as backend.

        Args:
            train_log: pandas Dataframe with the loss and metrics calculated during training on the training dataset
            val_log: pandas Dataframe with the loss and metrics calculated during training on the validation dataset
            loss_or_metric: the loss or the metric to visualize
            start: the epoch from where you want to display the curve
            title: the name of the window (and title) of the plot in the visdom interface
        """
        train_log = train_log.copy()
        train_log['Dataset'] = "Training"
        val_log = val_log.copy()
        val_log['Dataset'] = "Validation"
        log = pd.concat([train_log, val_log])
        log = log[log.index >= start].reset_index().rename(columns={'index': 'Epoch'})
        fig = px.line(log, x="Epoch", y=loss_or_metric, color="Dataset", title=title)
        fig.show()
        return

    def save(self, new_exp_name: str = None) -> None:
        """
        It saves a checkpoint for the model and the optimizer with the settings of the experiments, the results,
        and the training and validation learning curves. The folder has the name of the experiment and is located in
        {attribute model_pardir}/models

        Args:
            new_exp_name: optionally save the model in a folder with this name to branch the model.
             The Trainer object won't modify self.exp_name
        """
        self.model.eval()
        paths = self.paths(model_pardir=self.model_pardir, exp_name=new_exp_name or self.exp_name, epoch=self.epoch)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        # Write instead of append to be but safe from bugs
        for df_name in ['train_log', 'val_log', 'saved_test_metrics']:
            self.__getattribute__(df_name).to_csv(paths[df_name])
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
                warnings.warn('No saved models found. Training from scratch.', UserWarning)
                return
            self.epoch = max(past_epochs)
        paths = self.paths(model_pardir=self.model_pardir, exp_name=self.exp_name, epoch=self.epoch)
        self.model.load_state_dict(torch.load(paths['model'], map_location=torch.device(self.device)))
        try:
            self.optimizer.load_state_dict(torch.load(paths['optim'], map_location=torch.device(self.device)))
        except ValueError as err:
            warnings.warn('Optimizer has not been correctly loaded:')
            print(err)
        for df_name in ['train_log', 'val_log']:
            try:
                df = pd.read_csv(paths[df_name], index_col=0)
            except FileNotFoundError:
                df = pd.DataFrame()
            # filter out future epochs from the logs
            df = df[df.index <= self.epoch]
            self.__setattr__(df_name, df)
        print('Loaded: ', paths['model'])
        return

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
        for df_name in ['train_log', 'val_log', 'saved_test_metrics']:
            paths[df_name] = os.path.join(directory, f'{df_name}.csv')
        # paths for the model and the optimizer checkpoints
        for pt_file in ['model', 'optim']:
            paths[pt_file] = os.path.join(directory, f'{pt_file}_epoch{epoch}.pt')
        return paths
