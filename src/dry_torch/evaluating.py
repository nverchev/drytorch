import sys
import abc
import warnings
from typing import TypeVar, Generic

from typing_extensions import override

import torch

from src.dry_torch import descriptors
from src.dry_torch import exceptions
from src.dry_torch import events
from src.dry_torch import aggregators
from src.dry_torch import apply_ops
from src.dry_torch import protocols as p
from src.dry_torch import registering

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


class Evaluation(p.EvaluationProtocol,
                 Generic[_Input, _Target, _Output],
                 metaclass=abc.ABCMeta):
    max_stored_output: int = sys.maxsize
    partition: descriptors.Split
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracker: contain the module and the optimizing strategy.
        loaders: dictionary with loaders for the training, and optionally,
         the validation and test datasets.
        loss_calc: the _loss_calc function, which needs to return batched values
         as in LossAndMetricsProtocol.
        metrics_calc: the test metrics function, returning TestMetricsProtocol.
         If None, _loss_calc will be used instead.
        mixed_precision: whether to use mixed precision computing.
         Optional, default to False.

    Attributes:
        max_stored_output:
        the maximum number of outputs to store when testing.
        update_frequency:
        number of times the progress bar updates in one epoch.
        outputs_list:
            An instance of TorchDictList that stores the last test evaluation.
        _store_outputs: if the flag is active store the module outputs in the
            outputs_list attribute. Default to False.

    Methods:
        train:
        run the training session,
        optionally quickly evaluate on the validation dataset.
        test: evaluate on the specified partition of the dataset.
        hook_before_training_epoch:
        property for adding a hook before running the training session.
        hook_after_training_epoch:
        property for adding a hook after running the training session.
    """

    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            mixed_precision: bool = False,
            name: str = '',
    ) -> None:
        self.model = model
        self.name = name
        self._loader = loader
        self._calculator = metrics_calc
        device_is_cuda = self.model.device.type == 'cuda'
        self._mixed_precision = mixed_precision and device_is_cuda
        self._metrics = aggregators.TorchAverager()
        self.outputs_list = list[_Output]()
        self._last_metrics: dict[str, float] = {}
        return

    @property
    def metrics(self) -> dict[str, float]:
        if not self._last_metrics:
            self._last_metrics = self._metrics.reduce_all()
        return self._last_metrics

    def clear_metrics(self):
        self._metrics.clear()
        self._last_metrics = {}

    @abc.abstractmethod
    def __call__(self, store_outputs: bool = False) -> None:
        ...

    def log_metrics(self) -> None:
        events.MetricsCreation(model_name=self.model.name,
                               source=self.name,
                               epoch=self.model.epoch,
                               metrics=self.metrics)
        return

    def _run_epoch(self, store_outputs: bool):
        self.outputs_list.clear()
        self.clear_metrics()
        pbar = events.EpochBar(self.name, self._loader)
        for batch in self._loader:
            inputs, targets = apply_ops.apply_to(batch, self.model.device)
            outputs = self._run_forward(inputs)
            self._calculator.calculate(outputs, targets)
            self._run_backwards()
            self._metrics += self._calculator.metrics
            self._calculator.reset_calculated()
            pbar.update_pbar(self._metrics.reduce_all())
            if store_outputs:
                self._store(outputs)

        self.log_metrics()

    def _run_forward(self, inputs: _Input) -> _Output:
        with torch.autocast(device_type=self.model.device.type,
                            enabled=self._mixed_precision):
            return self.model(inputs)

    def _run_backwards(self) -> None:
        pass

    def _store(self, outputs: _Output) -> None:
        try:
            outputs = apply_ops.apply_cpu_detach(outputs)
        except (exceptions.FuncNotApplicableError,
                exceptions.NamedTupleOnlyError) as err:
            warnings.warn(exceptions.CannotStoreOutputWarning(str(err)))
        else:
            self.outputs_list.append(outputs)

    def __str__(self) -> str:
        return f'Base Evaluator for {self.model.name}.'


class Diagnostic(Evaluation[_Input, _Target, _Output]):
    partition = descriptors.Split.TRAIN

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        self.model.module.eval()
        self._run_epoch(store_outputs)
        return


class Validation(Evaluation[_Input, _Target, _Output]):
    partition = descriptors.Split.VAL

    @registering.register_kwargs
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            mixed_precision: bool = False,
            name: str = '',
    ) -> None:
        super().__init__(model,
                         name=name,
                         loader=loader,
                         metrics_calc=metrics_calc,
                         mixed_precision=mixed_precision)
        return

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        self.model.module.eval()
        self._run_epoch(store_outputs)
        return

    def __str__(self) -> str:
        return f'Validator for {self.model.name}.'


class Test(Evaluation[_Input, _Target, _Output]):
    partition = descriptors.Split.TEST

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracker: contain the module and the optimizing strategy.
        loaders: dictionary with loaders for the training, and optionally,
         the validation and test datasets.
        loss_calc: the _loss_calc function, which needs to return batched values
         as in LossAndMetricsProtocol.
        metrics_calc: the test metrics function, returning TestMetricsProtocol.
         If None, _loss_calc will be used instead.
        mixed_precision: whether to use mixed precision computing.
         Optional, default to False.

    Attributes:
        max_stored_output:
        the maximum number of outputs to store when testing.
        update_frequency:
        number of times the progress bar updates in one epoch.
        outputs_list:
            An instance of TorchDictList that stores the last test evaluation.
        _store_outputs: if the flag is active store the module outputs in the
            outputs_list attribute. Default to False.

    Methods:
        train:
        run the training session,
        optionally quickly evaluate on the validation dataset.
        test: evaluate on the specified partition of the dataset.
        hook_before_training_epoch:
        property for adding a hook before running the training session.
        hook_after_training_epoch:
        property for adding a hook after running the training session.
    """

    @registering.register_kwargs
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            name: str = '',
    ) -> None:
        super().__init__(model,
                         name=name,
                         loader=loader,
                         metrics_calc=metrics_calc)
        # self._checkpoint = io.LogIO(model.name)
        return

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        events.StartTest(self.model.name, self.name)
        self.model.module.eval()
        self._run_epoch(store_outputs)
        # self._checkpoint.save()
        return

    # def __str__(self) -> str:
    #     return (
    #         f'{repr(self.model_tracker.default_names[self.__class__.__name__])}'
    #         'for tracker {self.model.name}.'
    #     )
