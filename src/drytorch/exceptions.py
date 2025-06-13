"""Module containing internal exceptions."""

import pathlib
import traceback
from typing import Any, Optional

import torch


class DryTorchException(Exception):
    """Exception from the drytorch package."""
    msg: str = ''

    def __init__(self, *args: Any) -> None:
        super().__init__(self.msg.format(*args))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls.__name__ = '[drytorch] ' + cls.__name__
        super().__init_subclass__(**kwargs)
        return


class DryTorchWarning(UserWarning):
    """Warning from the drytorch package."""
    msg: str = ''

    def __init__(self, *args: Any) -> None:
        super().__init__(self.msg.format(*args))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls.__name__ = '[drytorch] ' + cls.__name__
        super().__init_subclass__(**kwargs)
        return


class TrackerException(DryTorchException):
    msg = '[{}] {}'

    def __init__(self, tracker: Any, tracker_msg: str) -> None:
        self.tracker = tracker
        super().__init__(tracker.__class__.__name__, tracker_msg)


class AccessOutsideScopeError(DryTorchException):
    msg = 'Operation only allowed within an experiment scope.'


class ActiveExperimentNotRegistered(DryTorchException):
    msg = 'Experiment {} is not registered as sub experiment of class {}.'

    def __init__(self, sub_exp_cls: type, main_exp_cls: type) -> None:
        self.sub_exp_cls = sub_exp_cls
        self.main_exp_cls = main_exp_cls
        super().__init__(sub_exp_cls.__name__, main_exp_cls.__name__)


class CheckpointNotInitializedError(DryTorchException):
    msg = 'The checkpoint did not register any model.'


class ConvergenceError(DryTorchException):
    msg = 'The module did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        self.criterion = criterion
        super().__init__(criterion)


class ExperimentIsNotActiveError(DryTorchException):
    msg = 'Experiment {} has no active instances.'

    def __init__(self, experiment_cls: type) -> None:
        self.experiment_cls = experiment_cls
        super().__init__(experiment_cls.__name__)


class FuncNotApplicableError(DryTorchException):
    msg = 'Cannot apply function {} on type {}.'

    def __init__(self, func_name: str, type_name: str) -> None:
        self.func_name = func_name
        self.type_name = type_name
        super().__init__(func_name, type_name)


class LossNotScalarError(DryTorchException):
    msg = 'Loss must be a scalar but got Tensor of shape {}.'

    def __init__(self, size: torch.Size) -> None:
        self.size = size
        super().__init__(size)


class MetricNotFoundError(DryTorchException):
    msg = 'No metric {}found in {}.'

    def __init__(self, source_name: str, metric_name: str) -> None:
        self.source_name = source_name
        self.metric_name = metric_name + ' ' if metric_name else ''
        super().__init__(self.metric_name, source_name)


class MissingParamError(DryTorchException):
    msg = 'Parameter groups in input learning rate miss parameters {}.'

    def __init__(self,
                 module_names: list[str],
                 lr_param_groups: list[str]) -> None:
        self.module_names = module_names
        self.lr_param_groups = lr_param_groups
        self.missing = set(module_names) - set(lr_param_groups)
        super().__init__(self.missing)


class ModelNotRegisteredError(DryTorchException):
    msg = 'Model {} has not been registered in experiment {}.'

    def __init__(self, model_name: str, exp_name: str) -> None:
        self.model_name = model_name
        super().__init__(str(model_name), exp_name)


class ModelNotFoundError(DryTorchException):
    msg = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        self.checkpoint_directory = checkpoint_directory
        super().__init__(checkpoint_directory)


class ModuleAlreadyRegisteredError(DryTorchException):
    msg = 'Module has already been registered from model {} in experiment {}.'

    def __init__(self, model_name: str, exp_name: str) -> None:
        self.model_name = model_name
        super().__init__(str(model_name), exp_name)


class NameAlreadyRegisteredError(DryTorchException):
    msg = 'Name {} has already been registered in the current experiment.'

    def __init__(self, name: str) -> None:
        super().__init__(name)


class NamedTupleOnlyError(DryTorchException):
    msg = 'The only accepted subtypes of tuple are namedtuple classes. Got {}.'

    def __init__(self, tuple_type: str) -> None:
        self.tuple_type = tuple_type
        super().__init__(tuple_type)


class NestedScopeError(DryTorchException):
    msg = 'Cannot start Experiment {} within Experiment {} scope.'

    def __init__(self, current_exp_name: str, new_exp_name: str) -> None:
        self.current_exp_name = current_exp_name
        self.new_exp_name = new_exp_name
        super().__init__(current_exp_name, new_exp_name)


class NoActiveExperimentError(DryTorchException):
    msg = 'No experiment {}has been started.'

    def __init__(self, experiment_class: Optional[type] = None) -> None:
        self.experiment_class = experiment_class
        if experiment_class is not None:
            specify_string = f'of class {experiment_class.__class__.__name__} '
        else:
            specify_string = ''
        super().__init__(specify_string)


class NoConfigurationError(DryTorchException):
    msg = 'No configuration available for the experiment.'


class DatasetHasNoLengthError(DryTorchException):
    msg = 'Dataset does not implement __len__ method.'


class ResultNotAvailableError(DryTorchException):
    msg = 'The result will be available only after the hook has been called.'


class SubExperimentNotRegisteredError(DryTorchException):
    msg = 'SubExperiment {} has not been registered to any MainExperiment.'

    def __init__(self, experiment_cls: type) -> None:
        self.experiment_cls = experiment_cls
        super().__init__(experiment_cls.__name__)


class TrackerAlreadyRegisteredError(DryTorchException):
    msg = 'Tracker {} already registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        self.tracker_name = tracker_name
        super().__init__(tracker_name, exp_name)


class TrackerNotRegisteredError(DryTorchException):
    msg = 'Tracker {} not registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        self.tracker_name = tracker_name
        super().__init__(tracker_name, exp_name)


class CannotStoreOutputWarning(DryTorchWarning):
    msg = 'Impossible to store output because the following error.\n{}'

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__(str(error))


class ComputedBeforeUpdatedWarning(DryTorchWarning):
    msg = 'The ``compute`` method of {} was called before its updating.'

    def __init__(self, calculator: Any) -> None:
        self.calculator = calculator
        super().__init__(calculator.__class__.__name__)


class FailedOptionalImportWarning(DryTorchWarning):
    msg = 'Failed to import optional dependency {}. Install for better support.'

    def __init__(self, package_name: str, error: BaseException) -> None:
        self.error = error
        self.package_name = package_name
        super().__init__(self.msg.format(package_name))


class OptimizerNotLoadedWarning(DryTorchWarning):
    msg = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__(error)


class PastEpochWarning(DryTorchWarning):
    msg = 'Training until epoch {} stopped: current epoch is already {}.'

    def __init__(self, selected_epoch: int, current_epoch: int) -> None:
        self.selected_epoch = selected_epoch
        self.current_epoch = current_epoch
        super().__init__(selected_epoch, current_epoch)


class RecursionWarning(DryTorchWarning):
    msg = 'Impossible to extract metadata because there are recursive objects.'


class TerminatedTrainingWarning(DryTorchWarning):
    msg = 'Attempted to train module after termination.'


class TrackerError(DryTorchWarning):
    msg = 'Tracker {} encountered the following error and was skipped:\n{}'

    def __init__(self, subscriber_name: str, error: BaseException) -> None:
        self.subscriber_name = subscriber_name
        self.error = error
        formatted_traceback = traceback.format_exc()
        super().__init__(subscriber_name, formatted_traceback)
