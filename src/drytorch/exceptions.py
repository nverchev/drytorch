"""Module containing internal exceptions for the drytorch package."""

import pathlib
import traceback

from typing import Any

import torch


class DryTorchError(Exception):
    """Base exception class for all drytorch package exceptions."""

    msg: str = ''

    def __init__(self, *args: Any) -> None:
        """Constructor.

        Args:
            *args: arguments to be formatted into the message template.
        """
        super().__init__(self.msg.format(*args))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically prefix subclass names with [drytorch]."""
        cls.__name__ = '[drytorch] ' + cls.__name__
        super().__init_subclass__(**kwargs)
        return


class DryTorchWarning(UserWarning):
    """Base warning class for all drytorch package warnings."""

    msg: str = ''

    def __init__(self, *args: Any) -> None:
        """Constructor.

        Args:
            *args: arguments to be formatted into the message template.
        """
        super().__init__(self.msg.format(*args))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically prefix subclass names with [drytorch]."""
        cls.__name__ = '[drytorch] ' + cls.__name__
        super().__init_subclass__(**kwargs)
        return


class TrackerError(DryTorchError):
    """Exception raised by tracker objects during experiment tracking."""

    msg = '[{}] {}'

    def __init__(self, tracker: Any, tracker_msg: str) -> None:
        """Constructor.

        Args:
            tracker: the tracker object that encountered the error.
            tracker_msg: the error message from the tracker.
        """
        self.tracker = tracker
        super().__init__(tracker.__class__.__name__, tracker_msg)


class AccessOutsideScopeError(DryTorchError):
    """Raised when an operation is attempted outside an experiment scope."""

    msg = 'Operation only allowed within an experiment scope.'


class CheckpointNotInitializedError(DryTorchError):
    """Raised when attempting to use a checkpoint without a registered model."""

    msg = 'The checkpoint did not register any model.'


class ConvergenceError(DryTorchError):
    """Raised when a module fails to converge during training."""

    msg = 'The module did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        """Constructor.

        Args:
            criterion: the convergence criterion that was not met.
        """
        self.criterion = criterion
        super().__init__(criterion)


class FuncNotApplicableError(DryTorchError):
    """Raised when a function cannot be applied to a specific type."""

    msg = 'Cannot apply function {} on type {}.'

    def __init__(self, func_name: str, type_name: str) -> None:
        """Constructor.

        Args:
            func_name: the name of the function that cannot be applied.
            type_name: the name of the type that doesn't support the function.
        """
        self.func_name = func_name
        self.type_name = type_name
        super().__init__(func_name, type_name)


class InvalidBatchError(DryTorchError):
    """Raised when the batch size of a loader is invalid."""

    msg = 'Batch size must be a positive integer. Got {}.'

    def __init__(self, batch_size: int | None) -> None:
        """Constructor.

        Args:
            batch_size: the requested number of element in the mini-batch.
        """
        self.batch_size = batch_size
        super().__init__(batch_size)


class LossNotScalarError(DryTorchError):
    """Raised when a loss value is not a scalar tensor."""

    msg = 'Loss must be a scalar but got Tensor of shape {}.'

    def __init__(self, size: torch.Size) -> None:
        """Constructor.

        Args:
            size: the actual size of the non-scalar loss tensor.
        """
        self.size = size
        super().__init__(size)


class MetricNotFoundError(DryTorchError):
    """Raised when a requested metric is not found in the specified source."""

    msg = 'No metric {}found in {}.'

    def __init__(self, source_name: str, metric_name: str) -> None:
        """Constructor.

        Args:
            source_name: the name of the source where the metric was not found.
            metric_name: the name of the metric that was not found.
        """
        self.source_name = source_name
        self.metric_name = metric_name + ' ' if metric_name else ''
        super().__init__(self.metric_name, source_name)


class MissingParamError(DryTorchError):
    """Raised when parameter groups are missing required parameters."""

    msg = 'Parameter groups in input learning rate miss parameters {}.'

    def __init__(
        self, module_names: list[str], lr_param_groups: list[str]
    ) -> None:
        """Constructor.

        Args:
            module_names: list of module names that should have parameters.
            lr_param_groups: group names in the parameter learning rate config.
        """
        self.module_names = module_names
        self.lr_param_groups = lr_param_groups
        self.missing = set(module_names) - set(lr_param_groups)
        super().__init__(self.missing)


class ModelNotRegisteredError(DryTorchError):
    """Raised when trying to access a model that hasn't been registered."""

    msg = 'Model {} has not been registered in experiment {}.'

    def __init__(self, model_name: str, exp_name: str) -> None:
        """Constructor.

        Args:
            model_name: the name of the model that was not registered.
            exp_name: current experiment.
        """
        self.model_name = model_name
        super().__init__(str(model_name), exp_name)


class ModelNotFoundError(DryTorchError):
    """Raised when no saved model is found in the checkpoint directory."""

    msg = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        """Constructor.

        Args:
            checkpoint_directory: the directory path where no model was found.
        """
        self.checkpoint_directory = checkpoint_directory
        super().__init__(checkpoint_directory)


class ModuleAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register an already registered module."""

    msg = 'Module has already been registered from model {} in experiment {}.'

    def __init__(self, model_name: str, exp_name: str) -> None:
        """Constructor.

        Args:
            model_name: the name of the model that is already registered.
            exp_name: the name of the experiment where the model is registered.
        """
        self.model_name = model_name
        super().__init__(str(model_name), exp_name)


class NameAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register a name already in use."""

    msg = 'Name {} has already been registered in the current experiment.'

    def __init__(self, name: str) -> None:
        """Constructor.

        Args:
            name: the name that is already registered.
        """
        super().__init__(name)


class NamedTupleOnlyError(DryTorchError):
    """Raised when operations require a named tuple and not a subclass."""

    msg = 'The only accepted subtypes of tuple are namedtuple classes. Got {}.'

    def __init__(self, tuple_type: str) -> None:
        """Constructor.

        Args:
            tuple_type: the actual type of the tuple that was provided.
        """
        self.tuple_type = tuple_type
        super().__init__(tuple_type)


class NestedScopeError(DryTorchError):
    """Raised when attempting to nest an experiment scope within another one."""

    msg = 'Cannot start Experiment {} within Experiment {} scope.'

    def __init__(self, current_exp_name: str, new_exp_name: str) -> None:
        """Constructor.

        Args:
            current_exp_name: the name of the currently active experiment.
            new_exp_name: the name of the experiment that cannot be started.
        """
        self.current_exp_name = current_exp_name
        self.new_exp_name = new_exp_name
        super().__init__(current_exp_name, new_exp_name)


class NoActiveExperimentError(DryTorchError):
    """Raised when no experiment is currently active."""

    msg = 'No experiment {}has been started.'

    def __init__(self, experiment_class: type | None = None) -> None:
        """Constructor.

        Args:
            experiment_class: specifies experiment class.
        """
        self.experiment_class = experiment_class
        if experiment_class is not None:
            specify_string = f'of class {experiment_class.__class__.__name__} '
        else:
            specify_string = ''
        super().__init__(specify_string)


class NoSpecificationError(DryTorchError):
    """Raised when no configuration is available for the experiment."""

    msg = 'No specification available for the experiment.'


class DatasetHasNoLengthError(DryTorchError):
    """Raised when a dataset does not implement the __len__ method."""

    msg = 'Dataset does not implement __len__ method.'


class ResultNotAvailableError(DryTorchError):
    """Raised when trying to access a result before the hook has been called."""

    msg = 'The result will be available only after the hook has been called.'


class SubExperimentNotRegisteredError(DryTorchError):
    """Raised when a sub-experiment has not been registered."""

    msg = 'SubExperiment {} has not been registered to any MainExperiment.'

    def __init__(self, experiment_cls: type) -> None:
        """Constructor.

        Args:
            experiment_cls: the sub-experiment class that is not registered.
        """
        self.experiment_cls = experiment_cls
        super().__init__(experiment_cls.__name__)


class TrackerAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register an already registered tracker."""

    msg = 'Tracker {} already registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        """Constructor.

        Args:
            tracker_name: the name of the tracker that is already registered.
            exp_name: the name of the experiment where to register the tracker.
        """
        self.tracker_name = tracker_name
        super().__init__(tracker_name, exp_name)


class TrackerNotRegisteredError(DryTorchError):
    """Raised when trying to access a tracker that is not registered."""

    msg = 'Tracker {} not registered in any active experiment.'

    def __init__(self, tracker_name: str) -> None:
        """Constructor.

        Args:
            tracker_name: the name of the tracker that is not registered.
        """
        self.tracker_name = tracker_name
        super().__init__(tracker_name)


class CannotStoreOutputWarning(DryTorchWarning):
    """Warning raised when output cannot be stored due to an error."""

    msg = 'Impossible to store output because the following error.\n{}'

    def __init__(self, error: BaseException) -> None:
        """Constructor.

        Args:
            error: the error that prevented output storage.
        """
        self.error = error
        super().__init__(str(error))


class ComputedBeforeUpdatedWarning(DryTorchWarning):
    """Warning raised when compute method is called before updating."""

    msg = 'The ``compute`` method of {} was called before its updating.'

    def __init__(self, calculator: Any) -> None:
        """Constructor.

        Args:
            calculator: the calculator object that was computed before updating.
        """
        self.calculator = calculator
        super().__init__(calculator.__class__.__name__)


class FailedOptionalImportWarning(DryTorchWarning):
    """Warning raised when an optional dependency fails to import."""

    msg = 'Failed to import optional dependency {}. Install for better support.'

    def __init__(self, package_name: str) -> None:
        """Constructor.

        Args:
            package_name: the name of the package that failed to import.
        """
        self.package_name = package_name
        super().__init__(self.msg.format(package_name))


class OptimizerNotLoadedWarning(DryTorchWarning):
    """Warning raised when the optimizer has not been correctly loaded."""

    msg = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        """Constructor.

        Args:
            error: the error that occurred while loading the optimizer.
        """
        self.error = error
        super().__init__(error)


class PastEpochWarning(DryTorchWarning):
    """Warning raised when training is requested for a past epoch."""

    msg = 'Training until epoch {} stopped: current epoch is already {}.'

    def __init__(self, selected_epoch: int, current_epoch: int) -> None:
        """Constructor.

        Args:
            selected_epoch: the epoch that training was requested until.
            current_epoch: the current epoch number.
        """
        self.selected_epoch = selected_epoch
        self.current_epoch = current_epoch
        super().__init__(selected_epoch, current_epoch)


class RecursionWarning(DryTorchWarning):
    """Warning raised when recursive objects obstruct metadata extraction."""

    msg = 'Impossible to extract metadata because there are recursive objects.'


class TerminatedTrainingWarning(DryTorchWarning):
    """Warning raised when training is attempted after termination."""

    msg = 'Attempted to train module after termination.'


class TrackerExceptionWarning(DryTorchWarning):
    """Warning raised when a tracker encounters an error and is skipped."""

    msg = 'Tracker {} encountered the following error and was skipped:\n{}'

    def __init__(self, subscriber_name: str, error: BaseException) -> None:
        """Constructor.

        Args:
            subscriber_name: the name of the tracker that encountered the error.
            error: the error that occurred in the tracker.
        """
        self.subscriber_name = subscriber_name
        self.error = error
        formatted_traceback = traceback.format_exc()
        super().__init__(subscriber_name, formatted_traceback)
