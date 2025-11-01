"""Module containing internal exceptions for the drytorch package."""

import pathlib
import traceback

from typing import Any, Final

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
        cls.__name__: str = '[drytorch] ' + cls.__name__
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
        self.criterion: Final = criterion
        super().__init__(criterion)


class DatasetHasNoLengthError(DryTorchError):
    """Raised when a dataset does not implement the __len__ method."""

    msg = 'Dataset does not implement __len__ method.'


class EpochNotFoundError(DryTorchError):
    """Raised when no saved model is found in the checkpoint directory."""

    msg = 'No checkpoints for epoch {} found in {}.'

    def __init__(self, epoch: int, checkpoint_directory: pathlib.Path) -> None:
        """Constructor.

        Args:
            epoch: the epoch that was not found.
            checkpoint_directory: the directory path where no model was found.
        """
        self.model_directory: Final = checkpoint_directory
        super().__init__(epoch, checkpoint_directory)


class FuncNotApplicableError(DryTorchError):
    """Raised when a function cannot be applied to a specific type."""

    msg = 'Cannot apply function {} on type {}.'

    def __init__(self, func_name: str, type_name: str) -> None:
        """Constructor.

        Args:
            func_name: the name of the function that cannot be applied.
            type_name: the name of the type that doesn't support the function.
        """
        self.func_name: Final = func_name
        self.type_name: Final = type_name
        super().__init__(func_name, type_name)


class LossNotScalarError(DryTorchError):
    """Raised when a loss value is not a scalar tensor."""

    msg = 'Loss must be a scalar but got Tensor of shape {}.'

    def __init__(self, size: torch.Size) -> None:
        """Constructor.

        Args:
            size: the actual size of the non-scalar loss tensor.
        """
        self.size: Final = size
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
        self.source_name: Final = source_name
        self.metric_name: Final = metric_name + ' ' if metric_name else ''
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
        self.module_names: Final = module_names
        self.lr_param_groups: Final = lr_param_groups
        self.missing: Final = set(module_names) - set(lr_param_groups)
        super().__init__(self.missing)


class ModuleAlreadyRegisteredError(DryTorchError):
    """Raised when trying to access a model that has already been registered."""

    msg = 'Module from model {} is already registered in experiment {} run {}.'

    def __init__(self, model_name: str, exp_name: str, run_id: str) -> None:
        """Constructor.

        Args:
            model_name: the name of the model that was not registered.
            exp_name: the name of the current experiment.
            run_id: the current run's id.
        """
        self.model_name: Final = model_name
        self.exp_name: Final = exp_name
        self.run_id: Final = run_id
        super().__init__(model_name, exp_name, run_id)


class ModuleNotRegisteredError(DryTorchError):
    """Raised an actor tries to access a module that hasn't been registered."""

    msg = 'Module from model {} is not registered in the current run {} - {}.'

    def __init__(self, model_name: str, exp_name: str, run_id: str) -> None:
        """Constructor.

        Args:
            model_name: the name of the model that was not registered.
            exp_name: the name of the current experiment.
            run_id: the current run's id.
        """
        self.model_name: Final = model_name
        self.exp_name: Final = exp_name
        self.run_id: Final = run_id
        super().__init__(model_name, exp_name, run_id)


class ModelNotFoundError(DryTorchError):
    """Raised when no saved model is found in the checkpoint directory."""

    msg = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        """Constructor.

        Args:
            checkpoint_directory: the directory path where no model was found.
        """
        self.checkpoint_directory: Final = checkpoint_directory
        super().__init__(checkpoint_directory)


class NameAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register a name already in use."""

    msg = 'Name {} has already been registered in the current run.'

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
        self.tuple_type: Final = tuple_type
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
        self.current_exp_name: Final = current_exp_name
        self.new_exp_name: Final = new_exp_name
        super().__init__(current_exp_name, new_exp_name)


class NoActiveExperimentError(DryTorchError):
    """Raised when no experiment is currently active."""

    msg = 'No experiment {}has been started.'

    def __init__(
        self,
        experiment_name: str | None = None,
        experiment_class: type | None = None,
    ) -> None:
        """Constructor.

        Args:
            experiment_name: specifies experiment's name.
            experiment_class: specifies experiment's name.
        """
        self.experiment_class: Final = experiment_class
        if experiment_name is not None:
            specify_string = f'named {experiment_name} '
        elif experiment_class is not None:
            specify_string = f'of class {experiment_class.__class__.__name__} '
        else:
            specify_string = ''

        super().__init__(specify_string)


class ResultNotAvailableError(DryTorchError):
    """Raised when trying to access a result before the hook has been called."""

    msg = 'The result will be available only after the hook has been called.'


class TrackerAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register an already registered tracker."""

    msg = 'Tracker {} already registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        """Constructor.

        Args:
            tracker_name: the name of the tracker that is already registered.
            exp_name: the name of the experiment where to register the tracker.
        """
        self.tracker_name: Final = tracker_name
        super().__init__(tracker_name, exp_name)


class TrackerNotActiveError(DryTorchError):
    """Raised when trying to access a tracker that is not registered."""

    msg = 'Tracker {} not registered in any active experiment.'

    def __init__(self, tracker_name: str) -> None:
        """Constructor.

        Args:
            tracker_name: the name of the tracker that is not registered.
        """
        self.tracker_name: Final = tracker_name
        super().__init__(tracker_name)


class CannotStoreOutputWarning(DryTorchWarning):
    """Warning raised when output cannot be stored due to an error."""

    msg = 'Impossible to store output because the following error.\n{}'

    def __init__(self, error: BaseException) -> None:
        """Constructor.

        Args:
            error: the error that prevented output storage.
        """
        self.error: Final = error
        super().__init__(str(error))


class ComputedBeforeUpdatedWarning(DryTorchWarning):
    """Warning raised when compute method is called before updating."""

    msg = 'The ``compute`` method of {} was called before its updating.'

    def __init__(self, calculator: Any) -> None:
        """Constructor.

        Args:
            calculator: the calculator object that was computed before updating.
        """
        self.calculator: Final = calculator
        super().__init__(calculator.__class__.__name__)


class FailedOptionalImportWarning(DryTorchWarning):
    """Warning raised when an optional dependency fails to import."""

    msg = 'Failed to import optional dependency {}. Install for better support.'

    def __init__(self, package_name: str) -> None:
        """Constructor.

        Args:
            package_name: the name of the package that failed to import.
        """
        self.package_name: Final = package_name
        super().__init__(package_name)


class NoPreviousRunsWarning(DryTorchWarning):
    """Attempted to resume the last run, but none were found."""

    msg = 'No previous runs found. Starting a new one.'


class NotExistingRunWarning(DryTorchWarning):
    """Attempted to resume a not existing run."""

    msg = 'Run with id {} not found. Starting a new one.'

    def __init__(self, run_id: str) -> None:
        """Constructor.

        Args:
            run_id: the id of the run that was not found.
        """
        self.run_id: Final = run_id
        super().__init__(run_id)


class OptimizerNotLoadedWarning(DryTorchWarning):
    """Warning raised when the optimizer has not been correctly loaded."""

    msg = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        """Constructor.

        Args:
            error: the error that occurred while loading the optimizer.
        """
        self.error: Final = error
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
        self.selected_epoch: Final = selected_epoch
        self.current_epoch: Final = current_epoch
        super().__init__(selected_epoch, current_epoch)


class RecursionWarning(DryTorchWarning):
    """Warning raised when recursive objects obstruct metadata extraction."""

    msg = 'Impossible to extract metadata because there are recursive objects.'


class RunNotStartedWarning(DryTorchWarning):
    """Warning raised when a run is stopped before being started."""

    msg = """Attempted to stop a Run instance that is not active."""


class RunAlreadyCompletedWarning(DryTorchWarning):
    """Warning raised when a run is stopped after completion."""

    msg = """Attempted to stop a Run instance that is already completed."""


class RunAlreadyRunningWarning(DryTorchWarning):
    """Warning raised when a run is started when already running."""

    msg = """Attempted to start a Run instance that is already running."""


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
        self.subscriber_name: Final = subscriber_name
        self.error: Final = error
        formatted_traceback: Final = traceback.format_exc()
        super().__init__(subscriber_name, formatted_traceback)
