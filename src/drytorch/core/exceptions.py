"""Module containing internal exceptions for the drytorch package."""

import pathlib
import traceback

from typing import Any, ClassVar, Final

import torch


__all__ = [
    'AccessOutsideScopeError',
    'CannotStoreOutputWarning',
    'CheckpointNotInitializedError',
    'ComputedBeforeUpdatedWarning',
    'ComputedMetricsTypeError',
    'ConvergenceError',
    'DatasetHasNoLengthError',
    'DeviceMismatchError',
    'DistributedDatasetNotDivisibleWarning',
    'DistributedStorageWarning',
    'DryTorchError',
    'DryTorchWarning',
    'EpochNotFoundError',
    'ExperimentalFeatureWarning',
    'FailedOptionalImportWarning',
    'FuncNotApplicableError',
    'LossNotScalarError',
    'MetricNotFoundError',
    'MissingParamError',
    'ModelDeviceMismatchError',
    'ModelNotFoundError',
    'ModuleAlreadyRegisteredError',
    'ModuleNotDistributedWarning',
    'ModuleNotRegisteredError',
    'NameAlreadyRegisteredError',
    'NamedTupleOnlyError',
    'NestedScopeError',
    'NoActiveExperimentError',
    'NoPreviousRunsWarning',
    'NotExistingRunWarning',
    'ObjectiveSyncWarning',
    'OptimizerNotLoadedWarning',
    'PastEpochWarning',
    'RecursionWarning',
    'ResultNotAvailableError',
    'RunAlreadyCompletedWarning',
    'RunAlreadyRecordedError',
    'RunAlreadyRunningWarning',
    'RunNotRecordedError',
    'RunNotStartedWarning',
    'TerminatedTrainingWarning',
    'TrackerAlreadyRegisteredError',
    'TrackerError',
    'TrackerExceptionWarning',
    'TrackerNotActiveError',
]


class DryTorchError(Exception):
    """Base exception class for all drytorch package exceptions."""

    _template: ClassVar[str] = ''

    def __init__(self, *args: Any) -> None:
        """Initialize.

        Args:
            *args: arguments to be formatted into the message template.
        """
        super().__init__(self._template.format(*args))


class DryTorchWarning(UserWarning):
    """Base warning class for all drytorch package warnings."""

    _template: ClassVar[str] = ''

    def __init__(self, *args: Any) -> None:
        """Initialize.

        Args:
            *args: arguments to be formatted into the message template.
        """
        super().__init__(self._template.format(*args))


class TrackerError(DryTorchError):
    """Exception raised by tracker objects during experiment tracking."""

    _template = '[{}] {}'

    def __init__(self, tracker: Any, tracker_msg: str) -> None:
        """Initialize.

        Args:
            tracker: the tracker object that encountered the error.
            tracker_msg: the error message from the tracker.
        """
        self.tracker = tracker
        super().__init__(tracker.__class__.__name__, tracker_msg)


class AccessOutsideScopeError(DryTorchError):
    """Raised when an operation is attempted outside an experiment scope."""

    _template = 'Operation only allowed within an experiment scope.'


class CheckpointNotInitializedError(DryTorchError):
    """Raised when attempting to use a checkpoint without a registered model."""

    _template = 'The checkpoint did not register any model.'


class ComputedMetricsTypeError(DryTorchError):
    """Raised when computed metrics have an unexpected type."""

    _template = (
        'Expected computed metrics as a Mapping[str, Tensor] or Tensor. Got {}.'
    )

    def __init__(self, computed_metrics_type: type) -> None:
        """Initialize.

        Args:
            computed_metrics_type: the actual type of the computed metrics.
        """
        self.computed_metrics_type: Final = computed_metrics_type
        super().__init__(computed_metrics_type.__name__)


class ConvergenceError(DryTorchError):
    """Raised when a module fails to converge during training."""

    _template = 'The module did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        """Initialize.

        Args:
            criterion: the convergence criterion that was not met.
        """
        self.criterion: Final = criterion
        super().__init__(criterion)


class DatasetHasNoLengthError(DryTorchError):
    """Raised when a dataset does not implement the __len__ method."""

    _template = 'Dataset does not implement __len__ method.'


class DeviceMismatchError(DryTorchError):
    """Raised when the metrics device does not match the expected device."""

    _template = 'Metric {} is stored on {} but expected on {}.'

    def __init__(
        self,
        metric_name: str,
        metric_device: torch.device,
        target_device: torch.device,
    ) -> None:
        """Initialize.

        Args:
            metric_name: the name of the metric.
            metric_device: the device of the output tensor.
            target_device: the device of the model.
        """
        self.metric_name: Final = metric_name
        self.metric_device: Final = metric_device
        self.target_device: Final = target_device
        super().__init__(metric_name, metric_device, target_device)


class EpochNotFoundError(DryTorchError):
    """Raised when no saved model is found in the checkpoint directory."""

    _template = 'No checkpoints for epoch {} found in {}.'

    def __init__(self, epoch: int, checkpoint_directory: pathlib.Path) -> None:
        """Initialize.

        Args:
            epoch: the epoch that was not found.
            checkpoint_directory: the directory path where no model was found.
        """
        self.model_directory: Final = checkpoint_directory
        super().__init__(epoch, checkpoint_directory)


class FuncNotApplicableError(DryTorchError):
    """Raised when a function cannot be applied to a specific type."""

    _template = 'Cannot apply function {} on type {}.'

    def __init__(self, func_name: str, type_name: str) -> None:
        """Initialize.

        Args:
            func_name: the name of the function that cannot be applied.
            type_name: the name of the type that doesn't support the function.
        """
        self.func_name: Final = func_name
        self.type_name: Final = type_name
        super().__init__(func_name, type_name)


class LossNotScalarError(DryTorchError):
    """Raised when a loss value is not a scalar tensor."""

    _template = 'Loss must be a scalar but got Tensor of shape {}.'

    def __init__(self, size: torch.Size) -> None:
        """Initialize.

        Args:
            size: the actual size of the non-scalar loss tensor.
        """
        self.size: Final = size
        super().__init__(size)


class MetricNotFoundError(DryTorchError):
    """Raised when a requested metric is not found in the specified source."""

    _template = 'No metric {}found in {}.'

    def __init__(self, source_name: str, metric_name: str) -> None:
        """Initialize.

        Args:
            source_name: the name of the source where the metric was not found.
            metric_name: the name of the metric that was not found.
        """
        self.source_name: Final = source_name
        self.metric_name: Final = metric_name + ' ' if metric_name else ''
        super().__init__(self.metric_name, source_name)


class MissingParamError(DryTorchError):
    """Raised when parameter groups are missing required parameters."""

    _template = 'Parameter groups in input learning rate miss parameters {}.'

    def __init__(
        self, module_names: list[str], lr_param_groups: list[str]
    ) -> None:
        """Initialize.

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

    _template = (
        'Module from model {} is already registered in experiment {} run {}.'
    )

    def __init__(self, model_name: str, exp_name: str, run_id: str) -> None:
        """Initialize.

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

    _template = (
        'Module from model {} is not registered in the current run {} - {}.'
    )

    def __init__(self, model_name: str, exp_name: str, run_id: str) -> None:
        """Initialize.

        Args:
            model_name: the name of the model that was not registered.
            exp_name: the name of the current experiment.
            run_id: the current run's id.
        """
        self.model_name: Final = model_name
        self.exp_name: Final = exp_name
        self.run_id: Final = run_id
        super().__init__(model_name, exp_name, run_id)


class ModelDeviceMismatchError(DryTorchError):
    """Raised when the metrics device does not match the model device."""

    _template = (
        "In multiprocessing, parameters' and outputs' device type must match."
    )


class ModelNotFoundError(DryTorchError):
    """Raised when no saved model is found in the checkpoint directory."""

    _template = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        """Initialize.

        Args:
            checkpoint_directory: the directory path where no model was found.
        """
        self.checkpoint_directory: Final = checkpoint_directory
        super().__init__(checkpoint_directory)


class NameAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register a name already in use."""

    _template = 'Name {} has already been registered in the current run.'

    def __init__(self, name: str) -> None:
        """Initialize.

        Args:
            name: the name that is already registered.
        """
        super().__init__(name)


class NamedTupleOnlyError(DryTorchError):
    """Raised when operations require a named tuple and not a subclass."""

    _template = (
        'The only accepted subtypes of tuple are namedtuple classes. Got {}.'
    )

    def __init__(self, tuple_type: str) -> None:
        """Initialize.

        Args:
            tuple_type: the actual type of the tuple that was provided.
        """
        self.tuple_type: Final = tuple_type
        super().__init__(tuple_type)


class NestedScopeError(DryTorchError):
    """Raised when attempting to nest an experiment scope within another one."""

    _template = 'Cannot start Experiment {} within Experiment {} scope.'

    def __init__(self, current_exp_name: str, new_exp_name: str) -> None:
        """Initialize.

        Args:
            current_exp_name: the name of the currently active experiment.
            new_exp_name: the name of the experiment that cannot be started.
        """
        self.current_exp_name: Final = current_exp_name
        self.new_exp_name: Final = new_exp_name
        super().__init__(current_exp_name, new_exp_name)


class NoActiveExperimentError(DryTorchError):
    """Raised when no experiment is currently active."""

    _template = 'No experiment {}has been started.'

    def __init__(
        self,
        experiment_name: str | None = None,
        experiment_class: type | None = None,
    ) -> None:
        """Initialize.

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

    _template = (
        'The result will be available only after the hook has been called.'
    )


class TrackerAlreadyRegisteredError(DryTorchError):
    """Raised when attempting to register an already registered tracker."""

    _template = 'Tracker {} already registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        """Initialize.

        Args:
            tracker_name: the name of the tracker that is already registered.
            exp_name: the name of the experiment where to register the tracker.
        """
        self.tracker_name: Final = tracker_name
        super().__init__(tracker_name, exp_name)


class TrackerNotActiveError(DryTorchError):
    """Raised when trying to access a tracker that is not registered."""

    _template = 'Tracker {} not registered in any active experiment.'

    def __init__(self, tracker_name: str) -> None:
        """Initialize.

        Args:
            tracker_name: the name of the tracker that is not registered.
        """
        self.tracker_name: Final = tracker_name
        super().__init__(tracker_name)


class CannotStoreOutputWarning(DryTorchWarning):
    """Warning raised when output cannot be stored due to an error."""

    _template = 'Impossible to store output because the following error.\n{}'

    def __init__(self, error: BaseException) -> None:
        """Initialize.

        Args:
            error: the error that prevented output storage.
        """
        self.error: Final = error
        super().__init__(str(error))


class ComputedBeforeUpdatedWarning(DryTorchWarning):
    """Warning raised when compute method is called before updating."""

    _template = 'The ``compute`` method of {} was called before its updating.'

    def __init__(self, calculator: Any) -> None:
        """Initialize.

        Args:
            calculator: the calculator object that was computed before updating.
        """
        self.calculator: Final = calculator
        super().__init__(calculator.__class__.__name__)


class DistributedDatasetNotDivisibleWarning(DryTorchWarning):
    """Warning raised when the dataset cannot be equally distributed."""

    _template = (
        '{} has encountered the following issue with distributed evaluation: \n'
        'The dataset size: {} is not divisible by the number of processes: {}. '
        'Some samples will be evaluated twice, and metrics may not be reliable.'
    )

    def __init__(self, name: str, len_dataset: int, n_processes: int) -> None:
        """Initialize.

        Args:
            name: the name of the actor experiencing the issue.
            len_dataset: the size of the dataset.
            n_processes: the number of processes used in distributed processing.
        """
        self.actor: Final = name
        self.dataset_size: Final = len_dataset
        self.num_processes: Final = n_processes
        super().__init__(name, len_dataset, n_processes)


class DistributedStorageWarning(DryTorchWarning):
    """Warning raised when the distributed storage is not synchronized."""

    _template = 'The storage of the distributed model is not synchronized:\n{}.'

    def __init__(self, error: BaseException) -> None:
        """Initialize.

        Args:
            error: the error that occurred while synchronizing the storage.
        """
        self.error: Final = error
        super().__init__(str(error))


class ExperimentalFeatureWarning(DryTorchWarning):
    """Warning raised when an experimental feature is used."""

    _template = '{} is an experimental feature and may change in the future.'

    def __init__(self, feature: str) -> None:
        """Initialize.

        Args:
            feature: the experimental feature that was used.
        """
        self.feature: Final = feature
        super().__init__(feature)


class FailedOptionalImportWarning(DryTorchWarning):
    """Warning raised when an optional dependency fails to import."""

    _template = (
        'Failed to import optional dependency {}. Install for better support.'
    )

    def __init__(self, package_name: str) -> None:
        """Initialize.

        Args:
            package_name: the name of the package that failed to import.
        """
        self.package_name: Final = package_name
        super().__init__(package_name)


class ModuleNotDistributedWarning(DryTorchWarning):
    """Warning raised when a model is not distributed."""

    _template = 'Distributed wrapper not detected: model weights may diverge.'


class NoPreviousRunsWarning(DryTorchWarning):
    """Attempted to resume the last run, but none were found."""

    _template = 'No previous runs found. Starting a new one.'


class NotExistingRunWarning(DryTorchWarning):
    """Attempted to resume a not existing run."""

    _template = 'Run with id {} not found. Starting a new one.'

    def __init__(self, run_id: str) -> None:
        """Initialize.

        Args:
            run_id: the id of the run that was not found.
        """
        self.run_id: Final = run_id
        super().__init__(run_id)


class ObjectiveSyncWarning(DryTorchWarning):
    """Warning for metric synchronization configuration issues."""

    _template = (
        'Objective synchronization encountered issue: {}. Recommend to: {} .'
    )

    def __init__(self, issue: str, recommend: str) -> None:
        """Initialize.

        Args:
            issue: the issue that was encountered with the objective.
            recommend: the recommended action to fix the issue.
        """
        self.issue: Final = issue
        self.recommend: Final = recommend
        super().__init__(issue, recommend)


class OptimizerNotLoadedWarning(DryTorchWarning):
    """Warning raised when the optimizer has not been correctly loaded."""

    _template = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        """Initialize.

        Args:
            error: the error that occurred while loading the optimizer.
        """
        self.error: Final = error
        super().__init__(error)


class PastEpochWarning(DryTorchWarning):
    """Warning raised when training is requested for a past epoch."""

    _template = 'Training until epoch {} stopped: current epoch is already {}.'

    def __init__(self, selected_epoch: int, current_epoch: int) -> None:
        """Initialize.

        Args:
            selected_epoch: the epoch that training was requested until.
            current_epoch: the current epoch number.
        """
        self.selected_epoch: Final = selected_epoch
        self.current_epoch: Final = current_epoch
        super().__init__(selected_epoch, current_epoch)


class RecursionWarning(DryTorchWarning):
    """Warning raised when recursive objects obstruct metadata extraction."""

    _template = (
        'Impossible to extract metadata because there are recursive objects.'
    )


class RunAlreadyRecordedError(DryTorchError):
    """Error raised when attempting to record a run multiple times."""

    _template = (
        'Run {} already recorded in experiment {}. Use resume=True to resume.'
    )

    def __init__(self, run_id: str, exp_name: str) -> None:
        """Initialize.

        Args:
            run_id: the id of the run that is already recorded.
            exp_name: the name of the experiment where to record the run.
        """
        self.run_id: Final = run_id
        self.exp_name: Final = exp_name
        super().__init__(run_id, exp_name)


class RunAlreadyCompletedWarning(DryTorchWarning):
    """Warning raised when a run is stopped after completion."""

    _template = (
        """Attempted to stop a Run instance that is already completed."""
    )


class RunAlreadyRunningWarning(DryTorchWarning):
    """Warning raised when a run is started when already running."""

    _template = """Attempted to start a Run instance that is already running."""


class RunNotStartedWarning(DryTorchWarning):
    """Warning raised when a run is stopped before being started."""

    _template = """Attempted to stop a Run instance that is not active."""


class RunNotRecordedError(DryTorchError):
    """Raised when attempting to update a run that is not registered."""

    _template = 'Run with id {} is not recorded.'

    def __init__(self, run_id: str) -> None:
        """Constructor.

        Args:
            run_id: the id of the run that is not registered.
        """
        self.run_id: Final = run_id
        super().__init__(run_id)


class TerminatedTrainingWarning(DryTorchWarning):
    """Warning raised when training is attempted after termination."""

    _template = 'Attempted to train module after termination.'


class TrackerExceptionWarning(DryTorchWarning):
    """Warning raised when a tracker encounters an error and is skipped."""

    _template = (
        'Tracker {} encountered the following error and was skipped:\n{}'
    )

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
