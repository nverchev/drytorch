"""Library specific exceptions."""

from typing import Any
import pathlib


class DryTorchException(Exception):
    """Exception from the dry_torch package."""
    msg: str = ''

    def __init__(self, *args: Any) -> None:
        super().__init__(self.msg.format(*args))


class ConvergenceError(DryTorchException, ValueError):
    msg = 'The module did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        self.criterion = criterion
        super().__init__(criterion)


class FuncNotApplicableError(DryTorchException, TypeError):
    msg = 'Cannot apply function {} on type {}.'

    def __init__(self, func_name: str, type_name: str) -> None:
        self.func_name = func_name
        self.type_name = type_name
        super().__init__(func_name, type_name)


class LibraryNotAvailableError(DryTorchException, ImportError):
    msg = 'Library {} is not installed.'

    def __init__(self, library_name: str) -> None:
        self.library_name = library_name
        super().__init__(library_name)


class MetricsNotAVectorError(DryTorchException, ValueError):
    msg = 'Value must be scalar or one-dimensional but got Tensor of shape {}.'

    def __init__(self, shapes: list[int]) -> None:
        self.shapes = shapes
        super().__init__(shapes)


class MetricNotFoundError(DryTorchException, ValueError):
    msg = 'No metric {}found in {}.'

    def __init__(self, source_name: str, metric_name: str) -> None:
        self.source_name = source_name
        self.metric_name = metric_name + ' ' if metric_name else ''
        super().__init__(self.metric_name, source_name)


class MissingParamError(DryTorchException, ValueError):
    msg = 'Parameter groups {} in input learning rate miss some parameters.'

    def __init__(self,
                 model_architecture: str,
                 lr_param_groups: list[str]) -> None:
        self.model_architecture = model_architecture
        self.lr_param_groups = lr_param_groups
        super().__init__(lr_param_groups)


class ModelFirstError(DryTorchException, TypeError):
    msg = 'First argument of type {} does not follow ModelProtocol.'

    def __init__(self, not_a_model: Any) -> None:
        self.not_a_model = not_a_model
        super().__init__(type(not_a_model))


class ModelNotFoundError(DryTorchException, FileNotFoundError):
    msg = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        self.checkpoint_directory = checkpoint_directory
        super().__init__(checkpoint_directory)


class ModuleAlreadyRegisteredError(DryTorchException, ValueError):
    msg = 'Module already registered in experiment {}.'

    def __init__(self, exp_name: str) -> None:
        super().__init__(exp_name)


class NamedTupleOnlyError(DryTorchException, TypeError):
    msg = ('The only accepted subtypes of tuple are namedtuple constructs. '
           'Got {}.')

    def __init__(self, tuple_type: str) -> None:
        self.tuple_type = tuple_type
        super().__init__(tuple_type)


class NoActiveExperimentError(DryTorchException, AttributeError):
    msg = 'No experiment has been started.'


class NoConfigError(DryTorchException, AttributeError):
    msg = 'No config found in experiment.'


class DatasetHasNoLengthError(DryTorchException, AttributeError):
    msg = 'Dataset does not implement __len__ method.'


class TrackerAlreadyRegisteredError(DryTorchException, ValueError):
    msg = 'Tracker {} already registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        self.tracker_name = tracker_name
        super().__init__(tracker_name, exp_name)


class ResultNotAvailableError(DryTorchException, ValueError):
    msg = 'The result will be available only after the hook has been called.'


class TrackerNotRegisteredError(DryTorchException, ValueError):
    msg = 'Tracker {} not registered in experiment {}.'

    def __init__(self, tracker_name: str, exp_name: str) -> None:
        self.tracker_name = tracker_name
        super().__init__(tracker_name, exp_name)


class CannotStoreOutputWarning(DryTorchException, UserWarning):
    msg = 'Impossible to store output because the following error.\n{}'

    def __init__(self, err_msg: str) -> None:
        self.err_msg = err_msg
        super().__init__(err_msg)


class ComputedBeforeUpdatedWarning(DryTorchException, UserWarning):
    msg = 'The ``compute`` method of {} was called before its updating.'

    def __init__(self, calculator: Any) -> None:
        self.calculator = calculator
        super().__init__(calculator.__class__.__name__)


class OptimizerNotLoadedWarning(DryTorchException, UserWarning):
    msg = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__(error)


class PastEpochWarning(DryTorchException, UserWarning):
    msg = 'Training until epoch {} stopped: current epoch is already {}.'

    def __init__(self, selected_epoch: int, current_epoch: int) -> None:
        self.selected_epoch = selected_epoch
        self.current_epoch = current_epoch
        super().__init__(selected_epoch, current_epoch)


class RecursionWarning(DryTorchException, UserWarning):
    msg = 'Impossible to extract metadata because there are recursive objects.'


class TerminatedTrainingWarning(DryTorchException, UserWarning):
    msg = 'Attempted to train module after termination.'


class TrackerError(DryTorchException, UserWarning):
    msg = 'Tracker {} encountered the following error: \n {}'

    def __init__(self, subscriber_name: str, error: Exception) -> None:
        self.subscriber_name = subscriber_name
        self.error = error
        super().__init__(subscriber_name, str(error))
