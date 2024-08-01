from typing import Iterable, Any, TypeVar, Hashable
import pathlib

_K = TypeVar('_K', bound=Hashable)


class DryTorchException(BaseException):
    msg: str = ''

    def __init__(self, *args: Any) -> None:
        super().__init__(self.msg.format(*args))


class AccessBeforeCalculateError(DryTorchException, AttributeError):
    msg = 'Results must be precomputed with the calculate method.'

    def __init__(self) -> None:
        super().__init__()


class AlreadyBoundError(DryTorchException, RuntimeError):
    msg = 'Model {} is already bound to class {}.'

    def __init__(self, model_name: str, cls_str: str) -> None:
        self.model_name = model_name
        self.cls_str = cls_str
        super().__init__(model_name, cls_str)


class AlreadyRegisteredError(DryTorchException, ValueError):
    msg = ('Registering model_name {} was unsuccessful:'
           ' model_name already registered in experiment {}.')

    def __init__(self, name: str, exp_name: str) -> None:
        self.name = name
        super().__init__(name, exp_name)


class BoundedModelTypeError(DryTorchException, TypeError):
    msg = 'First argument of type {} does not follow ModelProtocol.'

    def __init__(self, not_a_model: Any) -> None:
        self.not_a_model = not_a_model
        super().__init__(type(not_a_model))


class ConvergenceError(DryTorchException, ValueError):
    msg = 'The module did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        self.criterion = criterion
        super().__init__(criterion)


class DifferentBatchSizeError(DryTorchException, ValueError):
    msg = ('Input tensors should have the same batch size'
           ' but got different values: {}.')

    def __init__(self, iterable: Iterable[int]) -> None:
        self.list = list(iterable)
        super().__init__(self.list)


class FuncNotApplicableError(DryTorchException, TypeError):
    msg = 'Cannot apply function {} on type {}.'

    def __init__(self, func_name: str, type_name: str) -> None:
        self.func_name = func_name
        self.type_name = type_name
        super().__init__(func_name, type_name)


class KeysAlreadySetError(DryTorchException, KeyError):
    msg = 'DictList keys are already set to {} and cannot be modified.'

    def __init__(self,
                 input_keys: Iterable[_K],
                 current_keys: Iterable[_K]) -> None:
        self.input_keys = list(input_keys)
        self.current_keys = list(current_keys)
        super().__init__(current_keys)


class LibraryNotAvailableError(DryTorchException, ImportError):
    msg = 'Library {} is not installed.'

    def __init__(self, library_name: str) -> None:
        self.library_name = library_name
        super().__init__(library_name)


class LibraryNotSupportedError(DryTorchException, ValueError):
    msg = 'Library {} is not supported.'

    def __init__(self, library_name: str) -> None:
        self.library_name = library_name
        super().__init__(library_name)


class MetricNotFoundError(DryTorchException, ValueError):
    msg = 'Metric {} is not present in the training log.'

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        super().__init__(metric_name)


class MissingParamError(DryTorchException, ValueError):
    msg = ('Parameter groups {} in input learning rate'
           ' do not contain all module parameters.')

    def __init__(self,
                 model_architecture: str,
                 lr_param_groups: list[str]) -> None:
        self.model_architecture = model_architecture
        self.lr_param_groups = lr_param_groups
        super().__init__(lr_param_groups)


class ModelNotExistingError(DryTorchException, ValueError):
    msg = ('Accessing module {} was unsuccessful:'
           ' module not registered in experiment {}.')

    def __init__(self, name: str, exp_name: str) -> None:
        self.name = name
        super().__init__(name, exp_name)


class ModelNotFoundError(DryTorchException, FileNotFoundError):
    msg = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        self.checkpoint_directory = checkpoint_directory
        super().__init__(checkpoint_directory)


class MustSupportIndex(DryTorchException, TypeError):
    msg = "Object of type {} has not a '__index__' method."

    def __init__(self, not_supporting_index: Any) -> None:
        self.not_supporting_index = not_supporting_index
        super().__init__(type(not_supporting_index).__name__)


class NamedTupleOnlyError(DryTorchException, TypeError):
    msg = ('The only accepted subtypes of tuple are namedtuple constructs. '
           'Got {}.')

    def __init__(self, tuple_type: str) -> None:
        self.tuple_type = tuple_type
        super().__init__(tuple_type)


class NoConfigError(DryTorchException, AttributeError):
    msg = 'No config found in experiment.'


class NoLengthError(DryTorchException, AttributeError):
    msg = 'Dataset does not implement __len__ method.'


class NotASliceError(DryTorchException, TypeError):
    msg = 'Expected slice. Got object of type {}.'

    def __init__(self, not_a_slice: Any) -> None:
        self.not_a_slice = not_a_slice
        super().__init__(type(not_a_slice))


class NotATensorError(DryTorchException, TypeError):
    msg = 'Object {} of type {} is not a Tensor.'

    def __init__(self, not_a_tensor: Any, name: str = '') -> None:
        self.name = name
        self.not_a_tensor = not_a_tensor
        super().__init__(name, type(not_a_tensor))


class NotBatchedError(DryTorchException, ValueError):
    msg = 'Value must be scalar or one-dimensional but got Tensor of shape {}.'

    def __init__(self, shapes: list[int]) -> None:
        self.shapes = shapes
        super().__init__(shapes)


class NotBoundedError(DryTorchException, RuntimeError):
    msg = 'There is no object of class {} bounded to module {}.'

    def __init__(self, model_name: str, cls_str: str) -> None:
        self.model_name = model_name
        self.cls_str = cls_str
        super().__init__(cls_str, model_name)


class NoToDictMethodError(DryTorchException, AttributeError):
    msg = 'Object of type {} does not have a to_dict() method.'

    def __init__(self, no_to_dict: Any) -> None:
        self.no_to_dict = no_to_dict
        super().__init__(type(no_to_dict))


class PartitionNotFoundError(DryTorchException, ValueError):
    msg = 'Impossible to load {} dataset: partition {} not found.'

    def __init__(self, partition: str) -> None:
        self.partition = partition
        super().__init__(partition, partition)


class AlreadyTestedWarning(DryTorchException, RuntimeWarning):
    msg = 'Test has already been executed.'


class CannotStoreOutputWarning(DryTorchException, RuntimeWarning):
    msg = 'Impossible to store output because the following error.\n{}'

    def __init__(self, err_msg: str) -> None:
        self.err_msg = err_msg
        super().__init__(err_msg)


class OptimizerNotLoadedWarning(DryTorchException, RuntimeWarning):
    msg = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__(error)


class RecursionWarning(DryTorchException, RuntimeWarning):
    msg = 'Impossible to extract metadata because there are recursive objects.'


class VisdomConnectionWarning(DryTorchException, RuntimeWarning):
    msg = 'Visdom connection refused by server.'
