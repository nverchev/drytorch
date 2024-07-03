from typing import Iterable, Any, TypeVar, Hashable, Callable, Type
import pathlib

_K = TypeVar('_K', bound=Hashable)


class DryTorchException(BaseException):
    msg: str

    def __init__(self, message: str = '') -> None:
        super().__init__(message or self.msg)


class AccessBeforeCalculateError(AttributeError, DryTorchException):
    msg = 'Results must be precomputed with the calculate method'


class AlreadyBoundedError(RuntimeError, DryTorchException):
    msg = 'There is already an object of class {} bounded to module {}'

    def __init__(self, model_name: str, cls_str: str) -> None:
        self.model_name = model_name
        self.cls_str = cls_str
        super().__init__(self.msg.format(cls_str, model_name))


class BoundedModelTypeError(TypeError, DryTorchException):
    msg = 'First argument of type {} does not follow ModelProtocol'

    def __init__(self, not_a_model: Any) -> None:
        self.not_a_model = not_a_model
        super().__init__(self.msg.format(type(not_a_model)))


class ConvergenceError(ValueError, DryTorchException):
    msg = 'The module did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        self.criterion = criterion
        super().__init__(self.msg.format(criterion))


class DifferentBatchSizeError(ValueError, DryTorchException):
    msg = ('Input tensors should have the same batch size'
           ' but got different values: {}.')

    def __init__(self, iterable: Iterable[int]) -> None:
        self.list = list(iterable)
        super().__init__(self.msg.format(self.list))


class FuncNotApplicableError(TypeError, DryTorchException):
    msg = 'Cannot apply {} on Datatype {}.'

    def __init__(self, func: Callable, datatype=Type) -> None:
        self.func = func
        self.type_name = datatype.__name__
        super().__init__(self.msg.format(func, self.type_name))


class KeysAlreadySetError(KeyError, DryTorchException):
    msg = 'DictList keys are already set to {} and cannot be modified.'

    def __init__(self,
                 input_keys: Iterable[_K],
                 current_keys: Iterable[_K]) -> None:
        self.input_keys = list(input_keys)
        self.current_keys = list(current_keys)
        super().__init__(self.msg.format(current_keys))


class LibraryNotAvailableError(ImportError, DryTorchException):
    msg = 'Library {} is not installed.'

    def __init__(self, library_name: str) -> None:
        self.library_name = library_name
        super().__init__(self.msg.format(library_name))


class LibraryNotSupportedError(ValueError, DryTorchException):
    msg = 'Library {} is not supported.'

    def __init__(self, library_name: str) -> None:
        self.library_name = library_name
        super().__init__(self.msg.format(library_name))


class MissingParamError(ValueError, DryTorchException):
    msg = ('Parameter groups {} in input learning rate'
           ' do not contain all module parameters.')

    def __init__(self,
                 model_architecture: str,
                 lr_param_groups: list[str]) -> None:
        self.model_architecture = model_architecture
        self.lr_param_groups = lr_param_groups
        super().__init__(self.msg.format(lr_param_groups))


class ModelAlreadyRegisteredError(ValueError, DryTorchException):
    msg = ('Registering model_name {} was unsuccessful:'
           ' model_name already registered in experiment {}.')

    def __init__(self, name: str, exp_name: str) -> None:
        self.name = name
        super().__init__(self.msg.format(name, exp_name))


class ModelNotExistingError(ValueError, DryTorchException):
    msg = ('Accessing module {} was unsuccessful:'
           ' module not registered in experiment {}.')

    def __init__(self, name: str, exp_name: str) -> None:
        self.name = name
        super().__init__(self.msg.format(name, exp_name))


class ModelNotFoundError(FileNotFoundError, DryTorchException):
    msg = 'No saved module found in {}.'

    def __init__(self, checkpoint_directory: pathlib.Path) -> None:
        self.checkpoint_directory = checkpoint_directory
        super().__init__(self.msg.format(checkpoint_directory))


class NoLengthError(AttributeError, DryTorchException):
    msg = 'Dataset does not implement __len__ method.'


class NotATensorError(TypeError, DryTorchException):
    msg = 'Object {} of type {} is not a Tensor.'

    def __init__(self, not_a_tensor: Any, name: str = '') -> None:
        self.name = name
        self.not_a_tensor = not_a_tensor
        super().__init__(self.msg.format(name, type(not_a_tensor)))


class NotBoundedError(RuntimeError, DryTorchException):
    msg = 'There is no object of class {} bounded to module {}'

    def __init__(self, model_name: str, cls_str: str) -> None:
        self.model_name = model_name
        self.cls_str = cls_str
        super().__init__(self.msg.format(cls_str, model_name))


class PartitionNotFoundError(ValueError, DryTorchException):
    msg = 'Impossible to load {} dataset: partition {} not found.'

    def __init__(self, partition: str) -> None:
        self.partition = partition
        super().__init__(self.msg.format(partition, partition))


class OptimizerNotLoadedWarning(RuntimeWarning, DryTorchException):
    msg = 'The optimizer has not been correctly loaded:\n{}'

    def __init__(self, error: BaseException) -> None:
        super().__init__(self.msg.format(error))


class VisdomConnectionWarning(RuntimeWarning, DryTorchException):
    msg = 'Visdom connection refused by server.'

