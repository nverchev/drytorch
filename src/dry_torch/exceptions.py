from __future__ import annotations

from typing import Iterable, Any, TypeVar, Hashable

_K = TypeVar('_K', bound=Hashable)


class KeysAlreadySet(KeyError):
    msg = 'DictList keys are already set to {} and cannot be modified.'

    def __init__(self,
                 input_keys: Iterable[_K],
                 current_keys: Iterable[_K]) -> None:
        self.input_keys = list(input_keys)
        self.current_keys = list(current_keys)
        super().__init__(self.msg.format(current_keys))


class DifferentBatchSizeError(ValueError):
    msg = ('Input tensors should have the same batch size'
           ' but got different values: {}.')

    def __init__(self, iterable: Iterable[int]) -> None:
        self.list = list(iterable)
        super().__init__(self.msg.format(self.list))


class NotATensorError(TypeError):
    msg = 'Object {} of type {} is not a Tensor.'

    def __init__(self, not_a_tensor: Any, name: str = '') -> None:
        self.name = name
        self.not_a_tensor = not_a_tensor
        super().__init__(self.msg.format(name, type(not_a_tensor)))


class ModelNotFoundError(ValueError):
    msg = ('Accessing model {} was unsuccessful:'
           ' model not registered in experiment {}.')

    def __init__(self, name: str, exp_name: str) -> None:
        self.name = name
        super().__init__(self.msg.format(name, exp_name))


class ModelAlreadyRegisteredError(ValueError):
    msg = ('Registering name {} was unsuccessful:'
           ' name already registered in experiment {}.')

    def __init__(self, name: str, exp_name: str) -> None:
        self.name = name
        super().__init__(self.msg.format(name, exp_name))



class PartitionNotFoundError(ValueError):
    msg = 'Impossible to load {} dataset: partition {} not found.'

    def __init__(self, partition: str) -> None:
        self.partition = partition
        super().__init__(self.msg.format(partition))


class BoundedModelTypeError(TypeError):
    msg = 'First argument of type {} does not follow ModelOptimizerProtocol'

    def __init__(self, not_a_model: Any) -> None:
        self.not_a_model = not_a_model
        super().__init__(self.msg.format(type(not_a_model)))


class MissingParamError(ValueError):
    msg = ('Parameter groups {} in input learning rate'
           ' do not contain all model parameters.')

    def __init__(self,
                 model_architecture: str,
                 lr_param_groups: list[str]) -> None:
        self.model_architecture = model_architecture
        self.lr_param_groups = lr_param_groups
        super().__init__(self.msg.format(lr_param_groups))


class AlreadyBoundedError(RuntimeError):
    msg = 'There is already an object of class {} bounded to model {}'

    def __init__(self, model_name: str, cls: str) -> None:
        self.model_name = model_name
        self.cls = cls
        super().__init__(self.msg.format(cls, model_name))


class ConvergenceError(ValueError):
    msg = 'The model did not converge (criterion is {}).'

    def __init__(self, criterion: float) -> None:
        self.criterion = criterion
        super().__init__(self.msg.format(criterion))
