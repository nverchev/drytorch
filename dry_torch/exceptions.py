from __future__ import annotations

from typing import TypeVar, Iterable, Hashable, Any

K = TypeVar('K', bound=Hashable)


class ListKeyError(KeyError):

    def __init__(self, input_keys: Iterable[K], current_keys: Iterable[K]) -> None:
        self.input_keys = input_keys
        self.current_keys = current_keys
        iter_keys = iter(input_keys)
        try:
            key = next(iter_keys)
        except StopIteration:
            raise ValueError('This exception should not be raised when input_keys is empty.')
        try:
            next(iter_keys)
            message = f'Input key {key} does not match the keys already present in the list {current_keys}.'
        except StopIteration:
            message = f'Input keys {input_keys} do not match the keys already present in the list {current_keys}.'
        super().__init__(message)


class DifferentValueError(ValueError):
    def __init__(self, iterable: Iterable[int]) -> None:
        self.list = list(iterable)
        if len(self.list) < 2:
            raise ValueError('This exception should not be raised when when the iterable has less than 2 elements.')
        else:
            message = f'Iterable with values {list(self.list)} contains different values.'
        super().__init__(message)


class NotATensorError(TypeError):

    def __init__(self, not_a_tensor: Any, name: str = '') -> None:
        self.name = name
        self.not_a_tensor = not_a_tensor
        super().__init__(f' Object {name} of type {type(not_a_tensor)} is not a Tensor.')
