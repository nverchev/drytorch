from __future__ import annotations
from typing import Generic, TypeVar, Iterable, SupportsIndex, Self, Optional, overload
from typing_extensions import override
from collections import UserList
from collections.abc import KeysView

K = TypeVar('K')  # Type variable for keys
V = TypeVar('V')  # Type variable for values


class ListKeyError(KeyError):

    def __init__(self, input_keys: Iterable[K], current_keys: Iterable[K]) -> None:
        self.input_keys = input_keys
        self.current_keys = current_keys
        iter_keys = iter(input_keys)
        try:
            key = next(iter_keys)
        except StopIteration:
            raise ValueError('This exception should not be raised when input_keys is empty')
        try:
            next(iter_keys)
            message = f'Input key {key} does not match the keys already present in the list {current_keys}'
        except StopIteration:
            message = f'Input keys {input_keys} do not match the keys already present in the list {current_keys}'
        super().__init__(message)


class DictList(UserList, Generic[K, V]):
    """
    A list-like data structure that stores dictionaries with a fixed set of keys.

    Args:
        iterable: An iterable of dictionaries to initialize the list with.

    Attributes:
        list_keys: A tuple of the keys that are present in the list.

    Raises:
        ListKeyError if the input dictionaries have keys that are not present in the list.

    """

    def __init__(self, iterable: Iterable[dict[K, V]] = zip(), /) -> None:
        self.list_keys: tuple[K, ...] = ()
        super().__init__(self.validate_iterable(iterable))

    @override
    def __add__(self, other: Iterable[dict[K, V]], /) -> Self:
        out = self.copy()
        out.extend(other)
        return out

    @override
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> dict[K, V]:
        ...

    @override
    @overload
    def __getitem__(self, input_slice: slice, /) -> Self:
        ...

    @override
    def __getitem__(self, index_or_slice: SupportsIndex | slice, /) -> dict[K, V] | Self:
        """
        Standard __getitem__ implementation that converts stored values back into dictionaries.
        """
        if isinstance(index_or_slice, slice):
            return self.__class__(map(self.item_to_dict, super().__getitem__(index_or_slice)))
        return self.item_to_dict(super().__getitem__(index_or_slice))

    @override
    def __setitem__(self, key, value):
        """
        Standard __setitem__ implementation that validates the input.
        """
        super().__setitem__(key, self.validate_dict(value))

    @override
    def __repr__(self) -> str:
        return f'DictList({self.to_dict().__repr__()}'

    @override
    def append(self, input_dict: dict[K, V], /) -> None:
        """
        Standard append implementation that validates the input.
        """
        super().append(self.validate_dict(input_dict))

    @override
    def extend(self, iterable: Iterable[dict[K, V]], /) -> None:
        """
        Standard extend implementation that validates the input.
        """
        if isinstance(iterable, self.__class__):  # prevents self-alimenting extension
            if self.list_keys == iterable.list_keys:  # more efficient implementation for the common case
                self.data.extend(iterable.data)
                return
        super().extend(self.validate_iterable(iterable))

    @override
    def insert(self, index: int, input_dict: dict[K, V], /):
        """
        Standard insert implementation that validates the input.
        """
        super().insert(index, self.validate_dict(input_dict))

    @override
    def pop(self, index: int = -1, /) -> dict[K, V]:
        """
        Standard pop implementation that converts stored values back into dictionaries.
        """
        return self.item_to_dict(super().pop(index))

    def keys(self) -> KeysView[K]:
        """
        Usual syntax for keys in a dictionary.
        """
        return {key: None for key in self.list_keys}.keys()

    def get(self, key: K, /, default: Optional[V] = None) -> list[V] | list[None]:
        """
        Analogous to the get method in dictionaries.

        Args:
            key: The key for which to retrieve the values.
            default: The default value to return if the key is not present in the list.

        Returns:
            The values for the key in the list or a list of default values.
        """
        try:
            return [item[key] for item in self]
        except KeyError:
            if default is None:
                return [None] * len(self)
            return [default for _ in range(len(self))]  # make sure items are not the same object

    def validate_dict(self, input_dict: dict[K, V], /) -> tuple[V, ...]:
        """
        Validate an input dictionary's keys.

        Args:
            input_dict: The input dictionary to validate.

        Side Effects:
            creates self.list_keys if it does not already exist.

        Returns:
            The values from the input dictionary ordered consistently with the existing keys.

        Raises:
            ListKeyError if the input dictionary has keys that are not present in the list.

        """
        if self.list_keys:
            if set(self.list_keys) != set(input_dict.keys()):
                raise ListKeyError(input_dict.keys(), self.list_keys)
        else:
            self.list_keys = tuple(input_dict.keys())
        return tuple(input_dict[key] for key in self.list_keys)

    def validate_iterable(self, iterable: Iterable[dict[K, V]], /) -> Iterable[tuple[V, ...]]:
        """
        Validate an iterable of input dictionaries.

        Args:
            iterable: The input iterable of dictionaries to validate.

        Returns:
            an Iterable with validated dictionaries.
        """
        return map(self.validate_dict, iterable)

    def item_to_dict(self, item: tuple) -> dict[K, V]:
        """
        Convert a stored item back into a dictionary.

        Returns:
            The dictionary corresponding to the item.
        """
        return dict(zip(self.list_keys, item))

    def to_dict(self) -> dict[K, list[V]]:
        """
        Convert the list into a dictionary.

        Returns:
            The dictionary representation of the list.
        """
        return {key: [item[index] for item in self.data] for index, key in enumerate(self.list_keys)}
