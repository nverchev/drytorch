"""Module with classes for aggregating and storing structured data. """

import collections
from typing import Generic, Hashable, Iterable, Iterator, KeysView, Mapping
from typing import MutableSequence, Optional, Self, SupportsIndex, TypeVar
from typing import ValuesView, overload

import numpy as np
import numpy.typing as npt
import torch

from dry_torch import descriptors
from dry_torch import exceptions
from dry_torch import protocols as p

_K = TypeVar('_K', bound=Hashable)
_V = TypeVar('_V')


class TorchAggregate:
    """
    This class averages tensor values from dict-like objects.

    This class accepts only tensors with no more than one non-squeezable
    dimension, typically the one for the batch. By keeping counts of the
    samples, it allows a precise sample average in case of unequal batches.

    Args:
        iterable: a tensor valued dictionary or equivalent iterable.
        kwargs: keyword tensor arguments (alternative syntax).

    Attributes:
        aggregate: a dictionary with the summed tensors.
        counts: a dictionary with the count of summed tensors.
    """
    __slots__ = ('aggregate', 'counts')

    def __init__(self,
                 iterable: Iterable[tuple[str, torch.Tensor]] = (),
                 /,
                 **kwargs: torch.Tensor):
        self.aggregate = collections.defaultdict[str, float](float)
        self.counts = collections.defaultdict[str, int](int)
        for key, value in iterable:
            self[key] = value

        for key, value in kwargs.items():
            self[key] = value

    def __add__(self, other: Self | Mapping[str, torch.Tensor]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)
        out = self.__copy__()
        out += other
        return out

    def __iadd__(self, other: Self | Mapping[str, torch.Tensor]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)

        for key, value in other.aggregate.items():
            self.aggregate[key] += value

        for key, count in other.counts.items():
            self.counts[key] += count
        return self

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.aggregate[key] = self._aggregate(value)
        self.counts[key] = self._count(value)
        return

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.aggregate == other.aggregate and self.counts == other.counts

    def reduce(self) -> dict[str, float]:
        """Return the averages values as floating points."""
        return {key: value / self.counts[key]
                for key, value in self.aggregate.items()}

    def __copy__(self) -> Self:
        copied = self.__class__()
        copied.aggregate = self.aggregate.copy()
        copied.counts = self.counts.copy()
        return copied

    @staticmethod
    def _count(value: torch.Tensor) -> int:
        return value.numel()

    @staticmethod
    def _aggregate(value: torch.Tensor) -> float:
        try:
            return value.sum(0).item()
        except RuntimeError:
            raise exceptions.NotBatchedError(list(value.shape))

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(counts={self.counts})'


class DictList(MutableSequence, Generic[_K, _V]):
    """
    A wrapper around a list of tuples with dict-like functionalities.

    Differently from a list of NamedTuples, the keys are set dynamically at
    class instantiation. After that, the keys are immutable. If there are no,
    arguments during instantiation, the setting of the keys is postponed.

    Args:
        iterable: an iterable of dictionaries with the same keys.

    Raises:
        KeysAlreadySetError if the dictionaries have different keys.
    """

    def __init__(self, iterable: Iterable[dict[_K, _V]] = zip(), /) -> None:
        self._keys: tuple[_K, ...] = ()
        self._tuple_list = list(self._validate_iterable(iterable))

    def append(self, input_dict: dict[_K, _V], /) -> None:
        """Validate and append argument to the list."""
        self._tuple_list.append(self._validate_dict(input_dict))

    def clear(self) -> None:
        """Clear the list keeping the keys."""
        self._tuple_list.clear()

    def copy(self) -> Self:
        """Shallow copy of the instance."""
        cloned_self = self.__class__()
        cloned_self.set_keys(self._keys)
        cloned_self._tuple_list = self._tuple_list.copy()
        return cloned_self

    def extend(self, iterable: Iterable[dict[_K, _V]], /) -> None:
        """Validate and extend list."""
        if isinstance(iterable, self.__class__):
            # prevents self-alimenting extension
            if self._keys == iterable._keys:
                # more efficient implementation for the common case
                self._tuple_list.extend(iterable._tuple_list)
                return
        self._tuple_list.extend(self._validate_iterable(iterable))

    def keys(self) -> KeysView[_K]:
        """The keys associated to the tuples of the list."""
        # workaround to produce a KeysView object
        return {key: None for key in self._keys}.keys()

    def insert(self, index: SupportsIndex, input_dict: dict[_K, _V], /) -> None:
        """Validate and insert into the list."""
        self._tuple_list.insert(index, self._validate_dict(input_dict))

    def pop(self, index: SupportsIndex = -1, /) -> dict[_K, _V]:
        """Pop from the list."""
        return self._item_to_dict(self._tuple_list.pop(index))

    def get(self,
            key: _K,
            /,
            default: Optional[_V] = None) -> list[_V] | list[None]:
        """
        Analogous to the get method in dictionaries.

        Args:
            key: the key for which to retrieve the values.
            default: the value to return if the key is not present in the list.

        Returns:
            the values for the key in the list or a list of default values.
        """
        try:
            return [item[key] for item in self]
        except KeyError:
            if default is None:
                return [None] * len(self)
            # make sure items are not the same object
            return [default for _ in range(len(self))]

    def set_keys(self, value=Iterable[_K]) -> None:
        if self._keys:
            raise exceptions.KeysAlreadySetError(self._keys, value)
        else:
            self._keys = tuple(value)
        return

    def _validate_dict(self,
                       input_dict: dict[_K, _V],
                       /) -> tuple[_V, ...]:
        if set(self._keys) != set(input_dict.keys()):
            self.set_keys(tuple(input_dict.keys()))
            # throws a KeysAlreadySet if keys are already present.
        return tuple(input_dict[key] for key in self._keys)

    def _validate_iterable(self,
                           iterable: Iterable[dict[_K, _V]],
                           /) -> Iterable[tuple[_V, ...]]:
        return map(self._validate_dict, iterable)

    def _item_to_dict(self, item: tuple) -> dict[_K, _V]:
        return dict(zip(self._keys, item))

    def to_dict(self) -> dict[_K, list[_V]]:
        """
        Convert the list of tuples into a dictionary of lists.

        Returns:
            the dictionary representation of the object.
        """
        # more efficient than self.get
        return {key: [item[index] for item in self._tuple_list]
                for index, key in enumerate(self._keys)}

    def __add__(self, other: Iterable[dict[_K, _V]], /) -> Self:
        out = self.copy()
        out.extend(other)
        return out

    def __contains__(self, item) -> bool:
        return self._keys.__contains__(item)

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        self._tuple_list.__delitem__(index)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, self.__class__) and
                self._keys == other._keys and
                self._tuple_list == other._tuple_list)

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> dict[_K, _V]:
        ...

    @overload
    def __getitem__(self, input_slice: slice, /) -> Self:
        ...

    def __getitem__(
            self,
            index_or_slice: SupportsIndex | slice,
            /,
    ) -> dict[_K, _V] | Self:
        if isinstance(index_or_slice, slice):
            sliced = map(self._item_to_dict,
                         self._tuple_list.__getitem__(index_or_slice))
            return self.__class__(sliced)
        return self._item_to_dict(self._tuple_list.__getitem__(index_or_slice))

    def __len__(self) -> int:
        return self._tuple_list.__len__()

    def __iter__(self) -> Iterator[dict[_K, _V]]:
        self_iter = self._tuple_list.__iter__()
        for item in self_iter:
            yield self._item_to_dict(item)

    @overload
    def __setitem__(
            self,
            index_or_slice: SupportsIndex,
            value: dict[_K, _V],
            /
    ) -> None:
        ...

    @overload
    def __setitem__(
            self,
            index_or_slice: slice,
            value: Iterable[dict[_K, _V]],
            /
    ) -> None:
        ...

    def __setitem__(
            self,
            index_or_slice: SupportsIndex | slice,
            value: dict[_K, _V] | Iterable[dict[_K, _V]],
            /,
    ) -> None:
        if isinstance(value, dict):
            if not isinstance(index_or_slice, SupportsIndex):
                raise exceptions.MustSupportIndex(index_or_slice)

            tuple_value = self._validate_dict(value)
            self._tuple_list.__setitem__(index_or_slice, tuple_value)
        else:
            if not isinstance(index_or_slice, slice):
                raise exceptions.NotASliceError(index_or_slice)

            iterable_value = self._validate_iterable(value)
            self._tuple_list.__setitem__(index_or_slice, iterable_value)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.to_dict()}'


class NumpyDictList(DictList[str, npt.NDArray | tuple[npt.NDArray, ...]]):
    """
    Subclass that extracts NDArray and NDArray tuples from the batched output. .
    """

    @classmethod
    def from_batch(cls, tensor_batch: p.OutputType) -> Self:
        """
        Class constructor that performs checks, unnesting and type conversion.

        Batched outputs are first translated into a dictionary, with only one
        key-value pair if the outputs are a Tensor, a list or a tuple.
        Otherwise, the outputs must have a to_dict method yielding Tensors or
        iterables of Tensors. The tensors in the iterables also contain batched
        values, and need to be unnested.
        After checking that the tensors have the same batch dimension, they are
        transformed into Numpy arrays. Finally, they are unnested so that each
        element of the DictList corresponds to the outputs of one sample.

        Args:
            tensor_batch: structure containing tensors of batched values.
        """
        instance = cls()
        tensor_dict: Mapping[str, torch.Tensor | Iterable[torch.Tensor]]
        if isinstance(tensor_batch, (torch.Tensor, list, tuple)):
            tensor_dict = dict(outputs=tensor_batch)
        elif isinstance(tensor_batch, p.HasToDictProtocol):
            tensor_dict = tensor_batch.to_dict()
        else:
            raise exceptions.NoToDictMethodError(tensor_batch)

        instance.set_keys(tuple(tensor_dict.keys()))
        instance._tuple_list = cls._enlist(tensor_dict.values())
        return instance

    @classmethod
    def _enlist(
            cls,
            tensor_values: ValuesView[torch.Tensor | Iterable[torch.Tensor]],
            /,
    ) -> list[tuple[npt.NDArray | tuple[npt.NDArray, ...], ...]]:
        _check_tensor_have_same_length(tensor_values)
        numpy_values = map(_to_numpy, tensor_values)
        return list(zip(*map(_conditional_zip, numpy_values)))


def _to_numpy(
        elem: torch.Tensor | Iterable[torch.Tensor],
) -> npt.NDArray | Iterable[npt.NDArray]:
    if isinstance(elem, torch.Tensor):
        return _tensor_to_numpy(elem)
    else:
        return (_tensor_to_numpy(item) for item in elem)


def _tensor_to_numpy(elem: torch.Tensor) -> npt.NDArray:
    return elem.detach().cpu().numpy()


def _conditional_zip(
        elem: npt.NDArray | Iterable[npt.NDArray],
) -> npt.NDArray | Iterator[tuple[npt.NDArray, ...]]:
    return elem if isinstance(elem, np.ndarray) else zip(*elem)


def _check_tensor_have_same_length(
        tensor_values=ValuesView[descriptors.Tensors],
) -> None:
    if not len(tensor_values):
        return

    # this set should only have at most one element
    tensor_len_set: set[int] = set()
    for value in tensor_values:
        if isinstance(value, torch.Tensor):
            tensor_len_set.add(len(value))
        elif isinstance(value, (list, tuple)):
            for elem in value:
                if isinstance(elem, torch.Tensor):
                    tensor_len_set.add(len(elem))
                else:
                    raise exceptions.NotATensorError(elem)
        else:
            raise exceptions.NotATensorError(value)

        if len(tensor_len_set) > 1:
            raise exceptions.DifferentBatchSizeError(tensor_len_set)
    return
