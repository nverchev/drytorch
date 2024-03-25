from __future__ import annotations

from typing import Any, Generic, TypeVar, Iterable, Optional

T = TypeVar('T', bound=Iterable)  # typically torch.Tensor


class DifferentLengthsError(ValueError):

    def __init__(self, lengths_set: list[int]) -> None:
        self.lengths_set = lengths_set
        message = 'Lists do not have the same lengths'
        super().__init__(message)


class DifferentNestedError(ValueError):

    def __init__(self, message) -> None:
        super().__init__(message)


class PartiallyNestedError(ValueError):

    def __init__(self, type_list: list[type]) -> None:
        self.type_list = type_list
        message = 'List contains mixture of lists and no lists'
        super().__init__(message)


def buildDictList(names: list[str]):
    class DictList(list, Generic[T]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.names = names

    return DictList



class DictList(dict, Generic[T]):
    """
    Dictionary with possibly nested lists of objects (specifically Tensors).
    It ensures that the lists keep their index reference.
    Objects should not be lists
    """

    def __init__(self, **kwargs: list[T] | list[list[T]]) -> None:
        """
        Args: a dictionary mapping to list and list of lists of objects of tipe T
        """
        super().__init__(**kwargs)
        self.simple_list_keys, self.nested_list_keys = self.which_nested(self)
        self.len = self.list_len(self)

    def __setitem__(self, key: Any, value: list[T] | list[list[T]]):
        """
        It performs various checks before adding an item
        Side Effect:
            A key - value pair is added
        Raises:
           PartiallyNestedError if value contains both lists and non-list
           DifferentLengthsError if value has a length incompatible with self
        """
        len_list = [len(elem) for elem in value if isinstance(elem, list)]
        if len(len_list) == 0:
            len_value = len(value)
        elif len(len_list) == len(value):
            len_value, *other_len = set(len_list)
            if other_len:
                raise DifferentLengthsError([*len_list])
        else:
            raise PartiallyNestedError([type(elem) for elem in value])
        if len_value == len(self):
            super().__setitem__(key, value)
        else:
            raise DifferentLengthsError([len_value, self.len])
        return

    def setdefault(self, key: Any, default: Optional[T | list[T]] = None) -> list[T] | list[list[T]]:
        """
        It sets a default value.
        Args:
             key: the key to get or to create
             default: a default value in case the key is missing.
        Returns:
            the value for the key
        Side Effects:
            set key to default if it does not exist
        Raises:
            ValueError if default is None
        """
        if default is None:
            raise ValueError('Default value must be specified')
        # this check to help type checking
        elif isinstance(default, list):
            self[key] = [default for _ in range(len(self))]
        else:
            self[key] = [default for _ in range(len(self))]
        return self[key]

    def extract(self, ind: int) -> dict[Any, T | list[T]]:
        """
        It extracts an index element from the (nested) lists
        Args:
             ind: the value of the index
        Returns:
            out_dict: a dct with an element for normal lists, a list of elements for nested lists
        """
        out_dict: dict[str, T | list[T]] = {}

        for key in self.simple_list_keys:
            out_dict[key] = self[key][ind]

        for key in self.nested_list_keys:
            out_dict[key] = [elem[ind] for elem in self[key]]

        return out_dict

    def append_dict(self, dct: dict[Any, T | list[T]]) -> None:
        """
        It appends (or creates) dict of tensor or list of tensors
        Args:
            dct: a dictionary to append.
        Side Effect:
            the lists in self are extended by 1 element
        """
        dict_list = DictList(**self.enlist_dict(dct))

        if not self:
            self._init_like(dict_list)
        else:
            self._check_nested(dict_list)

        for key in self.simple_list_keys:
            self[key].append(dct[key])

        for key in self.nested_list_keys:
            for elem, new_elem in zip(self[key], dct[key]):
                elem.append(new_elem)

        self.len += 1

    def extend_dict(self, dict_list: DictList[T]) -> None:
        """
        It extends (or creates) dict of tensor or list of tensors
        Args:
            dict_list: the DictList that will extend self.
        Side Effect:
            the lists in self are extended
        Raises:
            DifferentLengthsError if lists and sub-lists (in case of nested lists) have different lengths
        """
        if not self:
            self._init_like(dict_list)
        else:
            self._check_nested(dict_list)

        for key in self.simple_list_keys:
            self[key].extend(dict_list[key])

        for key in self.nested_list_keys:
            for elem, new_elem in zip(self[key], dict_list[key]):
                elem.extend(new_elem)

        self.len += dict_list.len

    def reset(self) -> None:
        """
        Reset self but keeping the nested structure
        """
        self._init_like(self)

    def _check_nested(self, other: DictList[T]) -> None:
        """
        It checks that the structure of other matches the one of self
        Args:
            other: the DictList whose structure we want to check.
        Raises:
            DifferentNestedError is other has a different structure.
        """
        unmatched_simple = self.simple_list_keys.symmetric_difference(other.simple_list_keys)
        unmatched_nested = self.nested_list_keys.symmetric_difference(other.nested_list_keys)
        if unmatched_simple or unmatched_nested:
            raise DifferentNestedError(f'Unmatched keys: {unmatched_simple | unmatched_nested}')
        for key in self.nested_list_keys:
            if len(self[key] != len(other[key])):
                raise DifferentNestedError(f'Different number of elements for key {key}: {self[key]} and {other[key]}')

    def _init_like(self, dict_list: DictList[T]) -> None:
        """
        It initializes self with the keys and the nested structure of another DictList
        Args:
            dict_list: the DictList whose structure we want to replicate.
        Side Effect:
            self is initialized
        """

        for key in dict_list.simple_list_keys:
            empty_list: list[T] = []
            self[key] = empty_list

        for key in dict_list.nested_list_keys:
            empty_lists: list[list[T]] = [[] for _ in dict_list[key]]
            self[key] = empty_lists

        self.len = dict_list.len
        self.simple_list_keys = dict_list.simple_list_keys
        self.nested_list_keys = dict_list.nested_list_keys

    def __len__(self) -> int:
        """
        The length corresponds to the available indexes in the extract method.
        Returns:
            the length of all the (nested) lists
        """
        return self.len

    @staticmethod
    def enlist_dict(dct: dict[Any, T | list[T]]) -> dict[Any, list[T] | list[list[T]]]:
        """
        Instantiate class from a dictionary by adding elements in a list
        Returns:
            the instantiated object
        """
        out_dict: dict[Any, list[T] | list[list[T]]] = {}
        for key, value in dct.items():
            if isinstance(value, list):
                out_dict[key] = [[elem] for elem in value]
            else:
                out_dict[key] = [value]
        return out_dict

    @staticmethod
    def which_nested(dict_list: DictList[T]) -> tuple[set[Any], set[Any]]:
        """
        It initializes self with the keys and the nested structure of another DictList
        Args:
            dict_list: a target DictList
        Raises:
           PartiallyNestedError if a mapped list contains both lists and non-list
        Returns:
            simple_list_keys: a list of the keys mapping to simple lists
            nested_list_keys: a list of the keys mapping to nested lists

        """
        simple_list_keys: set[Any] = set()
        nested_list_keys: set[Any] = set()

        for key, value in dict_list.items():
            if all(isinstance(elem, list) for elem in value):
                nested_list_keys.add(key)
            elif any(isinstance(elem, list) for elem in value):
                raise PartiallyNestedError([type(elem) for elem in value])
            else:
                simple_list_keys.add(key)

        return simple_list_keys, nested_list_keys

    @staticmethod
    def list_len(dict_list: DictList[T]) -> int:
        """
        It checks that all the mapped simple lists and the sub-lists of the nested lists have the same length.
        It returns its value
        Args:
            dict_list: a target DictList
        Raises:
           DifferentLengthsError if the length of the lists and of the sub-lists is not the same
        Returns:
            only_len: the length of all the list and sub-lists
        """
        if not dict_list.keys():
            return 0

        # this set should only have at most one element
        list_len_set: set[int] = set()

        for key in dict_list.simple_list_keys:
            list_len_set.add(len(dict_list[key]))

        for key in dict_list.nested_list_keys:
            list_len_set.update({len(elem) for elem in dict_list[key]})

        only_len, *other_len = list_len_set
        if other_len:
            raise DifferentLengthsError(list(list_len_set))
        return only_len
