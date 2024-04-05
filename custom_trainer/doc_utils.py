from typing import TypeVar, Callable

CallableVar = TypeVar('CallableVar', bound=Callable)


def add_docstring(doc: str) -> Callable[[CallableVar], CallableVar]:
    """
    Decorator factory that builds a decorator for a given docstring.

    Args:
        doc: the docstring to add.
    Returns:
        the decorator responsible for adding the docstring.
    """

    def wrapper(method: CallableVar) -> CallableVar:
        """
        Add a docstring to a callable (a method usually).

        Args:
            method: the docstring to be added to the callable
        Returns:
            the callable with the added docstring
        """
        method.__doc__ = doc
        return method

    return wrapper
