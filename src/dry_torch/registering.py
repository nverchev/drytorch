"""Module with functions to connect a Model-like instance to other classes."""

from functools import wraps
from typing import Callable, Concatenate, Any, TypeVar, ParamSpec
import warnings

from dry_torch import exceptions
from dry_torch import io
from dry_torch import protocols as p
from dry_torch import repr_utils
from dry_torch import tracking

_Input_contra = TypeVar('_Input_contra',
                        bound=p.InputType,
                        contravariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=p.OutputType,
                     covariant=True)

_P = ParamSpec('_P')
_RT = TypeVar('_RT')

_REGISTERED_MODELS: dict[int, tracking.Experiment] = {}


def _cache_register_model(
        func: Callable[[p.ModelProtocol], None]
) -> Callable[[p.ModelProtocol], None]:
    @wraps(func)
    def wrapper(model: p.ModelProtocol) -> None:
        exp = tracking.Experiment.current()
        model_identifier = id(model.module)
        if model_identifier in _REGISTERED_MODELS:
            exp_name = _REGISTERED_MODELS[model_identifier].name
            raise exceptions.ModuleAlreadyRegisteredError(exp_name)

        _REGISTERED_MODELS[model_identifier] = exp
        return func(model)

    return wrapper


@_cache_register_model
def register_model(model: p.ModelProtocol, /) -> None:
    """
    Function needed to save train or test a Model-like instance.

    It registers a model to an experiment, giving a context that encapsulates
    the development and testing of the model. When registering a model, you
    create a ModelTracker object which contains logs and metadata of the model.
    To encourage best practises, a model can only be registered to a single
    experiment.

    Args:
        model: the model to register.

    Side Effects:
        Creation of a ModelTracker instance.
        Metadata about the model is dumped to a file.
    """
    exp = tracking.Experiment.current()
    name = model.name
    exp.tracker[name] = tracking.ModelTracker(name)
    class_str_snake = model.__class__.__name__
    metadata = {name: repr_utils.LiteralStr(repr(model.module))}
    exp.tracker[name].metadata[class_str_snake] = metadata
    if exp.save_metadata:
        io.dump_metadata(name, class_str_snake)


def extract_metadata(to_document: dict[str, Any],
                     max_size: int = 3) -> dict[str, Any]:
    """
    Wrapper of recursive_repr that catches Recursion Errors

    Args:
        to_document: a dictionary of objects to document.
        max_size: maximum number of documented items in an obj.
    """
    # get the recursive representation of the objects.
    try:
        metadata = {k: repr_utils.recursive_repr(v, max_size=max_size)
                    for k, v in to_document.items()}
    except RecursionError:
        warnings.warn(exceptions.RecursionWarning())
        metadata = {}
    return metadata


def register_kwargs(
        func: Callable[
            Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
            _RT],
) -> Callable[
    Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P], _RT,
]:
    """
    Decorator that collects metadata related to a model.

    This decorator should be applied to the __init__ method of a class that
    operates on the model. The model should be the first positional argument
    after the instance of the class. The keyword arguments are added to the
    metadata related to the class.

    Args:
        func: typically the __init__ method of a class.
    Returns:
        the same input function.
    """

    @wraps(func)
    def wrapper(instance: Any,
                model: p.ModelProtocol[_Input_contra, _Output_co],
                *args: _P.args,
                **kwargs: _P.kwargs) -> _RT:
        if not isinstance(model, p.ModelProtocol):
            raise exceptions.ModelFirstError(model)

        exp = tracking.Experiment.current()
        model_tracker = exp.tracker[model.name]
        if not exp.save_metadata:
            return func(instance, model, *args, **kwargs)
        if args:
            warnings.warn(exceptions.NotDocumentedArgs())

        cls_str = instance.__class__.__name__

        cls_count = model_tracker.default_names.setdefault(
            cls_str,
            tracking.DefaultName(cls_str)
        )
        name = cls_count()
        if 'name' in kwargs:
            name = str(kwargs.pop('name'))
        metadata = extract_metadata(kwargs, exp.max_items_repr)
        if name in model_tracker.metadata:
            raise exceptions.NameAlreadyExistsError(name, model.name)
        model_tracker.metadata[name] = metadata
        kwargs['name'] = io.dump_metadata(model.name, name)
        return func(instance, model, *args, **kwargs)

    return wrapper
