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
        model_identifier = id(model)
        if model_identifier in _REGISTERED_MODELS:
            exp_name = _REGISTERED_MODELS[model_identifier].name
            raise exceptions.AlreadyRegisteredError(model.name, exp_name)

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
    model_repr = model.module.__repr__()
    exp.tracker[name] = tracking.ModelTracker(name, model_repr=model_repr)
    io.dump_metadata(name)


def extract_metadata(to_document: dict[str, Any],
                     max_size: int = 3) -> dict[str, Any]:
    """
    Wrapper of recursive_repr that catches Recursion Errors

    Args:
        to_document: a dictionary of objects to document.
        max_size: maximum number of documented items in a obj.
    """
    # get the recursive representation of the objects.
    try:
        metadata = {k: repr_utils.recursive_repr(v, max_size=max_size)
                    for k, v in to_document.items()}
    except RecursionError:
        warnings.warn(exceptions.RecursionWarning())
        metadata = {}
    return metadata


def add_metadata(model_tracker: tracking.ModelTracker,
                 max_items_repr,
                 object_name: str,
                 to_document: dict[str, Any]) -> None:
    """
     Add metadata related to an object operating on a model to its ModelTracker.

     Args:
         model_tracker: the ModelTracker instance where to add metadata.
         max_items_repr: maximum number of documented items in a obj.
         object_name: the name of the object operating on a model.
         to_document: a dictionary of arguments to document.
     """
    extracted_metadata = extract_metadata(to_document, max_items_repr)
    model_tracker.metadata[object_name] = extracted_metadata
    io.dump_metadata(model_tracker.name)
    return


def bind_to_model(
        func: Callable[
            Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
            _RT],
) -> Callable[
    Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
    _RT,
]:
    """
    Decorator that binds a model to a class.

    Only one class can be bound to a model at the one time.
    This decorator should be applied to the __init__ method of a class that
    operates on the model. The __init__ should only accept keyword arguments
    and only the model as a positional only argument.
    The keyword arguments are added to the metadata related to the class.

    Args:
        func: typically the __init__ method of a class.
    Returns:
        the same input function.
    """

    @wraps(func)
    def wrapper(instance: Any,
                model: p.ModelProtocol[_Input_contra, _Output_co],
                **kwargs: _P.kwargs) -> _RT:
        if not isinstance(model, p.ModelProtocol):
            raise exceptions.BoundedModelTypeError(model)

        exp = tracking.Experiment.current()
        model_tracker = exp.tracker[model.name]
        cls_str = instance.__class__.__name__
        bound_cls_str = model_tracker.binding
        if model_tracker.binding is not None:
            raise exceptions.AlreadyBoundError(model.name, bound_cls_str)

        model_tracker.binding = cls_str
        cls_count = model_tracker.default_names.setdefault(
            cls_str,
            tracking.DefaultName(cls_str)
        )
        if exp.allow_extract_metadata:
            add_metadata(model_tracker, exp.max_items_repr, cls_count(), kwargs)

        return func(instance, model, **kwargs)

    return wrapper


def unbind(instance: Any,
           model: p.ModelProtocol[_Input_contra, _Output_co]) -> None:
    if not isinstance(model, p.ModelProtocol):
        raise exceptions.BoundedModelTypeError(model)

    model_tracker = tracking.Experiment.current().tracker[model.name]
    metadata = model_tracker.metadata
    cls_str = instance.__class__.__name__
    if cls_str != model_tracker.binding:
        raise exceptions.NotBoundedError(model.name, cls_str)

    model_tracker.binding = None
    cls_str_with_counter = repr(model_tracker.default_names[cls_str])
    metadata[cls_str_with_counter]['model_final_epoch'] = model_tracker.epoch
    return
