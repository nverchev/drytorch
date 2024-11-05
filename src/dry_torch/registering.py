"""Module with functions to connect a Model-like instance to other classes."""
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Any, TypeVar, ParamSpec

from src.dry_torch import log_events
from src.dry_torch import exceptions
from src.dry_torch import protocols as p
from src.dry_torch import tracking

_Input_contra = TypeVar('_Input_contra',
                        bound=p.InputType,
                        contravariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=p.OutputType,
                     covariant=True)

_P = ParamSpec('_P')
_RT = TypeVar('_RT')

_REGISTERED_MODELS: dict[int, tracking.Experiment] = {}


def register_model(model: p.ModelProtocol, /) -> None:
    """
    Function needed to save train or test a Model-like instance.

    It registers a model to an experiment, giving a context that encapsulates
    the development and testing of the model.

    Args:
        model: the model to register.

    Side Effects:
        Creation of a ModelTracker instance.
        Metadata about the model is dumped to a file.
    """
    exp = tracking.Experiment.current()
    model_identifier = id(model.module)
    if model_identifier in _REGISTERED_MODELS:
        exp_name = _REGISTERED_MODELS[model_identifier].name
        raise exceptions.ModuleAlreadyRegisteredError(exp_name)
    _REGISTERED_MODELS[model_identifier] = exp
    log_events.ModelCreation(model)


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

        if args:
            warnings.warn(exceptions.NotDocumentedArgs())

        class_name = instance.__class__.__name__
        if 'name' in kwargs:
            name = str(kwargs.pop('name'))
        else:
            name = class_name

        record = log_events.RecordMetadata(model.name, class_name, name, kwargs)
        kwargs['name'] = record.name
        return func(instance, model, *args, **kwargs)

    return wrapper
