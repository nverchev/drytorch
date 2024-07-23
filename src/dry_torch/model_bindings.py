from __future__ import annotations

import warnings
from functools import wraps
from typing import Callable, Concatenate, Any, TypeVar, ParamSpec

from dry_torch import protocols as p
from dry_torch import exceptions
from dry_torch import tracking
from dry_torch import repr_utils
from dry_torch import io

_Input_contra = TypeVar('_Input_contra',
                        bound=p.InputType,
                        contravariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=p.OutputType,
                     covariant=True)

_P = ParamSpec('_P')
_RT = TypeVar('_RT')

registered_models: dict[int, tracking.GenericExperiment] = {}


def cache_register_model(
        func: Callable[[p.ModelProtocol], None]
) -> Callable[[p.ModelProtocol], None]:
    @wraps(func)
    def wrapper(model: p.ModelProtocol) -> None:
        model_identifier = id(model)
        if model_identifier in registered_models:
            exp_name = registered_models[model_identifier].exp_name
            raise exceptions.AlreadyRegisteredError(model.name, exp_name)
        registered_models[model_identifier] = tracking.GenericExperiment.current()
        return func(model)

    return wrapper


@cache_register_model
def register_model(model: p.ModelProtocol) -> None:
    exp = tracking.GenericExperiment.current()
    name = model.name
    model_repr = model.module.__repr__()
    exp.tracker[name] = tracking.ModelTracker(name, model_repr=model_repr)
    io.dump_metadata(name)


def extract_metadata(attr_dict: dict[str, Any],
                     max_size: int = 3) -> dict[str, Any]:
    # tries to get the most informative representation of the metadata.
    try:
        metadata = {k: repr_utils.struc_repr(v, max_size=max_size)
                    for k, v in attr_dict.items()}
    except RecursionError:
        msg = 'Could not extract metadata because of recursive objects.'
        warnings.warn(msg)
        metadata = {}
    return metadata


def add_metadata(exp: tracking.GenericExperiment,
                 model_name: str,
                 object_name: str,
                 attr_dict: dict[str, Any]) -> None:
    if exp.allow_extract_metadata:
        # tries to get the most informative representation of the metadata.
        object_metadata = extract_metadata(attr_dict, exp.max_items_repr)
        exp.tracker[model_name].metadata[object_name] = object_metadata
        io.dump_metadata(model_name)


def bind_to_model(
        func: Callable[
            Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
            _RT],
) -> Callable[
    Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
    _RT,
]:
    """
    Decorator that extracts metadata from a function named arguments.

    Args:
        func: the function that we want to extract metadata from.
    Returns:
        Callable: the same input function.
    """

    @wraps(func)
    def wrapper(instance: Any,
                model: p.ModelProtocol[_Input_contra, _Output_co],
                *args: _P.args,
                **kwargs: _P.kwargs) -> _RT:
        if not isinstance(model, p.ModelProtocol):
            raise exceptions.BoundedModelTypeError(model)
        exp = tracking.GenericExperiment.current()
        model_tracking = exp.tracker[model.name]
        bindings = model_tracking.bindings
        cls_str = instance.__class__.__name__
        if cls_str in bindings:
            raise exceptions.AlreadyBoundedError(model.name, cls_str)
        cls_count = bindings.setdefault(cls_str, tracking.DefaultName(cls_str))
        add_metadata(exp, model.name, cls_count(), kwargs)
        return func(instance, model, *args, **kwargs)

    return wrapper


def unbind(instance: Any,
           model: p.ModelProtocol[_Input_contra, _Output_co]) -> None:
    if not isinstance(model, p.ModelProtocol):
        raise exceptions.BoundedModelTypeError(model)
    model_tracking = tracking.GenericExperiment.current().tracker[model.name]
    metadata = model_tracking.metadata
    cls_str = instance.__class__.__name__
    if cls_str not in model_tracking.bindings:
        raise exceptions.NotBoundedError(model.name, cls_str)
    cls_str_counter = repr(model_tracking.bindings.pop(cls_str))
    metadata[cls_str_counter]['model_final_epoch'] = model_tracking.epoch
    return
