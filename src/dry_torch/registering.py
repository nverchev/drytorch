"""
Module that register models and records when they are called.

Callers and models are registered in global variables that keep track of the
experiments at the time of calling. The experiment must be the same. Then the
Experiment class is called to create the log events.

Attributes:
    ALL_MODULES: A dictionary that maps module instances to experiments.
"""

from typing import Any

from torch import nn

from dry_torch import exceptions
from dry_torch import experiments
from dry_torch import protocols as p
from dry_torch import repr_utils

ALL_MODULES = dict[nn.Module, experiments.Experiment]()


def record_model_call(x: Any, model: p.ModelProtocol) -> None:
    """
    Records metadata in the current experiment.

    Args:
        x: The object to document.
        model: The model that the object calls.
    """
    exp = experiments.Experiment.current()
    name = getattr(x, 'name', '') or repr_utils.StrWithTS(x.__class__.__name__)
    exp.metadata_manager.record_model_call(name, model.name, x)
    module = model.module
    if module in ALL_MODULES:
        model_exp = ALL_MODULES[module]
        if exp is model_exp:
            return
    raise exceptions.ModelNotRegisteredError(model.name, exp.name)


def register_model(model: p.ModelProtocol) -> experiments.Experiment:
    """
    Records mode inl the current experiment.

    Args:
        model: The model to register.
    """
    exp = experiments.Experiment.current()
    module = model.module
    if module in ALL_MODULES:
        raise exceptions.ModuleAlreadyRegisteredError(model.name, exp.name)
    ALL_MODULES[module] = exp
    exp.metadata_manager.register_model(model)
    return exp
