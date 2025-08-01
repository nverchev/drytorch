"""Module registering models and records when they are called.

Callers and models are registered in global variables that keep track of the
experiments at the time of calling. The experiment must be the same. Then the
Experiment class is called to create the log events.

Attributes:
    ALL_MODULES: A dictionary that maps module instances to experiments.
"""

from typing import Any

from torch import nn

from drytorch import exceptions, experiments
from drytorch import protocols as p


ALL_MODULES = dict[nn.Module, experiments.Experiment]()


def register_source(source: Any, model: p.ModelProtocol) -> None:
    """Record metadata in the current experiment.

    Args:
        source: the object to document.
        model: the model that the object calls.
    """
    exp: experiments.Experiment = experiments.Experiment.current()
    exp.metadata_manager.register_source(source, model)
    module = model.module
    if module in ALL_MODULES:
        model_exp = ALL_MODULES[module]
        if exp is model_exp:
            return

    raise exceptions.ModelNotRegisteredError(model.name, exp.name)


def register_model(model: p.ModelProtocol) -> None:
    """Record mode in the current experiment.

    Args:
        model: the model to register.
    """
    exp: experiments.Experiment = experiments.Experiment.current()
    module = model.module
    if module in ALL_MODULES:
        raise exceptions.ModuleAlreadyRegisteredError(model.name, exp.name)

    ALL_MODULES[module] = exp
    exp.metadata_manager.register_model(model)
    return
