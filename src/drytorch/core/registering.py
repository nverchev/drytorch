"""Module registering models and records when they are called.

Callers and models are registered in global variables that keep track of the
experiments at the time of calling. The experiment must be the same. Then the
Experiment class is called to create the log events.

Attributes:
    ALL_MODULES: A dictionary that maps module instances to experiments.
"""

from typing import Any

from torch import nn

from drytorch.core import exceptions, experiments
from drytorch.core import protocols as p


ALL_MODULES = dict[nn.Module, experiments.Run[Any]]()


def register_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Record mode in the current experiment.

    Args:
        model: the model to register.
    """
    run: experiments.Run[Any] = experiments.Experiment.get_current().run
    module = model.module
    if module in ALL_MODULES:
        raise exceptions.ModuleAlreadyRegisteredError(model.name, run.run_id)

    ALL_MODULES[module] = run
    run.metadata_manager.register_model(model)
    return


def register_source(source: Any, model: p.ModelProtocol[Any, Any]) -> None:
    """Record metadata in the current experiment.

    Args:
        source: the object to document.
        model: the model that the object calls.
    """
    run: experiments.Run[Any] = experiments.Experiment.get_current().run
    run.metadata_manager.register_source(source, model)
    module = model.module
    if module in ALL_MODULES:
        model_exp = ALL_MODULES[module]
        if run is model_exp:
            return

    raise exceptions.ModelNotRegisteredError(model.name, run.run_id)
