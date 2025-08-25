"""Module registering models and records when they are called.

Actors and models are registered in global variables that keep track of the
experiments at the time of calling. The experiment must be the same. Then the
Experiment class is called to create the log events.

Attributes:
    ALL_MODULES: A dictionary that maps module instances to experiments.
"""

from typing import Any, Final

from torch import nn

from drytorch.core import exceptions, experiment
from drytorch.core import protocols as p


ALL_MODULES: Final = dict[nn.Module, experiment.Run[Any]]()
ALL_ACTORS: Final = dict[nn.Module, set[int]]()


def register_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Record mode in the current experiment.

    Args:
        model: the model to register.
    """
    run: experiment.Run[Any] = experiment.Experiment.get_current().run
    module = model.module
    if module in ALL_MODULES:
        return

    ALL_MODULES[module] = run
    run.metadata_manager.register_model(model)
    return


def register_actor(actor: Any, model: p.ModelProtocol[Any, Any]) -> None:
    """Record actor in the current experiment.

    Args:
        actor: the object to document.
        model: the model that the object acts on.
    """
    run: experiment.Run[Any] = experiment.Experiment.get_current().run
    module = model.module
    if module in ALL_MODULES and ALL_MODULES[module] is run:
        if module not in ALL_ACTORS:
            ALL_ACTORS[module] = set()

        if id(actor) not in ALL_ACTORS[module]:
            run.metadata_manager.register_actor(actor, model)
            ALL_ACTORS[module].add(id(actor))

        return

    raise exceptions.ModelNotRegisteredError(
        model.name, run.experiment.name, run.id
    )


def unregister_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Unregister a model and all its actors from the current experiment.

    Args:
        model: the model to register.
    """
    module = model.module
    if module in ALL_MODULES:
        del ALL_MODULES[module]

    run: experiment.Run[Any] = experiment.Experiment.get_current().run
    run.metadata_manager.unregister_model(model)
    return


def unregister_actor(actor: Any) -> None:
    """Unregister an actor from the current experiment.

    Args:
        actor: the object to document.
    """
    run: experiment.Run[Any] = experiment.Experiment.get_current().run
    run.metadata_manager.unregister_actor(actor)
    for actor_set in ALL_ACTORS.values():
        if id(actor) in actor_set:
            actor_set.remove(id(actor))
    return
