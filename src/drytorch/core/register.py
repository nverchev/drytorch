"""Module registering models and records when they are called.

Actors and models are registered in global variables that keep track of the
experiments at the time of calling. The experiment must be the same. Then the
Experiment class is called to create the log events.

Attributes:
    ALL_MODULES: A dictionary that maps module references to experiments.
"""

from typing import Any, Final

from drytorch.core import exceptions, experiment
from drytorch.core import protocols as p


ALL_MODULES: Final = dict[int, experiment.Run[Any]]()
ALL_ACTORS: Final = dict[int, set[int]]()


def register_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Register a module in the current experiment.

    Args:
        model: the model to register.
    """
    run: experiment.Run[Any] = experiment.Experiment.get_current().run
    id_module = id(model.module)
    if id_module in ALL_MODULES:
        raise exceptions.ModuleAlreadyRegisteredError(
            model.name, run.experiment.name, run.id
        )

    ALL_MODULES[id_module] = run
    run.metadata_manager.register_model(model)
    return


def register_actor(actor: Any, model: p.ModelProtocol[Any, Any]) -> None:
    """Register an actor in the current run.

    Args:
        actor: the object to document.
        model: the model that the object acts on.
    """
    run: experiment.Run[Any] = experiment.Experiment.get_current().run
    id_module = id(model.module)
    if id_module not in ALL_MODULES or ALL_MODULES[id_module] is not run:
        raise exceptions.ModuleNotRegisteredError(
            model.name, run.experiment.name, run.id
        )

    actors = ALL_ACTORS.setdefault(id_module, set())
    if id(actor) not in actors:
        run.metadata_manager.register_actor(actor, model)
        actors.add(id(actor))

    return


def unregister_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Unregister a module and all its actors from the current experiment.

    Args:
        model: the model to register.
    """
    id_module = id(model.module)
    if id_module in ALL_MODULES:
        del ALL_MODULES[id_module]

    if id_module in ALL_ACTORS:
        del ALL_ACTORS[id_module]

    try:
        run: experiment.Run[Any] = experiment.Experiment.get_current().run
    except exceptions.NoActiveExperimentError:
        pass
    else:
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
