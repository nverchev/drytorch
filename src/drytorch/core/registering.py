"""Module registering models and records when they are called.

Actors and models are registered in global variables that keep track of the
experiments at the time of calling. The experiment must be the same. Then the
Experiment class is called to create the log events.

Attributes:
    ALL_MODULES: A dictionary that maps module references to experiments.
"""

import dataclasses
import weakref

from typing import Any, Final

import torch

from drytorch.core import exceptions, experimenting
from drytorch.core import protocols as p


__all__ = [
    'ALL_ACTORS',
    'ALL_MODULES',
    'check_current_run',
    'register_actor',
    'register_model',
    'unregister_actor',
    'unregister_model',
]


@dataclasses.dataclass(frozen=True)
class _Owner:
    run: weakref.ReferenceType[experimenting.Run[Any]]
    exp_name: str
    run_id: str


ALL_MODULES: Final = weakref.WeakKeyDictionary[torch.nn.Module, _Owner]()
ALL_ACTORS: Final = weakref.WeakKeyDictionary[
    torch.nn.Module, weakref.WeakSet[Any]
]()


def check_current_run(model: p.ModelProtocol[Any, Any]) -> None:
    """Raise if the model's run is not the active one."""
    run: experimenting.Run[Any] = experimenting.Experiment.get_current().run
    module = model.module
    owner = ALL_MODULES.get(module)
    if owner is None:
        raise exceptions.ModuleNotRegisteredError(
            model.name, run.experiment.name, run.id
        )
    elif owner.run() is not run:
        raise exceptions.ModuleFromAnotherRunError(
            model.name,
            owner.exp_name,
            owner.run_id,
            run.experiment.name,
            run.id,
        )


def register_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Register a module in the current experiment.

    Args:
        model: the model to register.

    Raises:
        ModuleAlreadyRegisteredError: if the module is already registered.
    """
    run: experimenting.Run[Any] = experimenting.Experiment.get_current().run
    module = model.module
    if module in ALL_MODULES:
        owner = ALL_MODULES[module]
        raise exceptions.ModuleAlreadyRegisteredError(
            model.name, owner.exp_name, owner.run_id
        )

    ALL_MODULES[module] = _Owner(weakref.ref(run), run.experiment.name, run.id)
    run.metadata_manager.register_model(model)
    return


def register_actor(actor: Any, model: p.ModelProtocol[Any, Any]) -> None:
    """Register an actor in the current run.

    Args:
        actor: the object to document.
        model: the model that the object acts on.

    Raises:
        ModuleNotRegisteredError: if the module is not registered in the
            current experiment run.
    """
    check_current_run(model)
    run: experimenting.Run[Any] = experimenting.Experiment.get_current().run
    module = model.module

    actors = ALL_ACTORS.setdefault(module, weakref.WeakSet())
    if actor not in actors:
        run.metadata_manager.register_actor(actor, model)
        actors.add(actor)

    return


def unregister_model(model: p.ModelProtocol[Any, Any]) -> None:
    """Unregister a module and all its actors from the current experiment.

    Args:
        model: the model to register.
    """
    module = model.module
    owner = ALL_MODULES.pop(module, None)
    ALL_ACTORS.pop(module, None)
    if owner is not None:
        run = owner.run()
        if run is not None:
            run.metadata_manager.unregister_model(model)

    return


def unregister_actor(actor: Any) -> None:
    """Unregister an actor from the current experiment.

    Args:
        actor: the object to document.
    """
    try:
        run: experimenting.Run[Any] = experimenting.Experiment.get_current().run
        run.metadata_manager.unregister_actor(actor)
    except exceptions.NoActiveExperimentError:
        pass

    for actor_set in ALL_ACTORS.values():
        actor_set.discard(actor)

    return
