"""Module registering models and their actors in the current run.

An actor, such as a runner, registers when created, so it must be created in
the run that registered its model.

Attributes:
    ALL_MODULES: maps each registered module to the run that owns it.
    ALL_ACTORS: maps each registered module to the actors acting on it.
"""

import abc
import dataclasses
import weakref

from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar

import torch
import typing_extensions

from drytorch.core import exceptions, experimenting
from drytorch.core import protocols as p


if TYPE_CHECKING:
    from typing import _ProtocolMeta
else:  # typing_extensions metaclass subclasses the typing one
    _ProtocolMeta = type(typing_extensions.Protocol)


__all__ = [
    'ALL_ACTORS',
    'ALL_MODULES',
    'RegisteringMeta',
    'check_current_run',
    'register_actor',
    'register_model',
    'unregister_actor',
    'unregister_model',
]


class _ActorProtocol(Protocol):
    @property
    def model(self) -> p.ModelProtocol[Any, Any]: ...


_T = TypeVar('_T', bound=_ActorProtocol)


class RegisteringMeta(_ProtocolMeta, abc.ABCMeta):
    """Metaclass registering an actor after initialization."""

    def __call__(cls: type[_T], *args: Any, **kwargs: Any) -> _T:
        """Actor instantiation.

        Args:
            *args: the positional arguments of the class constructor.
            **kwargs: the keyword arguments of the class constructor.

        Returns:
            The instance.

        Side Effects:
            Registers the instance as an actor of its model.
        """
        instance: _T = type.__call__(cls, *args, **kwargs)
        register_actor(instance, instance.model)
        return instance


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
    elif (owner.exp_name, owner.run_id) != (run.experiment.name, run.id):
        raise exceptions.ModuleFromAnotherRunError(
            model.name,
            owner.exp_name,
            owner.run_id,
            run.experiment.name,
            run.id,
        )

    return


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
