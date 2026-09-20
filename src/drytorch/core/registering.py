"""Registration of models and their runners in the current run.

An runner, such as a runner, registers when created, so it must be created in
the run that registered its model.

Attributes:
    ALL_MODULES: maps each registered module to the run that owns it.
    ALL_RUNNERS: maps each registered module to the runners using it.
"""

import abc
import dataclasses
import weakref

from typing import Any, Final, TypeVar

import torch

from drytorch.core import exceptions, experimenting
from drytorch.core import protocols as p


__all__ = [
    'ALL_MODULES',
    'ALL_RUNNERS',
    'RegisteringMeta',
    'check_current_run',
    'register_model',
    'register_runner',
    'unregister_model',
    'unregister_runner',
]


_T = TypeVar('_T', bound=p.RunnerProtocol)


class RegisteringMeta(abc.ABCMeta):
    """Metaclass registering a runner after initialization."""

    def __call__(cls: type[_T], *args: Any, **kwargs: Any) -> _T:
        """Runner instantiation.

        Args:
            *args: the positional arguments of the class constructor.
            **kwargs: the keyword arguments of the class constructor.

        Returns:
            The instance.

        Side Effects:
            Registers the instance as a runner of its model.
        """
        instance: _T = type.__call__(cls, *args, **kwargs)
        register_runner(instance, instance.model)
        return instance


@dataclasses.dataclass(frozen=True)
class _Owner:
    run: weakref.ReferenceType[experimenting.Run[Any]]
    exp_name: str
    run_id: str


ALL_MODULES: Final = weakref.WeakKeyDictionary[torch.nn.Module, _Owner]()
ALL_RUNNERS: Final = weakref.WeakKeyDictionary[
    torch.nn.Module, weakref.WeakSet[Any]
]()


def check_current_run(model: p.ModelProtocol[Any, Any], /) -> None:
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


def register_model(model: p.ModelProtocol[Any, Any], /) -> None:
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

    ALL_MODULES[module] = _Owner(
        exp_name=run.experiment.name,
        run=weakref.ref(run),
        run_id=run.id,
    )
    run.metadata_manager.register_model(model)
    return


def register_runner(runner: Any, model: p.ModelProtocol[Any, Any], /) -> None:
    """Register a runner in the current run.

    Args:
        runner: the object to document.
        model: the model that the object runs.

    Raises:
        ModuleNotRegisteredError: if the module is not registered in the
            current experiment run.
    """
    check_current_run(model)
    run: experimenting.Run[Any] = experimenting.Experiment.get_current().run
    module = model.module

    runners = ALL_RUNNERS.setdefault(module, weakref.WeakSet())
    if runner not in runners:
        run.metadata_manager.register_runner(runner, model)
        runners.add(runner)

    return


def unregister_model(model: p.ModelProtocol[Any, Any], /) -> None:
    """Unregister a module and all its runners from the current experiment.

    Args:
        model: the model to register.
    """
    module = model.module
    owner = ALL_MODULES.pop(module, None)
    ALL_RUNNERS.pop(module, None)
    if owner is not None:
        run = owner.run()
        if run is not None:
            run.metadata_manager.unregister_model(model)

    return


def unregister_runner(runner: Any, /) -> None:
    """Unregister a runner from the current experiment.

    Args:
        runner: the object to document.
    """
    try:
        run: experimenting.Run[Any] = experimenting.Experiment.get_current().run
        run.metadata_manager.unregister_runner(runner)
    except exceptions.NoActiveExperimentError:
        pass

    for runner_set in ALL_RUNNERS.values():
        runner_set.discard(runner)

    return
