"""Tests for the "registering" module."""

import weakref

from typing import Any, Protocol

import typing_extensions

from typing_extensions import override

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.core.registering import (
    ALL_ACTORS,
    ALL_MODULES,
    RegisteringMeta,
    _Owner,
    register_actor,
    register_model,
    unregister_actor,
    unregister_model,
)


@pytest.fixture(autouse=True, scope='module')
def setup_module(module_mocker, mock_experiment, mock_run) -> None:
    """Fixture for a mock experiment."""
    mock_experiment.run = mock_run
    module_mocker.patch(
        'drytorch.Experiment.get_current', return_value=mock_experiment
    )
    return


class _SimpleCaller:
    name = 'simple_caller'


def test_register_model(mock_run, mock_model) -> None:
    """Test successful model registration."""
    manager = mock_run.metadata_manager
    register_model(mock_model)
    manager.register_model.assert_called_once_with(mock_model)
    assert mock_model.module in ALL_MODULES
    assert ALL_MODULES[mock_model.module].run() == mock_run


def test_register_model_with_existing_module(mock_run, mock_model) -> None:
    """Test successful model registration."""
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(mock_run), mock_run.experiment.name, mock_run.id
    )
    with pytest.raises(exceptions.ModuleAlreadyRegisteredError):
        register_model(mock_model)


def test_register_actor(mock_run, mock_model) -> None:
    """Test a successful actor registration."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(mock_run), mock_run.experiment.name, mock_run.id
    )
    register_actor(caller, mock_model)
    manager.register_actor.assert_called_once_with(caller, mock_model)
    assert caller in ALL_ACTORS[mock_model.module]


def test_register_actor_with_wrong_experiment(
    mocker, mock_run, mock_model
) -> None:
    """Test error if registering an actor on a model from another experiment."""
    other_experiment = mocker.Mock()
    other_experiment.experiment.name = 'other'
    other_experiment.id = 'other_id'
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(other_experiment), 'other', 'other_id'
    )
    with pytest.raises(exceptions.ModuleFromAnotherRunError):
        register_actor(_SimpleCaller(), mock_model)


def test_unregister_model(mock_run, mock_model) -> None:
    """Test successful model unregistration."""
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(mock_run), mock_run.experiment.name, mock_run.id
    )
    manager = mock_run.metadata_manager
    unregister_model(mock_model)
    manager.unregister_model.assert_called_once_with(mock_model)
    assert mock_model.module not in ALL_MODULES


def test_unregister_actor(mock_run) -> None:
    """Test successful actor unregistration."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager
    unregister_actor(caller)
    manager.unregister_actor.assert_called_once_with(caller)
    assert not any(caller in actors for actors in ALL_ACTORS.values())


class _TypingProtocol(Protocol):
    """Protocol from typing."""

    def first(self) -> None:
        """First method."""


class _ExtensionsProtocol(typing_extensions.Protocol):
    """Protocol from typing_extensions."""

    def second(self) -> None:
        """Second method."""


class _Actor(metaclass=RegisteringMeta):
    """Actor of a model."""

    model: p.ModelProtocol[Any, Any]

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        self.model = model
        return


class _ExtendedActor(_Actor):
    """Actor setting an attribute after the parent initialization."""

    extra: str

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        super().__init__(model)
        self.extra = 'extra'
        return


class _InheritedActor(_ExtendedActor):
    """Actor without its own initialization."""


class _FailingActor(_Actor):
    """Actor whose initialization fails after the parent initialization."""

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        super().__init__(model)
        raise ValueError('failing initialization')


class _ProtocolActor(_Actor, _TypingProtocol, _ExtensionsProtocol):
    """Actor inheriting protocols from typing and typing_extensions."""

    @override
    def first(self) -> None:
        return

    @override
    def second(self) -> None:
        return


class TestRegisteringMeta:
    """Tests for the metaclass registering the actors."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.registered_attributes: list[set[str]] = []
        self.mock_register_actor = mocker.patch(
            'drytorch.core.registering.register_actor',
            side_effect=lambda actor, _: self.registered_attributes.append(
                set(vars(actor))
            ),
        )
        return

    def test_registers_after_subclass_init(self, mock_model) -> None:
        """Test the actor registers once, after the subclass initialization."""
        actor = _ExtendedActor(mock_model)

        self.mock_register_actor.assert_called_once_with(actor, mock_model)
        assert 'extra' in self.registered_attributes[0]

    def test_registers_inherited_init(self, mock_model) -> None:
        """Test a subclass without initialization registers once."""
        actor = _InheritedActor(mock_model)

        self.mock_register_actor.assert_called_once_with(actor, mock_model)

    def test_failed_init_does_not_register(self, mock_model) -> None:
        """Test an actor is not registered when its initialization fails."""
        with pytest.raises(ValueError):
            _FailingActor(mock_model)

        self.mock_register_actor.assert_not_called()

    def test_protocol_inheritance(self, mock_model) -> None:
        """Test an actor inheriting typing and typing_extensions protocols."""
        actor = _ProtocolActor(mock_model)

        self.mock_register_actor.assert_called_once_with(actor, mock_model)
