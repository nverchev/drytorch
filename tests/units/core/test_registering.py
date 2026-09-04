"""Tests for the "registering" module."""

import weakref

import pytest

from drytorch.core import exceptions
from drytorch.core.registering import (
    ALL_ACTORS,
    ALL_MODULES,
    _Owner,
    register_actor,
    register_model,
    unregister_actor,
    unregister_model,
)


@pytest.fixture(autouse=True, scope='module')
def setup_module(
    module_mocker, tmpdir_factory, mock_experiment, mock_run
) -> None:
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
