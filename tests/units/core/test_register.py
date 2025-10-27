"""Tests for the "register" module."""

import pytest

from drytorch.core import exceptions
from drytorch.core.register import (
    ALL_ACTORS,
    ALL_MODULES,
    register_actor,
    register_model,
    unregister_actor,
    unregister_model,
)


@pytest.fixture(autouse=True, scope='module')
def setup_module(
    session_mocker, tmpdir_factory, mock_experiment, mock_run
) -> None:
    """Fixture for a mock experiment."""
    mock_experiment.run = mock_run
    session_mocker.patch(
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
    assert id(mock_model.module) in ALL_MODULES
    assert ALL_MODULES[id(mock_model.module)] == mock_run


def test_register_model_with_existing_module(mock_run, mock_model) -> None:
    """Test successful model registration."""
    ALL_MODULES[id(mock_model.module)] = mock_run
    with pytest.raises(exceptions.ModuleAlreadyRegisteredError):
        register_model(mock_model)


def test_register_actor(mock_run, mock_model) -> None:
    """Test a successful actor registration."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager
    ALL_MODULES[id(mock_model.module)] = mock_run
    register_actor(caller, mock_model)
    manager.register_actor.assert_called_once_with(caller, mock_model)
    assert id(caller) in ALL_ACTORS[id(mock_model.module)]


def test_register_actor_with_wrong_experiment(
    mocker, mock_run, mock_model
) -> None:
    """Test error if registering an actor on a model from another experiment."""
    other_experiment = mocker.Mock()
    ALL_MODULES[id(mock_model.module)] = other_experiment
    with pytest.raises(exceptions.ModuleNotRegisteredError):
        register_actor(_SimpleCaller(), mock_model)


def test_unregister_model(mock_run, mock_model) -> None:
    """Test successful model unregistration."""
    ALL_MODULES[id(mock_model.module)] = mock_run
    manager = mock_run.metadata_manager
    unregister_model(mock_model)
    manager.unregister_model.assert_called_once_with(mock_model)
    assert id(mock_model.module) not in ALL_MODULES


def test_unregister_actor(mock_run) -> None:
    """Test successful actor unregistration."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager
    unregister_actor(caller)
    manager.unregister_actor.assert_called_once_with(caller)
    assert id(caller) not in set().union(*ALL_ACTORS.values())
