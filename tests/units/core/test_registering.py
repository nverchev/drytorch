"""Tests for the "registering" module."""

import pytest

from drytorch.core import exceptions
from drytorch.core.registering import (
    ALL_MODULES,
    register_model,
    register_source)
from tests.units.core.conftest import mock_experiment


@pytest.fixture(autouse=True, scope='module')
def setup_module(session_mocker, tmpdir_factory, mock_experiment) -> None:
    """Fixture for a mock experiment."""
    session_mocker.patch('drytorch.Experiment.current',
                         return_value=mock_experiment)
    return


class _SimpleCaller:
    name = "simple_caller"


def test_record_model_call(mock_experiment, mock_model) -> None:
    """Test a successful record model call."""
    caller = _SimpleCaller()
    manager = mock_experiment._metadata_manager
    # Monkey patch model registration
    ALL_MODULES[mock_model.module] = mock_experiment
    register_source(caller, mock_model)

    manager.register_source.assert_called_once_with(caller, mock_model)


def test_register_model(mock_experiment, mock_model) -> None:
    """Test successful model registration."""
    manager = mock_experiment._metadata_manager
    register_model(mock_model)

    manager.register_model.assert_called_once_with(mock_model)
    assert mock_model.module in ALL_MODULES
    assert ALL_MODULES[mock_model.module] == mock_experiment


def test_register_model_already_registered(mock_experiment, mock_model) -> None:
    """Test error when registering a model that is already registered."""
    ALL_MODULES[mock_model.module] = mock_experiment

    with pytest.raises(exceptions.ModuleAlreadyRegisteredError):
        register_model(mock_model)


def test_record_model_call_unregistered_model(mock_experiment,
                                              mock_model) -> None:
    """Test error when recording a call for an unregistered model."""
    with pytest.raises(exceptions.ModelNotRegisteredError):
        register_source(_SimpleCaller(), mock_model)


def test_record_model_call_wrong_experiment(mocker,
                                            mock_experiment,
                                            mock_model) -> None:
    """Test error when recording a call for a model from another experiment."""
    other_experiment = mocker.Mock()
    ALL_MODULES[mock_model.module] = other_experiment

    with pytest.raises(exceptions.ModelNotRegisteredError):
        register_source(_SimpleCaller(), mock_model)
