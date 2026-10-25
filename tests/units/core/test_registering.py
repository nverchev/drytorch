"""Tests for the ``registering`` module."""

import weakref

from collections.abc import Mapping
from typing import Any

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.core.registering import (
    ALL_MODULES,
    ALL_RUNNERS,
    RegisteringMeta,
    TestRegisteringMeta,
    _Owner,
    register_model,
    register_runner,
    register_test,
    unregister_model,
    unregister_runner,
)


@pytest.fixture(autouse=True, scope='module')
def setup_module(module_mocker, mock_experiment, mock_run) -> None:
    """Fixture for a mock experiment."""
    mock_experiment.run = mock_run
    module_mocker.patch(
        'drytorch.Experiment.get_current', return_value=mock_experiment
    )
    return


class _DummyObjective:
    """Dummy objective supporting weak references."""


class _SimpleCaller:
    computed_metrics: Mapping[str, float]
    last_run_epoch: int
    model: Any
    name: str = 'simple_caller'
    objective: Any

    def __init__(self, objective: Any = None) -> None:
        self.computed_metrics = {}
        self.last_run_epoch = 0
        self.model = None
        self.objective = objective or _DummyObjective()
        return

    def __call__(self, *, store_outputs: bool = False) -> None:
        return


class _Actor(metaclass=RegisteringMeta):
    """Runner of a model."""

    model: p.ModelProtocol[Any, Any]

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        self.model = model
        return


class _ExtendedActor(_Actor):
    """Runner setting an attribute after the parent initialization."""

    extra: str

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        super().__init__(model)
        self.extra = 'extra'
        return


class _InheritedActor(_ExtendedActor):
    """Runner without its own initialization."""


class _FailingActor(_Actor):
    """Runner whose initialization fails after the parent initialization."""

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        super().__init__(model)
        raise ValueError('failing initialization')


class _TestActor(metaclass=TestRegisteringMeta):
    """Test runner of a model."""

    computed_metrics: Mapping[str, float]
    last_run_epoch: int
    model: p.ModelProtocol[Any, Any]
    name: str = 'Test'
    objective: Any

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        self.computed_metrics = {}
        self.last_run_epoch = 0
        self.model = model
        self.objective = _DummyObjective()
        return

    def __call__(self, *, store_outputs: bool = False) -> None:
        return


class TestRegisteringMeta:
    """Tests for the metaclass registering the runners."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.registered_attributes: list[set[str]] = []
        self.mock_register_runner = mocker.patch(
            'drytorch.core.registering.register_runner',
            side_effect=lambda runner, _: self.registered_attributes.append(
                set(vars(runner))
            ),
        )
        return

    def test_registers_after_subclass_init(self, mock_model) -> None:
        """Test the runner registers once, after the subclass initialization."""
        runner = _ExtendedActor(mock_model)

        self.mock_register_runner.assert_called_once_with(runner, mock_model)
        assert 'extra' in self.registered_attributes[0]

    def test_registers_inherited_init(self, mock_model) -> None:
        """Test a subclass without initialization registers once."""
        runner = _InheritedActor(mock_model)

        self.mock_register_runner.assert_called_once_with(runner, mock_model)

    def test_failed_init_does_not_register(self, mock_model) -> None:
        """Test a runner is not registered when its initialization fails."""
        with pytest.raises(ValueError):
            _FailingActor(mock_model)

        self.mock_register_runner.assert_not_called()


class TestTestRegisteringMeta:
    """Tests for the metaclass registering test runners."""

    def test_registers_test(self, mocker, mock_model) -> None:
        """Test the runner registers through register_test."""
        mock_register = mocker.patch('drytorch.core.registering.register_test')
        runner = _TestActor(mock_model)

        mock_register.assert_called_once_with(runner, mock_model)


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


def test_register_runner(mock_run, mock_model) -> None:
    """Test a successful runner registration."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager
    manager.register_runner.reset_mock()
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(mock_run), mock_run.experiment.name, mock_run.id
    )

    register_runner(caller, mock_model)

    manager.register_runner.assert_called_once_with(caller, mock_model)
    assert caller in ALL_RUNNERS[mock_model.module]


def test_register_runner_module_not_registered(mock_model) -> None:
    """Test error if registering a runner on an unregistered module."""
    with pytest.raises(exceptions.ModuleNotRegisteredError):
        register_runner(_SimpleCaller(), mock_model)


def test_register_runner_no_current_run(mocker, mock_model) -> None:
    """Test error if registering a runner with no active experiment."""
    mocker.patch(
        'drytorch.Experiment.get_current',
        side_effect=exceptions.NoActiveExperimentError('mock'),
    )
    with pytest.raises(exceptions.NoActiveExperimentError):
        register_runner(_SimpleCaller(), mock_model)


def test_register_runner_with_wrong_experiment(
    mocker, mock_run, mock_model
) -> None:
    """Test error if registering a runner on a model from another experiment."""
    other_experiment = mocker.Mock()
    other_experiment.experiment.name = 'other'
    other_experiment.id = 'other_id'
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(other_experiment), 'other', 'other_id'
    )

    with pytest.raises(exceptions.ModuleFromAnotherRunError):
        register_runner(_SimpleCaller(), mock_model)


def test_register_test(mock_run, mock_model) -> None:
    """Test test registration through metadata manager."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager
    manager.register_test.reset_mock()
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(mock_run), mock_run.experiment.name, mock_run.id
    )

    register_test(caller, mock_model)

    manager.register_test.assert_called_once_with(caller, mock_model)
    assert caller in ALL_RUNNERS[mock_model.module]


def test_register_test_module_not_registered(mock_model) -> None:
    """Test error if registering a test on an unregistered module."""
    with pytest.raises(exceptions.ModuleNotRegisteredError):
        register_test(_SimpleCaller(), mock_model)


def test_register_test_no_current_run(mocker, mock_model) -> None:
    """Test error if registering a test with no active experiment."""
    mocker.patch(
        'drytorch.Experiment.get_current',
        side_effect=exceptions.NoActiveExperimentError('mock'),
    )
    with pytest.raises(exceptions.NoActiveExperimentError):
        register_test(_SimpleCaller(), mock_model)


def test_register_test_with_wrong_experiment(
    mocker, mock_run, mock_model
) -> None:
    """Test error if registering a test on a model from another experiment."""
    other_experiment = mocker.Mock()
    other_experiment.experiment.name = 'other'
    other_experiment.id = 'other_id'
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(other_experiment), 'other', 'other_id'
    )

    with pytest.raises(exceptions.ModuleFromAnotherRunError):
        register_test(_SimpleCaller(), mock_model)


def test_unregister_model(mock_run, mock_model) -> None:
    """Test successful model unregistration."""
    ALL_MODULES[mock_model.module] = _Owner(
        weakref.ref(mock_run), mock_run.experiment.name, mock_run.id
    )
    manager = mock_run.metadata_manager

    unregister_model(mock_model)

    manager.unregister_model.assert_called_once_with(mock_model)
    assert mock_model.module not in ALL_MODULES


def test_unregister_runner(mock_run) -> None:
    """Test successful runner unregistration."""
    caller = _SimpleCaller()
    manager = mock_run.metadata_manager

    unregister_runner(caller)

    manager.unregister_runner.assert_called_once_with(caller)
    assert not any(caller in runners for runners in ALL_RUNNERS.values())
