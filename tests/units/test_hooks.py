"""Tests for the "hooks" module."""

from typing import Any

import pytest

from drytorch import schedulers
from drytorch.hooks import (
    EarlyStoppingCallback,
    HookRegistry,
    PruneCallback,
    ReduceLROnPlateau,
    RestartScheduleOnPlateau,
    StaticHook,
    call_every,
    saving_hook,
    static_hook_class,
)


Accuracy = 'Accuracy'
Criterion = 'Loss'


class TestHookRegistry:
    """Tests for HookRegistry class."""

    @pytest.fixture
    def registry(self) -> HookRegistry[Any]:
        """Set up the instance."""
        return HookRegistry()

    def test_register_single_hook(self, mocker, registry) -> None:
        """Test that a single hook can be registered and executed."""
        mock_hook = mocker.MagicMock()
        registry.register(mock_hook)
        registry.execute(mocker.MagicMock())  # Pass any instance
        mock_hook.assert_called_once()

    def test_register_all_hooks(self, mocker, registry) -> None:
        """Test that multiple hooks can be registered and executed in order."""
        mock_hook1 = mocker.MagicMock()
        mock_hook2 = mocker.MagicMock()
        registry.register_all([mock_hook1, mock_hook2])
        registry.execute(mocker.MagicMock())
        mock_hook1.assert_called_once()
        mock_hook2.assert_called_once()


def test_saving_hook(mock_trainer) -> None:
    """Test that saving_hook calls save_checkpoint on the instance."""
    hook = saving_hook
    hook(mock_trainer)
    mock_trainer.save_checkpoint.assert_called_once()  # type: ignore


def test_static_hook(mocker) -> None:
    """Test that static_hook wraps a void callable."""
    mock_callable = mocker.MagicMock()
    hook = StaticHook(mock_callable)
    hook(mocker.MagicMock())
    mock_callable.assert_called_once()


def test_static_class(mocker, mock_trainer) -> None:
    """Test that static_hook_class creates a callable hook."""
    mock_event = mocker.MagicMock()

    class _TestClass:
        def __init__(self, text: str, number: int = 1):
            self.text = text
            self.number = number

        def __call__(self) -> None:
            mock_event()

    hook = static_hook_class(_TestClass)('test')
    hook(mock_trainer)
    mock_event.assert_called_once()


def test_call_every(mocker, mock_trainer) -> None:
    """Test call_every executes the hook based on interval and trainer state."""
    mock_hook = mocker.MagicMock()
    hook = call_every(start=3, interval=3)(mock_hook)

    mock_hook.reset_mock()
    mock_trainer.model.epoch = 0
    hook(mock_trainer)
    mock_hook.assert_not_called()

    mock_hook.reset_mock()
    mock_trainer.model.epoch = 4
    hook(mock_trainer)
    mock_hook.assert_not_called()

    mock_trainer.model.epoch = 6
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)

    mock_hook.reset_mock()
    mock_trainer.terminate_training('This is a test.')
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)

class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    @pytest.fixture()
    def callback(self, mock_metric) -> EarlyStoppingCallback:
        """Set up a test instance."""
        return EarlyStoppingCallback(
            metric=mock_metric,
            patience=2,
        )

    def test_early_epoch_no_stop(self, mock_trainer, callback) -> None:
        """Test training continues if not enough epochs passed."""
        mock_trainer.model.epoch = 1
        callback(mock_trainer)
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_stops_on_plateau(self, mock_trainer, callback) -> None:
        """Test training stops after plateau."""
        mock_trainer.validation.objective.higher_is_better = True
        for _ in range(callback.monitor.patience + 1):
            callback(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore


class TestPruneCallback:
    """Tests for PruneCallback."""

    @pytest.fixture()
    def simple_pruning(self) -> dict[int, float]:
        """Set up a simple pruning instance."""
        return {3: 2, 5: 0.5}

    @pytest.fixture()
    def callback(self, mock_metric, simple_pruning) -> PruneCallback:
        """Set up a test instance."""
        return PruneCallback(
            thresholds=simple_pruning, metric=mock_metric, best_is='higher'
        )

    def test_no_pruning_before_threshold(self, mock_trainer, callback) -> None:
        """Test no pruning before defined epoch."""
        mock_trainer.model.epoch = 2
        callback(mock_trainer)
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_prunes_at_threshold(self, mock_trainer, callback) -> None:
        """Test pruning occurs when threshold condition is met."""
        mock_trainer.model.epoch = 5
        callback(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore


class TestReduceLROnPlateau:
    """Tests for ReduceLROnPlateau."""

    @pytest.fixture()
    def callback(self, mock_metric) -> ReduceLROnPlateau:
        """Set up a test instance."""
        return ReduceLROnPlateau(
            metric=mock_metric, patience=2, factor=0.01, cooldown=1
        )

    def test_reduces_lr_and_respects_cooldown(
        self, mocker, mock_trainer, callback
    ) -> None:
        """Test LR reduction and cooldown enforcement."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_scheme = mocker.Mock
        mock_trainer.learning_scheme.scheduler = scheduler

        for _ in range(callback.monitor.patience + 1):
            callback(mock_trainer)

        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore

        callback(mock_trainer)
        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore


class TestRestartScheduleOnPlateau:
    """Tests for RestartScheduleOnPlateau."""

    @pytest.fixture()
    def callback(self) -> RestartScheduleOnPlateau:
        """Set up a test instance."""
        return RestartScheduleOnPlateau(
            metric='mock_metric', patience=2, cooldown=1
        )

    def test_restarts_schedule_on_plateau(
        self, mocker, mock_trainer, callback
    ) -> None:
        """Test learning schedule restart after plateau."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_scheme = mocker.Mock
        mock_trainer.learning_scheme.scheduler = scheduler

        for _ in range(callback.monitor.patience + 2):
            callback(mock_trainer)

        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore
        args = mock_trainer.update_learning_rate.call_args  # type: ignore
        assert isinstance(args[1]['scheduler'], schedulers.WarmupScheduler)
