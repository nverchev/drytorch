"""Tests for the hooks module."""

import pytest

from typing import Any

from dry_torch import exceptions
from dry_torch import schedulers
from dry_torch.hooks import EarlyStoppingCallback, HookRegistry
from dry_torch.hooks import MetricMonitor, PruneCallback, ReduceLROnPlateau
from dry_torch.hooks import RestartScheduleOnPlateau, StaticHook
from dry_torch.hooks import call_every, saving_hook, static_hook_class

Accuracy = 'Accuracy'
Criterion = 'Loss'


class TestHookRegistry:
    def test_register_single_hook(self, mocker) -> None:
        """Test that a single hook can be registered and executed."""
        registry = HookRegistry[Any]()
        mock_hook = mocker.MagicMock()

        registry.register(mock_hook)
        registry.execute(mocker.MagicMock())  # Pass any instance

        mock_hook.assert_called_once()

    def test_register_all_hooks(self, mocker) -> None:
        """Test that multiple hooks can be registered and executed in order."""
        registry = HookRegistry[Any]()
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
    """Test that static_hook wraps a static callable."""
    mock_callable = mocker.MagicMock()
    hook = StaticHook(mock_callable)

    hook(mocker.MagicMock())  # Pass any instance

    mock_callable.assert_called_once()


def test_static_class(mocker, mock_trainer) -> None:
    """Test that static_hook wraps a static callable."""

    mock_event = mocker.MagicMock()

    class _TestClass:
        def __init__(self, text: str, number: int = 1):
            self.text = text
            self.number = number

        def __call__(self) -> None:
            mock_event()

    hook = static_hook_class(_TestClass)('test', number=1)
    hook(mock_trainer)
    mock_event.assert_called_once()


def test_call_every(mocker, mock_trainer) -> None:
    """Test call_every executes the hook based on interval and trainer state."""
    mock_hook = mocker.MagicMock()
    hook = call_every(start=3, interval=3)(mock_hook)

    # Test when epoch is before start
    mock_hook.reset_mock()
    mock_trainer.model.epoch = 0
    hook(mock_trainer)
    mock_hook.assert_not_called()

    # Test when epoch minus the start is not divisible by interval
    mock_hook.reset_mock()
    mock_trainer.model.epoch = 4
    hook(mock_trainer)
    mock_hook.assert_not_called()

    # Test when epoch minus the start is divisible by interval
    mock_trainer.model.epoch = 6
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)

    mock_hook.reset_mock()
    # Test when trainer is terminated
    mock_trainer.terminate_training('This is a test.')
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)


class TestMetricMonitor:
    """Test MetricMonitor class."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_metric):
        """Set up a MetricMonitor instance."""
        self.monitor = MetricMonitor(
            metric=mock_metric,
            min_delta=0.01,
            patience=2
        )

    def test_get_monitor(self, mock_trainer):
        """Test getting monitored values."""
        expected = mock_trainer.validation
        assert self.monitor._get_monitor(mock_trainer) == expected
        mock_trainer.validation = None
        expected = mock_trainer
        assert self.monitor._get_monitor(mock_trainer) == expected

    def test_init_with_string_metric(self):
        monitor = MetricMonitor(metric='test_metric')
        assert monitor.metric_name == 'test_metric'

    def test_init_with_metric_object(self, mock_metric):
        mock_metric.name = 'mock_metric'
        mock_metric.higher_is_better = True
        monitor = MetricMonitor(metric=mock_metric)
        assert monitor.metric_name == mock_metric.name
        assert monitor.best_is == 'higher'

    def test_negative_patience(self):
        """Test invalid patience."""
        with pytest.raises(ValueError):
            MetricMonitor(patience=-1)

    def test_best_result_not_available(self):
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = self.monitor.best_result

    def test_aggregate_fn_selection(self):
        assert self.monitor.average_fn([1, 2, 3]) == 3

    def test_is_improving_with_better_value(self):
        self.monitor.best_is = 'higher'
        self.monitor._warm_up = 0
        self.monitor._monitor_log.append(1.0)
        self.monitor._monitor_log.append(2.0)
        assert self.monitor.is_improving() is True

    def test_is_improving_with_worse_value(self):
        self.monitor.best_is = 'higher'
        self.monitor._warm_up = 0
        self.monitor._monitor_log.append(2.0)
        self.monitor._monitor_log.append(1.0)
        assert self.monitor.is_improving() is False

    def test_auto_best_is_determination(self):
        self.monitor.best_is = 'auto'
        self.monitor._warm_up = 0
        self.monitor._monitor_log.append(1.0)
        self.monitor._monitor_log.append(2.0)
        assert self.monitor.is_improving() is True
        assert self.monitor.best_is == 'higher'

    def test_improvement_with_tolerance(self):
        """Test improvement detection considering min_delta."""
        self.monitor.best_is = 'higher'
        self.monitor._warm_up = 0

        self.monitor._monitor_log.append(1.0)
        assert self.monitor.is_improving()

        # (1.009 is not > 1.0 + 0.01)
        self.monitor._monitor_log.append(1.009)
        assert not self.monitor.is_improving()

        # (1.011 is > 1.0 + 0.01)
        self.monitor._monitor_log.append(1.011)
        assert self.monitor.is_improving()


class TestEarlyStoppingCallback:
    @pytest.fixture(autouse=True)
    def setup(self, mock_metric):
        """Set up an EarlyStoppingCallback instance."""
        self.callback = EarlyStoppingCallback(
            metric=mock_metric,
            patience=2,
        )

    def test_early_epoch_no_stop(self, mock_trainer):
        mock_trainer.model.epoch = 1

        self.callback(mock_trainer)
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_stops_on_plateau(self, mock_trainer):
        mock_trainer.validation.objective.higher_is_better = True

        # best result + patience + terminate = 2 + patience
        for _ in range(self.callback.monitor.patience + 2):
            self.callback(mock_trainer)

        mock_trainer.terminate_training.assert_called_once()  # type: ignore


class TestPruneCallback:
    @pytest.fixture(autouse=True)
    def setup(self, mock_metric):
        """Set up PruneCallback instance."""
        self.pruning = {3: 1, 5: 2}
        self.callback = PruneCallback(
            pruning=self.pruning,
            metric=mock_metric,
        )

    def test_no_pruning_before_threshold(self, mock_trainer):
        mock_trainer.model.epoch = 2

        self.callback(mock_trainer)
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_prunes_at_threshold(self, mock_trainer):
        self.callback(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore


class TestReduceLROnPlateau:
    @pytest.fixture(autouse=True)
    def setup(self, mock_metric):
        """Set up a ReduceLROnPlateau instance."""
        self.callback = ReduceLROnPlateau(
            metric=mock_metric,
            patience=2,
            factor=0.01,
            cooldown=1
        )

    def test_reduces_lr_and_respects_cooldown(self, mocker, mock_trainer):
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_scheme = mocker.Mock
        mock_trainer.learning_scheme.scheduler = scheduler

        for _ in range(self.callback.monitor.patience + 2):
            self.callback(mock_trainer)

        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore

        self.callback(mock_trainer)
        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore


class TestRestartScheduleOnPlateau:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up a RestartScheduleOnPlateau instance."""
        self.callback = RestartScheduleOnPlateau(
            metric='mock_metric',
            patience=2,
            cooldown=1
        )

    def test_restarts_schedule_on_plateau(self, mocker, mock_trainer):
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_scheme = mocker.Mock
        mock_trainer.learning_scheme.scheduler = scheduler

        # best result + patience + terminate = 2 + patience
        for _ in range(self.callback.monitor.patience + 2):
            self.callback(mock_trainer)

        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore
        # Verify that new scheduler is WarmupScheduler
        args = mock_trainer.update_learning_rate.call_args  # type: ignore
        assert isinstance(args[1]["scheduler"], schedulers.WarmupScheduler)
