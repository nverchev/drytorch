"""Tests for the hooks module."""

import pytest

from typing import Any

from src.dry_torch import exceptions
from src.dry_torch.hooks import HookRegistry, saving_hook, static_hook
from src.dry_torch.hooks import static_hook_closure, call_every
from src.dry_torch.hooks import EarlyStoppingCallback

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
    hook = saving_hook()

    hook(mock_trainer)
    mock_trainer.save_checkpoint.assert_called_once()  # type: ignore


def test_static_hook(mocker) -> None:
    """Test that static_hook wraps a static callable."""
    mock_callable = mocker.MagicMock()
    hook = static_hook(mock_callable)

    hook(mocker.MagicMock())  # Pass any instance

    mock_callable.assert_called_once()


def test_docs():
    def do_nothing_hook() -> None:
        """Test docs."""
        pass

    hook = static_hook(do_nothing_hook)

    assert hook.__doc__ == 'Test docs.'


def test_static_hook_closure(mocker) -> None:
    """Test that static_hook_closure returns a static hook closure."""
    mock_callable = mocker.MagicMock(return_value=mocker.MagicMock())
    hook_closure = static_hook_closure(mock_callable)

    hook = hook_closure(arg1=10)
    hook(mocker.MagicMock())  # Pass any instance

    mock_callable.assert_called_once_with(arg1=10)
    mock_callable.return_value.assert_called_once()


def test_call_every(mocker, mock_trainer) -> None:
    """Test call_every executes the hook based on interval and trainer state."""
    mock_hook = mocker.MagicMock()
    hook = call_every(interval=2, hook=mock_hook)

    # Test when epoch is divisible by interval
    mock_trainer.model.epoch = 4
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)

    # Test when epoch is not divisible by interval
    mock_hook.reset_mock()
    mock_trainer.model.epoch = 3
    hook(mock_trainer)
    mock_hook.assert_not_called()

    # Test when trainer is terminated
    mock_trainer.terminate_training()
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)


class TestEarlyStoppingCallback:
    patience = 2

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the hook."""
        self.hook = EarlyStoppingCallback(metric=Criterion,
                                          patience=self.patience,
                                          start_from_epoch=0,
                                          min_delta=.01)

    def test_best_is_when_no_call(self, mock_trainer) -> None:
        """Test default aggregation functions."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = self.hook.best_result

    def test_default_aggregate_fn(self, mock_trainer) -> None:
        """Test default aggregation functions."""
        self.hook._best_is = 'higher'
        assert self.hook.aggregate_fn == max
        self.hook._aggregate_fn = None  # reset aggregate
        self.hook._best_is = 'lower'
        assert self.hook.aggregate_fn == min

    def test_trainer_evaluation(self, mock_trainer) -> None:
        """Test that it monitors training."""
        self.hook(mock_trainer)
        assert self.hook.best_result == 1

    def test_validation(self,
                        mocker,
                        mock_validation) -> None:
        """Test that it monitors validation."""
        hook = EarlyStoppingCallback(metric=Accuracy)
        trainer = mocker.Mock()
        trainer.validation = mock_validation
        trainer.model.epoch = 2
        hook(trainer)
        assert hook.best_result == 0.5

    def test_error_when_metric_not_found(self,
                                         mock_trainer,
                                         mock_validation) -> None:
        """Test that it raises MetricNotFoundError."""
        self.hook._metric_name = 'not_existing'
        with pytest.raises(exceptions.MetricNotFoundError):
            self.hook(mock_trainer)

    def test_update_best_result(self, mock_trainer) -> None:
        """Test that it updates best_result."""
        self.hook._best_is = 'higher'

        self.hook._monitor_log.append(0.)

        for _ in range(self.patience):
            self.hook(mock_trainer)

        assert self.hook.best_result == 1.

        self.hook._monitor_log.append(2.)
        self.hook(mock_trainer)

        assert self.hook.best_result == 2.

    def test_patience(self, mock_trainer) -> None:
        """Test that it does not terminate while patient."""
        self.hook._best_is = 'lower'
        self.hook._monitor_log.extend([2., 0.])

        for _ in range(self.patience - 1):
            self.hook._monitor_log.append(1.)

        self.hook(mock_trainer)

        mock_trainer.terminate_training.assert_not_called()  # type: ignore

        self.hook._monitor_log.clear()
        self.hook._monitor_log.extend([2., 0.])

        for _ in range(self.patience):
            self.hook._monitor_log.append(1.)

        self.hook(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore

    def test_pruning(self, mock_trainer) -> None:
        """Test that it prunes as indicated."""
        self.hook._best_is = 'lower'
        self.hook._pruning = {3: 0}
        self.hook._monitor_log.extend([2.] * self.patience)

        mock_trainer.model.epoch = 2
        self.hook(mock_trainer)
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

        self.hook._monitor_log.clear()
        self.hook._monitor_log.extend([2.] * self.patience)

        mock_trainer.model.epoch = 3
        self.hook(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore
