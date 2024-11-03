import pytest

from typing import Any

from src.dry_torch import exceptions
from src.dry_torch.hooks import HookRegistry, saving_hook, static_hook
from src.dry_torch.hooks import static_hook_closure, call_every
from src.dry_torch.hooks import EarlyStoppingCallback


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
    # noinspection PyUnresolvedReferences
    mock_trainer.save_checkpoint.assert_called_once()


def test_static_hook(mocker) -> None:
    """Test that static_hook wraps a static callable."""
    mock_callable = mocker.MagicMock()
    hook = static_hook(mock_callable)

    hook(mocker.MagicMock())  # Pass any instance

    mock_callable.assert_called_once()


def test_static_hook_closure(mocker) -> None:
    """Test that static_hook_closure returns a static hook closure."""
    mock_callable = mocker.MagicMock(return_value=mocker.MagicMock())
    hook_closure = static_hook_closure(mock_callable)

    hook = hook_closure(arg1=10)
    hook(mocker.MagicMock())  # Pass any instance

    mock_callable.assert_called_once_with(arg1=10)
    mock_callable.return_value.assert_called_once()


# noinspection PyPropertyAccess
def test_call_every(mocker, mock_trainer) -> None:
    """Test call_every executes the hook based on interval and trainer state."""
    mock_hook = mocker.MagicMock()
    hook = call_every(interval=2, hook=mock_hook, start=0)

    mock_trainer.terminated = False

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
    mock_trainer.terminated = True
    hook(mock_trainer)
    mock_hook.assert_called_once_with(mock_trainer)


class TestEarlyStoppingCallback:

    @pytest.fixture(autouse=True)
    def setup(self,
              mock_model,
              mock_learning_scheme,
              mock_loss_calculator,
              mock_loader):
        self.hook = EarlyStoppingCallback(
            metric_name='accuracy',
            monitor_validation=False,
            patience=5,
            best_is='higher'
        )

    def test_validation(self,
                        mock_trainer,
                        mock_validation) -> None:
        """Test that early_stopping_callback monitors validation."""
        # noinspection PyPropertyAccess
        mock_validation.metrics = {'accuracy': 1.0}
        mock_trainer.validation = mock_validation
        self.hook.monitor_validation = True

        self.hook(mock_trainer)

    def test_error_when_no_validation(self, mock_trainer) -> None:
        """Test that early_stopping_callback raises NoValidationError"""
        self.hook.monitor_validation = True
        mock_trainer.validation = None

        with pytest.raises(exceptions.NoValidationError):
            self.hook(mock_trainer)

    def test_error_when_metric_not_found(self,
                                         mock_trainer,
                                         mock_validation) -> None:
        """Test that early_stopping_callback raises MetricNotFoundError"""
        mock_trainer.validation = mock_validation
        with pytest.raises(exceptions.MetricNotFoundError):
            self.hook(mock_trainer)

    def test_pruning(self, mocker, mock_trainer) -> None:
        """Test that early_stopping_callback prunes as indicated."""

        self.hook.pruning = {15: 0.95}
        mock_terminated_event = mocker.patch(
            'src.dry_torch.events.TerminatedTraining'
        )

        # noinspection PyPropertyAccess
        mock_trainer.metrics = {'accuracy': 0.9}

        mock_trainer.model.epoch = 14
        self.hook(mock_trainer)

        # noinspection PyUnresolvedReferences
        mock_trainer.terminate_training.assert_not_called()
        mock_terminated_event.assert_not_called()

        mock_trainer.model.epoch = 15
        self.hook(mock_trainer)

        # noinspection PyUnresolvedReferences
        mock_trainer.terminate_training.assert_called_once()
        mock_terminated_event.assert_called_once_with(
            mock_trainer.model.epoch, 'early stopping'
        )
        assert self.hook.pruning_results[15] == 0.9

    def test_update_best_result(self, mock_trainer) -> None:
        """Test that early_stopping_callback updates best_result"""

        mock_trainer.model.epoch = 12
        # noinspection PyPropertyAccess
        mock_trainer.metrics = {'accuracy': 0.2}

        self.hook(mock_trainer)

        assert self.hook.best_result == 0.2

        # Second call with improved metric
        mock_trainer.metrics['accuracy'] = 0.45

        self.hook(mock_trainer)

        assert self.hook.best_result == 0.45

    def test_patience(self, mock_trainer) -> None:
        """Test that early_stopping_callback does not terminate."""
        # noinspection PyPropertyAccess

        mock_trainer.metrics = {'accuracy': 0.2}
        mock_trainer.model.epoch = 15

        self.hook(mock_trainer)  # First call sets the baseline

        # noinspection PyUnresolvedReferences
        mock_trainer.terminate_training.assert_not_called()

        # Call without improvement
        mock_trainer.metrics['accuracy'] = 0.1
        self.hook(mock_trainer)

        # noinspection PyUnresolvedReferences
        mock_trainer.terminate_training.assert_not_called()
