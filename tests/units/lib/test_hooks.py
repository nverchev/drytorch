"""Tests for the ``hooks`` module."""

import pytest

from drytorch.core import exceptions
from drytorch.lib import schedulers
from drytorch.lib.hooks import (
    CallEvery,
    EarlyStoppingCallback,
    EvaluationHook,
    HookChain,
    PruneCallback,
    ReduceLROnPlateau,
    RestartScheduleOnPlateau,
    SavingHook,
    TrainerHook,
)


class _CountingHook(TrainerHook):
    """Hook counting how many times a trainer called it."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.calls = 0
        return

    def __call__(self) -> None:
        """Count the call."""
        self.calls += 1
        return


class TestHookChain:
    """Tests for HookChain class."""

    def test_execute_in_order(self, mocker) -> None:
        """Test the hooks are executed in the given order."""
        calls: list[int] = []
        first = mocker.Mock(side_effect=lambda: calls.append(1))
        second = mocker.Mock(side_effect=lambda: calls.append(2))
        chain = HookChain([first, second])

        chain.execute()

        assert calls == [1, 2]

    def test_binds_its_hooks(self, mocker, mock_trainer) -> None:
        """Test binding the chain binds every hook."""
        hook = mocker.Mock()
        chain = HookChain([hook])

        chain.bind_trainer(mock_trainer)

        hook.bind_trainer.assert_called_once_with(mock_trainer)

    def test_hooks_copied(self, mocker) -> None:
        """Test changing the given hooks does not change the chain."""
        hook_list = [mocker.Mock()]
        chain = HookChain(hook_list)

        hook_list.append(mocker.Mock())

        assert len(chain.hooks) == 1


def test_saving_hook(mock_trainer) -> None:
    """Test the saving hook saves the checkpoint of its trainer."""
    hook = SavingHook()
    hook.bind_trainer(mock_trainer)

    hook()

    mock_trainer.save_checkpoint.assert_called_once()


def test_hook_not_bound() -> None:
    """Test calling a hook before a trainer binds it raises."""
    hook = SavingHook()

    with pytest.raises(exceptions.HookNotBoundError):
        hook()


def test_hook_bound_twice(mocker, mock_trainer) -> None:
    """Test binding a hook to a second trainer raises."""
    hook = SavingHook()
    hook.bind_trainer(mock_trainer)

    with pytest.raises(exceptions.HookAlreadyBoundError):
        hook.bind_trainer(mocker.Mock())


def test_evaluation_hook(mocker, mock_trainer) -> None:
    """Test the hook runs the evaluation it holds."""
    evaluation = mocker.Mock()
    hook = EvaluationHook(evaluation)
    hook.bind_trainer(mock_trainer)

    hook()

    evaluation.assert_called_once()


@pytest.mark.parametrize(
    ('epoch', 'called'), [(0, False), (4, False), (6, True)]
)
def test_periodic_hook(mocker, mock_trainer, epoch, called) -> None:
    """Test the hook is called on the epochs matching the interval."""
    mock_hook = mocker.MagicMock()
    hook = CallEvery(mock_hook, interval=3, start=3)
    mock_trainer.model.epoch = epoch
    hook.bind_trainer(mock_trainer)

    hook()

    assert mock_hook.called is called


def test_periodic_hook_terminated(mocker, mock_trainer) -> None:
    """Test the hook is called when the training is terminated."""
    mock_hook = mocker.MagicMock()
    hook = CallEvery(mock_hook, interval=3, start=3)
    mock_trainer.terminate_training('This is a test.')
    hook.bind_trainer(mock_trainer)

    hook()

    mock_hook.assert_called_once_with()


class TestCallEvery:
    """Tests for the CallEvery class."""

    def test_records_the_schedule(self) -> None:
        """Test the interval and the start are recorded with the hook."""
        wrapped = _CountingHook()

        hook = CallEvery(wrapped, interval=2, start=3)

        assert hook.hook is wrapped
        assert hook._get_name() == '_CountingHook'
        assert hook._get_parameters() == {'interval': 2, 'start': 3}

    @pytest.mark.parametrize(('interval', 'start'), [(0, 0), (1, -1)])
    def test_invalid_arguments(self, mocker, interval, start) -> None:
        """Test a non-positive interval or a negative start raise."""
        with pytest.raises(ValueError):
            CallEvery(mocker.Mock(), interval=interval, start=start)


def test_evaluation_runs_periodically(mocker, mock_trainer) -> None:
    """Test a scheduled evaluation runs every given number of epochs."""
    evaluation = mocker.Mock()
    hook = CallEvery(EvaluationHook(evaluation), interval=2)
    hook.bind_trainer(mock_trainer)

    for epoch in (1, 2, 3):
        mock_trainer.model.epoch = epoch
        hook()

    evaluation.assert_called_once()


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    @pytest.fixture()
    def callback(self, example_loss_name) -> EarlyStoppingCallback:
        """Set up a test instance."""
        return EarlyStoppingCallback(
            metric=example_loss_name,
            best_is='higher',
            patience=2,
        )

    def test_early_epoch_no_stop(self, mock_trainer, callback) -> None:
        """Test training continues if not enough epochs passed."""
        mock_trainer.model.epoch = 1

        callback.bind_trainer(mock_trainer)

        callback()

        mock_trainer.terminate_training.assert_not_called()

    def test_stops_on_plateau(self, mock_trainer, callback) -> None:
        """Test training stops after a plateau."""
        callback.bind_trainer(mock_trainer)

        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback()

        mock_trainer.terminate_training.assert_called_once()


class TestPruneCallback:
    """Tests for PruneCallback."""

    @pytest.fixture()
    def simple_pruning(self) -> dict[int, float]:
        """Set up a simple pruning instance."""
        return {3: 2, 5: 0.5}

    @pytest.fixture()
    def callback(self, example_loss_name, simple_pruning) -> PruneCallback:
        """Set up a test instance."""
        return PruneCallback(
            thresholds=simple_pruning,
            metric=example_loss_name,
            best_is='higher',
        )

    def test_no_pruning_before_threshold(self, mock_trainer, callback) -> None:
        """Test no pruning before the defined epoch."""
        mock_trainer.model.epoch = 2

        callback.bind_trainer(mock_trainer)

        callback()

        mock_trainer.terminate_training.assert_not_called()

    def test_prunes_at_threshold(self, mock_trainer, callback) -> None:
        """Test pruning occurs when the threshold condition is met."""
        mock_trainer.model.epoch = 5

        callback.bind_trainer(mock_trainer)

        callback()

        mock_trainer.terminate_training.assert_called_once()


class TestReduceLROnPlateau:
    """Tests for ReduceLROnPlateau."""

    @pytest.fixture()
    def callback(self, example_loss_name) -> ReduceLROnPlateau:
        """Set up a test instance."""
        return ReduceLROnPlateau(
            metric=example_loss_name,
            best_is='higher',
            cooldown=1,
            factor=0.01,
            patience=2,
        )

    def test_reduces_lr(self, mocker, mock_trainer, callback) -> None:
        """Test LR reduction after plateau patience is reached."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_schema = mocker.Mock
        mock_trainer.learning_schema.scheduler = scheduler

        callback.bind_trainer(mock_trainer)

        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback()

        mock_trainer.update_learning_rate.assert_called_once()

    def test_respects_cooldown(self, mocker, mock_trainer, callback) -> None:
        """Test LR update is suppressed during cooldown."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_schema = mocker.Mock
        mock_trainer.learning_schema.scheduler = scheduler

        callback.bind_trainer(mock_trainer)

        for _ in range(callback.monitor.metric_tracker.patience + 2):
            callback()

        mock_trainer.update_learning_rate.assert_called_once()


class TestRestartScheduleOnPlateau:
    """Tests for RestartScheduleOnPlateau."""

    @pytest.fixture()
    def callback(self, example_loss_name) -> RestartScheduleOnPlateau:
        """Set up a test instance."""
        return RestartScheduleOnPlateau(
            metric=example_loss_name, best_is='higher', cooldown=1, patience=2
        )

    def test_restarts_schedule_on_plateau(
        self, mocker, mock_trainer, callback
    ) -> None:
        """Test learning schedule restart after plateau."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_schema = mocker.Mock
        mock_trainer.learning_schema.scheduler = scheduler

        callback.bind_trainer(mock_trainer)

        for _ in range(callback.monitor.metric_tracker.patience + 2):
            callback()

        args = mock_trainer.update_learning_rate.call_args

        mock_trainer.update_learning_rate.assert_called_once()
        assert isinstance(args[1]['scheduler'], schedulers.AbstractScheduler)
