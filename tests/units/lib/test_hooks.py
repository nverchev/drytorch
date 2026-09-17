"""Tests for the ``hooks`` module."""

import pickle

import pytest

from drytorch.lib import schedulers
from drytorch.lib.hooks import (
    CallEvery,
    EarlyStoppingCallback,
    Hook,
    HookChain,
    PruneCallback,
    ReduceLROnPlateau,
    RestartScheduleOnPlateau,
    StaticHook,
    call_every,
    evaluation_hook,
    saving_hook,
    static_hook_class,
)


class _TestClass:
    def __init__(self, text: str, number: int = 1):
        self.text = text
        self.number = number
        return

    def __call__(self) -> None:
        return


def _record_trainer(trainer) -> None:
    _not_used = trainer
    return


class TestHookChain:
    """Tests for HookChain class."""

    def test_execute_in_order(self, mocker) -> None:
        """Test the hooks are executed in the given order."""
        calls: list[int] = []
        chain = HookChain(
            [lambda _: calls.append(1), lambda _: calls.append(2)]
        )

        chain.execute(mocker.Mock())

        assert calls == [1, 2]

    def test_hooks_copied(self, mocker) -> None:
        """Test changing the given hooks does not change the chain."""
        hook_list = [mocker.Mock()]
        chain = HookChain(hook_list)

        hook_list.append(mocker.Mock())

        assert len(chain.hooks) == 1


def test_saving_hook(mock_trainer) -> None:
    """Test that saving_hook calls save_checkpoint on the instance."""
    hook = saving_hook

    hook(mock_trainer)

    mock_trainer.save_checkpoint.assert_called_once()


def test_static_hook(mocker) -> None:
    """Test that static_hook wraps a void callable."""
    mock_callable = mocker.MagicMock()
    hook = StaticHook(mock_callable)

    hook(mocker.MagicMock())

    mock_callable.assert_called_once()


def test_static_class(mocker, mock_trainer) -> None:
    """Test that static_hook_class creates a callable hook."""
    mock_event = mocker.MagicMock()
    hook = static_hook_class(_TestClass)('test')
    mocker.patch.object(_TestClass, '__call__', mock_event)

    hook(mock_trainer)

    mock_event.assert_called_once()


def test_static_hook_pickleable(mocker, mock_trainer) -> None:
    """Test that static_hook_class produces pickleable hooks."""
    mock_event = mocker.MagicMock()
    mocker.patch.object(_TestClass, '__call__', mock_event)

    original_hook = static_hook_class(_TestClass)('test', number=42)
    reconstructed_hook = pickle.loads(pickle.dumps(original_hook))  # noqa: S301

    reconstructed_hook(mock_trainer)

    mock_event.assert_called_once()


@pytest.mark.parametrize(
    ('epoch', 'called'), [(0, False), (4, False), (6, True)]
)
def test_call_every(mocker, mock_trainer, epoch, called) -> None:
    """Test call_every executes the hook based on interval."""
    mock_hook = mocker.MagicMock()
    hook = call_every(start=3, interval=3)(mock_hook)
    mock_trainer.model.epoch = epoch

    hook(mock_trainer)

    assert mock_hook.called is called


def test_call_every_terminated(mocker, mock_trainer) -> None:
    """Test call_every executes the hook on terminated trainer."""
    mock_hook = mocker.MagicMock()
    hook = call_every(start=3, interval=3)(mock_hook)
    mock_trainer.terminate_training('This is a test.')

    hook(mock_trainer)

    mock_hook.assert_called_once_with(mock_trainer)


@pytest.mark.parametrize('hook_cls', [Hook, StaticHook])
def test_parameters_copied(mocker, hook_cls) -> None:
    """Test changing the given parameters does not change the hook."""
    parameters = {'key': 1}
    hook = hook_cls(mocker.Mock(), parameters=parameters)

    parameters['key'] = 2

    assert hook.parameters == {'key': 1}


class TestHook:
    """Tests for the Hook class."""

    def test_default_name(self) -> None:
        """Test the base hook name defaults to the function name."""
        hook = Hook(_record_trainer)

        assert hook.base_hook_name == '_record_trainer'

    def test_default_name_callable_object(self, mocker) -> None:
        """Test the base hook name defaults to the class of the callable."""
        hook = Hook(mocker.Mock())

        assert hook.base_hook_name == 'Mock'

    def test_saving_hook_name(self) -> None:
        """Test the saving hook is named after its function."""
        bound = saving_hook.bind(call_every(interval=2))

        assert saving_hook.base_hook_name == 'saving_hook'
        assert bound.base_hook_name == 'saving_hook'

    def test_bind_merges_parameters(self, mocker) -> None:
        """Test binding merges the parameters and keeps the base name."""
        hook = Hook(mocker.Mock(), base_hook_name='base', parameters={'a': 1})

        bound = hook.bind(call_every(interval=2, start=1))

        assert bound.parameters == {'a': 1, 'interval': 2, 'start': 1}
        assert bound.base_hook_name == 'base'


class TestStaticHook:
    """Tests for the StaticHook class."""

    def test_default_name(self) -> None:
        """Test the base hook name defaults to the class of the callable."""
        hook = StaticHook(_TestClass('text'))

        assert hook.base_hook_name == '_TestClass'


class TestCallEvery:
    """Tests for the CallEvery class."""

    def test_parameters(self, mocker) -> None:
        """Test interval and start are added to the parameters."""
        hook = CallEvery(
            mocker.Mock(), parameters={'a': 1}, interval=2, start=3
        )

        assert hook.parameters == {'a': 1, 'interval': 2, 'start': 3}

    @pytest.mark.parametrize(('interval', 'start'), [(0, 0), (1, -1)])
    def test_invalid_arguments(self, mocker, interval, start) -> None:
        """Test a non-positive interval or a negative start raise."""
        with pytest.raises(ValueError):
            CallEvery(mocker.Mock(), interval=interval, start=start)


def test_evaluation_hook(mocker, mock_trainer) -> None:
    """Test the evaluation runs every given number of epochs."""
    evaluation = mocker.Mock()
    hook = evaluation_hook(evaluation, every=2)

    for epoch in (1, 2, 3):
        mock_trainer.model.epoch = epoch
        hook(mock_trainer)

    evaluation.assert_called_once()


@pytest.mark.parametrize('every', [0, -1])
def test_evaluation_hook_not_positive(mocker, every) -> None:
    """Test a non-positive every raises ValueError."""
    with pytest.raises(ValueError):
        evaluation_hook(mocker.Mock(), every=every)


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

        callback(mock_trainer)

        mock_trainer.terminate_training.assert_not_called()

    def test_stops_on_plateau(self, mock_trainer, callback) -> None:
        """Test training stops after a plateau."""
        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback(mock_trainer)

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

        callback(mock_trainer)

        mock_trainer.terminate_training.assert_not_called()

    def test_prunes_at_threshold(self, mock_trainer, callback) -> None:
        """Test pruning occurs when the threshold condition is met."""
        mock_trainer.model.epoch = 5

        callback(mock_trainer)

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

        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback(mock_trainer)

        mock_trainer.update_learning_rate.assert_called_once()

    def test_respects_cooldown(self, mocker, mock_trainer, callback) -> None:
        """Test LR update is suppressed during cooldown."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_schema = mocker.Mock
        mock_trainer.learning_schema.scheduler = scheduler

        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback(mock_trainer)

        callback(mock_trainer)

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

        for _ in range(callback.monitor.metric_tracker.patience + 2):
            callback(mock_trainer)

        args = mock_trainer.update_learning_rate.call_args

        mock_trainer.update_learning_rate.assert_called_once()
        assert isinstance(args[1]['scheduler'], schedulers.AbstractScheduler)
