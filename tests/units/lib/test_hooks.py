"""Tests for the ``hooks`` module."""

import pytest

from drytorch.core import exceptions
from drytorch.lib import monitoring, schedulers
from drytorch.lib.hooks import (
    EarlyStopping,
    EvaluationHook,
    HookChain,
    Pruning,
    ReduceLROnPlateau,
    RestartScheduleOnPlateau,
    SavingHook,
    TerminateTraining,
    TrainerHook,
)


class _CountingHook(TrainerHook):
    """Hook counting how many times it ran."""

    def __init__(self, when: monitoring.Condition = monitoring.ALWAYS) -> None:
        """Initialize.

        Args:
            when: decides whether the hook runs.
        """
        super().__init__(when)
        self.runs = 0
        return

    def run(self) -> None:
        """Count the run."""
        self.runs += 1
        return


class TestHookChain:
    """Tests for the HookChain class."""

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


class TestTrainerHook:
    """Tests for the TrainerHook class."""

    def test_runs_when_the_condition_holds(self, mock_trainer) -> None:
        """Test the hook runs while its condition holds."""
        hook = _CountingHook(when=monitoring.After(mock_trainer.model.epoch))
        hook.bind_trainer(mock_trainer)

        hook()

        assert hook.runs == 1

    def test_skips_when_the_condition_fails(self, mock_trainer) -> None:
        """Test the hook does not run when its condition fails."""
        hook = _CountingHook(
            when=monitoring.After(mock_trainer.model.epoch + 1)
        )
        hook.bind_trainer(mock_trainer)

        hook()

        assert hook.runs == 0

    def test_records_the_epoch_it_ran(self, mock_trainer) -> None:
        """Test the hook remembers when it last ran."""
        hook = _CountingHook()
        hook.bind_trainer(mock_trainer)

        assert hook.last_run_epoch == 0

        hook()

        assert hook.last_run_epoch == mock_trainer.model.epoch

    def test_observes_before_running(self, mocker, mock_trainer) -> None:
        """Test the condition observes on every call."""
        when = mocker.Mock(return_value=False)
        hook = _CountingHook(when=when)
        hook.bind_trainer(mock_trainer)

        hook()

        when.observe.assert_called_once_with(hook)

    def test_not_bound(self) -> None:
        """Test calling a hook before a trainer binds it raises."""
        hook = SavingHook()

        with pytest.raises(exceptions.HookNotBoundError):
            hook()

    def test_bound_twice(self, mocker, mock_trainer) -> None:
        """Test binding a hook to a second trainer raises."""
        hook = SavingHook()
        hook.bind_trainer(mock_trainer)

        with pytest.raises(exceptions.HookAlreadyBoundError):
            hook.bind_trainer(mocker.Mock())

    def test_binds_its_condition(self, mocker, mock_trainer) -> None:
        """Test binding the hook binds its condition."""
        when = mocker.Mock()
        hook = _CountingHook(when=when)

        hook.bind_trainer(mock_trainer)

        when.bind_trainer.assert_called_once_with(mock_trainer)


def test_saving_hook(mock_trainer) -> None:
    """Test the saving hook saves the checkpoint of its trainer."""
    hook = SavingHook()
    hook.bind_trainer(mock_trainer)

    hook()

    mock_trainer.save_checkpoint.assert_called_once()


def test_evaluation_hook(mocker, mock_trainer) -> None:
    """Test the hook runs the evaluation it holds."""
    evaluation = mocker.Mock()
    hook = EvaluationHook(evaluation)
    hook.bind_trainer(mock_trainer)

    hook()

    evaluation.assert_called_once()


def test_evaluation_hook_runs_periodically(
    mocker, mock_trainer, mock_model
) -> None:
    """Test a scheduled evaluation runs every given number of epochs."""
    evaluation = mocker.Mock()
    hook = EvaluationHook(evaluation, when=monitoring.Every(2, start=2))
    hook.bind_trainer(mock_trainer)

    for epoch in (1, 2, 3):
        mock_model.epoch = epoch
        hook()

    evaluation.assert_called_once()


def test_terminate_training(mock_trainer) -> None:
    """Test the hook terminates the training of its trainer."""
    hook = TerminateTraining()
    hook.bind_trainer(mock_trainer)

    hook()

    mock_trainer.terminate_training.assert_called_once_with(
        'a hook terminated the training'
    )


class TestEarlyStopping:
    """Tests for the EarlyStopping hook."""

    @pytest.fixture
    def hook(self, example_loss_name) -> EarlyStopping:
        """Set up a test instance."""
        watch = monitoring.Watch(best_is='higher', metric=example_loss_name)
        return EarlyStopping(
            patience=2,
            start_from_epoch=1,
            watch=watch,
        )

    def test_plateau_property(self, hook) -> None:
        """Test the plateau property exposes the condition."""
        assert isinstance(hook.plateau, monitoring.Plateau)

    def test_early_epoch_no_stop(self, mock_trainer, mock_model, hook) -> None:
        """Test training continues if not enough epochs passed."""
        mock_model.epoch = 0
        hook.bind_trainer(mock_trainer)

        hook()

        mock_trainer.terminate_training.assert_not_called()

    def test_stops_on_plateau(
        self, mock_trainer, mock_model, hook, example_loss_name
    ) -> None:
        """Test training stops after a plateau."""
        hook.bind_trainer(mock_trainer)

        for epoch in range(1, 4):
            mock_model.epoch = epoch
            mock_trainer.computed_metrics = {example_loss_name: 1.0}
            mock_trainer.last_run_epoch = epoch
            hook()

        mock_trainer.terminate_training.assert_called_once()
        call_msg = mock_trainer.terminate_training.call_args[0][0]
        assert 'the metric stopped improving' in call_msg


class TestPruning:
    """Tests for the Pruning hook."""

    @pytest.fixture
    def hook(self, example_loss_name) -> Pruning:
        """Set up a test instance."""
        watch = monitoring.Watch(best_is='higher', metric=example_loss_name)
        return Pruning(
            {3: 2.0, 5: 0.5},
            watch=watch,
        )

    def test_no_pruning_before_threshold(
        self, mock_trainer, mock_model, hook, example_loss_name
    ) -> None:
        """Test no pruning before the defined epoch."""
        mock_model.epoch = 2
        mock_trainer.computed_metrics = {example_loss_name: 1.0}
        mock_trainer.last_run_epoch = 2
        hook.bind_trainer(mock_trainer)

        hook()

        mock_trainer.terminate_training.assert_not_called()

    def test_prunes_at_threshold(
        self, mock_trainer, mock_model, hook, example_loss_name
    ) -> None:
        """Test pruning occurs when the threshold condition is met."""
        mock_model.epoch = 5
        mock_trainer.computed_metrics = {example_loss_name: 0.2}
        mock_trainer.last_run_epoch = 5
        hook.bind_trainer(mock_trainer)

        hook()

        mock_trainer.terminate_training.assert_called_once()
        call_msg = mock_trainer.terminate_training.call_args[0][0]
        assert 'the metric missed a threshold' in call_msg


class TestReduceLROnPlateau:
    """Tests for the ReduceLROnPlateau hook."""

    @pytest.fixture
    def hook(self, example_loss_name) -> ReduceLROnPlateau:
        """Set up a test instance."""
        watch = monitoring.Watch(best_is='higher', metric=example_loss_name)
        return ReduceLROnPlateau(
            factor=0.01,
            patience=2,
            cooldown=1,
            watch=watch,
        )

    def test_plateau_property(self, hook) -> None:
        """Test the plateau property exposes the condition."""
        assert isinstance(hook.plateau, monitoring.Plateau)

    def test_reduces_lr(
        self, mocker, mock_trainer, mock_model, hook, example_loss_name
    ) -> None:
        """Test the learning rate is reduced after the patience."""
        mock_trainer.learning_schema = mocker.Mock()
        mock_trainer.learning_schema.scheduler = schedulers.ConstantScheduler()
        hook.bind_trainer(mock_trainer)

        for epoch in range(1, 4):
            mock_model.epoch = epoch
            mock_trainer.computed_metrics = {example_loss_name: 1.0}
            mock_trainer.last_run_epoch = epoch
            hook()

        mock_trainer.update_learning_rate.assert_called_once()

    def test_respects_cooldown(
        self, mocker, mock_trainer, mock_model, hook, example_loss_name
    ) -> None:
        """Test no further reduction happens during the cooldown."""
        mock_trainer.learning_schema = mocker.Mock()
        mock_trainer.learning_schema.scheduler = schedulers.ConstantScheduler()
        hook.bind_trainer(mock_trainer)

        for epoch in range(1, 5):
            mock_model.epoch = epoch
            mock_trainer.computed_metrics = {example_loss_name: 1.0}
            mock_trainer.last_run_epoch = epoch
            hook()

        mock_trainer.update_learning_rate.assert_called_once()

    def test_validation_every_two_epochs_single_reduction(
        self,
        mocker,
        mock_trainer,
        mock_model,
        mock_validation,
        example_loss_name,
    ) -> None:
        """Test validation every 2 epochs produces a single reduction."""
        mock_trainer.learning_schema = mocker.Mock()
        mock_trainer.learning_schema.scheduler = schedulers.ConstantScheduler()
        watch = monitoring.Watch(
            source=mock_validation, best_is='higher', metric=example_loss_name
        )
        hook = ReduceLROnPlateau(
            factor=0.1, patience=0, cooldown=0, watch=watch
        )
        hook.bind_trainer(mock_trainer)

        # Epoch 0: baseline
        mock_model.epoch = 0
        mock_validation.computed_metrics = {example_loss_name: 1.0}
        mock_validation.last_run_epoch = 0
        hook()

        # Epoch 1: no validation
        mock_model.epoch = 1
        mock_validation.computed_metrics = {}
        mock_validation.last_run_epoch = 0
        hook()

        # Epoch 2: plateau triggers reduction
        mock_model.epoch = 2
        mock_validation.computed_metrics = {example_loss_name: 1.0}
        mock_validation.last_run_epoch = 2
        hook()

        # Epoch 3: no validation, no second reduction
        mock_model.epoch = 3
        mock_validation.computed_metrics = {}
        mock_validation.last_run_epoch = 2
        hook()

        mock_trainer.update_learning_rate.assert_called_once()


class TestRestartScheduleOnPlateau:
    """Tests for the RestartScheduleOnPlateau hook."""

    @pytest.fixture
    def hook(self, example_loss_name) -> RestartScheduleOnPlateau:
        """Set up a test instance."""
        watch = monitoring.Watch(best_is='higher', metric=example_loss_name)
        return RestartScheduleOnPlateau(
            patience=2,
            watch=watch,
        )

    def test_plateau_property(self, hook) -> None:
        """Test the plateau property exposes the condition."""
        assert isinstance(hook.plateau, monitoring.Plateau)

    def test_restarts_schedule(
        self, mocker, mock_trainer, mock_model, hook, example_loss_name
    ) -> None:
        """Test the schedule restarts after a plateau."""
        mock_trainer.learning_schema = mocker.Mock()
        mock_trainer.learning_schema.scheduler = schedulers.ConstantScheduler()
        hook.bind_trainer(mock_trainer)

        for epoch in range(1, 4):
            mock_model.epoch = epoch
            mock_trainer.computed_metrics = {example_loss_name: 1.0}
            mock_trainer.last_run_epoch = epoch
            hook()

        arguments = mock_trainer.update_learning_rate.call_args
        mock_trainer.update_learning_rate.assert_called_once()
        assert isinstance(
            arguments[1]['scheduler'], schedulers.AbstractScheduler
        )
