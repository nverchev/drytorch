"""Tests for the ``monitoring`` module."""

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib.monitoring import (
    ALWAYS,
    After,
    Always,
    AndCondition,
    AtEpochs,
    Cooldown,
    Every,
    Monitor,
    NotCondition,
    OnTermination,
    OrCondition,
    Plateau,
    Watch,
    WorseThan,
)


class _Hook:
    """Minimal hook exposing what the conditions read."""

    def __init__(self, trainer, last_run_epoch: int = 0) -> None:
        self.trainer = trainer
        self.last_run_epoch = last_run_epoch
        return

    def __call__(self) -> None:
        return

    def bind_trainer(self, trainer) -> None:
        self.trainer = trainer
        return


@pytest.fixture
def hook(mock_trainer) -> _Hook:
    """A hook bound to the mock trainer."""
    return _Hook(mock_trainer)


def _observe(condition, hook, values, epochs) -> None:
    """Let the condition observe values, as its source computed them."""
    source = condition.monitor.source
    for epoch, value in zip(epochs, values, strict=True):
        source.computed_metrics = {condition.monitor.metric_name: value}
        source.last_run_epoch = epoch
        hook.trainer.model.epoch = epoch
        condition.observe(hook)

    return


class TestWatch:
    """Tests for the Watch class."""

    def test_resolve_default_source_and_metric(
        self, mock_trainer, mock_loss
    ) -> None:
        """Test resolve defaults to trainer and its loss."""
        watch = Watch()
        monitor = watch.resolve(mock_trainer)

        assert monitor.source is mock_trainer
        assert monitor.metric_name == mock_loss.name
        assert monitor.best_is == 'lower'

    def test_resolve_with_source(self, mock_trainer, mock_validation) -> None:
        """Test resolve with an explicit source."""
        watch = Watch(source=mock_validation)
        monitor = watch.resolve(mock_trainer)

        assert monitor.source is mock_validation
        assert monitor.metric_name == mock_validation.objective.name
        assert monitor.best_is == 'higher'

    def test_resolve_with_named_metric(self, mock_trainer) -> None:
        """Test resolve with metric given by name."""
        watch = Watch(metric='Accuracy')
        monitor = watch.resolve(mock_trainer)

        assert monitor.metric_name == 'Accuracy'
        assert monitor.best_is is None

    def test_resolve_with_objective(self, mocker, mock_trainer) -> None:
        """Test resolve with metric given as an objective."""
        objective = mocker.create_autospec(p.ObjectiveProtocol, instance=True)
        objective.name = 'Accuracy'
        objective.higher_is_better = True
        watch = Watch(metric=objective)
        monitor = watch.resolve(mock_trainer)

        assert monitor.metric_name == 'Accuracy'
        assert monitor.best_is == 'higher'

    def test_resolve_with_explicit_best_is(self, mock_trainer) -> None:
        """Test explicit best_is overrides objective direction."""
        watch = Watch(best_is='higher')
        monitor = watch.resolve(mock_trainer)

        assert monitor.best_is == 'higher'

    def test_resolve_unknown_metric_raises(self, mocker, mock_trainer) -> None:
        """Test resolve raises ValueError when metric name cannot be told."""
        mock_trainer.objective.name = ''
        watch = Watch()

        with pytest.raises(ValueError, match='Cannot tell which metric'):
            watch.resolve(mock_trainer)


class TestMonitor:
    """Tests for the Monitor class."""

    @pytest.fixture
    def monitor(self, mock_validation, example_loss_name) -> Monitor:
        """A monitor of the loss of a validation."""
        return Monitor(
            best_is='lower',
            metric_name=example_loss_name,
            source=mock_validation,
        )

    def test_record_values(
        self, monitor, example_epoch, example_loss_name, example_named_metrics
    ) -> None:
        """Test recording stores epoch and metric value."""
        monitor.record(example_epoch)

        assert monitor.epochs == (example_epoch,)
        assert monitor.history == (example_named_metrics[example_loss_name],)
        assert monitor.values == {
            example_epoch: example_named_metrics[example_loss_name]
        }
        assert monitor.value == example_named_metrics[example_loss_name]

    def test_record_once_per_epoch(self, monitor, example_epoch) -> None:
        """Test recording twice on the same epoch records only once."""
        monitor.record(example_epoch)
        monitor.record(example_epoch)

        assert monitor.epochs == (example_epoch,)

    def test_value_not_available_error(self, monitor) -> None:
        """Test accessing value before recording raises error."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = monitor.value

    def test_record_silent_skip_when_source_did_not_run(
        self, monitor, mock_validation, example_epoch
    ) -> None:
        """Test empty metrics when source did not run are skipped silently."""
        mock_validation.computed_metrics = {}
        mock_validation.last_run_epoch = 0

        monitor.record(example_epoch)

        assert monitor.epochs == ()
        assert monitor.history == ()

    def test_record_raises_when_source_ran_after_hook(
        self, monitor, mock_validation
    ) -> None:
        """Test error when source ran after hook in observed epoch."""
        mock_validation.computed_metrics = {}
        mock_validation.last_run_epoch = 0

        # Hook runs at epoch 1 before validation
        monitor.record(1)

        # Validation ran at epoch 1 after hook
        mock_validation.last_run_epoch = 1

        # Hook runs at epoch 2 before validation
        with pytest.raises(exceptions.SourceRunAfterHookError) as exc_info:
            monitor.record(2)

        assert exc_info.value.source_name == mock_validation.name
        assert exc_info.value.epoch == 1

    def test_record_metric_not_found(
        self, monitor, mock_validation, example_epoch
    ) -> None:
        """Test recording raises MetricNotFoundError if metric name missing."""
        mock_validation.computed_metrics = {'Other': 1.0}

        with pytest.raises(exceptions.MetricNotFoundError):
            monitor.record(example_epoch)

    @pytest.mark.parametrize(
        ('best_is', 'better'), [('lower', False), ('higher', True)]
    )
    def test_is_better(self, mock_validation, best_is, better) -> None:
        """Test is_better follows the direction."""
        monitor = Monitor(
            best_is=best_is,
            metric_name='Metric',
            source=mock_validation,
        )

        assert monitor.is_better(2.0, 1.0) is better

    def test_is_better_with_nan(self, monitor) -> None:
        """Test is_better returns False for NaN."""
        assert not monitor.is_better(float('nan'), 1.0)

    def test_is_better_without_direction_raises(self, mock_validation) -> None:
        """Test is_better raises ValueError when best_is is None."""
        monitor = Monitor(
            best_is=None,
            metric_name='Metric',
            source=mock_validation,
        )

        with pytest.raises(ValueError):
            monitor.is_better(1.0, 2.0)


class TestConditions:
    """Tests for simple conditions."""

    def test_always(self, hook) -> None:
        """Test Always holds unconditionally."""
        assert Always()(hook)
        assert ALWAYS(hook)

    @pytest.mark.parametrize(('epoch', 'holds'), [(1, False), (2, True)])
    def test_after(self, hook, mock_model, epoch, holds) -> None:
        """Test After holds from the given epoch on."""
        mock_model.epoch = epoch

        assert After(2)(hook) is holds

    @pytest.mark.parametrize(('epoch', 'holds'), [(2, True), (3, False)])
    def test_at_epochs(self, hook, mock_model, epoch, holds) -> None:
        """Test AtEpochs holds only at the given epochs."""
        mock_model.epoch = epoch

        assert AtEpochs([2, 5])(hook) is holds

    @pytest.mark.parametrize(
        ('epoch', 'holds'), [(0, False), (1, True), (4, False), (5, True)]
    )
    def test_every(self, hook, mock_model, epoch, holds) -> None:
        """Test Every holds every given number of epochs."""
        mock_model.epoch = epoch

        assert Every(2, start=1)(hook) is holds

    def test_every_invalid_arguments(self) -> None:
        """Test non-positive interval raises ValueError."""
        with pytest.raises(ValueError):
            Every(0)

    def test_every_on_termination(self, hook, mock_model, mock_trainer) -> None:
        """Test Every holds when trainer terminated."""
        mock_model.epoch = 4
        mock_trainer.terminated = True

        assert Every(3, start=1)(hook)

    @pytest.mark.parametrize(
        ('last_run', 'holds'), [(0, True), (4, False), (2, True)]
    )
    def test_cooldown(self, mock_trainer, mock_model, last_run, holds) -> None:
        """Test Cooldown holds when enough epochs passed since last run."""
        mock_model.epoch = 5
        hook = _Hook(mock_trainer, last_run_epoch=last_run)

        assert Cooldown(2)(hook) is holds

    def test_cooldown_invalid(self) -> None:
        """Test negative cooldown raises ValueError."""
        with pytest.raises(ValueError):
            Cooldown(-1)

    def test_on_termination(self, hook, mock_trainer) -> None:
        """Test OnTermination holds only when trainer is terminated."""
        assert not OnTermination()(hook)

        mock_trainer.terminated = True

        assert OnTermination()(hook)


class TestPlateau:
    """Tests for the Plateau condition."""

    @pytest.fixture
    def plateau(self, mock_trainer, example_loss_name) -> Plateau:
        """A condition following the loss with a patience of two."""
        watch = Watch(best_is='lower', metric=example_loss_name)
        condition = Plateau(patience=2, watch=watch)
        condition.bind_trainer(mock_trainer)
        return condition

    def test_invalid_patience(self) -> None:
        """Test negative patience raises ValueError."""
        with pytest.raises(ValueError):
            Plateau(patience=-1)

    def test_best_value_before_observation(self, plateau) -> None:
        """Test best_value raises ResultNotAvailableError before any value."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = plateau.best_value

    def test_fresh_value_rule(self, plateau, hook, mock_model) -> None:
        """Test Plateau holds only if monitor recorded at current epoch."""
        _observe(plateau, hook, [3.0, 2.0, 2.5, 2.5], [1, 2, 3, 4])
        mock_model.epoch = 4

        assert plateau(hook)

        # Epoch 5: source did not run
        mock_model.epoch = 5

        assert not plateau(hook)

    def test_holds_after_patience(self, plateau, hook, mock_model) -> None:
        """Test Plateau holds after patience records without improvement."""
        _observe(plateau, hook, [3.0, 2.0, 2.5], [1, 2, 3])
        mock_model.epoch = 3

        assert not plateau(hook)

        _observe(plateau, hook, [2.5], [4])
        mock_model.epoch = 4

        assert plateau(hook)

    def test_improvement_resets_patience(
        self, plateau, hook, mock_model
    ) -> None:
        """Test improvement resets patience countdown."""
        _observe(plateau, hook, [3.0, 2.5, 2.4, 1.0], [1, 2, 3, 4])
        mock_model.epoch = 4

        assert not plateau(hook)
        assert plateau.best_value == 1.0


class TestWorseThan:
    """Tests for the WorseThan condition."""

    @pytest.fixture
    def worse_than(self, mock_trainer, example_loss_name) -> WorseThan:
        """A condition comparing the loss with thresholds."""
        watch = Watch(best_is='lower', metric=example_loss_name)
        condition = WorseThan({2: 1.0, 3: None}, watch=watch)
        condition.bind_trainer(mock_trainer)
        return condition

    def test_ignores_other_epochs(self, worse_than, hook, mock_model) -> None:
        """Test condition returns False for epochs not in thresholds."""
        mock_model.epoch = 1
        _observe(worse_than, hook, [2.0], [1])

        assert not worse_than(hook)

    def test_threshold_none_returns_false(
        self, worse_than, hook, mock_model
    ) -> None:
        """Test None threshold returns False."""
        mock_model.epoch = 3
        _observe(worse_than, hook, [2.0], [3])

        assert not worse_than(hook)

    def test_holds_when_worse(self, worse_than, hook, mock_model) -> None:
        """Test condition holds when value misses threshold."""
        mock_model.epoch = 2
        _observe(worse_than, hook, [2.0], [2])

        assert worse_than(hook)

    def test_fresh_value_rule_raises_when_no_record(
        self, worse_than, hook, mock_model
    ) -> None:
        """Test WorseThan raises error if threshold epoch has no record."""
        mock_model.epoch = 2

        with pytest.raises(exceptions.ResultNotAvailableError):
            worse_than(hook)


class TestCombinations:
    """Tests for logical combinations of conditions."""

    def test_and(self, hook, mock_model) -> None:
        """Test And condition and & operator."""
        mock_model.epoch = 3

        condition = After(2) & AtEpochs([3])
        first, second = condition.operands

        assert isinstance(condition, AndCondition)
        assert condition(hook)
        assert isinstance(first, After)
        assert first.epoch == 2
        assert isinstance(second, AtEpochs)
        assert second.epochs == frozenset({3})

    def test_or(self, hook, mock_model) -> None:
        """Test Or condition and | operator."""
        mock_model.epoch = 3

        condition = After(4) | AtEpochs([3])
        first, second = condition.operands

        assert isinstance(condition, OrCondition)
        assert condition(hook)
        assert isinstance(first, After)
        assert first.epoch == 4
        assert isinstance(second, AtEpochs)
        assert second.epochs == frozenset({3})

    def test_not(self, hook, mock_model) -> None:
        """Test Not condition and ~ operator."""
        mock_model.epoch = 3

        condition = ~After(4)
        (operand,) = condition.operands

        assert isinstance(condition, NotCondition)
        assert condition(hook)
        assert isinstance(operand, After)
        assert operand.epoch == 4

    def test_bind_trainer_and_observe_delegation(
        self, hook, mock_trainer, example_loss_name, example_epoch
    ) -> None:
        """Test combined condition delegates to operands."""
        watch = Watch(best_is='lower', metric=example_loss_name)
        plateau = Plateau(watch=watch)
        combined = After(1) & plateau

        combined.bind_trainer(mock_trainer)
        combined.observe(hook)

        assert plateau.monitor.epochs == (example_epoch,)
