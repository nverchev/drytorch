"""Tests for the "hooks" module."""

from typing import Any, Literal

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import objectives, schedulers
from drytorch.lib.hooks import (
    EarlyStoppingCallback,
    HookRegistry,
    MetricExtractor,
    MetricMonitor,
    PruneCallback,
    ReduceLROnPlateau,
    RestartScheduleOnPlateau,
    StaticHook,
    call_every,
    get_last,
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


class TestMetricExtractor:
    """Tests for MetricExtractor class."""

    @pytest.fixture()
    def mock_metric_higher_is_better(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with higher_is_better = True."""
        mock = mocker.MagicMock()
        mock.higher_is_better = True
        mock.name = 'test_accuracy'
        return mock

    @pytest.fixture()
    def mock_metric_lower_is_better(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with higher_is_better = False."""
        mock = mocker.MagicMock()
        mock.higher_is_better = False
        mock.name = 'test_loss'
        return mock

    @pytest.fixture()
    def mock_metric_no_preference(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with no higher_is_better attribute."""
        mock = mocker.MagicMock()
        del mock.higher_is_better  # Remove the attribute
        mock.name = 'test_metric'
        return mock

    @pytest.fixture()
    def mock_metric_with_get_name(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with the _get_name method."""
        mock = mocker.MagicMock()
        mock._get_name = 'dynamic_name'
        del mock.name  # Remove name attribute
        return mock

    @pytest.fixture()
    def extractor_from_str(self, example_loss_name) -> MetricExtractor:
        """Set up an extractor with string metric."""
        return MetricExtractor(metric=example_loss_name)

    @pytest.fixture()
    def extractor_from_object(
        self, mock_metric_higher_is_better
    ) -> MetricExtractor:
        """Set up an extractor with a metric object."""
        return MetricExtractor(metric=mock_metric_higher_is_better)

    @pytest.fixture()
    def extractor_no_metric(self) -> MetricExtractor:
        """Set up an extractor with no specified metric."""
        return MetricExtractor()

    @pytest.fixture
    def mock_metric_tracker(self, mocker) -> objectives.MetricTracker:
        """Returns a mock object."""
        return mocker.create_autospec(objectives.MetricTracker)

    def test_init_with_string_metric(
        self, extractor_from_str, example_loss_name
    ) -> None:
        """Test instantiation with string metric."""
        assert extractor_from_str.metric_spec == example_loss_name
        assert extractor_from_str.metric_name is None  # Not resolved yet

    def test_init_with_metric_object(
        self, extractor_from_object, mock_metric_higher_is_better
    ) -> None:
        """Test instantiation with a metric object."""
        assert extractor_from_object.metric_spec == mock_metric_higher_is_better
        assert extractor_from_object.metric_name is None  # Not resolved yet

    def test_get_monitor_with_validation(
        self, extractor_from_str, mock_trainer, mock_validation
    ) -> None:
        """Test getting monitored values with validation available."""
        mock_trainer.validation = mock_validation
        monitor = extractor_from_str._get_monitor(mock_trainer)
        assert monitor == mock_trainer.validation

    def test_get_monitor_without_validation(
        self, extractor_from_str, mock_trainer
    ) -> None:
        """Test getting monitored values without validation."""
        mock_trainer.validation = None
        assert extractor_from_str._get_monitor(mock_trainer) == mock_trainer

    def test_get_monitor_with_optional_monitor(
        self, mock_trainer, mock_validation
    ) -> None:
        """Test getting monitored values with an optional monitor specified."""
        extractor = MetricExtractor(monitor=mock_validation)
        assert extractor._get_monitor(mock_trainer) == mock_validation

    def test_get_metric_name_from_string(self) -> None:
        """Test extracting metric name from string."""
        assert MetricExtractor._get_metric_name('test_loss') == 'test_loss'

    def test_get_metric_name_from_object_with_name(
        self, mock_metric_higher_is_better
    ) -> None:
        """Test extracting a name from an object with a name attribute."""
        name = MetricExtractor._get_metric_name(mock_metric_higher_is_better)
        assert name == 'test_accuracy'

    def test_get_metric_name_from_object_with_get_name(
        self, mock_metric_with_get_name
    ) -> None:
        """Test extracting a name from an object with the _get_name method."""
        assert (
            MetricExtractor._get_metric_name(mock_metric_with_get_name)
            == 'dynamic_name'
        )

    def test_get_metric_name_from_class_name(
        self, mocker, example_loss_name
    ) -> None:
        """Test extracting metric name from class name."""
        mock = mocker.MagicMock(spec=[])  # empty spec, no name or _get_name
        mock.__class__.__name__ = example_loss_name
        assert MetricExtractor._get_metric_name(mock) == example_loss_name

    def test_get_metric_best_is_higher(
        self, mock_metric_higher_is_better
    ) -> None:
        """Test getting best_is preference when higher is better."""
        assert (
            MetricExtractor._get_metric_best_is(mock_metric_higher_is_better)
            == 'higher'
        )

    def test_get_metric_best_is_lower(
        self, mock_metric_lower_is_better
    ) -> None:
        """Test getting best_is preference when lower is better."""
        assert (
            MetricExtractor._get_metric_best_is(mock_metric_lower_is_better)
            == 'lower'
        )

    def test_get_metric_best_is_none(self, mock_metric_no_preference) -> None:
        """Test getting best_is preference when not specified."""
        assert (
            MetricExtractor._get_metric_best_is(mock_metric_no_preference)
            is None
        )

    def test_get_metric_best_is_string_metric(self) -> None:
        """Test getting best_is preference for string metric."""
        assert MetricExtractor._get_metric_best_is('test_loss') is None

    def test_get_metric_best_is_none_metric(self) -> None:
        """Test getting best_is preference for None metric."""
        assert MetricExtractor._get_metric_best_is(None) is None

    def test_extract_metric_value_with_string_metric(
        self,
        extractor_from_str,
        mock_trainer,
        mock_metric_tracker,
        example_loss_name,
    ) -> None:
        """Test extracting metric value with string metric specification."""
        extractor_from_str.extract_metric_value(
            mock_trainer, mock_metric_tracker
        )
        # example_loss_name is the first metric name in example_named_metrics
        assert extractor_from_str._resolved_metric_name == example_loss_name

    def test_extract_metric_value_auto_select_first_metric(
        self,
        extractor_no_metric,
        mock_trainer,
        mock_metric_tracker,
        example_loss_name,
    ) -> None:
        """Test extracting metric value when no metric specified."""
        extractor_no_metric.extract_metric_value(
            mock_trainer, mock_metric_tracker
        )
        # example_loss_name is the first metric name in example_named_metrics
        assert extractor_no_metric._resolved_metric_name == example_loss_name

    def test_extract_metric_value_metric_not_found(
        self,
        extractor_from_str,
        mock_trainer,
        mock_metric_tracker,
    ) -> None:
        """Test exception when the specified metric is not found."""
        mock_trainer.validation = None
        mock_trainer.computed_metrics.return_value = {}
        with pytest.raises(exceptions.MetricNotFoundError):
            extractor_from_str.extract_metric_value(
                mock_trainer, mock_metric_tracker
            )

    def test_get_metric_best_is_delegation(self, extractor_from_object) -> None:
        """Test get_metric_best_is delegates to a static method."""
        result = extractor_from_object.get_metric_best_is()
        assert result == 'higher'


class TestMetricMonitor:
    """Tests for MetricMonitor class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_extractor = mocker.create_autospec(MetricExtractor)
        self.mock_extractor.get_metric_best_is = mocker.Mock(
            return_value='higher'
        )
        self.mock_metric_tracker = mocker.create_autospec(
            objectives.MetricTracker
        )
        self.mock_metric_tracker_cls = mocker.patch(
            'drytorch.lib.objectives.MetricTracker'
        )
        self.mock_metric_extractor_cls = mocker.patch(
            'drytorch.lib.hooks.MetricExtractor'
        )
        self.mock_metric_tracker_cls.return_value = self.mock_metric_tracker
        self.mock_metric_extractor_cls.return_value = self.mock_extractor
        return

    def test_init_calls_metric_extractor_and_metric_tracker(self) -> None:
        """Test that the constructor calls the mocks."""
        metric = 'accuracy'
        min_delta = 0.01
        patience = 5
        best_is: Literal['auto', 'higher', 'lower'] = 'higher'

        MetricMonitor(
            metric=metric,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
        )

        self.mock_metric_extractor_cls.assert_called_once_with(
            metric=metric, monitor=None
        )
        self.mock_metric_tracker_cls.assert_called_once_with(  # type: ignore
            metric_name=metric,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=get_last,
        )

    def test_record_extractor_and_tracker_calls(self, mocker) -> None:
        """Test that record_metric_value calls the necessary methods."""
        instance = mocker.Mock()
        monitor = MetricMonitor[Any, Any](metric='accuracy')
        self.mock_extractor.extract_metric_value.return_value = 0.95
        monitor.record_metric_value(instance)

        self.mock_extractor.extract_metric_value.assert_called_once_with(
            instance, monitor.metric_tracker
        )
        self.mock_metric_tracker.add_value.assert_called_once_with(0.95)

    def test_properties_delegate_to_tracker(self):
        """Test properties delegate all calls to the internal tracker."""
        monitor = MetricMonitor(metric='accuracy')
        monitor.metric_tracker.best_value = 0.99
        self.mock_metric_tracker.filtered_value = 0.95
        monitor.metric_tracker.history = [0.9, 0.92, 0.95]

        assert monitor.best_value == 0.99
        assert monitor.filtered_value == 0.95
        assert monitor.history == [0.9, 0.92, 0.95]

    def test_is_improving_and_is_patient_delegate_to_tracker(self):
        """Test that is_improving and is_patient delegate to the tracker."""
        monitor = MetricMonitor(metric='accuracy')
        monitor.metric_tracker.is_improving.return_value = True
        monitor.metric_tracker.is_patient.return_value = False

        is_improving = monitor.is_improving()
        is_patient = monitor.is_patient()

        assert is_improving is True
        assert is_patient is False
        self.mock_metric_tracker.is_improving.assert_called_once()
        self.mock_metric_tracker.is_patient.assert_called_once()

    def test_is_better_delegates_to_tracker(self):
        """Test that is_better delegates to the tracker."""
        monitor = MetricMonitor(metric='accuracy')
        self.mock_metric_tracker.is_better.return_value = True  # type: ignore

        result = monitor.is_better(0.95, 0.9)

        assert result is True
        self.mock_metric_tracker.is_better.assert_called_once_with(0.95, 0.9)

    def test_metric_name_property_delegates_to_extractor_and_tracker(self):
        """Test that the metric_name property delegates correctly."""
        monitor = MetricMonitor(metric='accuracy')
        self.mock_extractor.metric_name = 'test_accuracy'
        self.mock_metric_tracker.metric_name = 'tracker_accuracy'

        assert monitor.metric_name == 'test_accuracy'


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    @pytest.fixture()
    def callback(self, example_loss_name) -> EarlyStoppingCallback:
        """Set up a test instance."""
        return EarlyStoppingCallback(
            metric=example_loss_name,
            patience=2,
        )

    def test_early_epoch_no_stop(self, mock_trainer, callback) -> None:
        """Test training continues if not enough epochs passed."""
        mock_trainer.model.epoch = 1
        callback(mock_trainer)
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_stops_on_plateau(self, mock_trainer, callback) -> None:
        """Test training stops after a plateau."""
        objective = mock_trainer.validation.objective  # type: ignore
        objective.higher_is_better = True
        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore


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
        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_prunes_at_threshold(self, mock_trainer, callback) -> None:
        """Test pruning occurs when the threshold condition is met."""
        mock_trainer.model.epoch = 5
        callback(mock_trainer)
        mock_trainer.terminate_training.assert_called_once()  # type: ignore


class TestReduceLROnPlateau:
    """Tests for ReduceLROnPlateau."""

    @pytest.fixture()
    def callback(self, example_loss_name) -> ReduceLROnPlateau:
        """Set up a test instance."""
        return ReduceLROnPlateau(
            metric=example_loss_name, patience=2, factor=0.01, cooldown=1
        )

    def test_reduces_lr_and_respects_cooldown(
        self, mocker, mock_trainer, callback
    ) -> None:
        """Test LR reduction and cooldown enforcement."""
        scheduler = schedulers.ConstantScheduler()
        mock_trainer.learning_schema = mocker.Mock
        mock_trainer.learning_schema.scheduler = scheduler

        for _ in range(callback.monitor.metric_tracker.patience + 1):
            callback(mock_trainer)

        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore

        callback(mock_trainer)
        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore


class TestRestartScheduleOnPlateau:
    """Tests for RestartScheduleOnPlateau."""

    @pytest.fixture()
    def callback(self, example_loss_name) -> RestartScheduleOnPlateau:
        """Set up a test instance."""
        return RestartScheduleOnPlateau(
            metric=example_loss_name, patience=2, cooldown=1
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

        mock_trainer.update_learning_rate.assert_called_once()  # type: ignore
        args = mock_trainer.update_learning_rate.call_args  # type: ignore
        assert isinstance(args[1]['scheduler'], schedulers.WarmupScheduler)
