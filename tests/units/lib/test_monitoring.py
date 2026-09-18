"""Tests for the ``monitoring`` module."""

from typing import Any, Literal

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib.monitoring import (
    MetricExtractor,
    MetricMonitor,
    MetricTracker,
    get_last,
)


class TestMetricExtractor:
    """Tests for MetricExtractor class."""

    @pytest.fixture
    def mock_metric_higher_is_better(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with higher_is_better = True."""
        mock = mocker.MagicMock()
        mock.higher_is_better = True
        mock.name = 'test_accuracy'
        return mock

    @pytest.fixture
    def mock_metric_lower_is_better(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with higher_is_better = False."""
        mock = mocker.MagicMock()
        mock.higher_is_better = False
        mock.name = 'test_loss'
        return mock

    @pytest.fixture
    def mock_metric_no_preference(self, mocker) -> p.ObjectiveProtocol:
        """Mock a metric object with no higher_is_better attribute."""
        mock = mocker.MagicMock()
        del mock.higher_is_better  # Remove the attribute
        mock.name = 'test_metric'
        return mock

    @pytest.fixture
    def extractor_from_str(self, example_loss_name) -> MetricExtractor:
        """Set up an extractor with string metric."""
        return MetricExtractor(metric=example_loss_name)

    @pytest.fixture
    def extractor_from_object(
        self, mock_metric_higher_is_better
    ) -> MetricExtractor:
        """Set up an extractor with a metric object."""
        return MetricExtractor(metric=mock_metric_higher_is_better)

    @pytest.fixture
    def extractor_no_metric(self) -> MetricExtractor:
        """Set up an extractor with no specified metric."""
        return MetricExtractor()

    @pytest.fixture
    def mock_metric_tracker(self, mocker) -> MetricTracker:
        """Returns a mock object."""
        return mocker.create_autospec(MetricTracker)

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

    def test_get_source_with_validation(
        self, extractor_from_str, mock_trainer, mock_validation
    ) -> None:
        """Test the source is the validation when there is one."""
        mock_trainer.validation = mock_validation

        source = extractor_from_str._get_source(mock_trainer)

        assert source == mock_trainer.validation

    def test_get_source_without_validation(
        self, extractor_from_str, mock_trainer
    ) -> None:
        """Test the source is the trainer without a validation."""
        mock_trainer.validation = None

        assert extractor_from_str._get_source(mock_trainer) == mock_trainer

    def test_get_source_with_a_given_source(
        self, mock_trainer, mock_validation
    ) -> None:
        """Test a given source is used instead of the default one."""
        extractor = MetricExtractor(source=mock_validation)

        assert extractor._get_source(mock_trainer) == mock_validation

    def test_get_metric_name_from_string(self) -> None:
        """Test extracting metric name from string."""
        assert MetricExtractor._get_metric_name('test_loss') == 'test_loss'

    def test_get_metric_name_from_object_with_name(
        self, mock_metric_higher_is_better
    ) -> None:
        """Test extracting a name from an object with a name attribute."""
        name = MetricExtractor._get_metric_name(mock_metric_higher_is_better)

        assert name == 'test_accuracy'

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

    def test_extract_metric_value_without_a_metric(
        self,
        extractor_no_metric,
        mock_trainer,
        mock_metric_tracker,
        example_loss_name,
    ) -> None:
        """Test the metric defaults to the objective of the source."""
        mock_trainer.validation.objective.name = example_loss_name

        extractor_no_metric.extract_metric_value(
            mock_trainer, mock_metric_tracker
        )

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
            MetricTracker, instance=True
        )
        self.mock_metric_tracker.best_is = 'auto'
        self.mock_metric_tracker_cls = mocker.patch(
            'drytorch.lib.monitoring.MetricTracker'
        )
        self.mock_metric_extractor_cls = mocker.patch(
            'drytorch.lib.monitoring.MetricExtractor'
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
            metric=metric, source=None
        )
        self.mock_metric_tracker_cls.assert_called_once_with(
            metric_name=metric,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=get_last,
        )

    def test_init_takes_best_is_from_metric(self) -> None:
        """Test an automatic direction comes from the metric when known."""
        MetricMonitor(metric='accuracy')

        call = self.mock_metric_tracker_cls.call_args
        assert call.kwargs['best_is'] == 'higher'

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

    def test_properties_delegate_to_tracker(self) -> None:
        """Test properties delegate all calls to the internal tracker."""
        monitor = MetricMonitor(metric='accuracy')
        monitor.metric_tracker.best_value = 0.99
        self.mock_metric_tracker.filtered_value = 0.95
        monitor.metric_tracker.history = [0.9, 0.92, 0.95]

        assert monitor.best_value == 0.99
        assert monitor.filtered_value == 0.95
        assert monitor.history == [0.9, 0.92, 0.95]

    def test_is_improving_and_is_patient_delegate_to_tracker(self) -> None:
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

    def test_is_better_delegates_to_tracker(self) -> None:
        """Test that is_better delegates to the tracker."""
        monitor = MetricMonitor(metric='accuracy')
        self.mock_metric_tracker.is_better.return_value = True

        result = monitor.is_better(0.95, 0.9)

        assert result is True
        self.mock_metric_tracker.is_better.assert_called_once_with(0.95, 0.9)

    def test_metric_name_property_delegates_to_extractor_and_tracker(
        self,
    ) -> None:
        """Test that the metric_name property delegates correctly."""
        monitor = MetricMonitor(metric='accuracy')
        self.mock_extractor.metric_name = 'test_accuracy'
        self.mock_metric_tracker.metric_name = 'tracker_accuracy'

        assert monitor.metric_name == 'test_accuracy'


class TestMetricTracker:
    """Tests for MetricTracker class."""

    @pytest.fixture
    def tracker_auto(self) -> MetricTracker:
        """Set up a basic test instance."""
        return MetricTracker(
            metric_name='test_loss', min_delta=0.01, patience=2
        )

    @pytest.fixture
    def tracker_higher_is_better(self) -> MetricTracker:
        """Set up a test instance with higher is better."""
        return MetricTracker(
            metric_name='test_acc', min_delta=0.01, patience=2, best_is='higher'
        )

    @pytest.fixture
    def tracker_lower_is_better(self) -> MetricTracker:
        """Set up a test instance with lower is better."""
        return MetricTracker(
            metric_name='test_loss', min_delta=0.01, patience=2, best_is='lower'
        )

    def test_init_auto(self, tracker_auto) -> None:
        """Test basic instantiation."""
        assert tracker_auto.metric_name == 'test_loss'
        assert tracker_auto.best_is == 'auto'
        assert tracker_auto.min_delta == 0.01
        assert tracker_auto.patience == 2
        assert len(tracker_auto.history) == 0

    def test_resolve_best_is(self, tracker_auto) -> None:
        """Test the direction is set only while it is automatic."""
        tracker_auto.resolve_best_is('higher')
        tracker_auto.resolve_best_is('lower')

        assert tracker_auto.resolved_best_is == 'higher'
        assert tracker_auto.best_is == 'auto'

    def test_resolved_best_is_not_resolved(self, tracker_auto) -> None:
        """Test the direction raises before it is resolved."""
        with pytest.raises(exceptions.DirectionNotResolvedError):
            _ = tracker_auto.resolved_best_is

    def test_resolve_metric_name(self) -> None:
        """Test the metric name is set only while it is unknown."""
        tracker = MetricTracker()

        tracker.resolve_metric_name('first')
        tracker.resolve_metric_name('second')

        assert tracker.metric_name == 'first'

    def test_negative_patience(self) -> None:
        """Test invalid patience."""
        with pytest.raises(ValueError):
            MetricTracker(patience=-1)

    def test_best_result_not_available(self, tracker_auto) -> None:
        """Test calling the best result before any values are added fails."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = tracker_auto.best_value

    def test_filtered_result_not_available(self, tracker_auto) -> None:
        """Test calling filtered_value before any values are added fails."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = tracker_auto.filtered_value

        tracker_auto.add_value(5.0)

        assert tracker_auto.filtered_value == 5.0
        assert tracker_auto.best_value == 5.0

    def test_add_value(self, tracker_auto) -> None:
        """Test adding values to history."""
        tracker_auto.add_value(1.0)
        tracker_auto.add_value(2.0)

        assert tracker_auto.history == [1.0, 2.0]

    def test_filtered_value_default(self, tracker_auto) -> None:
        """Test the default aggregation method (last value)."""
        tracker_auto.add_value(1.0)
        tracker_auto.add_value(2.0)
        tracker_auto.add_value(3.0)

        assert tracker_auto.filtered_value == 3.0

    def test_is_improving_with_better_value_higher(
        self, tracker_higher_is_better
    ) -> None:
        """Test is_improving for improvement when higher is better."""
        tracker_higher_is_better.add_value(1.0)
        tracker_higher_is_better.add_value(2.0)

        assert tracker_higher_is_better.is_improving() is True

    def test_is_improving_with_worse_value_higher(
        self, tracker_higher_is_better
    ) -> None:
        """Test is_improving for worse results when higher is better."""
        tracker_higher_is_better.add_value(2.0)
        tracker_higher_is_better.add_value(1.0)

        assert tracker_higher_is_better.is_improving() is False

    def test_is_improving_with_better_value_lower(
        self, tracker_lower_is_better
    ) -> None:
        """Test is_improving for improvement when lower is better."""
        tracker_lower_is_better.add_value(2.0)
        tracker_lower_is_better.add_value(1.0)

        assert tracker_lower_is_better.is_improving() is True

    def test_is_improving_with_worse_value_lower(
        self, tracker_lower_is_better
    ) -> None:
        """Test is_improving for worse results when lower is better."""
        tracker_lower_is_better.add_value(1.0)
        tracker_lower_is_better.add_value(2.0)

        assert tracker_lower_is_better.is_improving() is False

    @pytest.mark.parametrize(
        ('first', 'second', 'expected'),
        [(1.0, 2.0, 'higher'), (2.0, 1.0, 'lower')],
    )
    def test_auto_best_is_determination(
        self, tracker_auto, first, second, expected
    ) -> None:
        """Test the direction is guessed from the first two values."""
        tracker_auto.add_value(first)
        tracker_auto.add_value(second)

        with pytest.warns(exceptions.DirectionGuessedWarning):
            improving = tracker_auto.is_improving()

        assert improving is True
        assert tracker_auto.resolved_best_is == expected

    def test_improvement_with_tolerance(self, tracker_higher_is_better) -> None:
        """Test improvement detection considering min_delta."""
        tracker_higher_is_better.add_value(1.0)
        assert tracker_higher_is_better.is_improving()

        tracker_higher_is_better.add_value(1.009)
        assert not tracker_higher_is_better.is_improving()

        tracker_higher_is_better.add_value(1.011)
        assert tracker_higher_is_better.is_improving()

    def test_patience_countdown(self, tracker_higher_is_better) -> None:
        """Test patience countdown mechanism."""
        tracker_higher_is_better.add_value(2.0)
        assert tracker_higher_is_better.is_improving()
        assert tracker_higher_is_better.is_patient()

        tracker_higher_is_better.add_value(1.0)  # worse
        assert not tracker_higher_is_better.is_improving()
        assert tracker_higher_is_better.is_patient()

        tracker_higher_is_better.add_value(1.0)  # still worse
        assert not tracker_higher_is_better.is_improving()
        assert not tracker_higher_is_better.is_patient()

    def test_patience_reset_on_improvement(
        self, tracker_higher_is_better
    ) -> None:
        """Test patience resets when improvement occurs."""
        tracker_higher_is_better.add_value(1.0)
        tracker_higher_is_better.add_value(0.5)  # worse
        assert not tracker_higher_is_better.is_improving()
        assert tracker_higher_is_better.is_patient()

        tracker_higher_is_better.add_value(1.5)  # better
        assert tracker_higher_is_better.is_improving()
        assert tracker_higher_is_better.is_patient()

    def test_reset_patience_method(self, tracker_higher_is_better) -> None:
        """Test manual patience reset."""
        tracker_higher_is_better.add_value(2.0)
        tracker_higher_is_better.add_value(1.0)  # worse
        assert not tracker_higher_is_better.is_improving()

        tracker_higher_is_better.add_value(1.0)  # worse
        assert not tracker_higher_is_better.is_improving()
        assert not tracker_higher_is_better.is_patient()

        tracker_higher_is_better.reset_patience()
        assert tracker_higher_is_better.is_patient()

    def test_is_better_with_nan(self, tracker_higher_is_better) -> None:
        """Test is_better handles NaN values correctly."""
        assert not tracker_higher_is_better.is_better(float('nan'), 1.0)

    def test_single_value_always_improving(
        self, tracker_higher_is_better
    ) -> None:
        """Test that single values are always considered improving."""
        tracker_higher_is_better.add_value(1.0)

        assert tracker_higher_is_better.is_improving()
