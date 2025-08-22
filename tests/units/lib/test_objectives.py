"""Tests for the "objectives" module."""

from collections.abc import Callable

import torch

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib.objectives import (
    CompositionalLoss,
    Loss,
    Metric,
    MetricCollection,
    MetricTracker,
    dict_apply,
    repr_metrics,
)


_Tensor = torch.Tensor


@pytest.fixture(scope='module')
def metric_1() -> str:
    """Simple metric."""
    return 'Metric_1'


@pytest.fixture(scope='module')
def metric_2() -> str:
    """Another simple metric."""
    return 'Metric_2'


@pytest.fixture(scope='module')
def metric_fun_1(
    metric_1: str,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Simple metric fun."""
    return {metric_1: lambda x, y: x}


@pytest.fixture(scope='module')
def metric_fun_2(
    metric_2: str,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Another simple metric fun."""
    return {metric_2: lambda x, y: y}


class TestMetricCollection:
    """Tests for MetricCollection."""

    @pytest.fixture(scope='class')
    def metrics(self, metric_fun_1, metric_fun_2) -> MetricCollection:
        """Set up a MetricCollection instance with simple metric functions."""
        metric_fun_dict = metric_fun_1 | metric_fun_2
        return MetricCollection(**metric_fun_dict)

    def test_calculate(self, metric_1, metric_2, metrics) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}

        assert metrics.calculate(simple_outputs, simple_targets) == expected
        return

    def test_update_compute_and_reset(
        self, metric_1, metric_2, metrics
    ) -> None:
        """Test it stores, reduces, and resets metrics correctly."""
        simple_outputs_1 = torch.tensor(1)
        simple_targets_1 = torch.tensor(0)
        simple_outputs_2 = torch.tensor(3)
        simple_targets_2 = torch.tensor(2)

        metrics.update(simple_outputs_1, simple_targets_1)
        metrics.update(simple_outputs_2, simple_targets_2)
        expected = {metric_1: torch.tensor(2), metric_2: torch.tensor(1)}

        assert metrics.compute() == expected

        metrics.reset()
        with pytest.warns(exceptions.ComputedBeforeUpdatedWarning):
            assert {} == metrics.compute()

        metrics.update(simple_outputs_1, simple_targets_1)
        expected = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}

        assert metrics.compute() == expected

    def test_or(self, metric_1, metric_2, metrics) -> None:
        """Test | works as a union operator."""
        new_metric_fun_dict = {'NewMetric': lambda x, y: torch.tensor(0.5)}
        new_metrics = MetricCollection(**new_metric_fun_dict)

        combined_metrics = metrics | new_metrics

        expected_keys = {metric_1, metric_2, 'NewMetric'}
        assert set(combined_metrics.named_fn.keys()) == expected_keys

        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)
        combined_metrics.update(simple_outputs, simple_targets)

        expected = {
            metric_1: torch.tensor(1),
            metric_2: torch.tensor(0),
            'NewMetric': torch.tensor(0.5),
        }
        assert combined_metrics.compute() == expected


class TestMetric:
    """Tests for Metric."""

    @pytest.fixture(scope='class')
    def metric(self, metric_1, metric_fun_1) -> Metric:
        """Set up a Metric instance with a simple metric function."""
        self.simple_fun = next(iter(metric_fun_1.values()))
        return Metric(self.simple_fun, name=metric_1, higher_is_better=True)

    def test_calculate(self, metric_1, metric) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {metric_1: torch.tensor(1)}

        assert metric.calculate(simple_outputs, simple_targets) == expected
        return

    def test_or(self, metric_1, metric) -> None:
        """Test | works as a union operator."""
        new_metrics = Metric[_Tensor, _Tensor](
            lambda x, y: torch.tensor(0.5),
            name='NewMetric',
            higher_is_better=True,
        )

        combined_metrics = metric | new_metrics

        expected_keys = {metric_1, 'NewMetric'}
        assert set(combined_metrics.named_fn.keys()) == expected_keys

        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)
        combined_metrics.update(simple_outputs, simple_targets)

        expected = {metric_1: torch.tensor(1), 'NewMetric': torch.tensor(0.5)}
        assert combined_metrics.compute() == expected


class TestCompositionalLoss:
    """Tests for CompositionalLoss."""

    @pytest.fixture(scope='class')
    def example_metric_results(
        self, metric_1, metric_2
    ) -> dict[str, torch.Tensor]:
        """A possible calculated value for metrics."""
        return {
            metric_1: torch.tensor(1),
            metric_2: torch.tensor(2),
        }

    @pytest.fixture(scope='class')
    def loss_1(self, metric_1, metric_fun_1) -> CompositionalLoss:
        """Set up a base instance (as defined by the Loss subclass)."""
        # formula corresponds to what the formula components should look like
        return CompositionalLoss(
            lambda x: x[metric_1],
            formula=f'[{metric_1}]',
            higher_is_better=False,
            **metric_fun_1,
        )

    @pytest.fixture(scope='class')
    def loss_2(self, metric_2, metric_fun_2) -> CompositionalLoss:
        """Set up a second base instance (as defined by the Loss subclass)."""
        return CompositionalLoss(
            lambda x: x[metric_2],
            formula=f'[{metric_2}]',
            higher_is_better=False,
            **metric_fun_2,
        )

    @pytest.fixture(scope='class')
    def composed_loss_1(self, loss_1) -> CompositionalLoss:
        """Set up a CompositionalLoss instance with simple arguments."""
        return 2 * loss_1

    @pytest.fixture(scope='class')
    def composed_loss_2(self, loss_2) -> CompositionalLoss:
        """Set up a CompositionalLoss instance with simple arguments."""
        return 3 * loss_2

    def test_calculate(self, metric_1, composed_loss_1) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1.0)
        simple_targets = torch.tensor(0.0)
        expected = {
            'Combined Loss': torch.tensor(2.0),
            metric_1: torch.tensor(1.0),
        }
        assert (
            composed_loss_1.calculate(simple_outputs, simple_targets)
            == expected
        )

    def test_negate_loss(self, composed_loss_1, example_metric_results) -> None:
        """Test negation of a loss."""
        neg_loss = -composed_loss_1
        assert neg_loss.criterion(example_metric_results) == -2
        assert neg_loss.formula == '(-(2 x [Metric_1]))'

    def test_add_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test addition of two losses."""
        combined_loss = composed_loss_1 + composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 + 6
        assert combined_loss.formula == '(2 x [Metric_1] + 3 x [Metric_2])'

    def test_subtract_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test subtraction of two losses."""
        combined_loss = composed_loss_1 - -composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 + 6
        assert combined_loss.formula == '(2 x [Metric_1] - (-(3 x [Metric_2])))'

    def test_multiply_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test multiplication of two losses."""
        combined_loss = composed_loss_1 * composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 * 6
        assert combined_loss.formula == '(2 x [Metric_1]) x (3 x [Metric_2])'

    def test_divide_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test division of two losses."""
        combined_loss = composed_loss_1 / -composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 / -6
        expected = '(2 x [Metric_1]) x (1 / (-(3 x [Metric_2])))'
        assert combined_loss.formula == expected


class TestLoss:
    """Tests for Loss."""

    @pytest.fixture(scope='class')
    def example_metric_results(self, metric_1) -> dict[str, torch.Tensor]:
        """A possible calculated value for metrics."""
        return {metric_1: torch.tensor(2.0)}

    @pytest.fixture(scope='class')
    def loss_1(self, metric_1, metric_fun_1) -> Loss:
        """Set up a Loss instance with simple arguments."""
        return Loss(next(iter(metric_fun_1.values())), metric_1)

    def test_add_float(self, loss_1, example_metric_results) -> None:
        """Test addition by float."""
        combined_loss = loss_1 + 3
        assert combined_loss.criterion(example_metric_results) == 2 + 3
        assert combined_loss.formula == '(3 + [Metric_1])'

    def test_subtract_float(self, loss_1, example_metric_results) -> None:
        """Test subtraction by float."""
        combined_loss = loss_1 - 3
        assert combined_loss.criterion(example_metric_results) == 2 - 3
        assert combined_loss.formula == '(-3 + [Metric_1])'

    def test_multiply_float(self, loss_1, example_metric_results) -> None:
        """Test multiplication by float."""
        combined_loss = loss_1 * 3
        assert combined_loss.criterion(example_metric_results) == 2 * 3
        assert combined_loss.formula == '(3 x [Metric_1])'

    def test_divide_float(self, loss_1, example_metric_results) -> None:
        """Test division by float."""
        combined_loss = loss_1 / 3
        assert combined_loss.criterion(example_metric_results) == 2 / 3
        assert combined_loss.formula == '(0.3333333333333333 x [Metric_1])'

    def test_float_add(self, loss_1, example_metric_results) -> None:
        """Test addition to float."""
        combined_loss = 3 + loss_1
        assert combined_loss.criterion(example_metric_results) == 3 + 2
        assert combined_loss.formula == '(3 + [Metric_1])'

    def test_float_subtract(self, loss_1, example_metric_results) -> None:
        """Test subtraction to float."""
        combined_loss = 3 - loss_1
        assert combined_loss.criterion(example_metric_results) == 3 - 2
        assert combined_loss.formula == '(3 - [Metric_1])'

    def test_float_multiply(self, loss_1, example_metric_results) -> None:
        """Test multiplication to float."""
        combined_loss = 3 * loss_1
        assert combined_loss.criterion(example_metric_results) == 3 * 2
        assert combined_loss.formula == '(3 x [Metric_1])'

    def test_float_divide(self, loss_1, example_metric_results) -> None:
        """Test division to float."""
        combined_loss = 3 / loss_1
        assert combined_loss.criterion(example_metric_results) == 3 / 2
        assert combined_loss.formula == '(3 x (1 / [Metric_1]))'

    def test_positive_exp(self, loss_1, example_metric_results) -> None:
        """Test exponentiation by positive float."""
        combined_loss = loss_1**2
        assert combined_loss.criterion(example_metric_results) == 2**2
        assert combined_loss.formula == '([Metric_1]^2)'

    def test_negative_exp(self, loss_1, example_metric_results) -> None:
        """Test exponentiation by negative float."""
        combined_loss = loss_1**-2
        assert combined_loss.criterion(example_metric_results) == 2 ** (-2)
        assert combined_loss.formula == '(1 / [Metric_1]^2)'


def test_dict_apply(mocker) -> None:
    """Test it applies each function in the dict to outputs and targets."""
    mock_fun1 = mocker.MagicMock(return_value=torch.tensor(0.5))
    mock_fun2 = mocker.MagicMock(return_value=torch.tensor(0.8))
    dict_fun = {'fun1': mock_fun1, 'fun2': mock_fun2}

    mock_outputs = mocker.MagicMock()
    mock_targets = mocker.MagicMock()

    result = dict_apply(dict_fun, mock_outputs, mock_targets)

    assert result == {'fun1': torch.tensor(0.5), 'fun2': torch.tensor(0.8)}
    mock_fun1.assert_called_once_with(mock_outputs, mock_targets)
    mock_fun2.assert_called_once_with(mock_outputs, mock_targets)


@pytest.mark.parametrize(
    'compute_return, class_name, expected',
    [
        # Case 1: Mapping of metrics
        (
            {'metric_1': torch.tensor(1), 'metric_2': torch.tensor(2)},
            None,
            {'metric_1': 1, 'metric_2': 2},
        ),
        # Case 2: Single tensor
        (
            torch.tensor(0.5),
            'metric_1',
            {'metric_1': 0.5},
        ),
        # Case 3: None
        (
            None,
            None,
            {},
        ),
    ],
)
def test_repr_metrics(mocker, compute_return, class_name, expected):
    """Test the repr_metrics function with various compute return values."""
    # Mock the calculator
    mock_calculator = mocker.MagicMock(spec=p.ObjectiveProtocol)
    mock_calculator.compute.return_value = compute_return

    if class_name:
        mock_calculator.__class__.__name__ = class_name

    # Call the function and assert the result
    result = repr_metrics(mock_calculator)
    assert result == expected


class TestMetricTracker:
    """Tests for MetricTracker class."""

    @pytest.fixture()
    def tracker_auto(self) -> MetricTracker:
        """Set up a basic test instance."""
        return MetricTracker(
            metric_name='test_loss', min_delta=0.01, patience=2
        )

    @pytest.fixture()
    def tracker_higher_is_better(self) -> MetricTracker:
        """Set up a test instance with higher is better."""
        return MetricTracker(
            metric_name='test_acc', min_delta=0.01, patience=2, best_is='higher'
        )

    @pytest.fixture()
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

    def test_negative_patience(self) -> None:
        """Test invalid patience."""
        with pytest.raises(ValueError):
            MetricTracker(patience=-1)

    def test_best_result_not_available(self, tracker_auto) -> None:
        """Test calling best result before any values are added fails."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = tracker_auto.best_value

    def test_add_value(self, tracker_auto) -> None:
        """Test adding values to history."""
        tracker_auto.add_value(1.0)
        assert len(tracker_auto.history) == 1
        assert tracker_auto.history[0] == 1.0

        tracker_auto.add_value(2.0)
        assert len(tracker_auto.history) == 2
        assert tracker_auto.history[1] == 2.0

    def test_filtered_value_default(self, tracker_auto) -> None:
        """Test default aggregation method (last value)."""
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

    def test_auto_best_is_determination_higher(self, tracker_auto) -> None:
        """Test auto-determination when values are increasing."""
        tracker_auto.add_value(1.0)
        tracker_auto.add_value(2.0)
        assert tracker_auto.is_improving() is True
        assert tracker_auto.best_is == 'higher'

    def test_auto_best_is_determination_lower(self, tracker_auto) -> None:
        """Test auto-determination when values are decreasing."""
        tracker_auto.add_value(2.0)
        tracker_auto.add_value(1.0)
        assert tracker_auto.is_improving() is True
        assert tracker_auto.best_is == 'lower'

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
