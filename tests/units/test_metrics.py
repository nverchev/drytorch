"""Tests for the "metrics" module."""

from collections.abc import Callable

import torch

import pytest

from drytorch import exceptions
from drytorch import protocols as p
from drytorch.metrics import (
    CompositionalLoss,
    Loss,
    Metric,
    MetricCollection,
    MetricMonitor,
    dict_apply,
    repr_metrics,
)


_Tensor = torch.Tensor


@pytest.fixture(scope="module")
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
        assert set(combined_metrics.named_metric_fun.keys()) == expected_keys

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
        assert set(combined_metrics.named_metric_fun.keys()) == expected_keys

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
        expected = {'Loss': torch.tensor(2.0), metric_1: torch.tensor(1.0)}
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
        return Loss(next(iter(metric_fun_1.values())), name=metric_1)

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


class TestMetricMonitor:
    """Tests for MetricMonitor class."""

    @pytest.fixture()
    def monitor_from_str_metric(self, mock_metric) -> MetricMonitor:
        """Set up a test instance."""
        return MetricMonitor(
            metric=mock_metric.name, min_delta=0.01, patience=2
        )

    @pytest.fixture()
    def monitor_from_metric_object(self, mock_metric) -> MetricMonitor:
        """Set up a test instance."""
        return MetricMonitor(metric=mock_metric, min_delta=0.01, patience=2)

    def test_init_with_string_metric(
        self, monitor_from_str_metric, mock_metric
    ) -> None:
        """Test instantiating class with a string for the metric."""
        assert monitor_from_str_metric.metric_name == mock_metric.name
        assert monitor_from_str_metric.best_is == 'auto'

    def test_init_with_metric_object(
        self, monitor_from_metric_object, mock_metric
    ) -> None:
        """Test instantiating class with a metric-like object."""
        assert monitor_from_metric_object.metric_name == mock_metric.name
        assert monitor_from_metric_object.best_is == 'higher'

    def test_negative_patience(self) -> None:
        """Test invalid patience."""
        with pytest.raises(ValueError):
            MetricMonitor(patience=-1)

    def test_get_monitor(self, mock_trainer, monitor_from_str_metric) -> None:
        """Test getting monitored values."""
        expected = mock_trainer.validation
        assert monitor_from_str_metric._get_monitor(mock_trainer) == expected
        mock_trainer.validation = None
        expected = mock_trainer
        assert monitor_from_str_metric._get_monitor(mock_trainer) == expected

    def test_best_result_not_available(self, monitor_from_str_metric) -> None:
        """Test calling best result before the monitor has started fails."""
        with pytest.raises(exceptions.ResultNotAvailableError):
            _ = monitor_from_str_metric.best_value

    def test_aggregate_fn_selection(self, monitor_from_str_metric) -> None:
        """Test default aggregation method."""
        assert monitor_from_str_metric.filter_fn([1, 2, 3]) == 3

    def test_is_improving_with_better_value(
        self, monitor_from_str_metric
    ) -> None:
        """Test is_improving returns True for improvement."""
        monitor_from_str_metric.best_is = 'higher'
        monitor_from_str_metric.patience = 0
        monitor_from_str_metric.history.append(1.0)
        monitor_from_str_metric.history.append(2.0)
        assert monitor_from_str_metric.is_improving() is True

    def test_is_improving_with_worse_value(
        self, monitor_from_metric_object
    ) -> None:
        """Test is_improving returns False for worse result."""
        monitor_from_metric_object.best_is = 'higher'
        monitor_from_metric_object.patience = 0
        monitor_from_metric_object.history.append(2.0)
        monitor_from_metric_object.history.append(1.0)
        assert monitor_from_metric_object.is_improving() is False

    def test_auto_best_is_determination(self, monitor_from_str_metric) -> None:
        """Test auto-determination of whether higher is better."""
        monitor_from_str_metric.best_is = 'auto'
        monitor_from_str_metric.patience = 0
        monitor_from_str_metric.history.append(1.0)
        monitor_from_str_metric.history.append(2.0)
        assert monitor_from_str_metric.is_improving() is True
        assert monitor_from_str_metric.best_is == 'higher'

    def test_improvement_with_tolerance(self, monitor_from_str_metric) -> None:
        """Test improvement detection considering min_delta."""
        monitor_from_str_metric.best_is = 'higher'
        monitor_from_str_metric.patience = 0
        monitor_from_str_metric.history.append(1.0)
        assert monitor_from_str_metric.is_improving()

        monitor_from_str_metric.history.append(1.009)
        assert not monitor_from_str_metric.is_improving()

        monitor_from_str_metric.history.append(1.011)
        assert monitor_from_str_metric.is_improving()
