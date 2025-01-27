"""Tests for the calculating module."""
from collections.abc import Callable

import pytest
import torch

from src.dry_torch import exceptions
from src.dry_torch import protocols as p
from src.dry_torch.calculating import Metric, MetricCollection
from src.dry_torch.calculating import Loss, CompositionalLoss
from src.dry_torch.calculating import dict_apply, from_torchmetrics
from src.dry_torch.calculating import repr_metrics


@pytest.fixture()
def metric_1() -> str:
    """Simple metric."""
    return 'Metric_1'


@pytest.fixture()
def metric_2() -> str:
    """Another simple metric."""
    return 'Metric_2'


@pytest.fixture()
def metric_fun_1(metric_1: str) -> dict[str, Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor]
]:
    """Simple metric fun."""
    return {metric_1: lambda x, y: x}


@pytest.fixture()
def metric_fun_2(metric_2: str) -> dict[str, Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor]
]:
    """Another simple metric fun."""
    return {metric_2: lambda x, y: y}


class TestFromTorchMetrics:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up a torchmetrics-based test fixture."""

        # Conditionally import torchmetrics
        torchmetrics = pytest.importorskip("torchmetrics")

        self.metric_a = torchmetrics.Accuracy(task='binary')
        self.metric_b = torchmetrics.MeanSquaredError()
        self.additive_metric = 2 * self.metric_a + self.metric_b
        self.metric = from_torchmetrics(self.additive_metric)
        return

    def test_update_and_compute(self) -> None:
        """Test it correctly updates and computes metrics as dictionaries."""
        mock_outputs = torch.tensor([0.1, 0.2])
        mock_targets = torch.tensor([1, 0])

        self.metric.update(mock_outputs, mock_targets)
        result = self.metric.compute()

        # Note that update modifies the state of the input metric
        expected = {self.metric_a.__class__.__name__: self.metric_a.compute(),
                    self.metric_b.__class__.__name__: self.metric_b.compute()}

        assert result == expected

    def test_forward(self) -> None:
        """Test that forward still outputs a Tensor with the correct value."""
        mock_outputs = torch.tensor([0.1, 0.2])
        mock_targets = torch.tensor([1, 0])

        # Update the wrapped metric
        result = self.metric.forward(mock_outputs, mock_targets)

        # Note that forward modifies the state of the input metric
        expected = self.additive_metric.compute()

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, expected)


class TestMetricCollection:
    @pytest.fixture(autouse=True)
    def setup(self, metric_fun_1, metric_fun_2) -> None:
        """Set up a MetricCollection instance with simple metric functions."""

        metric_fun_dict = metric_fun_1 | metric_fun_2
        self.metrics = MetricCollection(**metric_fun_dict)
        return

    def test_calculate(self, metric_1, metric_2) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}

        assert self.metrics.calculate(simple_outputs,
                                      simple_targets) == expected
        return

    def test_update_compute_and_reset(self, metric_1, metric_2) -> None:
        """Test it stores, reduces, and resets metrics correctly."""
        simple_outputs_1 = torch.tensor(1)
        simple_targets_1 = torch.tensor(0)
        simple_outputs_2 = torch.tensor(3)
        simple_targets_2 = torch.tensor(2)

        self.metrics.update(simple_outputs_1, simple_targets_1)
        self.metrics.update(simple_outputs_2, simple_targets_2)
        expected = {metric_1: torch.tensor(2), metric_2: torch.tensor(1)}

        assert self.metrics.compute() == expected

        self.metrics.reset()
        with pytest.warns(exceptions.ComputedBeforeUpdatedWarning):
            assert {} == self.metrics.compute()

        self.metrics.update(simple_outputs_1, simple_targets_1)
        expected = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}

        assert self.metrics.compute() == expected

    def test_or(self, metric_1, metric_2) -> None:
        """Test | works as a union operator."""

        new_metric_fun_dict = {'NewMetric': lambda x, y: torch.tensor(0.5)}
        new_metrics = MetricCollection(**new_metric_fun_dict)

        combined_metrics = self.metrics | new_metrics

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

    @pytest.fixture(autouse=True)
    def setup(self, metric_1, metric_fun_1) -> None:
        """Set up a Metric instance with a simple metric function."""
        self.simple_fun = next(iter(metric_fun_1.values()))
        self.metrics = Metric(metric_1, self.simple_fun)
        return

    def test_calculate(self, metric_1) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {metric_1: torch.tensor(1)}

        assert self.metrics.calculate(simple_outputs,
                                      simple_targets) == expected
        return

    def test_or(self, metric_1) -> None:
        """Test | works as a union operator."""

        new_metrics = Metric('NewMetric', lambda x, y: torch.tensor(0.5))

        combined_metrics = self.metrics | new_metrics

        expected_keys = {metric_1, 'NewMetric'}
        assert set(combined_metrics.named_metric_fun.keys()) == expected_keys

        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)
        combined_metrics.update(simple_outputs, simple_targets)

        expected = {metric_1: torch.tensor(1), 'NewMetric': torch.tensor(0.5)}
        assert combined_metrics.compute() == expected


class TestCompositionalLoss:
    @pytest.fixture(autouse=True)
    def setup(self, metric_1, metric_2, metric_fun_1, metric_fun_2) -> None:
        """Set up a CompositionalLoss instance with simple arguments."""
        self.loss_1 = CompositionalLoss(lambda x: 2 * x[metric_1],
                                        **metric_fun_1)
        self.loss_2 = CompositionalLoss(lambda x: 3 * x[metric_2],
                                        **metric_fun_2)
        self.example_metric_results = {metric_1: torch.tensor(1),
                                       metric_2: torch.tensor(2)}
        return

    def test_calculate(self, metric_1) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {'Loss': torch.tensor(2), metric_1: torch.tensor(1)}

        assert self.loss_1.calculate(simple_outputs, simple_targets) == expected
        return

    def test_add_losses(self) -> None:
        """Test addition of two losses."""
        combined_loss = self.loss_1 + self.loss_2
        assert combined_loss.criterion(self.example_metric_results) == 2 + 6

    def test_subtract_losses(self) -> None:
        """Test subtraction of two losses."""
        combined_loss = self.loss_1 - self.loss_2
        assert combined_loss.criterion(self.example_metric_results) == 2 - 6

    def test_multiply_losses(self) -> None:
        """Test multiplication of two losses."""
        combined_loss = self.loss_1 * self.loss_2
        assert combined_loss.criterion(self.example_metric_results) == 2 * 6

    def test_divide_losses(self) -> None:
        """Test division of two losses."""
        combined_loss = self.loss_1 / self.loss_2
        assert combined_loss.criterion(self.example_metric_results) == 2 / 6

    def test_negate_loss(self) -> None:
        """Test negation of a loss."""
        neg_loss = -self.loss_1
        assert neg_loss.criterion(self.example_metric_results) == -2


class TestLoss:
    @pytest.fixture(autouse=True)
    def setup(self, metric_1, metric_fun_1) -> None:
        """Set up a Loss instance with simple arguments."""
        self.loss_1 = Loss(metric_1, next(iter(metric_fun_1.values())))
        self.example_metric_results = {metric_1: torch.tensor(1)}
        return

    def test_add_float(self) -> None:
        """Test addition by float."""
        combined_loss = self.loss_1 + 3
        assert combined_loss.criterion(self.example_metric_results) == 1 + 3

    def test_subtract_float(self) -> None:
        """Test subtraction by float."""
        combined_loss = self.loss_1 - 3
        assert combined_loss.criterion(self.example_metric_results) == 1 - 3

    def test_multiply_float(self) -> None:
        """Test multiplication by float."""
        combined_loss = self.loss_1 * 3
        assert combined_loss.criterion(self.example_metric_results) == 1 * 3

    def test_divide_float(self) -> None:
        """Test division by float."""
        combined_loss = self.loss_1 / 3
        assert combined_loss.criterion(self.example_metric_results) == 1 / 3

    def test_float_add(self) -> None:
        """Test addition to float."""
        combined_loss = 3 + self.loss_1
        assert combined_loss.criterion(self.example_metric_results) == 3 + 1

    def test_float_subtract(self) -> None:
        """Test subtraction by float."""
        combined_loss = 3 - self.loss_1
        assert combined_loss.criterion(self.example_metric_results) == 3 - 1

    def test_float_multiply(self) -> None:
        """Test multiplication by float."""
        combined_loss = 3 * self.loss_1
        assert combined_loss.criterion(self.example_metric_results) == 3 * 1

    def test_float_divide(self) -> None:
        """Test division by float."""
        combined_loss = self.loss_1 / 3
        assert combined_loss.criterion(self.example_metric_results) == 1 / 3


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
    "compute_return, class_name, expected",
    [
        # Case 1: Mapping of metrics
        (
                {"metric_1": torch.tensor(1), "metric_2": torch.tensor(2)},
                None,
                {"metric_1": 1, "metric_2": 2},
        ),
        # Case 2: Single tensor
        (
                torch.tensor(0.5),
                "metric_1",
                {"metric_1": 0.5},
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
    mock_calculator = mocker.MagicMock(spec=p.MetricCalculatorProtocol)
    mock_calculator.compute.return_value = compute_return

    if class_name:
        mock_calculator.__class__.__name__ = class_name

    # Call the function and assert the result
    result = repr_metrics(mock_calculator)
    assert result == expected
