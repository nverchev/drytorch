"""Tests for the calculating module"""

import pytest
import torch

from src.dry_torch import exceptions
from src.dry_torch.calculating import MetricsCalculator
from src.dry_torch.calculating import LossCalculator
from src.dry_torch.calculating import CompositeLossCalculator, dict_apply


class TestMetricsCalculator:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up a MetricsCalculator instance with mock metric functions."""
        metric_fun_dict = {
            'accuracy': mocker.MagicMock(return_value=torch.tensor(0.8))
        }
        self.calc = MetricsCalculator(**metric_fun_dict)
        return

    def test_calculate_metrics(self, mocker) -> None:
        """Test that MetricsCalculator calculates and stores metrics."""
        mock_outputs = mocker.MagicMock()
        mock_targets = mocker.MagicMock()

        self.calc.calculate(mock_outputs, mock_targets)

        # Ensure that dict_apply was called correctly
        assert self.calc.metrics == {'accuracy': torch.tensor(0.8)}
        return

    def test_reset_calculated(self) -> None:
        """Test reset_calculated sets _metrics to None."""
        self.calc._metrics = {'accuracy': torch.tensor(0.8)}
        self.calc.reset_calculated()
        assert self.calc._metrics is None
        return

    def test_access_before_calculate(self) -> None:
        """Test accessing metrics before calculation raises an error."""
        with pytest.raises(exceptions.AccessBeforeCalculateError):
            _ = self.calc.metrics
        return


class TestLossCalculator:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up a LossCalculator instance with mock arguments."""
        loss_fun = mocker.MagicMock(return_value=torch.tensor(0.5))
        metric_fun_dict = {
            'accuracy': mocker.MagicMock(return_value=torch.tensor(0.8))
        }
        self.calc = LossCalculator(loss_fun, **metric_fun_dict)
        return

    def test_calculate_loss_and_metrics(self, mocker) -> None:
        """Test that LossCalculator calculates and stores loss and metrics."""
        mock_outputs = mocker.MagicMock()
        mock_targets = mocker.MagicMock()

        self.calc.calculate(mock_outputs, mock_targets)

        # Check that loss and metrics were calculated
        assert self.calc.criterion == torch.tensor(0.5)
        assert self.calc.metrics == {'Criterion': torch.tensor(0.5),
                                     'accuracy': torch.tensor(0.8)}

    def test_criterion_access_before_calculate(self) -> None:
        """Test accessing criterion before calculation raises an error."""
        with pytest.raises(exceptions.AccessBeforeCalculateError):
            _ = self.calc.criterion

    def test_reset_calculated(self) -> None:
        """Test reset_calculated sets _metrics and _criterion to None."""
        self.calc._metrics = {'accuracy': torch.tensor(0.8)}
        self.calc._criterion = torch.tensor(1.0)
        self.calc.reset_calculated()
        assert self.calc._metrics is None
        assert self.calc._criterion is None


class TestCompositeLossCalculator:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up a CompositeLossCalculator instance with mock arguments."""
        components = [
            ('loss1', 0.6, mocker.MagicMock(return_value=torch.tensor(0.5))),
            ('loss2', 0.4, mocker.MagicMock(return_value=torch.tensor(0.3))),
        ]
        metric_fun_dict = {
            'accuracy': mocker.MagicMock(return_value=torch.tensor(0.8))
        }
        self.calc = CompositeLossCalculator(*components, **metric_fun_dict)
        return

    def test_calculate_composite_loss_and_metrics(self, mocker) -> None:
        """Test that it calculates weighted loss and metrics."""
        mock_outputs = mocker.MagicMock()
        mock_targets = mocker.MagicMock()

        self.calc.calculate(mock_outputs, mock_targets)

        # Calculate expected criterion based on components
        expected_criterion = 0.6 * 0.5 + 0.4 * 0.3
        assert torch.allclose(self.calc.criterion,
                              torch.tensor(expected_criterion))
        expected_metrics = {
            'Criterion': torch.tensor(expected_criterion),
            'loss1': torch.tensor(0.5),
            'loss2': torch.tensor(0.3),
            'accuracy': torch.tensor(0.8),
        }

        for key, expected_value in expected_metrics.items():
            actual_value = self.calc.metrics[key]
            assert torch.allclose(actual_value, expected_value)


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
