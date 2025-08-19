"""Tests for the "from torchmetrics" module."""

from typing import TYPE_CHECKING

import torch

import pytest

from drytorch.contrib.torchmetrics import from_torchmetrics


if TYPE_CHECKING:
    import torchmetrics
else:
    torchmetrics = pytest.importorskip('torchmetrics')


class TestFromTorchMetrics:
    """Tests for integration with torchmetrics."""

    @pytest.fixture
    def metric_a(self) -> torchmetrics.Metric:
        """Fixture for metric A."""
        return torchmetrics.Accuracy(task='binary')

    @pytest.fixture
    def metric_b(self) -> torchmetrics.Metric:
        """Fixture for metric B."""
        return torchmetrics.MeanSquaredError()

    @pytest.fixture
    def additive_metric(self, metric_a, metric_b):
        """Fixture for additive metric."""
        return 2 * metric_a + metric_b

    @pytest.fixture
    def metric(self, additive_metric):
        """Fixture for wrapped metric."""
        return from_torchmetrics(additive_metric)

    @pytest.fixture
    def mock_outputs(self) -> torch.Tensor:
        """Fixture for mock outputs."""
        return torch.tensor([0.1, 0.2])

    @pytest.fixture
    def mock_targets(self) -> torch.Tensor:
        """Fixture for mock targets."""
        return torch.tensor([1, 0])

    def test_update_and_compute(
        self,
        metric,
        metric_a,
        metric_b,
        mock_outputs: torch.Tensor,
        mock_targets: torch.Tensor,
    ) -> None:
        """Test it correctly updates and computes metrics as dictionaries."""
        metric.update(mock_outputs, mock_targets)
        result = metric.compute()

        # update modifies the state of the input metric
        expected = {
            metric_a.__class__.__name__: metric_a.compute(),
            metric_b.__class__.__name__: metric_b.compute(),
        }

        assert result == expected

    def test_forward(
        self,
        metric,
        additive_metric,
        mock_outputs: torch.Tensor,
        mock_targets: torch.Tensor,
    ) -> None:
        """Test that forward still outputs a Tensor with the correct value."""
        result = metric.forward(mock_outputs, mock_targets)

        # forward modifies the state of the input metric
        expected = additive_metric.compute()

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, expected)

    def test_reset(
        self, metric, mock_outputs: torch.Tensor, mock_targets: torch.Tensor
    ) -> None:
        """Test that reset properly resets the underlying metrics."""
        metric.update(mock_outputs, mock_targets)
        pre_reset_result = metric.compute()
        assert pre_reset_result['BinaryAccuracy'] == 0.5

        metric.reset()

        # all correct prediction
        metric.update(mock_targets, mock_targets)
        post_reset_result = metric.compute()
        assert post_reset_result['BinaryAccuracy'] == 1
