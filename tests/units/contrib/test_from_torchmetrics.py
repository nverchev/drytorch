"""Tests for the "from torchmetrics" module."""

import pytest
import torch

from drytorch.contrib.from_torchmetrics import from_torchmetrics


class TestFromTorchMetrics:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up a torchmetrics-based test fixture."""

        # Conditionally import torchmetrics
        torchmetrics = pytest.importorskip('torchmetrics')

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
