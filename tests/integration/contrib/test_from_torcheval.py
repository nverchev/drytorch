"""Tests for the "from torcheval" module."""

from typing import TYPE_CHECKING

import torch

import pytest

from drytorch.contrib.torcheval import from_torcheval
from drytorch.core import protocols as p


if TYPE_CHECKING:
    from torcheval import metrics

else:
    metrics = pytest.importorskip('torcheval.metrics')

_Tensor = torch.Tensor


class TestFromTorchMetrics:
    """Tests for integration with torchmetrics."""

    @pytest.fixture
    def torcheval_metric(self) -> metrics.Metric:
        """Fixture for accuracy metric."""
        return metrics.BinaryAccuracy()

    @pytest.fixture
    def metric(self, torcheval_metric) -> p.ObjectiveProtocol[_Tensor, _Tensor]:
        """Fixture for accuracy metric."""
        return from_torcheval(torcheval_metric)

    @pytest.fixture
    def mock_outputs(self) -> torch.Tensor:
        """Fixture for mock outputs."""
        return torch.tensor([0.1, 0.2])

    @pytest.fixture
    def mock_targets(self) -> torch.Tensor:
        """Fixture for mock targets."""
        return torch.tensor([1, 0])

    def test_reset_update_and_compute(
        self,
        metric,
        mock_outputs: torch.Tensor,
        mock_targets: torch.Tensor,
    ) -> None:
        """Test it correctly updates and computes metrics as dictionaries."""
        metric.reset()
        metric.update(mock_outputs, mock_targets)
        result = metric.compute()
        assert isinstance(result, torch.Tensor)

    def test_sync(self, mocker, metric) -> None:
        """Test it correctly synchronizes metrics across processes."""
        sync = mocker.patch('torcheval.metrics.toolkit.sync_and_compute')
        metric.sync()

        sync.assert_called_once_with(metric.metric)
