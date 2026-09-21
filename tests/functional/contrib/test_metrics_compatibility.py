"""Tests that third-party metrics match the objective protocols."""

from typing import TYPE_CHECKING, Any

import torch

import pytest

from drytorch.core import protocols as p


if TYPE_CHECKING:
    import torchmetrics

    from torcheval import metrics
    from torchmetrics import metric as torchmetric

    from drytorch.contrib.torcheval import from_torcheval
    from drytorch.contrib.torchmetrics import from_torchmetrics
else:
    torchmetrics = pytest.importorskip('torchmetrics')
    metrics = pytest.importorskip('torcheval.metrics')
    torchmetric = pytest.importorskip('torchmetrics.metric')

    from drytorch.contrib.torcheval import from_torcheval
    from drytorch.contrib.torchmetrics import from_torchmetrics


_torcheval_objective: type[p.ObjectiveProtocol[Any, Any]] = (
    metrics.MeanSquaredError
)
_torchmetrics_direction: type[p.HigherIsBetterProtocol] = torchmetrics.Metric
_torchmetrics_loss: type[p.LossProtocol[torch.Tensor, torch.Tensor]] = (
    torchmetrics.MeanSquaredError
)
_torchmetrics_objective: type[p.ObjectiveProtocol[Any, Any]] = (
    torchmetrics.Metric
)


def _wrap_torcheval(
    metric: 'metrics.Metric[torch.Tensor | dict[str, torch.Tensor]]',
) -> p.SyncedObjectiveProtocol[torch.Tensor, torch.Tensor]:
    """Check the wrapper accepts a metric and returns a synced objective."""
    return from_torcheval(metric)


def _wrap_torchmetrics(
    metric: 'torchmetric.CompositionalMetric',
) -> p.SyncedLossProtocol[torch.Tensor, torch.Tensor]:
    """Check the wrapper accepts a composition and returns a synced loss."""
    return from_torchmetrics(metric)


@pytest.fixture
def torcheval_metric() -> 'metrics.MeanSquaredError':
    """A torcheval metric."""
    return metrics.MeanSquaredError()


@pytest.fixture
def torchmetrics_metric() -> 'torchmetrics.MeanSquaredError':
    """A torchmetrics metric."""
    return torchmetrics.MeanSquaredError()


@pytest.fixture
def wrapped_torcheval_metric(
    torcheval_metric,
) -> p.SyncedObjectiveProtocol[torch.Tensor, torch.Tensor]:
    """A torcheval metric wrapped for the library."""
    return from_torcheval(torcheval_metric)


@pytest.fixture
def wrapped_torchmetrics_metric(
    torchmetrics_metric,
) -> p.SyncedLossProtocol[torch.Tensor, torch.Tensor]:
    """A torchmetrics composition wrapped for the library."""
    return from_torchmetrics(1 + torchmetrics_metric)


@pytest.mark.parametrize(
    ('fixture_name', 'protocol'),
    [
        ('torcheval_metric', p.ObjectiveProtocol),
        ('torchmetrics_metric', p.HigherIsBetterProtocol),
        ('torchmetrics_metric', p.LossProtocol),
        ('torchmetrics_metric', p.ObjectiveProtocol),
        ('wrapped_torcheval_metric', p.HigherIsBetterProtocol),
        ('wrapped_torcheval_metric', p.NamedProtocol),
        ('wrapped_torcheval_metric', p.ObjectiveProtocol),
        ('wrapped_torcheval_metric', p.SyncProtocol),
        ('wrapped_torchmetrics_metric', p.HigherIsBetterProtocol),
        ('wrapped_torchmetrics_metric', p.LossProtocol),
        ('wrapped_torchmetrics_metric', p.NamedProtocol),
        ('wrapped_torchmetrics_metric', p.ObjectiveProtocol),
        ('wrapped_torchmetrics_metric', p.SyncProtocol),
    ],
)
def test_third_party_metric_matches_protocol(
    request, fixture_name, protocol
) -> None:
    """Test the third-party metric satisfies the protocol."""
    instance = request.getfixturevalue(fixture_name)

    assert isinstance(instance, protocol)


def test_torchmetrics_sync_preserves_computed_values(
    wrapped_torchmetrics_metric,
) -> None:
    """Test sync on a single process leaves computed values unchanged."""
    outputs = torch.tensor([0.1, 0.2])
    targets = torch.tensor([1.0, 0.0])
    wrapped_torchmetrics_metric.update(outputs, targets)
    before = wrapped_torchmetrics_metric.compute()

    wrapped_torchmetrics_metric.sync()
    after = wrapped_torchmetrics_metric.compute()

    assert before == after
