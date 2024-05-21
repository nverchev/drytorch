import pytest
import torch
from dry_torch import LossAndMetricsCalculator
from dry_torch import MetricsCalculator
from dry_torch.loss_and_metrics import Metrics
from dry_torch.loss_and_metrics import LossAndMetrics


def test_SlottedMetrics():
    slotted_metrics = Metrics(test=torch.tensor(1))
    assert slotted_metrics.metrics == {'Test': torch.tensor(1.0)}


def test_SlottedCriterion():
    slotted_criterion = LossAndMetrics(criterion=torch.tensor(2.0),
                                       test=torch.tensor(1.0))
    assert slotted_criterion.metrics == {'Criterion': torch.tensor(2.0),
                                         'Test': torch.tensor(1.0)}


def test_MetricsFunction():
    metrics_fun = MetricsCalculator(CrossEntropy=torch.nn.CrossEntropyLoss())
    metrics = metrics_fun(torch.FloatTensor([1.0]), torch.FloatTensor([2.0]))
    assert metrics.metrics == dict(CrossEntropy=torch.tensor(0.))


def test_LossFunction():
    loss_fun = LossAndMetricsCalculator(
        loss_fun=torch.nn.MSELoss(),
        CrossEntropy=torch.nn.CrossEntropyLoss()
    )
    loss = loss_fun(torch.FloatTensor([1.0]), torch.FloatTensor([2.0]))
    assert loss.criterion == torch.tensor(1.0)
    assert loss.metrics == dict(Criterion=torch.tensor(1.0),
                                CrossEntropy=torch.tensor(0.))
