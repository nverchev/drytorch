import pytest
import torch
from dry_torch import SimpleLossCalculator
from dry_torch import MetricsCalculator
from dry_torch import structures


def test_aggregate_dict() -> None:
    unit_tensor = torch.tensor(1)
    aggregate_metrics = structures.TorchAggregate()
    aggregate_metrics['key'] = unit_tensor
    assert (aggregate_metrics + aggregate_metrics).reduce()['Key'] == 1
    aggregate_metrics += aggregate_metrics
    assert aggregate_metrics.reduce()['Key'] == 1
    other_aggregate = structures.TorchAggregate()
    other_aggregate['key2'] = .5 * unit_tensor
    resulting_aggregate = aggregate_metrics + other_aggregate
    assert resulting_aggregate.reduce()['Key'] == 1
    assert resulting_aggregate.reduce()['Key2'] == 0.5
    return


def test_MetricsFunction() -> None:
    metrics_calc = MetricsCalculator(BCE=torch.nn.BCELoss(reduction='none'))
    metrics_calc.calculate(torch.FloatTensor([1.0]), torch.FloatTensor([0.0]))
    expected = dict(BCE=torch.tensor(100.))
    assert metrics_calc.metrics == expected
    return


def test_LossFunction() -> None:
    loss_calc = SimpleLossCalculator(
        loss_fun=torch.nn.MSELoss(reduction='none'),
        BCE=torch.nn.BCELoss(reduction='none')
    )
    loss_calc.calculate(torch.FloatTensor([1.0]), torch.FloatTensor([0.0]))
    assert loss_calc.criterion == torch.tensor(1.0)
    expected = dict(criterion=torch.tensor(1.0), BCE=torch.tensor(100.))
    assert loss_calc.metrics == expected
    return
