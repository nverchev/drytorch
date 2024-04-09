import pytest

from custom_trainer.metrics_and_loss_base_classes import LossAndMetrics
import torch


class SlottedLoss(LossAndMetrics):
    __slots__: tuple[str, ...] = ('criterion', 'test')


my_loss1 = SlottedLoss(criterion=torch.tensor(1.0), test=torch.tensor(1.0))
my_loss2 = LossAndMetrics(criterion=torch.tensor(1.0))


def ShowMetrics(loss_and_metrics: LossAndMetrics) -> None:
    print(loss_and_metrics.metrics)
