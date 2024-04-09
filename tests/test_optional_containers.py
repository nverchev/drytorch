import pytest

from custom_trainer.dict_list import TorchDictList
from custom_trainer.optional_containers import (PreferablySlotted, BaseLossAndMetricsContainer,
                                                BaseMetricsContainer, OutputsContainer)
import torch

from custom_trainer.trainer import LossAndMetricsProtocol, MetricsProtocol


def print_metrics(loss_and_metrics: MetricsProtocol) -> None:
    print(loss_and_metrics.metrics)


def print_criterion_and_metrics(loss_and_metrics: LossAndMetricsProtocol) -> None:
    print(loss_and_metrics.criterion)
    print(loss_and_metrics.metrics)


def test_PreferablySlotted():
    class Slotted(PreferablySlotted):
        __slots__ = ('test',)

    slotted = Slotted()
    slotted.test = 1
    with pytest.raises(AttributeError):
        Slotted(test1=1)
    with pytest.raises(AttributeError):
        Slotted(test=1, test2=1)
    assert not getattr(slotted, '__dict__', False)

    class NonSlottedContainer(PreferablySlotted):
        pass

    non_slotted = NonSlottedContainer(test=1)
    assert getattr(non_slotted, '__dict__', False)
    assert slotted.__slots__ == non_slotted.__slots__
    slotted.clear()
    for slot in slotted.__slots__:
        assert getattr(slotted, slot) is None


def test_Metrics():
    class MyMetrics(BaseMetricsContainer):
        __slots__ = ('test',)

    slotted_metrics = MyMetrics(test=torch.tensor(1.0))
    assert slotted_metrics.metrics == {'test': torch.tensor(1.0)}
    print_metrics(slotted_metrics)


def test_LossAndMetrics():
    class NonSlottedLoss(BaseLossAndMetricsContainer):
        pass

    non_slotted_loss = NonSlottedLoss(criterion=torch.FloatTensor([[1.0]]))
    assert non_slotted_loss.criterion == torch.FloatTensor([[1.0]])
    with pytest.raises(TypeError):
        NonSlottedLoss(test=1.0)

    print_criterion_and_metrics(non_slotted_loss)


def test_Outputs():
    class MyOutputs(OutputsContainer):
        __slots__ = ('tensor1', 'list_tensor')

    input_dict_torch = MyOutputs(tensor1=torch.ones(2), list_tensor=[torch.zeros(2)])

    torch_dict_list = TorchDictList.from_batch(input_dict_torch)
    assert torch_dict_list.to_dict() == {'list_tensor': [[torch.tensor(0.)], [torch.tensor(0.)]],
                                         'tensor1': [torch.tensor(1.), torch.tensor(1.)]}
