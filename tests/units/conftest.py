"""Configuration module with mockups."""

import torch

import pytest

from drytorch import Experiment
from drytorch.core import protocols as p


experiment_current_original = Experiment.get_current


@pytest.fixture
def mock_model(mocker) -> p.ModelProtocol[torch.Tensor, torch.Tensor]:
    """Fixture for a mock model."""
    mock = mocker.create_autospec(p.ModelProtocol, instance=True)
    mock.epoch = 0
    mock.name = 'mock_model'
    mock.module = torch.nn.Linear(1, 1)
    mock.device = torch.device('cpu')
    mock.increment_epoch = mocker.Mock()
    mock.checkpoint = mocker.Mock()
    mock.mixed_precision = False
    return mock


@pytest.fixture
def mock_scheduler(mocker) -> p.SchedulerProtocol:
    """Fixture for a mock scheduler."""
    mock = mocker.create_autospec(
        spec=p.SchedulerProtocol, instance=True, side_effect=lambda x, y: x
    )
    return mock


@pytest.fixture
def mock_learning_schema(mocker, mock_scheduler) -> p.LearningProtocol:
    """Fixture for a mock learning scheme."""
    mock = mocker.create_autospec(spec=p.LearningProtocol, instance=True)
    mock.base_lr = 0.001
    mock.scheduler = mock_scheduler
    mock.optimizer_cls = torch.optim.SGD
    mock.optimizer_defaults = {}
    mock.gradient_op = lambda x: None
    return mock


@pytest.fixture
def mock_metric(
    mocker,
) -> p.ObjectiveProtocol[torch.Tensor, torch.Tensor]:
    """Fixture for a mock metric calculator."""
    mock = mocker.create_autospec(p.ObjectiveProtocol, instance=True)
    mock.name = 'mock_metric'
    mock.compute = mocker.Mock(return_value={'mock_metric': torch.tensor(0.5)})
    mock.higher_is_better = True
    mock.__deepcopy__ = mocker.Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_loss(mocker) -> p.LossProtocol[torch.Tensor, torch.Tensor]:
    """Fixture for a mock loss calculator."""
    mock = mocker.create_autospec(spec=p.LossProtocol, instance=True)
    mock_loss_value = torch.FloatTensor([1])
    mock_loss_value.requires_grad = True
    mock.forward = mocker.Mock(return_value=mock_loss_value)
    mock.name = 'Loss'
    mock.compute = mocker.Mock(return_value={'Loss': torch.tensor(1)})
    mock.__deepcopy__ = mocker.Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_loader(mocker) -> p.LoaderProtocol[tuple[torch.Tensor, torch.Tensor]]:
    """Fixture for a mock loader."""
    mock = mocker.create_autospec(spec=p.LoaderProtocol, instance=True)
    mock.batch_size = 32
    mock.__len__ = mocker.Mock(return_value=7)
    mock.dataset = mocker.Mock()
    mock.dataset.__len__ = mocker.Mock(return_value=500)
    tensor = torch.FloatTensor([1])
    mock.__iter__ = mocker.Mock(
        side_effect=lambda: iter([(tensor, tensor)] * 3)
    )
    return mock


@pytest.fixture
def mock_validation(
    mocker, mock_metric, example_named_metrics
) -> p.MonitorProtocol:
    """Fixture for a mock validation."""
    mock = mocker.create_autospec(spec=p.MonitorProtocol, instance=True)
    mock.name = 'mock_validation'
    mock.objective = mock_metric
    mock.computed_metrics = example_named_metrics
    return mock


@pytest.fixture
def mock_trainer(
    mocker, mock_model, mock_loss, mock_validation, example_named_metrics
) -> p.TrainerProtocol:
    """Fixture for a mock trainer."""
    mock = mocker.create_autospec(p.TrainerProtocol, instance=True)
    mock.model = mock_model
    mock.name = 'mock_trainer'
    mock.model.epoch = 3
    mock.objective = mock_loss
    mock.validation = mock_validation
    mock.terminated = False
    mock.__class__.computed_metrics = mocker.PropertyMock(
        return_value=example_named_metrics
    )

    def _terminate_training(reason: str):
        mock.terminated = True
        _unused = reason

    mock.terminate_training = mocker.Mock(side_effect=_terminate_training)
    return mock
