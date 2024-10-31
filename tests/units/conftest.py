import pytest

import torch

from src.dry_torch import protocols as p


@pytest.fixture
def mock_model(mocker) -> p.ModelProtocol:
    mock = mocker.create_autospec(p.ModelProtocol, instance=True)
    mock.epoch = 0
    mock.name = "mock_model"
    mock.module = torch.nn.Linear(1, 1)
    mock.increment_epoch = mocker.Mock()
    return mock


@pytest.fixture
def mock_scheduler(mocker):
    mock = mocker.create_autospec(spec=p.SchedulerProtocol, instance=True)
    return mock


@pytest.fixture
def mock_learning_scheme(mocker,
                         mock_scheduler) -> p.LearningProtocol:
    mock = mocker.create_autospec(spec=p.LearningProtocol, instance=True)
    mock.lr = 0.
    mock.scheduler = mock_scheduler
    mock.optimizer_cls = torch.optim.SGD
    mock.optimizer_defaults = {}
    return mock


@pytest.fixture
def mock_loss_calculator(mocker) -> p.LossCalculatorProtocol:
    mock = mocker.create_autospec(spec=p.LossCalculatorProtocol, instance=True)
    mock.criterion = 0.1
    return mock


@pytest.fixture
def mock_loader(mocker) -> p.LoaderProtocol:
    return mocker.create_autospec(spec=p.LoaderProtocol, instance=True)


@pytest.fixture
def mock_validation(mocker) -> p.EvaluationProtocol:
    return mocker.create_autospec(spec=p.EvaluationProtocol, instance=True)

