import pytest

import torch

from src.dry_torch import protocols as p
from src.dry_torch import Experiment


@pytest.fixture
def mock_experiment(mocker):
    """Fixture for a mock experiment."""
    mock_experiment = mocker.create_autospec(Experiment, instance=True)
    mock_experiment.name = 'mock_experiment'
    mocker.patch("src.dry_torch.tracking.Experiment.current",
                 return_value=mock_experiment)
    return mock_experiment


@pytest.fixture
def mock_model(mocker) -> p.ModelProtocol[torch.Tensor, torch.Tensor]:
    """Fixture for a mock model."""
    mock = mocker.create_autospec(p.ModelProtocol, instance=True)
    mock.epoch = 0
    mock.name = "mock_model"
    mock.module = torch.nn.Linear(1, 1)
    mock.increment_epoch = mocker.Mock()
    return mock


@pytest.fixture
def mock_scheduler(mocker):
    """Fixture for a mock scheduler."""
    mock = mocker.create_autospec(spec=p.SchedulerProtocol, instance=True)
    return mock


@pytest.fixture
def mock_learning_scheme(mocker,
                         mock_scheduler) -> p.LearningProtocol:
    """Fixture for a mock learning scheme."""
    mock = mocker.create_autospec(spec=p.LearningProtocol, instance=True)
    mock.lr = 0.
    mock.scheduler = mock_scheduler
    mock.optimizer_cls = torch.optim.SGD
    mock.optimizer_defaults = {}
    return mock


@pytest.fixture
def mock_metrics_calc(mocker):
    """Fixture for a mock metric calculator."""
    return mocker.create_autospec(p.MetricsCalculatorProtocol, instance=True)


@pytest.fixture
def mock_loss_calculator(mocker) -> p.LossCalculatorProtocol:
    """Fixture for a mock loss calculator."""
    mock = mocker.create_autospec(spec=p.LossCalculatorProtocol, instance=True)
    mock.criterion = 0.1
    return mock


@pytest.fixture
def mock_loader(mocker) -> p.LoaderProtocol:
    """Fixture for a mock loader."""
    return mocker.create_autospec(spec=p.LoaderProtocol, instance=True)


@pytest.fixture
def mock_trainer(mocker, mock_model) -> p.TrainerProtocol:
    """Fixture for a mock trainer."""
    mock = mocker.create_autospec(p.TrainerProtocol, instance=True)
    mock.model = mock_model
    mock.name = 'mock_trainer'
    return mock


@pytest.fixture
def mock_validation(mocker) -> p.EvaluationProtocol:
    """Fixture for a mock validation."""
    mock = mocker.create_autospec(spec=p.EvaluationProtocol, instance=True)
    mock.name = 'mock_validation'
    return mock
