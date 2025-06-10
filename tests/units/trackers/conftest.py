"""Configuration module defining example mocked events."""

import pytest
import datetime
import io
from typing import Generator

from dry_torch import log_events


@pytest.fixture
def start_experiment_mock_event(mocker,
                                tmp_path,
                                example_exp_name) -> log_events.StartExperiment:
    """Mock StartExperiment event instance."""
    mock = mocker.create_autospec(log_events.StartExperiment)
    # Set default attribute values
    mock.exp_name = example_exp_name
    mock.exp_dir = tmp_path / example_exp_name
    mock.exp_version = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    mock.config = None
    return mock


@pytest.fixture
def stop_experiment_mock_event(mocker,
                               example_exp_name) -> log_events.StopExperiment:
    """Mock StopExperiment event instance."""
    mock = mocker.create_autospec(log_events.StopExperiment)
    mock.exp_name = example_exp_name
    return mock


@pytest.fixture
def model_creation_mock_event(
        mocker,
        example_model_name,
        example_metadata
) -> log_events.ModelCreation:
    """Mock ModelCreation event instance."""
    mock = mocker.create_autospec(log_events.ModelCreation)
    mock.model_name = example_model_name
    mock.model_version = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    mock.metadata = example_metadata
    return mock


@pytest.fixture
def call_model_mock_event(mocker,
                          example_source_name,
                          example_model_name,
                          example_metadata) -> log_events.CallModel:
    """Mock CallModel event instance."""
    mock = mocker.create_autospec(log_events.CallModel)
    mock.source_name = example_source_name
    mock.source_version = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    mock.model_name = example_model_name
    mock.model_version = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    mock.metadata = example_metadata
    return mock


@pytest.fixture
def save_model_mock_event(mocker, example_model_name) -> log_events.SaveModel:
    """Mock SaveModel event instance."""
    mock = mocker.create_autospec(log_events.SaveModel)
    mock.model_name = example_model_name
    mock.definition = 'checkpoint'
    mock.location = '/path/to/checkpoints/model_epoch_10.pt'
    mock.epoch = 10
    return mock


@pytest.fixture
def load_model_mock_event(mocker, example_model_name) -> log_events.LoadModel:
    """Mock LoadModel event instance."""
    mock = mocker.create_autospec(log_events.LoadModel)
    mock.model_name = example_model_name
    mock.definition = 'checkpoint'
    mock.location = '/path/to/checkpoints/model_epoch_10.pt'
    mock.epoch = 10
    return mock


@pytest.fixture
def start_training_mock_event(mocker,
                              example_source_name,
                              example_model_name) -> log_events.StartTraining:
    """Mock StartTraining event instance."""
    mock = mocker.create_autospec(log_events.StartTraining)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.start_epoch = 0
    mock.end_epoch = 100
    return mock


@pytest.fixture
def start_epoch_mock_event(mocker,
                           example_source_name,
                           example_model_name) -> log_events.StartEpoch:
    """Mock StartEpoch event instance."""
    mock = mocker.create_autospec(log_events.StartEpoch)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.epoch = 5
    mock.end_epoch = 100
    return mock


@pytest.fixture
def end_epoch_mock_event(mocker,
                         example_source_name,
                         example_model_name) -> log_events.EndEpoch:
    """Mock EndEpoch event instance."""
    mock = mocker.create_autospec(log_events.EndEpoch)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.epoch = 100
    return mock


@pytest.fixture
def iterate_batch_mock_event(mocker,
                             example_source_name) -> log_events.IterateBatch:
    """Mock IterateBatch event instance."""
    mock = mocker.create_autospec(log_events.IterateBatch)
    mock.source_name = example_source_name
    mock.num_iter = 5
    mock.batch_size = 32
    mock.dataset_size = 1600
    mock.push_updates = []
    return mock


@pytest.fixture
def terminated_training_mock_event(
        mocker,
        example_model_name,
        example_source_name,
) -> log_events.TerminatedTraining:
    """Mock TerminatedTraining event instance."""
    mock = mocker.create_autospec(log_events.TerminatedTraining)
    mock.model_name = example_model_name
    mock.source_name = example_source_name
    mock.epoch = 45
    mock.reason = 'testing termination'
    return mock


@pytest.fixture
def end_training_mock_event(mocker,
                            example_source_name) -> log_events.EndTraining:
    """Mock EndTraining event instance."""
    mock = mocker.create_autospec(log_events.EndTraining)
    mock.source_name = example_source_name
    return mock


@pytest.fixture
def start_test_mock_event(mocker,
                          example_source_name,
                          example_model_name) -> log_events.StartTest:
    """Mock StartTest event instance."""
    mock = mocker.create_autospec(log_events.StartTest)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    return mock


@pytest.fixture
def end_test_mock_event(mocker,
                        example_source_name,
                        example_model_name) -> log_events.EndTest:
    """Mock EndTest event instance."""
    mock = mocker.create_autospec(log_events.EndTest)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    return mock


@pytest.fixture
def epoch_metrics_mock_event(mocker,
                             example_source_name,
                             example_model_name,
                             example_named_metrics) -> log_events.Metrics:
    """Mock Metrics event instance."""
    mock = mocker.create_autospec(log_events.Metrics)
    mock.model_name = example_model_name
    mock.source_name = example_source_name
    mock.epoch = 10
    mock.metrics = example_named_metrics
    return mock


@pytest.fixture
def update_learning_rate_mock_event(
        mocker,
        example_source_name,
        example_model_name,
) -> log_events.UpdateLearningRate:
    """Mock UpdateLearningRate event instance."""
    mock = mocker.create_autospec(log_events.UpdateLearningRate)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.epoch = 5
    mock.base_lr = 0.0001
    mock.scheduler_name = 'CosineAnnealingLR'
    return mock


@pytest.fixture
def string_stream() -> Generator[io.StringIO, None, None]:
    """StringIO object for capturing output in a string."""
    output = io.StringIO()
    yield output
    output.close()
    return
