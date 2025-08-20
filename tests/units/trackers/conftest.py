"""Configuration module defining example mocked events."""

import io

from collections.abc import Generator

import pytest

from drytorch.core import log_events


@pytest.fixture(autouse=True)
def allow_event_creation_outside_scope() -> None:
    """Allows the creation of events outside an experiment."""
    log_events.Event.set_auto_publish(lambda x: None)
    return


@pytest.fixture
def start_experiment_mock_event(
    mocker,
    tmp_path,
    example_exp_name,
    example_run_ts,
    example_run_id,
    example_tags,
    example_config,
) -> log_events.StartExperimentEvent:
    """Mock StartExperiment event instance."""
    mock = mocker.create_autospec(log_events.StartExperimentEvent)
    # Set default attribute values
    mock.config = example_config
    mock.exp_name = example_exp_name
    mock.run_id = example_run_id
    mock.run_ts = example_run_ts
    mock.row_id = example_run_id
    mock.run_dir = tmp_path
    mock.resumed = False
    mock.tags = example_tags
    return mock


@pytest.fixture
def stop_experiment_mock_event(
    mocker, example_exp_name
) -> log_events.StopExperimentEvent:
    """Mock StopExperiment event instance."""
    mock = mocker.create_autospec(log_events.StopExperimentEvent)
    mock.exp_name = example_exp_name
    return mock


@pytest.fixture
def model_registration_mock_event(
    mocker, example_model_name, example_architecure_repr, example_model_ts
) -> log_events.ModelRegistrationEvent:
    """Mock ModelRegistration event instance."""
    mock = mocker.create_autospec(log_events.ModelRegistrationEvent)
    mock.model_name = example_model_name
    mock.model_ts = example_model_ts
    mock.architecure_repr = example_architecure_repr
    return mock


@pytest.fixture
def actor_registration_mock_event(
    mocker,
    example_source_name,
    example_source_ts,
    example_model_name,
    example_model_ts,
    example_metadata,
) -> log_events.ActorRegistrationEvent:
    """Mock SourceRegistration event instance."""
    mock = mocker.create_autospec(log_events.ActorRegistrationEvent)
    mock.actor_name = example_source_name
    mock.actor_ts = example_source_ts
    mock.model_name = example_model_name
    mock.model_ts = example_model_ts
    mock.metadata = example_metadata
    return mock


@pytest.fixture
def save_model_mock_event(
    mocker, example_model_name
) -> log_events.SaveModelEvent:
    """Mock SaveModel event instance."""
    mock = mocker.create_autospec(log_events.SaveModelEvent)
    mock.model_name = example_model_name
    mock.definition = 'checkpoint'
    mock.location = '/path/to/checkpoints/model_epoch_10.pt'
    mock.epoch = 10
    return mock


@pytest.fixture
def load_model_mock_event(
    mocker, example_model_name
) -> log_events.LoadModelEvent:
    """Mock LoadModel event instance."""
    mock = mocker.create_autospec(log_events.LoadModelEvent)
    mock.model_name = example_model_name
    mock.definition = 'checkpoint'
    mock.location = '/path/to/checkpoints/model_epoch_10.pt'
    mock.epoch = 10
    return mock


@pytest.fixture
def start_training_mock_event(
    mocker,
    example_source_name,
    example_model_name,
) -> log_events.StartTrainingEvent:
    """Mock StartTraining event instance."""
    mock = mocker.create_autospec(log_events.StartTrainingEvent)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.start_epoch = 0
    mock.end_epoch = 100
    return mock


@pytest.fixture
def start_epoch_mock_event(
    mocker, example_source_name, example_model_name
) -> log_events.StartEpochEvent:
    """Mock StartEpoch event instance."""
    mock = mocker.create_autospec(log_events.StartEpochEvent)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.epoch = 5
    mock.end_epoch = 100
    return mock


@pytest.fixture
def end_epoch_mock_event(
    mocker, example_source_name, example_model_name
) -> log_events.EndEpochEvent:
    """Mock EndEpoch event instance."""
    mock = mocker.create_autospec(log_events.EndEpochEvent)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    mock.epoch = 100
    return mock


@pytest.fixture
def iterate_batch_mock_event(
    mocker,
    example_source_name,
) -> log_events.IterateBatchEvent:
    """Mock IterateBatch event instance."""
    mock = mocker.create_autospec(log_events.IterateBatchEvent)
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
) -> log_events.TerminatedTrainingEvent:
    """Mock TerminatedTraining event instance."""
    mock = mocker.create_autospec(log_events.TerminatedTrainingEvent)
    mock.model_name = example_model_name
    mock.source_name = example_source_name
    mock.epoch = 45
    mock.reason = 'testing termination'
    return mock


@pytest.fixture
def end_training_mock_event(
    mocker, example_source_name
) -> log_events.EndTrainingEvent:
    """Mock EndTraining event instance."""
    mock = mocker.create_autospec(log_events.EndTrainingEvent)
    mock.source_name = example_source_name
    return mock


@pytest.fixture
def start_test_mock_event(
    mocker, example_source_name, example_model_name
) -> log_events.StartTestEvent:
    """Mock StartTest event instance."""
    mock = mocker.create_autospec(log_events.StartTestEvent)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    return mock


@pytest.fixture
def end_test_mock_event(
    mocker, example_source_name, example_model_name
) -> log_events.EndTestEvent:
    """Mock EndTest event instance."""
    mock = mocker.create_autospec(log_events.EndTestEvent)
    mock.source_name = example_source_name
    mock.model_name = example_model_name
    return mock


@pytest.fixture
def epoch_metrics_mock_event(
    mocker, example_source_name, example_model_name, example_named_metrics
) -> log_events.MetricEvent:
    """Mock Metrics event instance."""
    mock = mocker.create_autospec(log_events.MetricEvent)
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
) -> log_events.LearningRateEvent:
    """Mock UpdateLearningRate event instance."""
    mock = mocker.create_autospec(log_events.LearningRateEvent)
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
