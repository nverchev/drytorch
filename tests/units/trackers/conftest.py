"""Configuration module defining example events."""
import pytest

import datetime
import io
import pathlib
from typing import Any, Mapping, Generator

from dry_torch import log_events


@pytest.fixture
def string_stream() -> Generator[io.StringIO, None, None]:
    """Provides a StringIO object for capturing progress bar output."""
    output = io.StringIO()
    yield output
    output.close()


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Provides sample metadata for events that require it."""
    return {
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "CrossEntropy",
        "architecture": "ResNet18"
    }


@pytest.fixture
def sample_metrics() -> dict[str, float]:
    """Provides sample metrics for events that require them."""
    return {
        "loss": 0.456,
        "accuracy": 0.892,
        "precision": 0.878,
        "recall": 0.901,
        "f1_score": 0.889
    }


@pytest.fixture(scope='package')
def start_experiment_event(tmpdir_factory) -> log_events.StartExperiment:
    """Provides a StartExperiment event instance."""
    temp_dir = tmpdir_factory.mktemp('experiments')
    return log_events.StartExperiment(
        exp_name='test_experiment',
        exp_dir=pathlib.Path(temp_dir) / 'test_experiment',
        exp_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        config=None
    )


@pytest.fixture
def stop_experiment_event() -> log_events.StopExperiment:
    """Provides a StopExperiment event instance."""
    return log_events.StopExperiment(
        exp_name="test_experiment"
    )


@pytest.fixture
def model_creation_event(
        sample_metadata: dict[str, Any]) -> log_events.ModelCreation:
    """Provides a ModelCreation event instance."""
    return log_events.ModelCreation(
        model_name="test_model",
        model_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        metadata=sample_metadata
    )


@pytest.fixture
def call_model_event(sample_metadata: dict[str, Any]) -> log_events.CallModel:
    """Provides a CallModel event instance."""
    return log_events.CallModel(
        source_name="model_caller",
        source_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        model_name="test_model",
        model_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        metadata=sample_metadata
    )


@pytest.fixture
def save_model_event() -> log_events.SaveModel:
    """Provides a SaveModel event instance."""
    return log_events.SaveModel(
        model_name="test_model",
        definition="checkpoint",
        location="/path/to/checkpoints/model_epoch_10.pt",
        epoch=10
    )


@pytest.fixture
def load_model_event() -> log_events.LoadModel:
    """Provides a LoadModel event instance."""
    return log_events.LoadModel(
        model_name="test_model",
        definition="checkpoint",
        location="/path/to/checkpoints/model_epoch_10.pt",
        epoch=10
    )


@pytest.fixture
def start_training_event() -> log_events.StartTraining:
    """Provides a StartTraining event instance."""
    return log_events.StartTraining(
        source_name='test_source',
        model_name='test_model',
        start_epoch=0,
        end_epoch=100
    )


@pytest.fixture
def start_epoch_event() -> log_events.StartEpoch:
    """Provides a StartEpoch event instance."""
    return log_events.StartEpoch(source_name='test_source',
                                 model_name='test_model',
                                 epoch=5,
                                 end_epoch=100)


@pytest.fixture
def end_epoch_event() -> log_events.EndEpoch:
    """Provides an EndEpoch event instance."""
    return log_events.EndEpoch(source_name='test_source',
                               model_name='test_model',
                               epoch=100)


@pytest.fixture
def iterate_batch_event() -> log_events.IterateBatch:
    """Provides an IterateBatch event instance."""
    return log_events.IterateBatch(
        source_name='test_source',
        num_iter=5,
        batch_size=32,
        dataset_size=1600,
        push_updates=[]
    )


@pytest.fixture
def terminated_training_event() -> log_events.TerminatedTraining:
    """Provides a TerminatedTraining event instance."""
    return log_events.TerminatedTraining(
        model_name='test_model',
        source_name='test_source',
        epoch=45,
        reason='testing termination'
    )


@pytest.fixture
def end_training_event() -> log_events.EndTraining:
    """Provides an EndTraining event instance."""
    return log_events.EndTraining(source_name='test_source')


@pytest.fixture
def start_test() -> log_events.StartTest:
    """Provides a Test event instance."""
    return log_events.StartTest(source_name='test_Test',
                                model_name='test_model')


@pytest.fixture
def end_test() -> log_events.EndTest:
    """Provides a Test event instance."""
    return log_events.EndTest(source_name='test_Test', model_name='test_model')


@pytest.fixture
def epoch_metrics_event(
        sample_metrics: Mapping[str, float]) -> log_events.Metrics:
    """Provides a FinalMetrics event instance."""
    return log_events.Metrics(
        model_name='test_model',
        source_name='test_source',
        epoch=10,
        metrics=sample_metrics,
    )


@pytest.fixture
def update_learning_rate_event() -> log_events.UpdateLearningRate:
    """Provides an UpdateLearningRate event instance."""
    return log_events.UpdateLearningRate(
        model_name='test_model',
        source_name='test_source',
        epoch=5,
        base_lr=0.0001,
        scheduler_name="CosineAnnealingLR"
    )
