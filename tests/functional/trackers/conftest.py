"""Configuration module defining example events."""

import pytest

import datetime
import io
import pathlib
from typing import Generator

from dry_torch import log_events


@pytest.fixture(scope='package', autouse=True)
def allow_event_creation_outside_scope() -> None:
    """Allows the creation of events outside an experiment"""
    log_events.Event.set_auto_publish(lambda x: None)
    return


@pytest.fixture()
def start_experiment_event(tmp_path,
                           example_exp_name) -> log_events.StartExperiment:
    """Provides a StartExperiment event instance."""
    return log_events.StartExperiment(
        exp_name=example_exp_name,
        exp_dir=pathlib.Path(tmp_path) / example_exp_name,
        exp_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        config=None,
    )


@pytest.fixture
def stop_experiment_event(example_exp_name) -> log_events.StopExperiment:
    """Provides a StopExperiment event instance."""
    return log_events.StopExperiment(exp_name=example_exp_name)


@pytest.fixture
def model_creation_event(example_model_name,
                         example_metadata) -> log_events.ModelCreation:
    """Provides a ModelCreation event instance."""
    return log_events.ModelCreation(
        model_name=example_model_name,
        model_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        metadata=example_metadata,
    )


@pytest.fixture
def call_model_event(example_source_name,
                     example_model_name,
                     example_metadata) -> log_events.CallModel:
    """Provides a CallModel event instance."""
    return log_events.CallModel(
        source_name=example_source_name,
        source_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        model_name=example_model_name,
        model_version=datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        metadata=example_metadata,
    )


@pytest.fixture
def save_model_event(example_model_name) -> log_events.SaveModel:
    """Provides a SaveModel event instance."""
    return log_events.SaveModel(
        model_name=example_model_name,
        definition='checkpoint',
        location='/path/to/checkpoints/model_epoch_10.pt',
        epoch=10,
    )


@pytest.fixture
def load_model_event(example_model_name) -> log_events.LoadModel:
    """Provides a LoadModel event instance."""
    return log_events.LoadModel(
        model_name=example_model_name,
        definition='checkpoint',
        location='/path/to/checkpoints/model_epoch_10.pt',
        epoch=10,
    )


@pytest.fixture
def start_training_event(example_source_name,
                         example_model_name) -> log_events.StartTraining:
    """Provides a StartTraining event instance."""
    return log_events.StartTraining(
        source_name=example_source_name,
        model_name=example_model_name,
        start_epoch=0,
        end_epoch=100,
    )


@pytest.fixture
def start_epoch_event(example_source_name,
                      example_model_name) -> log_events.StartEpoch:
    """Provides a StartEpoch event instance."""
    return log_events.StartEpoch(source_name=example_source_name,
                                 model_name=example_model_name,
                                 epoch=5,
                                 end_epoch=100)


@pytest.fixture
def end_epoch_event(example_source_name,
                    example_model_name) -> log_events.EndEpoch:
    """Provides an EndEpoch event instance."""
    return log_events.EndEpoch(source_name=example_source_name,
                               model_name=example_model_name,
                               epoch=100)


@pytest.fixture
def iterate_batch_event(example_source_name) -> log_events.IterateBatch:
    """Provides an IterateBatch event instance."""
    return log_events.IterateBatch(
        source_name=example_source_name,
        num_iter=5,
        batch_size=32,
        dataset_size=1600,
        push_updates=[],
    )


@pytest.fixture
def terminated_training_event(
        example_model_name,
        example_source_name,
) -> log_events.TerminatedTraining:
    """Provides a TerminatedTraining event instance."""
    return log_events.TerminatedTraining(
        model_name=example_model_name,
        source_name=example_source_name,
        epoch=45,
        reason='test event',
    )


@pytest.fixture
def end_training_event(example_source_name) -> log_events.EndTraining:
    """Provides an EndTraining event instance."""
    return log_events.EndTraining(source_name=example_source_name)


@pytest.fixture
def start_test_event(example_source_name,
                     example_model_name) -> log_events.StartTest:
    """Provides a Test event instance."""
    return log_events.StartTest(source_name=example_source_name,
                                model_name=example_model_name)


@pytest.fixture
def end_test_event(example_source_name,
                   example_model_name) -> log_events.EndTest:
    """Provides a Test event instance."""
    return log_events.EndTest(source_name=example_source_name,
                              model_name=example_model_name)


@pytest.fixture
def epoch_metrics_event(example_source_name,
                        example_model_name,
                        example_named_metrics) -> log_events.Metrics:
    """Provides a FinalMetrics event instance."""
    return log_events.Metrics(
        model_name=example_model_name,
        source_name=example_source_name,
        epoch=10,
        metrics=example_named_metrics,
    )


@pytest.fixture
def update_learning_rate_event(
        example_source_name,
        example_model_name,
) -> log_events.UpdateLearningRate:
    """Provides an UpdateLearningRate event instance."""
    return log_events.UpdateLearningRate(
        source_name=example_source_name,
        model_name=example_model_name,
        epoch=5,
        base_lr=0.0001,
        scheduler_name='CosineAnnealingLR',
    )


@pytest.fixture
def string_stream() -> Generator[io.StringIO, None, None]:
    """Provides a StringIO object for capturing progress bar output."""
    output = io.StringIO()
    yield output
    output.close()
    return


@pytest.fixture
def event_workflow(
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        save_model_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        end_test_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        save_model_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_test_event,
    )
    return event_tuple
