"""Configuration module defining example events."""

import datetime
import io
import pathlib

from collections.abc import Generator

import pytest

from drytorch import log_events


@pytest.fixture(scope='package', autouse=True)
def allow_event_creation_outside_scope() -> None:
    """Allows the creation of events outside an experiment."""
    log_events.Event.set_auto_publish(lambda x: None)
    return


@pytest.fixture()
def start_experiment_event(tmp_path,
                           example_exp_name,
                           example_config) -> log_events.StartExperimentEvent:
    """Provides a StartExperiment event instance."""
    return log_events.StartExperimentEvent(
        exp_name=example_exp_name,
        exp_dir=pathlib.Path(tmp_path) / example_exp_name,
        exp_version=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        config=example_config,
    )


@pytest.fixture
def stop_experiment_event(example_exp_name) -> log_events.StopExperimentEvent:
    """Provides a StopExperiment event instance."""
    return log_events.StopExperimentEvent(exp_name=example_exp_name)


@pytest.fixture
def model_creation_event(example_model_name,
                         example_metadata) -> log_events.ModelRegistrationEvent:
    """Provides a ModelCreation event instance."""
    return log_events.ModelRegistrationEvent(
        model_name=example_model_name,
        model_version=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        metadata=example_metadata,
    )


@pytest.fixture
def call_model_event(example_source_name,
                     example_model_name,
                     example_metadata) -> log_events.SourceRegistrationEvent:
    """Provides a CallModel event instance."""
    return log_events.SourceRegistrationEvent(
        source_name=example_source_name,
        source_version=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        model_name=example_model_name,
        model_version=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        metadata=example_metadata,
    )


@pytest.fixture
def save_model_event(example_model_name, example_epoch) -> log_events.SaveModelEvent:
    """Provides a SaveModel event instance."""
    return log_events.SaveModelEvent(
        model_name=example_model_name,
        definition='checkpoint',
        location=f'/path/to/checkpoints/model_epoch_{example_epoch}.pt',
        epoch=example_epoch,
    )


@pytest.fixture
def load_model_event(example_model_name, example_epoch) -> log_events.LoadModelEvent:
    """Provides a LoadModel event instance."""
    return log_events.LoadModelEvent(
        model_name=example_model_name,
        definition='checkpoint',
        location='/path/to/checkpoints/model_epoch_{example_epoch}.pt',
        epoch=example_epoch,
    )


@pytest.fixture
def start_training_event(example_source_name,
                         example_model_name,
                         example_epoch) -> log_events.StartTrainingEvent:
    """Provides a StartTraining event instance."""
    return log_events.StartTrainingEvent(
        source_name=example_source_name,
        model_name=example_model_name,
        start_epoch=example_epoch,
        end_epoch=example_epoch + 3,
    )


@pytest.fixture
def start_epoch_event(example_source_name,
                      example_model_name,
                      example_epoch) -> log_events.StartEpochEvent:
    """Provides a StartEpoch event instance."""
    return log_events.StartEpochEvent(source_name=example_source_name,
                                 model_name=example_model_name,
                                 epoch=example_epoch,
                                 end_epoch=example_epoch + 3)


@pytest.fixture
def end_epoch_event(example_source_name,
                    example_model_name,
                    example_epoch) -> log_events.EndEpochEvent:
    """Provides an EndEpoch event instance."""
    return log_events.EndEpochEvent(source_name=example_source_name,
                               model_name=example_model_name,
                               epoch=example_epoch)


@pytest.fixture
def iterate_batch_event(example_source_name) -> log_events.IterateBatchEvent:
    """Provides an IterateBatch event instance."""
    return log_events.IterateBatchEvent(
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
        example_epoch,
) -> log_events.TerminatedTrainingEvent:
    """Provides a TerminatedTraining event instance."""
    return log_events.TerminatedTrainingEvent(
        model_name=example_model_name,
        source_name=example_source_name,
        epoch=example_epoch,
        reason='test event',
    )


@pytest.fixture
def end_training_event(example_source_name) -> log_events.EndTrainingEvent:
    """Provides an EndTraining event instance."""
    return log_events.EndTrainingEvent(source_name=example_source_name)


@pytest.fixture
def start_test_event(example_source_name,
                     example_model_name) -> log_events.StartTestEvent:
    """Provides a Test event instance."""
    return log_events.StartTestEvent(source_name=example_source_name,
                                model_name=example_model_name)


@pytest.fixture
def end_test_event(example_source_name,
                   example_model_name) -> log_events.EndTestEvent:
    """Provides a Test event instance."""
    return log_events.EndTestEvent(source_name=example_source_name,
                              model_name=example_model_name)


@pytest.fixture
def epoch_metrics_event(example_source_name,
                        example_model_name,
                        example_named_metrics,
                        example_epoch) -> log_events.MetricEvent:
    """Provides a FinalMetrics event instance."""
    return log_events.MetricEvent(
        model_name=example_model_name,
        source_name=example_source_name,
        epoch=example_epoch,
        metrics=example_named_metrics,
    )


@pytest.fixture
def update_learning_rate_event(
        example_source_name,
        example_model_name,
        example_epoch,
) -> log_events.LearningRateEvent:
    """Provides an UpdateLearningRate event instance."""
    return log_events.LearningRateEvent(
        source_name=example_source_name,
        model_name=example_model_name,
        epoch=example_epoch,
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
