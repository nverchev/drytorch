"""Configuration module defining example events."""

import dataclasses
import io
import pathlib

from collections.abc import Generator

import pytest

from drytorch.core import log_events


@pytest.fixture(autouse=True)
def allow_event_creation_outside_scope() -> None:
    """Allows the creation of events outside an experiment."""
    log_events.Event.set_auto_publish(lambda x: None)
    return


@pytest.fixture()
def start_experiment_event(
    tmp_path,
    example_exp_name,
    example_config,
    example_run_ts,
    example_run_id,
) -> log_events.StartExperimentEvent:
    """Provides a StartExperiment event instance."""
    return log_events.StartExperimentEvent(
        config=example_config,
        exp_name=example_exp_name,
        run_ts=example_run_ts,
        run_id=example_run_id,
        par_dir=pathlib.Path(tmp_path),
        tags=['my_tag'],
    )


@pytest.fixture
def stop_experiment_event(example_exp_name) -> log_events.StopExperimentEvent:
    """Provides a StopExperiment event instance."""
    return log_events.StopExperimentEvent(exp_name=example_exp_name)


@pytest.fixture
def model_registration_event(
    example_model_name, example_architecure_repr, example_model_ts
) -> log_events.ModelRegistrationEvent:
    """Provides a ModelRegistration event instance."""
    return log_events.ModelRegistrationEvent(
        model_name=example_model_name,
        model_ts=example_model_ts,
        architecure_repr=example_architecure_repr,
    )


@pytest.fixture
def actor_registration_event(
    example_source_name,
    example_source_ts,
    example_model_name,
    example_model_ts,
    example_metadata,
) -> log_events.ActorRegistrationEvent:
    """Provides a SourceRegistration event instance."""
    return log_events.ActorRegistrationEvent(
        actor_name=example_source_name,
        actor_ts=example_source_ts,
        model_name=example_model_name,
        model_ts=example_model_ts,
        metadata=example_metadata,
    )


@pytest.fixture
def save_model_event(
    example_model_name, example_epoch
) -> log_events.SaveModelEvent:
    """Provides a SaveModel event instance."""
    return log_events.SaveModelEvent(
        model_name=example_model_name,
        definition='checkpoint',
        location=f'/path/to/checkpoints/model_epoch_{example_epoch}.pt',
        epoch=example_epoch,
    )


@pytest.fixture
def load_model_event(
    example_model_name, example_epoch
) -> log_events.LoadModelEvent:
    """Provides a LoadModel event instance."""
    return log_events.LoadModelEvent(
        model_name=example_model_name,
        definition='checkpoint',
        location='/path/to/checkpoints/model_epoch_{example_epoch}.pt',
        epoch=example_epoch,
    )


@pytest.fixture
def start_training_event(
    example_source_name, example_model_name, example_epoch
) -> log_events.StartTrainingEvent:
    """Provides a StartTraining event instance."""
    return log_events.StartTrainingEvent(
        source_name=example_source_name,
        model_name=example_model_name,
        start_epoch=example_epoch,
        end_epoch=example_epoch + 3,
    )


@pytest.fixture
def start_epoch_event(
    example_source_name, example_model_name, example_epoch
) -> log_events.StartEpochEvent:
    """Provides a StartEpoch event instance."""
    return log_events.StartEpochEvent(
        source_name=example_source_name,
        model_name=example_model_name,
        epoch=example_epoch,
        end_epoch=example_epoch + 3,
    )


@pytest.fixture
def end_epoch_event(
    example_source_name, example_model_name, example_epoch
) -> log_events.EndEpochEvent:
    """Provides an EndEpoch event instance."""
    return log_events.EndEpochEvent(
        source_name=example_source_name,
        model_name=example_model_name,
        epoch=example_epoch,
    )


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
        reason='test event.',
    )


@pytest.fixture
def end_training_event(example_source_name) -> log_events.EndTrainingEvent:
    """Provides an EndTraining event instance."""
    return log_events.EndTrainingEvent(source_name=example_source_name)


@pytest.fixture
def start_test_event(
    example_source_name, example_model_name
) -> log_events.StartTestEvent:
    """Provides a Test event instance."""
    return log_events.StartTestEvent(
        source_name=example_source_name, model_name=example_model_name
    )


@pytest.fixture
def end_test_event(
    example_source_name, example_model_name
) -> log_events.EndTestEvent:
    """Provides a Test event instance."""
    return log_events.EndTestEvent(
        source_name=example_source_name, model_name=example_model_name
    )


@pytest.fixture
def metrics_event(
    example_source_name,
    example_model_name,
    example_named_metrics,
    example_epoch,
) -> log_events.MetricEvent:
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


@pytest.fixture
def event_workflow(
    start_experiment_event,
    model_registration_event,
    load_model_event,
    actor_registration_event,
    start_training_event,
    start_epoch_event,
    iterate_batch_event,
    metrics_event,
    end_epoch_event,
    update_learning_rate_event,
    save_model_event,
    terminated_training_event,
    end_training_event,
    start_test_event,
    end_test_event,
    stop_experiment_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    initial_epoch = start_training_event.start_epoch
    second_start_epoch_event = dataclasses.replace(
        start_epoch_event, epoch=start_epoch_event.epoch + 1
    )
    second_epoch_metrics_event = dataclasses.replace(
        metrics_event, epoch=metrics_event.epoch + 1
    )
    second_end_epoch_event = dataclasses.replace(
        end_epoch_event, epoch=end_epoch_event.epoch + 1
    )
    save_model_event = dataclasses.replace(
        save_model_event, epoch=start_training_event.start_epoch + 1
    )
    new_location = save_model_event.location.replace(
        str(initial_epoch), str(initial_epoch + 1)
    )
    update_learning_rate_event = dataclasses.replace(
        update_learning_rate_event, epoch=update_learning_rate_event.epoch + 1
    )
    save_model_event = dataclasses.replace(
        save_model_event, location=new_location
    )
    third_start_epoch_event = dataclasses.replace(
        start_epoch_event, epoch=start_epoch_event.epoch + 2
    )
    third_epoch_metrics_event = dataclasses.replace(
        metrics_event, epoch=metrics_event.epoch + 2
    )
    third_end_epoch_event = dataclasses.replace(
        end_epoch_event, epoch=end_epoch_event.epoch + 2
    )
    test_metrics_event = dataclasses.replace(
        metrics_event, epoch=metrics_event.epoch + 2
    )
    terminated_training_event = dataclasses.replace(
        terminated_training_event, epoch=start_training_event.start_epoch + 2
    )

    event_tuple = (
        start_experiment_event,
        model_registration_event,
        load_model_event,
        actor_registration_event,
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        metrics_event,
        end_epoch_event,
        second_start_epoch_event,
        iterate_batch_event,
        second_epoch_metrics_event,
        second_end_epoch_event,
        update_learning_rate_event,
        save_model_event,
        third_start_epoch_event,
        iterate_batch_event,
        third_epoch_metrics_event,
        third_end_epoch_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        iterate_batch_event,
        test_metrics_event,
        end_test_event,
        stop_experiment_event,
    )
    return event_tuple
