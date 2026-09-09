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
def stop_experiment_event(
    example_exp_name, example_run_id
) -> log_events.StopExperimentEvent:
    """Provides a StopExperiment event instance."""
    return log_events.StopExperimentEvent(
        exp_name=example_exp_name, run_id=example_run_id
    )


@pytest.fixture
def pause_experiment_event(
    example_exp_name, example_run_id
) -> log_events.PauseExperimentEvent:
    """Provides a PauseExperiment event instance."""
    return log_events.PauseExperimentEvent(
        exp_name=example_exp_name, run_id=example_run_id
    )


@pytest.fixture
def continue_experiment_event(
    example_exp_name, example_run_id
) -> log_events.ContinueExperimentEvent:
    """Provides a ContinueExperiment event instance."""
    return log_events.ContinueExperimentEvent(
        exp_name=example_exp_name, run_id=example_run_id
    )


@pytest.fixture
def model_registration_event(
    example_model_name, example_architecure_repr, example_model_ts
) -> log_events.ModelRegistrationEvent:
    """Provides a ModelRegistration event instance."""
    return log_events.ModelRegistrationEvent(
        model_name=example_model_name,
        model_ts=example_model_ts,
        architecture_repr=example_architecure_repr,
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
        n_iter=5,
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
    pause_experiment_event,
    continue_experiment_event,
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
        pause_experiment_event,
        continue_experiment_event,
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


@pytest.fixture
def simulated_learning_curves() -> dict[str, list[float]]:
    """Provides a realistic 10-epoch multi-metric history."""
    return {
        'train_loss': [
            1.8,
            1.4,
            1.1,
            0.85,
            0.65,
            0.50,
            0.40,
            0.32,
            0.25,
            0.20,
        ],
        'train_accuracy': [
            0.35,
            0.50,
            0.62,
            0.70,
            0.78,
            0.84,
            0.88,
            0.91,
            0.94,
            0.96,
        ],
        'val_loss': [
            1.5,
            1.25,
            1.0,
            0.85,
            0.75,
            0.70,
            0.68,
            0.67,
            0.66,
            0.65,
        ],
        'val_accuracy': [
            0.45,
            0.55,
            0.65,
            0.72,
            0.77,
            0.80,
            0.82,
            0.83,
            0.84,
            0.85,
        ],
        'val_f1_score': [
            0.42,
            0.53,
            0.63,
            0.70,
            0.75,
            0.78,
            0.80,
            0.81,
            0.82,
            0.83,
        ],
    }


@pytest.fixture
def plotting_workflow(
    example_model_name: str,
    simulated_learning_curves: dict[str, list[float]],
    start_experiment_event: log_events.StartExperimentEvent,
    stop_experiment_event: log_events.StopExperimentEvent,
    pause_experiment_event: log_events.PauseExperimentEvent,
    continue_experiment_event: log_events.ContinueExperimentEvent,
) -> tuple[log_events.Event, ...]:
    """Generates a complete 10-epoch training and test event sequence."""
    model_name = example_model_name
    train_source = 'train'
    val_source = 'val'
    test_source = 'test'

    train_loss = simulated_learning_curves['train_loss']
    train_acc = simulated_learning_curves['train_accuracy']
    val_loss = simulated_learning_curves['val_loss']
    val_acc = simulated_learning_curves['val_accuracy']
    val_f1 = simulated_learning_curves['val_f1_score']

    events: list[log_events.Event] = [start_experiment_event]

    for epoch in range(1, 11):
        events.append(
            log_events.MetricEvent(
                model_name=model_name,
                source_name=train_source,
                epoch=epoch,
                metrics={
                    'loss': train_loss[epoch - 1],
                    'accuracy': train_acc[epoch - 1],
                },
            )
        )
        events.append(
            log_events.MetricEvent(
                model_name=model_name,
                source_name=val_source,
                epoch=epoch,
                metrics={
                    'loss': val_loss[epoch - 1],
                    'accuracy': val_acc[epoch - 1],
                    'f1_score': val_f1[epoch - 1],
                },
            )
        )
        events.append(
            log_events.EndEpochEvent(
                source_name=train_source,
                model_name=model_name,
                epoch=epoch,
            )
        )
        if epoch == 5:
            events.append(pause_experiment_event)
            events.append(continue_experiment_event)

    # Final test evaluation events
    events.append(
        log_events.MetricEvent(
            model_name=model_name,
            source_name=test_source,
            epoch=10,
            metrics={'loss': 0.69, 'accuracy': 0.83, 'f1_score': 0.81},
        )
    )
    events.append(
        log_events.EndTestEvent(source_name=test_source, model_name=model_name)
    )
    events.append(stop_experiment_event)
    return tuple(events)
