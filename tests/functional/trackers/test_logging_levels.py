"""Tests the logs of a typical session for all the logging levels."""

import pytest

import logging
from typing import Generator

from dry_torch import log_events
from dry_torch.trackers.logging import BuiltinLogger
from dry_torch.trackers.logging import INFO_LEVELS
from dry_torch.trackers.logging import set_verbosity

expected_internal_level = """Running experiment: test_model.
Loading test_model checkpoint at epoch 10.
Training test_model started.
====> Epoch   5/100:
     test_model: 	loss=4.560000e-01	accuracy=8.920000e-01
====> Epoch   5/100:
     test_model: 	loss=4.560000e-01	accuracy=8.920000e-01
test_source: Updated test_model optimizer at epoch 5.
New learning rate: 0.0001.
New scheduler: CosineAnnealingLR.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
Saving test_model checkpoint in: /path/to/checkpoints/model_epoch_10.pt.
Experiment: test_model stopped.
"""

expected_metrics_level = """Running experiment: test_model.
Loading test_model checkpoint at epoch 10.
Training test_model started.
====> Epoch   5/100:
     test_model: 	loss=4.560000e-01	accuracy=8.920000e-01
====> Epoch   5/100:
     test_model: 	loss=4.560000e-01	accuracy=8.920000e-01
test_source: Updated test_model optimizer at epoch 5.
New learning rate: 0.0001.
New scheduler: CosineAnnealingLR.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
Saving test_model checkpoint in: /path/to/checkpoints/model_epoch_10.pt.
"""

expected_epoch_level = """Running experiment: test_model.
Loading test_model checkpoint at epoch 10.
Training test_model started.
====> Epoch   5/100:
====> Epoch   5/100:
test_source: Updated test_model optimizer at epoch 5.
New learning rate: 0.0001.
New scheduler: CosineAnnealingLR.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
Saving test_model checkpoint in: /path/to/checkpoints/model_epoch_10.pt.
"""

expected_param_update_level = """Running experiment: test_model.
Loading test_model checkpoint at epoch 10.
Training test_model started.
test_source: Updated test_model optimizer at epoch 5.
New learning rate: 0.0001.
New scheduler: CosineAnnealingLR.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
Saving test_model checkpoint in: /path/to/checkpoints/model_epoch_10.pt.
"""

expected_checkpoint_level = """Running experiment: test_model.
Loading test_model checkpoint at epoch 10.
Training test_model started.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
Saving test_model checkpoint in: /path/to/checkpoints/model_epoch_10.pt.
"""

expected_experiment_level = """Running experiment: test_model.
Training test_model started.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
"""

expected_training_level = """Training test_model started.
test_source: Training test_model terminated at epoch 45. Reason: test event.
Training ended.
Testing test_model started.
"""

expected_test_level = 'Testing test_model started.\n'

expected_dict = {
    INFO_LEVELS.internal: expected_internal_level,
    INFO_LEVELS.metrics: expected_metrics_level,
    INFO_LEVELS.epoch: expected_epoch_level,
    INFO_LEVELS.param_update: expected_param_update_level,
    INFO_LEVELS.checkpoint: expected_checkpoint_level,
    INFO_LEVELS.experiment: expected_experiment_level,
    INFO_LEVELS.training: expected_training_level,
    INFO_LEVELS.test: expected_test_level,
}


@pytest.fixture
def logger() -> logging.Logger:
    """Fixture for the library logger."""
    return logging.getLogger('dry_torch')


@pytest.fixture
def stream_handler(string_stream) -> logging.StreamHandler:
    """StreamHandler with library formatter."""
    return logging.StreamHandler(string_stream)


@pytest.fixture
def setup(
        request,
        logger,
        string_stream,
        stream_handler,
) -> Generator[None, None, None]:
    """Set up a logger with temporary configuration."""
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    yield

    logger.handlers.clear()
    logger.handlers.extend(original_handlers)
    logger.setLevel(original_level)
    return


@pytest.fixture
def event_workflow(
        start_experiment_event,
        model_creation_event,
        load_model_event,
        start_training_event,
        start_epoch_event,
        call_model_event,
        iterate_batch_event,
        epoch_metrics_event,
        update_learning_rate_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        end_test_event,
        save_model_event,
        stop_experiment_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_experiment_event,
        model_creation_event,
        load_model_event,
        start_training_event,
        call_model_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        update_learning_rate_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        end_test_event,
        save_model_event,
        stop_experiment_event,
    )
    return event_tuple


@pytest.mark.parametrize('info_level', list(INFO_LEVELS))
def test_typical_workflow(setup,
                          event_workflow,
                          string_stream,
                          info_level):
    set_verbosity(info_level)
    tracker = BuiltinLogger()
    for event in event_workflow:
        tracker.notify(event)
    assert string_stream.getvalue() == expected_dict[info_level]
