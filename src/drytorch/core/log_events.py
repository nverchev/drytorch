"""Classes containing logging event classes."""

from __future__ import annotations

import dataclasses
import datetime
import pathlib

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from drytorch.core import exceptions


@dataclasses.dataclass(frozen=True)
class Event:
    """Class for logging events."""

    _auto_publish: ClassVar[Callable[[Event], None] | None] = None

    def __post_init__(self) -> None:
        """Publish the event on creation."""
        if Event._auto_publish is None:
            raise exceptions.AccessOutsideScopeError()
        Event._auto_publish(self)
        return

    @staticmethod
    def set_auto_publish(func: Callable[[Event], None] | None) -> None:
        """Specify how to notify subscribers upon creation."""
        Event._auto_publish = func
        return


@dataclasses.dataclass(frozen=True)
class StartExperimentEvent(Event):
    """Event logged when an experiment starts.

    Attributes:
        config: configuration for the experiment.
        exp_name: the name of the experiment.
        run_ts: run's timestamp.
        run_id: identifier of the run.
        resumed: this run was resumed.
        par_dir: the parent directory for the experiment.
        tags: descriptors for the experiment's variation (e.g., "lr=0.01").
    """

    config: Any
    exp_name: str
    run_ts: datetime.datetime
    run_id: str
    resumed: bool = False
    par_dir: pathlib.Path = pathlib.Path()
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class StopExperimentEvent(Event):
    """Event logged when an experiment stops.

    Attributes:
        exp_name: the name of the experiment.
    """

    exp_name: str


@dataclasses.dataclass(frozen=True)
class ModelRegistrationEvent(Event):
    """Event logged when a model is created.

    Attributes:
        model_name: the name of the model.
        model_ts: the model's timestamp.
        architecure_repr: the model'.
    """

    model_name: str
    model_ts: datetime.datetime
    architecure_repr: str


@dataclasses.dataclass(frozen=True)
class ActorRegistrationEvent(Event):
    """Event logged when a source has been registered.

    Attributes:
        actor_name: the name of the caller.
        actor_ts: the source's timestamp.
        model_name: the name of the model that was called.
        model_ts: the model's timestamp.
        metadata: additional metadata about the caller.
    """

    actor_name: str
    actor_ts: datetime.datetime
    model_name: str
    model_ts: datetime.datetime
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class SaveModelEvent(Event):
    """Event logged when a checkpoint is saved.

    Attributes:
        model_name: the name of the model.
        definition: specifies what was saved.
        location: the location where the model is saved.
        epoch: the epoch at which the model was saved.
    """

    model_name: str
    definition: str
    location: str
    epoch: int


@dataclasses.dataclass(frozen=True)
class LoadModelEvent(Event):
    """Event logged when a model is loaded.

    Attributes:
        model_name: the name of the model.
        definition: specifies what was loaded.
        location: the location where the model is loaded from.
        epoch: the epoch at which the model was loaded.
    """

    model_name: str
    definition: str
    location: str
    epoch: int


@dataclasses.dataclass(frozen=True)
class StartTrainingEvent(Event):
    """Event logged when training starts.

    Attributes:
        source_name: the object that is training a model.
        model_name: the name of the model.
        start_epoch: the starting epoch of the training.
        end_epoch: the ending epoch of the training.
    """

    source_name: str
    model_name: str
    start_epoch: int
    end_epoch: int


@dataclasses.dataclass(frozen=True)
class StartEpochEvent(Event):
    """Event logged when an epoch starts.

    Attributes:
        source_name: the name of the object that is training a model.
        model_name: the name of the model.
        epoch: the epoch number.
        end_epoch: the final epoch number for the current training session.
    """

    source_name: str
    model_name: str
    epoch: int
    end_epoch: int | None = None


@dataclasses.dataclass(frozen=True)
class EndEpochEvent(Event):
    """Event logged when an epoch ends.

    Attributes:
        source_name: the name of the object that is training a model.
        model_name: the name of the model.
        epoch: the epoch that was trained.
    """

    source_name: str
    model_name: str
    epoch: int


@dataclasses.dataclass(frozen=True)
class IterateBatchEvent(Event):
    """Event logged to create during batch iteration.

    Attributes:
        source_name: the object calling the iteration.
        batch_size: the size of the mini-batch.
        num_iter: the number of iterations planned.
        dataset_size: the size of the dataset.
        push_updates: callbacks from loggers that require push updates.
    """

    source_name: str
    batch_size: int | None
    num_iter: int
    dataset_size: int
    push_updates: list[Callable[[Mapping[str, Any]], None]] = dataclasses.field(
        default_factory=list
    )

    def update(self, metrics: Mapping[str, Any]) -> None:
        """Push the updated metrics to the loggers.

        Args:
            metrics: calculated values by metric name.
        """
        for update in self.push_updates:
            update(metrics)
        return


@dataclasses.dataclass(frozen=True)
class TerminatedTrainingEvent(Event):
    """Event logged when training is terminated.

    Attributes:
        source_name: the name object calling the termination.
        model_name: the name of the model.
        epoch: the epoch at which training was terminated.
        reason: the cause of the termination.
    """

    source_name: str
    model_name: str
    epoch: int
    reason: str


@dataclasses.dataclass(frozen=True)
class EndTrainingEvent(Event):
    """Event logged when training ends.

    Attributes:
        source_name: The name of the object that is training a model.
    """

    source_name: str


@dataclasses.dataclass(frozen=True)
class StartTestEvent(Event):
    """Event logged when a test is started.

    Attributes:
        source_name: the name of the object calling the test.
        model_name: the name of the model.
    """

    source_name: str
    model_name: str


@dataclasses.dataclass(frozen=True)
class EndTestEvent(Event):
    """Event logged when a test is ended.

    Attributes:
        source_name: the name of the object calling the test.
        model_name: the name of the model.
    """

    source_name: str
    model_name: str


@dataclasses.dataclass(frozen=True)
class MetricEvent(Event):
    """Event logged when metrics from the dataset are aggregated.

    Attributes:
        model_name: the name of the model.
        source_name: the name of the object that computed the metrics.
        epoch: the number of epochs the model was trained.
        metrics: the aggregated metrics.
    """

    model_name: str
    source_name: str
    epoch: int
    metrics: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class LearningRateEvent(Event):
    """Event logged when the learning rate is updated.

    Attributes:
        model_name: the name of the model.
        source_name: the name of the object that computed the metrics.
        epoch: the number of epochs the model was trained.
        base_lr: new value(s) for the learning rate(s).
        scheduler_name: the representation of the scheduler.
    """

    model_name: str
    source_name: str
    epoch: int
    base_lr: Mapping[str, float] | float | None = None
    scheduler_name: str | None = None
