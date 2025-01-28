"""Classes that contain event information for logging purposes."""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
import pathlib
from typing import Any, Optional, Mapping


class Event(metaclass=abc.ABCMeta):
    """Class for logging events"""
    _auto_publish: Callable[[Event], None] = lambda self: None

    def __post_init__(self):
        self.__class__._auto_publish(self)

    @classmethod
    def set_auto_publish(cls, func: Callable[[Event], None]) -> None:
        """Specify how to notify subscribers upon creation."""
        cls._auto_publish = func
        return


@dataclasses.dataclass
class StartExperiment(Event):
    """
    Event logged when an experiment starts.

    Attributes:
        exp_name: The name of the experiment.
        exp_dir: The directory where the experiment is stored.
    """
    exp_name: str
    exp_dir: pathlib.Path


@dataclasses.dataclass
class StopExperiment(Event):
    """
    Event logged when an experiment stops.

    Attributes:
        exp_name: The name of the experiment.
    """
    exp_name: str


@dataclasses.dataclass
class ModelCreation(Event):
    """
    Event logged when a model is created.

    Attributes:
        model_name: The name of the model.
        metadata: Additional metadata about the model.
    """
    model_name: str
    metadata: dict[str, Any]


@dataclasses.dataclass
class CallModel(Event):
    """
    Event logged when a model is called by another class (caller).

    Attributes:
        name: The name of the caller.
        model_name: The name of the model that was called.
        metadata: Additional metadata about the caller.
    """
    name: str
    model_name: str
    metadata: dict[str, Any]


@dataclasses.dataclass
class SaveModel(Event):
    """
    Event logged when a checkpoint is saved.

    Attributes:
        model_name: The name of the model.
        definition: Specifies what was saved.
        location: The location where the checkpoint is saved.
        epoch: The epoch at which the checkpoint was saved.
    """
    model_name: str
    definition: str
    location: str
    epoch: int


@dataclasses.dataclass
class LoadCheckpoint(Event):
    """Event logged when a checkpoint is loaded.

    Attributes:
        model_name: The name of the model.
        definition (str): Specifies what was is loaded.
        location: The location where the checkpoint is loaded from.
        epoch: The epoch at which the checkpoint was loaded.
    """
    model_name: str
    definition: str
    location: str
    epoch: int


@dataclasses.dataclass
class StartTraining(Event):
    """
    Event logged when training starts.

    Attributes:
        model_name: The name of the model.
        start_epoch: The starting epoch of the training.
        end_epoch: The ending epoch of the training.
    """
    model_name: str
    start_epoch: int
    end_epoch: int


@dataclasses.dataclass
class StartEpoch(Event):
    """
    Event logged when an epoch starts.

    Attributes:
        epoch: The epoch number.
        final_epoch: The final epoch number for the current training session.
    """
    epoch: int
    final_epoch: Optional[int] = None


@dataclasses.dataclass
class EndEpoch(Event):
    """Event logged when an epoch ends."""
    pass


@dataclasses.dataclass
class IterateBatch(Event):
    """
    Event logged to create during batch iteration.

    Attributes:
        source: The object calling the iteration.
        num_iter: The number of iterations planned.
        dataset_size: The size of the dataset.
        push_updates: callbacks from loggers that require push updates.
    """
    source: str
    num_iter: int
    dataset_size: int
    push_updates: list[
        Callable[[Mapping[str, Any]], None]
    ] = dataclasses.field(default_factory=list)

    def update(self, metrics: Mapping[str, Any]) -> None:
        """
        Push the updated metrics to the loggers.

        Args:
            metrics: Mapping from the metric names to the calculated values.
        """
        for update in self.push_updates:
            update(metrics)
        return


@dataclasses.dataclass
class ModelDidNotConverge(Event):
    """
    Event logged when a model failed to converge.

    Attributes:
        exception: The error explaining why the model did not converge.
    """
    exception: BaseException


@dataclasses.dataclass
class TerminatedTraining(Event):
    """
    Event logged when training is terminated.

    Attributes:
        epoch: The epoch at which training was terminated.
        cause: The cause of the termination.
    """
    epoch: int
    cause: Optional[str] = None


@dataclasses.dataclass
class EndTraining(Event):
    """Event logged when training ends."""
    pass


@dataclasses.dataclass
class Test(Event):
    """
    Event logged when a test is performed.

    Attributes:
        model_name: The name of the model.
        test_name: The name of the test.
    """
    model_name: str
    test_name: str


@dataclasses.dataclass
class FinalMetrics(Event):
    """
    Event logged when metrics from the dataset are aggregated.

    Attributes:
        model_name: The name of the model.
        source: The name of the object that computed the metrics.
        epoch: The number of epoch the model was trained.
        metrics: The aggregated metrics.
    """
    model_name: str
    source: str
    epoch: int
    metrics: Mapping[str, float]
