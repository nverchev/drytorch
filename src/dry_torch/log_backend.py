import abc
import pathlib
from typing import Optional


class LogBackend(abc.ABC):

    @abc.abstractmethod
    def __call__(self,
                    model_name: str,
                    source: str,
                    partition: str,
                    epoch: int,
                    metrics: dict[str, float]) -> None:
        ...

    @abc.abstractmethod
    def plot(self,
             model_name: str,
             partition: str,
             metric: str) -> None:
        ...


class ExperimentLog(abc.ABC):
    """Abstract base class for experiment backend implementations."""

    @abc.abstractmethod
    def create_log(self,
                   local_path: Optional[pathlib.Path],
                   exp_name: str) -> LogBackend:
        ...


class NoBackend(LogBackend):

    def __call__(self,
                    model_name: str,
                    source: str,
                    partition: str,
                    epoch: int,
                    metrics: dict[str, float]) -> None:
        pass

    def plot(self,
             model_name: str,
             partition: str,
             metric: str) -> None:
        pass


class NoLog(ExperimentLog):

    def create_log(self,
                   local_path: Optional[pathlib.Path],
                   exp_name: str) -> LogBackend:
        return NoBackend()
