"""Module for an abstract Dumper class to handle paths to files."""

import abc
import functools
import pathlib
from typing import Optional

from dry_torch import log_events
from dry_torch import tracking


class AbstractDumper(tracking.Tracker, metaclass=abc.ABCMeta):
    """Tracker that for a custom directory."""

    def __init__(self, par_dir: Optional[pathlib.Path] = None):
        """
        Args:
            par_dir: Directory where to dump metadata. Defaults uses the one of
                the current experiment.
        """
        super().__init__()
        self._par_dir = par_dir
        self._exp_dir: Optional[pathlib.Path] = None

    @property
    def par_dir(self) -> pathlib.Path:
        """Return the directory where the files will be saved."""
        if self._exp_dir is None:
            raise RuntimeError('Accessed outside experiment scope.')
        if self._par_dir is None:
            path = self._exp_dir
        else:
            path = self._par_dir
        return path

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        self._exp_dir = event.exp_dir

    @notify.register
    def _(self, _: log_events.StopExperiment) -> None:
        self._exp_dir = None
