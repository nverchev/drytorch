"""Module containing a TensorBoard tracker."""

import functools
import pathlib
from typing import Optional
from typing_extensions import override
import warnings

import torch.utils.tensorboard

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch.trackers import base_classes


class TensorBoard(base_classes.Dumper):
    """
    Tracker that wraps the TensorBoard SummaryWriter.

    Class Attributes:
        folder_name: name of the folder containing the output.

    Attributes:
        resume_run: load previous session having the same directory.
    """
    folder_name = 'tensorboard_runs'

    def __init__(
            self,
            par_dir: Optional[pathlib.Path] = None,
            resume_run: bool = False
    ) -> None:
        """
        Args:
            par_dir: the directory where to dump metadata. Defaults to the
                one for the current experiment.
            resume_run: load previous session having the same directory.
        """
        super().__init__(par_dir)
        self.resume_run = resume_run
        self._writer: Optional[torch.utils.tensorboard.SummaryWriter] = None

    @property
    def writer(self) -> torch.utils.tensorboard.SummaryWriter:
        """The active SummaryWriter instance."""
        if self._writer is None:
            raise exceptions.AccessOutsideScopeError()

        return self._writer

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        super().notify(event)
        if self.resume_run:
            retrieved = self._get_last_run(self.par_dir)
            if retrieved is None:
                msg = 'TensorBoard: No previous runs. Starting a new one.'
                warnings.warn(msg)
                root_dir = self.par_dir / self.folder_name
            else:
                root_dir = retrieved

        else:
            root_dir = self.par_dir / self.folder_name

        self._writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=root_dir.as_posix(),
        )
        if event.config:
            try:
                self.writer.add_hparams(hparam_dict=event.config,
                                        metric_dict={})
            except TypeError:
                pass

        return

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self.writer.close()
        self._writer = None
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        for name, value in event.metrics.items():
            full_name = f'{event.model_name}/{event.source_name}-{name}'
            self.writer.add_scalar(full_name, value, global_step=event.epoch)

        return super().notify(event)

    @staticmethod
    def _get_last_run(main_dir: pathlib.Path) -> Optional[pathlib.Path]:
        all_dirs = [d for d in main_dir.iterdir() if d.is_dir()]
        if not all_dirs:
            return None

        return max(all_dirs, key=lambda d: d.stat().st_ctime)
