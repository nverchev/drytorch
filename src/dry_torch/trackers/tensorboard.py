"""Define the TensorBoard tracker."""

import functools
import pathlib
from typing import Optional
from typing_extensions import override
import warnings

import torch.utils.tensorboard

from dry_torch import log_events
from dry_torch import tracking
from dry_torch import exceptions


class TensorBoard(tracking.Tracker):
    """Tracker for TensorBoard library."""
    folder_name = 'tensorboard_runs'

    def __init__(
            self,
            par_dir: pathlib.Path = pathlib.Path(folder_name),
            resume_run: bool = False
    ) -> None:
        super().__init__()
        self.par_dir = par_dir
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

        # Determine log directory
        if self.resume_run:
            retrieved = self._get_last_run()
            if retrieved is None:
                msg = 'TensorBoard: No previous runs. Starting a new one.'
                warnings.warn(msg)
                root_dir = self.par_dir / event.exp_name
            else:
                root_dir = retrieved

        else:
            root_dir = self.par_dir / event.exp_name

        # Initialize TensorBoard SummaryWriter
        self._writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=root_dir.as_posix(),
        )

        # Log experiment configuration as hyperparameters
        if event.config:
            try:
                self.writer.add_hparams(hparam_dict=event.config,
                                        metric_dict={})
            except TypeError:
                pass
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self.writer.close()
        self._writer = None
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:

        # Log scalar metrics
        for name, value in event.metrics.items():
            full_name = f'{event.model_name}/{event.source_name}-{name}'
            self.writer.add_scalar(full_name, value, global_step=event.epoch)

    def _get_last_run(self) -> Optional[pathlib.Path]:

        # Get all subdirectories
        all_dirs = [d for d in self.par_dir.iterdir() if d.is_dir()]

        if not all_dirs:
            # If no previous runs exist, return None
            return None

        # Get the most recent directory based on creation time
        return max(all_dirs, key=lambda d: d.stat().st_ctime)
