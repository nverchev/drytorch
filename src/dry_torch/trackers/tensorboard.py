import functools
import pathlib
from typing import Optional
from typing_extensions import override
import torch.utils.tensorboard

from dry_torch import log_events
from dry_torch import tracking
import warnings


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
        self.writer: Optional[torch.utils.tensorboard.SummaryWriter] = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:

        # Determine log directory
        if self.resume_run:
            retrieved = self.get_last_run()
            if retrieved is None:
                msg = 'TensorBoard: No previous runs. Starting a new one.'
                warnings.warn(msg)
                root_dir = self.par_dir / event.exp_name
            else:
                root_dir = retrieved

        else:
            root_dir = self.par_dir / event.exp_name

        # Initialize TensorBoard SummaryWriter
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=root_dir.as_posix(),
        )

        # Log experiment configuration as hyperparameters
        if event.config:
            try:
                self.writer.add_hparams(hparam_dict=event.config,
                                        metric_dict={})
            except TypeError:
                pass

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        if not self.writer:
            return

        # Log scalar metrics
        for name, value in event.metrics.items():
            full_name = f'{event.model_name:s}/{event.source:s}-{name}'
            self.writer.add_scalar(full_name, value, global_step=event.epoch)

    def get_last_run(self) -> Optional[pathlib.Path]:

        if not self.par_dir.exists():
            return None

        # Get all subdirectories
        all_dirs = [d for d in self.par_dir.iterdir() if d.is_dir()]

        if not all_dirs:
            # If no previous runs exist, return None
            return None

        # Get the most recent directory based on creation time
        return max(all_dirs, key=lambda d: d.stat().st_ctime)
