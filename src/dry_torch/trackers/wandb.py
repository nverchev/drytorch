"""Tracker wrapping wandb (Weights and Biases)."""

import functools
import pathlib
from typing import Optional
from typing_extensions import override

import wandb
from wandb.sdk import wandb_run
from wandb.sdk import wandb_settings

from dry_torch import log_events
from dry_torch import tracking


class Wandb(tracking.Tracker):
    """Tracker for wandb library."""

    def __init__(
            self,
            settings: wandb_settings.Settings = wandb_settings.Settings(),
            resume_previous_run: bool = False,
    ) -> None:
        """
        Args:
            settings: Settings object from wandb containing all init arguments.
            resume_previous_run: resume previous run from the project.
        """
        super().__init__()
        self.settings = settings
        self.resume_previous_run = resume_previous_run
        self.run: wandb_run.Run | None = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        # prioritize settings
        project = self.settings.project or format(event.exp_name, 's')
        root_dir = self.settings.root_dir or event.exp_dir
        run_id: Optional[str] = self.settings.run_id
        if self.resume_previous_run:
            runs = wandb.Api().runs(project)
            run_id = runs[len(runs) - 1].id

        self.run = wandb.init(id=run_id,
                              dir=root_dir,
                              project=project,
                              config=event.config,
                              settings=self.settings,
                              resume='allow')

    @notify.register
    def _(self, _: log_events.StopExperiment) -> None:
        wandb.finish()
        self.run = None

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        if self.run is None:
            raise RuntimeError('Access outside experiment scope.')
        plot_names = {f'{event.model_name:s}/{event.source:s}-{name}': value
                      for name, value in event.metrics.items()}
        self.run.log(plot_names, step=event.epoch)
        return
