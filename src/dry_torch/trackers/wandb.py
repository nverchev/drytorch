"""Tracker wrapping wandb (Weights and Biases)."""

import functools
from typing import Optional
from typing_extensions import override

import wandb
from wandb.sdk import wandb_run
from wandb.sdk import wandb_settings

from dry_torch import log_events
from dry_torch import tracking
from dry_torch import exceptions


class Wandb(tracking.Tracker):
    """
    Tracker for the wandb library.
    
    Attributes:
        resume_run: resume previous run from the project.
    """

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
        self.resume_run = resume_previous_run
        self._settings = settings
        self._run: wandb_run.Run | None = None

    @property
    def run(self) -> wandb_run.Run:
        """Wandb run."""
        if self._run is None:
            raise exceptions.AccessOutsideScopeError()
        return self._run

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        # prioritize settings
        project = self._settings.project or event.exp_name
        root_dir = self._settings.root_dir or event.exp_dir
        run_id: Optional[str] = self._settings.run_id
        if self.resume_run:
            runs = wandb.Api().runs(project)
            run_id = runs[len(runs) - 1].id

        self._run = wandb.init(id=run_id,
                               dir=root_dir,
                               project=project,
                               config=event.config,
                               settings=self._settings,
                               resume='allow')
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        wandb.finish()
        self._run = None
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        if self.run is None:
            raise exceptions.AccessOutsideScopeError()
        plot_names = {f'{event.model_name}/{event.source_name}-{name}': value
                      for name, value in event.metrics.items()}
        self.run.log(plot_names, step=event.epoch)
        return super().notify(event)
