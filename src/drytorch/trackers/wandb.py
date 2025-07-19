"""Module containing a tracker calling Weights and Biases."""

import functools
import pathlib

import wandb

from typing_extensions import override
from wandb.sdk import wandb_run, wandb_settings

from drytorch import exceptions, log_events
from drytorch.trackers.base_classes import Dumper


class Wandb(Dumper):
    """Tracker that wraps a run for the wandb library.

    Attributes:
        resume_run: resume the previous run from the project.
    """
    _default_settings = wandb_settings.Settings()

    def __init__(
            self,
            par_dir: pathlib.Path | None = None,
            settings: wandb_settings.Settings = _default_settings,
            resume_run: bool = False,
    ) -> None:
        """Constructor.

        Args:
            par_dir: the directory where to dump metadata. Overwrites settings.
                Defaults to the one for the current experiment.
            settings: settings object from wandb containing all init arguments.
            resume_run: resume the previous run from the project.
        """
        super().__init__(par_dir)
        self.resume_run = resume_run
        self._settings = settings
        self._run: wandb_run.Run | None = None

    @property
    def run(self) -> wandb_run.Run:
        """Active wandb run instance."""
        if self._run is None:
            raise exceptions.AccessOutsideScopeError()

        return self._run

    @override
    def clean_up(self) -> None:
        wandb.finish()
        self._run = None
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        # determine main directory
        super().notify(event)
        project = self._settings.project or event.exp_name
        run_id = self._settings.run_id or event.exp_version.replace(":", "_")
        if self.resume_run:
            runs = wandb.Api().runs(project)
            run_id = runs[len(runs) - 1].id

        self._run = wandb.init(id=run_id,
                               dir=self.par_dir,
                               project=project,
                               config=event.config,
                               settings=self._settings,
                               resume='allow')
        return

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        if self.run is None:
            raise exceptions.AccessOutsideScopeError()

        plot_names = {f'{event.model_name}/{event.source_name}-{name}': value
                      for name, value in event.metrics.items()}
        # noinspection PyTypeChecker, PydanticTypeChecker
        self.run.log(plot_names, step=event.epoch)
        return super().notify(event)
