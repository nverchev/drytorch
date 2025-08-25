"""Module containing a tracker calling Weights and Biases."""

import functools
import pathlib
import warnings

from typing import ClassVar

import wandb

from typing_extensions import override
from wandb.sdk import wandb_run, wandb_settings
from wandb.sdk.wandb_settings import Settings

from drytorch.core import exceptions, log_events
from drytorch.trackers.base_classes import Dumper
from drytorch.utils.repr_utils import recursive_repr


class Wandb(Dumper):
    """Tracker that wraps a run for the wandb library."""

    _default_settings: ClassVar[wandb_settings.Settings] = (
        wandb_settings.Settings()
    )
    folder_name = 'wandb'

    def __init__(
        self,
        par_dir: pathlib.Path | None = None,
        settings: wandb_settings.Settings = _default_settings,
    ) -> None:
        """Constructor.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            settings: settings object from wandb containing all init arguments.
        """
        super().__init__(par_dir)
        self._settings: Settings = settings
        self._run: wandb_run.Run | None = None
        return

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
    def _(self, event: log_events.StartExperimentEvent) -> None:
        super().notify(event)
        project = self._settings.project or event.exp_name
        group = self._settings.run_group or event.exp_name
        run_id = ''
        if event.resumed:
            api = wandb.Api()
            entity = self._settings.entity or api.default_entity
            runs = api.runs(
                f'{entity}/{project}',
                filters={'group': event.exp_name},
            )
            if runs:
                run_id = runs[0].id
            else:
                msg = 'Wandb: No previous runs. Starting a new one.'
                warnings.warn(msg, exceptions.DryTorchWarning, stacklevel=2)

        if self._settings.run_id:
            run_id = self._settings.run_id

        if not run_id:
            run_id = event.exp_name + '_' + event.run_id

        self._run = wandb.init(
            id=run_id,
            dir=self.par_dir.as_posix(),
            project=project,
            group=group,
            config=recursive_repr(event.config),
            tags=event.tags,
            settings=self._settings,
            resume='allow' if event.resumed else None,
        )
        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        if self.run is None:
            raise exceptions.AccessOutsideScopeError()

        plot_names = {
            f'{event.model_name}/{event.source_name}-{name}': value
            for name, value in event.metrics.items()
        }
        # noinspection PyTypeChecker
        self.run.log(plot_names, step=event.epoch)
        return super().notify(event)
