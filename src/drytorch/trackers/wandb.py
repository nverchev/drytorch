"""Module containing a tracker calling Weights and Biases."""

import copy
import functools
import pathlib
import warnings

from typing import ClassVar

import wandb

from typing_extensions import override
from wandb.sdk import wandb_run, wandb_settings

from drytorch.core import exceptions, log_events
from drytorch.trackers.base_classes import Dumper
from drytorch.utils import repr_utils


__all__ = [
    'Wandb',
    'WandbWarning',
]


class WandbWarning(exceptions.DryTorchWarning):
    """Warning class for wandb."""

    _template = 'Wandb: {}'


class Wandb(Dumper):
    """Tracker that wraps a run for the wandb library."""

    _default_settings: ClassVar[wandb_settings.Settings] = (
        wandb_settings.Settings()
    )
    folder_name = 'wandb'

    _settings: wandb_settings.Settings
    _run: wandb_run.Run | None
    _defined_metrics: set[str]
    _stashed_runs: dict[str, wandb_run.Run]

    def __init__(
        self,
        par_dir: pathlib.Path | None = None,
        settings: wandb_settings.Settings = _default_settings,
    ) -> None:
        """Initialize.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            settings: settings object from wandb containing all init arguments.
        """
        super().__init__(par_dir)
        self._settings = copy.copy(settings)
        if self._settings.reinit not in ('create_new', 'default', True):
            msg = "Forcing reinit='create_new' to support paused runs."
            warnings.warn(WandbWarning(msg), stacklevel=2)

        self._settings.reinit = 'create_new'

        self._run = None
        self._defined_metrics = set()
        self._stashed_runs = {}
        return

    @property
    def run(self) -> wandb_run.Run:
        """Active wandb run instance.

        Raises:
            AccessOutsideScopeError: if no run has been started yet.
        """
        if self._run is None:
            raise exceptions.AccessOutsideScopeError()

        return self._run

    @override
    def clean_up(self) -> None:
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as e:
            warnings.warn(
                WandbWarning(f'Error during cleanup: {e}'), stacklevel=1
            )
        finally:
            self._run = None
            self._defined_metrics = set()

        return super().clean_up()

    @override
    def close(self) -> None:
        for run in self._stashed_runs.values():
            try:
                run.finish()
            except Exception as e:
                warnings.warn(
                    WandbWarning(f'Error finishing stashed run: {e}'),
                    stacklevel=1,
                )

        self._stashed_runs.clear()
        return super().close()

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
            try:
                run_id = runs[0].id
            except (IndexError, ValueError):
                msg = 'No previous runs. Starting a new one.'
                warnings.warn(WandbWarning(msg), stacklevel=2)

        if self._settings.run_id:
            run_id = self._settings.run_id

        if not run_id:
            run_id = event.exp_name + '_' + event.run_id

        repr_config = repr_utils.recursive_repr(event.config, depth=1000)
        self._run = wandb.init(
            id=run_id,
            dir=self.par_dir.as_posix(),
            project=project,
            group=group,
            config=repr_config,
            tags=event.tags,
            settings=self._settings,
            resume='allow' if event.resumed else None,
        )
        self._defined_metrics = set()
        return

    @notify.register
    def _(self, event: log_events.PauseExperimentEvent) -> None:
        if self._run is not None:
            self._stashed_runs[event.run_id] = self._run
            self._run = None
            self._defined_metrics = set()

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ContinueExperimentEvent) -> None:
        if event.run_id not in self._stashed_runs:
            raise exceptions.NoStashedStateError(self, event.run_id)

        self._run = self._stashed_runs.pop(event.run_id)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        """Process metric events.

        Raises:
            AccessOutsideScopeError: if called outside an active run scope.
        """
        plot_names = {
            f'{event.model_name}/{event.source_name}-{name}': value
            for name, value in event.metrics.items()
        }
        step_key = f'Progress/{event.model_name}'
        plot_step = {step_key: event.epoch}

        # define new metrics only once
        for name in plot_names:
            if name not in self._defined_metrics:
                self.run.define_metric(name, step_metric=step_key)
                self._defined_metrics.add(name)

        self.run.log(plot_names | plot_step)
        return super().notify(event)
