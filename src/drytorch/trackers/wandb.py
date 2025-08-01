"""Module containing a tracker calling Weights and Biases."""

import functools
import pathlib
import warnings

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
    def _(self, event: log_events.StartExperimentEvent) -> None:
        # determine main directory
        super().notify(event)
        project = self._settings.project or event.exp_name
        if event.variation is None or self._par_dir is None:
            dir_ = self.par_dir
        else:
            self.par_dir.parent

        if self._settings.run_id:
            run_id = self._settings.run_id
        else:
            run_id = event.exp_ts.replace(":", "_")
            if event.variation:
                run_id = event.variation + '_' + run_id

        if self.resume_run:
            runs = wandb.Api().runs(project)
            run_id = runs[len(runs) - 1].id

        config_has_dict = hasattr(event.config, '__dict__')
        config_is_recognizable = isinstance(event.config, (dict, str))
        if config_has_dict or config_is_recognizable:
            config = event.config
        else:
            config = None
            msg = 'Config format not supported and will not be logged.'
            warnings.warn(msg, exceptions.DryTorchWarning, stacklevel=2)

        self._run = wandb.init(id=run_id,
                               dir=dir_,
                               project=project,
                               config=config,
                               tags=event.tags,
                               settings=self._settings,
                               resume='allow')
        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        if self.run is None:
            raise exceptions.AccessOutsideScopeError()

        plot_names = {f'{event.model_name}/{event.source_name}-{name}': value
                      for name, value in event.metrics.items()}
        # noinspection PyTypeChecker, PydanticTypeChecker
        self.run.log(plot_names, step=event.epoch)
        return super().notify(event)
