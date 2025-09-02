"""Module containing the HydraLink tracker."""

import functools
import pathlib
import shutil

import hydra
import hydra.core.hydra_config

from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.trackers import base_classes
from drytorch.utils import repr_utils


TS_FMT = repr_utils.CreatedAtMixin.ts_fmt


class HydraLink(base_classes.Dumper):
    """Link current Hydra metadata to the experiment.

    Class Attributes:
        hydra_folder: the folder where the logs are grouped.

    Attributes:
        hydra_dir: the directory where hydra saves the run.
    """

    folder_name = 'hydra'

    def __init__(
        self, par_dir: pathlib.Path | None = None, copy_hydra: bool = True
    ) -> None:
        """Constructor.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            copy_hydra: if True, copy the hydra folder content at the end of the
                experiment's scope, replacing the link folder.
        """
        super().__init__(par_dir)
        # get hydra configuration
        hydra_config = hydra.core.hydra_config.HydraConfig.get()
        str_dir = hydra_config.runtime.output_dir
        self.hydra_dir: pathlib.Path = pathlib.Path(str_dir)
        if not self.hydra_dir.exists():
            raise exceptions.TrackerError(self, 'Hydra has not started.')
        self._copy_hydra: bool = copy_hydra
        return

    @override
    def clean_up(self) -> None:
        try:
            run_dir = self._get_run_dir(False)
            if self._copy_hydra and run_dir.is_symlink():
                run_dir.unlink()
                shutil.copytree(self.hydra_dir, run_dir)
        except exceptions.AccessOutsideScopeError:
            pass
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        # call super method to create par_dir first
        super().notify(event)
        self._run_id = event.run_ts.strftime(TS_FMT)  # use ts instead of id
        link = self._get_run_dir(mkdir=False)
        link.parent.mkdir(exist_ok=True, parents=True)
        link.symlink_to(self.hydra_dir, target_is_directory=True)
        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)
