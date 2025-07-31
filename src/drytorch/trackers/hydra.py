"""Module containing the HydraLink tracker."""

import functools
import pathlib
import shutil

import hydra

from typing_extensions import override

from drytorch import exceptions, log_events
from drytorch.trackers import base_classes


class HydraLink(base_classes.Dumper):
    """Link current Hydra metadata to the experiment.

    Class Attributes:
        hydra_folder: the folder where the logs are grouped.

    Attributes:
        hydra_dir: the directory where hydra saves the run.
    """

    hydra_folder = 'hydra_runs'

    def __init__(
        self, par_dir: pathlib.Path | None = None, copy_hydra: bool = True
    ) -> None:
        """Constructor.

        Args:
            par_dir: the directory where to dump metadata. Defaults to the
                experiment folder.
            copy_hydra: if True, copy the hydra folder content at the end of the
                experiment's scope, replacing the link folder.
        """
        super().__init__(par_dir)
        # get hydra configuration
        hydra_config = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
        str_dir = hydra_config.runtime.output_dir
        self.hydra_dir = pathlib.Path(str_dir)
        if not self.hydra_dir.exists():
            raise exceptions.TrackerError(self, 'Hydra has not started.')
        self._exp_version: str | None = None
        self._copy_hydra = copy_hydra
        return

    @property
    def dir(self) -> pathlib.Path:
        """Return the directory where the files will be saved."""
        if self._exp_version is None:
            raise exceptions.AccessOutsideScopeError()
        else:
            hydra_local_folder = self.par_dir / self.hydra_folder
            hydra_local_folder.mkdir(exist_ok=True, parents=True)
            return hydra_local_folder / self._exp_version

    @override
    def clean_up(self) -> None:
        try:
            if self._copy_hydra and self.dir.is_symlink():
                self.dir.unlink()
                shutil.copytree(self.hydra_dir, self.dir)
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
        self._exp_version = event.exp_ts
        self.dir.symlink_to(self.hydra_dir, target_is_directory=True)
        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        self._exp_version = None
        return super().notify(event)
