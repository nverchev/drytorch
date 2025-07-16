"""Module containing the HydraLink tracker."""

import functools
import pathlib
import shutil
import datetime
from typing import Optional
from typing_extensions import override

import hydra

from drytorch import exceptions
from drytorch import log_events
from drytorch.trackers import base_classes


class HydraLink(base_classes.Dumper):
    """
    Link current Hydra metadata to the experiment.

    Class Attributes:
        hydra_folder: the folder where the logs are grouped.

    Attributes:
        hydra_dir: the directory where hydra saves the run.
    """
    hydra_folder = 'hydra_runs'

    def __init__(self,
                 par_dir: Optional[pathlib.Path] = None,
                 copy_hydra: bool = True) -> None:
        """
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
            raise exceptions.TrackerException(self, 'Hydra has not started.')

        self._dir: pathlib.Path | None = None
        self._copy_hydra = copy_hydra

    @property
    def dir(self) -> pathlib.Path:
        """Return the directory where the files will be saved."""
        if self._dir is None:
            link_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            hydra_local_folder = self.par_dir / self.hydra_folder
            hydra_local_folder.mkdir(exist_ok=True)
            new_dir = hydra_local_folder / link_name
            self._dir = new_dir
            return new_dir
        else:
            return self._dir

    @override
    def clean_up(self) -> None:
        try:
            if self._copy_hydra and self.dir.is_symlink():
                self.dir.unlink()
                shutil.copytree(self.hydra_dir, self.dir)
        except exceptions.AccessOutsideScopeError:
            pass

        return

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        # call super method to create par_dir first
        super().notify(event)
        self.dir.symlink_to(self.hydra_dir, target_is_directory=True)
        return

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self.clean_up()
        return super().notify(event)
