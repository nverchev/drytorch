"""Module containing the HydraLink tracker."""

import functools
import pathlib
import shutil
from typing import Optional
from typing_extensions import override

import hydra

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch.trackers import base_classes
from dry_torch.trackers import logging


class HydraLink(base_classes.AbstractDumper):
    """
    Link current Hydra metadata to experiment.

    Attributes:
        hydra_folder: the folder where the logs are grouped.
        hydra_dir: the directory where hydra saves the run.
        link_name: the name of the folder with the link.
    """

    def __init__(self,
                 par_dir: Optional[pathlib.Path] = None,
                 copy_hydra: bool = True) -> None:
        """
        Args:
            par_dir: the directory where to dump metadata. Defaults to the
            copy_hydra: if True, copy the hydra folder content at the end of the
                experiment's scope, replacing the link folder.
        """
        super().__init__(par_dir)
        self.hydra_folder = 'hydra_runs'
        # get hydra configuration
        hydra_config = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
        str_dir = hydra_config.runtime.output_dir
        self.hydra_dir = pathlib.Path(str_dir)
        if not self.hydra_dir.exists():
            raise exceptions.TrackerException(self, 'Hydra has not started.')

        self._copy_hydra = copy_hydra
        self.link_name = 'run'
        self._counter = 0

    @property
    def dir(self) -> pathlib.Path:
        """Return the directory where the files will be saved."""
        if self._counter:
            link_name = self.link_name + f'_{self._counter}'
        else:
            link_name = self.link_name

        hydra_local_folder = self.par_dir / self.hydra_folder
        hydra_local_folder.mkdir(exist_ok=True)
        return hydra_local_folder / link_name

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        self._exp_dir = event.exp_dir

        while True:
            if self.dir.exists():
                self._counter += 1
            else:
                break

        self.dir.symlink_to(self.hydra_dir, target_is_directory=True)
        logging.enable_propagation()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        if self._copy_hydra:
            self.dir.unlink()
            shutil.copytree(self.hydra_dir, self.dir)

        self._exp_dir = None
        self._counter = 0
        logging.disable_propagation()
        return super().notify(event)
