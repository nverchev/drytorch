"""Contains the HydraLink tracker."""

import functools
import pathlib
import shutil
from typing import Optional

import hydra
from typing_extensions import override

from dry_torch import log_events
from dry_torch.trackers import abstract_dumper
from dry_torch.trackers import logging


class HydraLink(abstract_dumper.AbstractDumper):
    """
    Tracker that links that organize Hydra logs.

    Attributes:
        hydra_folder: folder where the logs are grouped.
        hydra_dir: directory where hydra saves the run.
        link_name: name of the folder with the link.
    """

    def __init__(self,
                 par_dir: Optional[pathlib.Path] = None,
                 copy_hydra: bool = True) -> None:
        """
        Args:
            par_dir: parent directory for experiment data.
            copy_hydra: if True, copy the hydra folder content at the end of the
                run replacing the link folder
        """
        super().__init__(par_dir)
        self.hydra_folder = 'hydra_runs'
        # get hydra configuration
        hydra_config = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
        str_dir = hydra_config.runtime.output_dir
        self.hydra_dir = pathlib.Path(str_dir)
        if not self.hydra_dir.exists():
            raise RuntimeError('Hydra has not started.')

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

        return self.par_dir / self.hydra_folder / link_name

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
        return

    @notify.register
    def _(self, _event: log_events.StopExperiment) -> None:
        if self._copy_hydra:
            self.dir.unlink()
            shutil.copytree(self.hydra_dir, self.dir)
        self._exp_dir = None
        self._counter = 0
        logging.disable_propagation()
