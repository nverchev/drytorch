import functools
import pathlib
import shutil

import hydra

from src.dry_torch import tracking
from src.dry_torch import log_events
from src.dry_torch import repr_utils
from src.dry_torch.trackers import builtin_logger


class HydraLink(tracking.Tracker):

    def __init__(self, par_dir: pathlib.Path, link_name: str = 'hydra') -> None:
        super().__init__()
        self.link_dir: pathlib.Path
        self.par_dir = par_dir
        self._default_link_name = repr_utils.DefaultName(link_name)
        # noinspection PyUnresolvedReferences
        str_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.hydra_dir = pathlib.Path(str_dir)
        if not self.hydra_dir.exists():
            raise RuntimeError('Hydra has not started.')
        builtin_logger.enable_propagation()

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        exp_name = event.exp_name
        exp_dir = self.par_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        while True:
            link_dir = exp_dir / self._default_link_name()
            if not link_dir.exists():
                link_dir.symlink_to(self.hydra_dir, target_is_directory=True)
                break
        self.link_dir = link_dir
        return

    @notify.register
    def _(self, _event: log_events.StopExperiment) -> None:
        self.link_dir.unlink()
        shutil.copytree(self.hydra_dir, self.link_dir)
