"""Tracker wrapping wandb (Weights and Biases)."""

import functools
import pathlib
from typing import Optional
from typing_extensions import override

import wandb
from wandb.sdk import wandb_run
from wandb.sdk import wandb_settings


from dry_torch import log_events
from dry_torch import tracking


class Wandb(tracking.Tracker):
    """Tracker for wandb library."""

    def __init__(
            self,
            par_dir: Optional[pathlib.Path] = None,
            settings: wandb_settings.Settings = wandb_settings.Settings()
    ) -> None:
        super().__init__()
        self.par_dir = par_dir
        self.settings = settings
        self.run: wandb_run.Run | None = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        # prioritize settings
        project = event.exp_name if self.settings.project is None else None
        root_dir = event.exp_dir if self.par_dir is None else None
        self.run = wandb.init(dir=root_dir,
                              project=project,
                              config=event.config,
                              settings=self.settings)

    @notify.register
    def _(self, _: log_events.StopExperiment) -> None:
        wandb.finish()
        self.run = None

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        plot_names = {f'{event.model_name}/{event.source}-{name}': value
                      for name, value in event.metrics.items()}
        self.run.log(plot_names, step=event.epoch)
        return

    # TODO: add image support
    # @notify.register
    # def _(self, event: log_events.Images) -> None:
    #     if not self.initialized or not self.log_images:
    #         return
    #
    #     try:
    #         images_dict = {
    #             f'{event.model_name}/{event.source}-{name}': wandb.Image(img)
    #             for name, img in event.images.items()}
    #         wandb.log(images_dict, step=event.step)
    #     except Exception as e:
    #         logger.error(f"Failed to log images: {str(e)}")
