import io



import pathlib
from typing import Optional

from tempfile import TemporaryDirectory
import wandb

from dry_torch import log_backend


class WandbBackend(log_backend.LogBackend):
    """Logging backend implementation using SQLite with SQLAlchemy."""

    def __init__(self, output_dir: pathlib.Path | str, exp_name: str) -> None:
        wandb.finish()
        wandb.init(project=exp_name, dir=output_dir)

    def __call__(self,
                 model_name: str,
                 source: str,
                 partition: str,
                 epoch: int,
                 metrics: dict[str, float]) -> None:

        wandb.log({model_name: {source: metrics}}, step=epoch, commit=False)

    def plot(self,
             model_name: str,
             partition: str,
             metric: str) -> None:
        pass


class WandbLog(log_backend.ExperimentLog):

    def create_log(self,
                   local_path: Optional[pathlib.Path],
                   exp_name: str) -> log_backend.LogBackend:

        if local_path is None:
            return WandbBackend(TemporaryDirectory().name, exp_name)
        return WandbBackend(local_path, exp_name)


def my_test():
    exp_log = WandbLog()
    backend1 = exp_log.create_log(None, exp_name='first')
    backend2 = exp_log.create_log(None, exp_name='second')
    backend1('model_name',
             'train',
             'split',
             1,
             {'loss': 0.1, 'accuracy': 0.9})
    backend2('another_model',
             'train',
             'split',
             1,
             {'loss': 0.1, 'accuracy': 0.9})
    backend1.plot('model_name', 'train',
             'split', )
    backend2.plot('model_name', 'train',
             'split', )
    backend1.plot('model_name', 'train',
             'split', )

my_test()
