import functools
import itertools
from typing import TextIO
import sys

from tqdm import auto

from src.dry_torch import log_events
from src.dry_torch import loading
from src.dry_torch import protocols as p
from src.dry_torch import tracking


class EpochBar:
    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}'

    def __init__(self,
                 loader: p.LoaderProtocol,
                 leave: bool,
                 out: TextIO,
                 desc: str) -> None:
        self.batch_size = loader.batch_size or 0
        self.dataset_len = loading.check_dataset_length(loader.dataset)
        total = loading.num_batches(self.dataset_len, self.batch_size)
        # noinspection PyTypeChecker
        self.pbar = auto.tqdm(total=total,
                              leave=leave,
                              file=out,
                              desc=desc,
                              bar_format=self.fmt)
        self.seen_str = 'Samples'
        self.epoch_seen = 0
        self.last_epoch = False

    def update(self, metrics: dict[str, float]) -> None:
        self.pbar.update(1)
        self.epoch_seen += self.batch_size
        if self.epoch_seen >= self.dataset_len:
            self.epoch_seen = self.dataset_len
            self.last_epoch = True
        monitor_seen: dict[str, int] = {self.seen_str: self.epoch_seen}
        monitor_metric = {metric_name: f'{metric_value:.3e}'
                          for metric_name, metric_value in metrics.items()}
        monitor_dict = monitor_seen | monitor_metric
        self.pbar.set_postfix(monitor_dict, refresh=False)
        if self.last_epoch:
            self.pbar.close()
        return


class TrainingBar:
    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}'

    def __init__(self,
                 start_epoch: int,
                 end_epoch: int,
                 out: TextIO,
                 disable: bool) -> None:
        # noinspection PyTypeChecker
        self.pbar = auto.trange(start_epoch,
                                end_epoch,
                                desc='Epoch:',
                                leave=False,
                                position=0,
                                file=out,
                                disable=disable)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def update(self, current_epoch: int) -> None:
        self.pbar.update(1)
        self.pbar.set_description(
            f'Epoch: {current_epoch} / {self.end_epoch}')
        return

    def close(self) -> None:
        self.pbar.close()
        return


class TqdmLogger(tracking.Tracker):

    def __init__(self,
                 leave: bool = False,
                 enable_training_bar: bool = False,
                 out: TextIO = sys.stdout) -> None:
        super().__init__()
        self.leave = leave
        self.out = out
        self.enable_training_bar = enable_training_bar

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EpochBar) -> None:
        desc = event.source.rjust(15)
        bar = EpochBar(event.loader,
                       leave=self.leave,
                       out=self.out,
                       desc=desc)
        event.push_updates.append(bar.update)
        return

    @notify.register
    def _(self, event: log_events.StartTraining) -> None:
        self._training_bar = TrainingBar(event.start_epoch,
                                         event.end_epoch,
                                         out=self.out,
                                         disable=not self.enable_training_bar)
        return

    @notify.register
    def _(self, event: log_events.StartEpoch) -> None:
        self._training_bar.update(event.epoch)
        return
