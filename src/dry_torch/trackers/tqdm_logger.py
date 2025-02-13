"""Creates a progress meter tailored for specific events."""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Mapping
import sys

from tqdm import auto  # type: ignore

from dry_torch import log_events
from dry_torch import tracking

if TYPE_CHECKING:
    from _typeshed import SupportsWrite


class EpochBar:
    """Bar that displays current epoch's metrics and progress."""
    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}'

    def __init__(self,
                 batch_size: int | None,
                 num_iter: int,
                 num_samples: int,
                 leave: bool,
                 out: SupportsWrite[str],
                 desc: str) -> None:
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_iter = num_iter
        self.pbar = auto.tqdm(total=num_iter,
                              leave=leave,
                              file=out,
                              desc=desc,
                              bar_format=self.fmt,
                              colour='green')
        self.seen_str = 'Samples'
        self.epoch_seen = 0
        return

    def update(self, metrics: Mapping[str, Any]) -> None:
        """Adds batch and metric information to the bar."""
        monitor_seen: dict[str, int | str]
        last_epoch = self.pbar.n == self.num_iter - 1
        if self.batch_size is not None:
            if last_epoch:
                self.epoch_seen = self.num_samples
            else:
                self.epoch_seen += self.batch_size
            monitor_seen = {self.seen_str: self.epoch_seen}
        else:
            monitor_seen = {self.seen_str: '?'}
        monitor_metric = {metric_name: f'{metric_value:.3e}'
                          for metric_name, metric_value in metrics.items()}
        monitor_dict = monitor_seen | monitor_metric
        self.pbar.set_postfix(monitor_dict, refresh=False)
        self.pbar.update()
        if last_epoch:
            self.pbar.close()
        return


class TrainingBar:
    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}'

    def __init__(self,
                 start_epoch: int,
                 end_epoch: int,
                 out: SupportsWrite[str],
                 disable: bool) -> None:
        self.pbar = auto.trange(start_epoch,
                                end_epoch,
                                desc='Epoch:',
                                leave=False,
                                position=0,
                                file=out,
                                disable=disable,
                                colour='blue')
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def update(self, current_epoch: int) -> None:
        self.pbar.update()
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
                 out: SupportsWrite[str] = sys.stdout) -> None:
        super().__init__()
        self.leave = leave
        self.out = out
        self.enable_training_bar = enable_training_bar
        self.training_bar: TrainingBar | None = None

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.IterateBatch) -> None:
        desc = format(event.source, 's').rjust(15)
        bar = EpochBar(event.batch_size,
                       event.num_iter,
                       event.dataset_size,
                       leave=self.leave,
                       out=self.out,
                       desc=desc)
        event.push_updates.append(bar.update)
        return

    @notify.register
    def _(self, event: log_events.StartTraining) -> None:
        self.training_bar = TrainingBar(event.start_epoch,
                                        event.end_epoch,
                                        out=self.out,
                                        disable=not self.enable_training_bar)
        return

    @notify.register
    def _(self, event: log_events.StartEpoch) -> None:
        if self.training_bar is not None:
            self.training_bar.update(event.epoch)
        return
