"""Creates a progress meter tailored for specific events."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Mapping
import sys

from tqdm import auto  # type: ignore
from typing_extensions import override

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
        """
        Args:
            batch_size: how many samples are in one batch.
            num_iter: number of expected iterations.
            num_samples: total number of samples.
            leave: whether to leave the bar in after the epoch.
            out: stream where to flush the bar.
            desc: description to contextualize the bar.
        """
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
        """
        Updates the bar and displays last batch metrics values.

        Args:
            metrics: the values from the last batch by metric name.
        """
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
    """Class that creates a bar for the training progress."""

    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}'

    def __init__(self,
                 start_epoch: int,
                 end_epoch: int,
                 out: SupportsWrite[str],
                 disable: bool) -> None:
        """
        Args:
            start_epoch: the epoch from which the bar should start.
            end_epoch: the epoch where the bar should end.
            out: stream where to flush the bar.
            disable: if true this class will not produce any output.
        """
        self._pbar = auto.trange(start_epoch,
                                 end_epoch,
                                 desc='Epoch:',
                                 leave=False,
                                 position=0,
                                 file=out,
                                 disable=disable,
                                 colour='blue')
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch

    def update(self, current_epoch: int) -> None:
        """
        Updates the bar and display the current epoch.

        Args:
            current_epoch: the current epoch.
        """
        self._pbar.update()
        description = f'Epoch: {current_epoch} / {self._end_epoch}'
        self._pbar.set_description(description)
        return

    def close(self) -> None:
        """Close the tqdm bar."""
        self._pbar.close()
        return


class TqdmLogger(tracking.Tracker):
    """Tracker that creates an epoch progress bar."""

    def __init__(self,
                 leave: bool = False,
                 enable_training_bar: bool = False,
                 out: SupportsWrite[str] = sys.stdout) -> None:
        """
        Args:
            leave: whether to leave the bar after completion.
            enable_training_bar: create a bar for the overall training progress.
            out: stream where to flush the bar.

        Note:
            enable the training bar only if out support two progress bars and
            there is no other logger or printer streaming there.
        """

        super().__init__()
        self._leave = leave
        self._out = out
        self._enable_training_bar = enable_training_bar
        self._training_bar: TrainingBar | None = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.IterateBatch) -> None:
        desc = format(event.source, 's').rjust(15)
        bar = EpochBar(event.batch_size,
                       event.num_iter,
                       event.dataset_size,
                       leave=self._leave,
                       out=self._out,
                       desc=desc)
        event.push_updates.append(bar.update)
        return

    @notify.register
    def _(self, event: log_events.StartTraining) -> None:
        self._training_bar = TrainingBar(event.start_epoch,
                                         event.end_epoch,
                                         out=self._out,
                                         disable=not self._enable_training_bar)
        return

    @notify.register
    def _(self, event: log_events.StartEpoch) -> None:
        if self._training_bar is not None:
            self._training_bar.update(event.start_epoch)
        return
