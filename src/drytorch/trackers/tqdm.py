"""Module containing a tqdm tracker for progress bars."""

from __future__ import annotations

import functools
import sys

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import tqdm

from typing_extensions import override

from drytorch.core import log_events, track


if TYPE_CHECKING:
    from _typeshed import SupportsWrite


class EpochBar:
    """Bar that displays the current epoch's metrics and progress.

    This class is also used to display metrics and progress during evaluation.

    Class Attributes:
        fmt: the formatting of the bar.
        seen_str: the name for the elements of the batches.
        color: the color of the bar.

    Attributes:
        pbar: the wrapped tqdm bar.
    """

    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}'
    seen_str = 'Samples'
    color = 'green'

    def __init__(
        self,
        batch_size: int | None,
        num_iter: int,
        num_samples: int,
        leave: bool,
        file: SupportsWrite[str],
        desc: str,
    ) -> None:
        """Constructor.

        Args:
            batch_size: how many samples are in one batch.
            num_iter: the number of expected iterations.
            num_samples: the total number of samples.
            leave: whether to leave the bar in after the epoch.
            file: the stream where to flush the bar.
            desc: description to contextualize the bar.
        """
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._num_iter = num_iter
        self.pbar = tqdm.tqdm(
            total=num_iter,
            leave=leave,
            file=file,
            desc=desc,
            bar_format=self.fmt,
            colour=self.color,
        )
        self._epoch_seen = 0
        return

    def update(self, metrics: Mapping[str, Any]) -> None:
        """Update the bar and displays last batch metrics values.

        Args:
            metrics: the values from the last batch by metric name.
        """
        monitor_seen: dict[str, int | str]
        last_epoch = self.pbar.n == self._num_iter - 1
        if self._batch_size is not None:
            if last_epoch:
                self._epoch_seen = self._num_samples
            else:
                self._epoch_seen += self._batch_size
            monitor_seen = {self.seen_str: self._epoch_seen}
        else:
            monitor_seen = {self.seen_str: '?'}

        monitor_metric = {
            metric_name: f'{metric_value:.3e}'
            for metric_name, metric_value in metrics.items()
        }
        monitor_dict = monitor_seen | monitor_metric
        self.pbar.set_postfix(monitor_dict, refresh=False)
        self.pbar.update()
        if last_epoch:
            self.pbar.close()

        return


class TrainingBar:
    """Create a bar for the training progress.

    Class Attributes:
        fmt: the formatting of the bar.
        desc: the name for the iteration.
        color: the color of the bar.

    Attributes:
        pbar: the wrapped tqdm bar.
    """

    fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}'
    desc = 'Epoch'
    color = 'blue'

    def __init__(
        self,
        start_epoch: int,
        end_epoch: int,
        file: SupportsWrite[str],
        leave: bool,
    ) -> None:
        """Constructor.

        Args:
            start_epoch: the epoch from which the bar should start.
            end_epoch: the epoch where the bar should end.
            file: the stream where to flush the bar.
            leave: If True, leave bar once the iterations have completed.
        """
        self.pbar = tqdm.trange(
            start_epoch,
            end_epoch,
            desc=f'{self.desc}:',
            leave=leave,
            position=0,
            file=file,
            bar_format=self.fmt,
            colour=self.color,
        )
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch

    def update(self, current_epoch: int) -> None:
        """Update the bar and display the current epoch.

        Args:
            current_epoch: the current epoch.
        """
        self.pbar.update()
        description = f'{self.desc}: {current_epoch} / {self._end_epoch}'
        self.pbar.set_description(description)
        return


class TqdmLogger(track.Tracker):
    """Create an epoch progress bar."""

    def __init__(
        self,
        leave: bool = True,
        enable_training_bar: bool = False,
        file: SupportsWrite[str] = sys.stderr,
    ) -> None:
        """Constructor.

        Args:
            leave: whether to leave the epoch bar after completion.
            enable_training_bar: create a bar for the overall training progress.
            file: the stream where to flush the bar.

        Note:
            enable the training bar only if two progress bars are supported,
            and there is no other logger or printer streaming there.
        """
        super().__init__()
        self._leave = leave
        self._file = file
        self._enable_training_bar = enable_training_bar
        self._training_bar: TrainingBar | None = None
        self._epoch_bar: EpochBar | None = None
        return

    @override
    def clean_up(self) -> None:
        self._clean_epoch_bar()
        self._clean_training_bar()
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.IterateBatchEvent) -> None:
        desc = event.source_name.rjust(15)
        leave = self._leave and self._training_bar is None
        self._epoch_bar = EpochBar(
            event.batch_size,
            event.num_iter,
            event.dataset_size,
            leave=leave,
            file=self._file,
            desc=desc,
        )
        event.push_updates.append(self._epoch_bar.update)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartTrainingEvent) -> None:
        if self._enable_training_bar:
            self._training_bar = TrainingBar(
                event.start_epoch,
                event.end_epoch,
                file=self._file,
                leave=self._leave,
            )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartEpochEvent) -> None:
        if self._training_bar is not None:
            self._training_bar.update(event.epoch)

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndEpochEvent) -> None:
        self._clean_epoch_bar()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.TerminatedTrainingEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndTrainingEvent) -> None:
        self.clean_up()
        return super().notify(event)

    def _clean_training_bar(self) -> None:
        if self._training_bar is not None:
            self._training_bar.pbar.close()
            self._training_bar = None

        return

    def _clean_epoch_bar(self) -> None:
        if self._epoch_bar is not None:
            self._epoch_bar.pbar.close()
            self._epoch_bar = None

        return
