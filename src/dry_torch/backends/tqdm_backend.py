import functools
from typing import TextIO
import sys

from tqdm import auto

from src.dry_torch import events
from src.dry_torch import loading
from src.dry_torch import protocols as p


class TqdmBar:

    def __init__(self, loader: p.LoaderProtocol, out: TextIO) -> None:
        self.batch_size = loader.batch_size or 0
        self.dataset_len = loading.check_dataset_length(loader.dataset)
        total = loading.num_batches(self.dataset_len, self.batch_size)
        # noinspection PyTypeChecker
        self.pbar = auto.tqdm(total=total, file=out)
        self.seen_str = 'Seen'
        self.epoch_seen = 0
        self.last_epoch = False

    def update(self, metric_name: str, metric_value: float) -> None:
            self.pbar.update(1)
            self.epoch_seen += self.batch_size
            if self.epoch_seen >= self.dataset_len:
                self.epoch_seen = self.dataset_len
                self.last_epoch = True
            monitor_seen: dict[str, int] = {self.seen_str: self.epoch_seen}
            monitor_metric = {metric_name: f'{metric_value:.3e}'}
            monitor_dict = monitor_seen | monitor_metric
            self.pbar.set_postfix(monitor_dict, refresh=False)
            if self.last_epoch:
                self.pbar.close()
            return


class Tqdm(events.Subscriber):

    def __init__(self, out: TextIO = sys.stdout) -> None:
        super().__init__()
        self.out = out

    @functools.singledispatchmethod
    def notify(self, event: events.Event) -> None:
        return

    @notify.register
    def _(self, event: events.EpochProgressBar) -> None:
        bar = TqdmBar(event.loader, out=self.out)
        event.push_updates.append(bar.update)
        return


