"""TensorBoard tracker."""

import functools
import importlib.util
import logging
import pathlib

from typing import ClassVar, Final

from tensorboard import notebook as tb_notebook
from torch.utils import tensorboard
from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.trackers import base_classes


__all__ = [
    'TensorBoard',
]


if importlib.util.find_spec('tensorboard') is None:
    _MSG = 'TensorBoard is not installed. Run `pip install tensorboard`.'
    raise ImportError(_MSG)

SERVER_LOG_LEVEL: Final = 26

_logger: Final = logging.getLogger(__name__)


class TensorBoard(base_classes.Dumper):
    """Tracker that wraps the TensorBoard SummaryWriter.

    When a run starts, the tracker logs the command that serves the results,
    ``tensorboard --logdir <par_dir>/tensorboard``, to run in another terminal.
    In a notebook, it shows TensorBoard inline instead.
    """

    folder_name: ClassVar[str] = 'tensorboard'

    _flush_secs: int
    _max_queue_size: int
    _stashed_writers: dict[str, tensorboard.SummaryWriter]
    _writer: tensorboard.SummaryWriter | None

    def __init__(
        self,
        *,
        par_dir: pathlib.Path | None = None,
        flush_secs: int = 120,
        max_queue_size: int = 10,
    ) -> None:
        """Initialize.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            flush_secs: tensorboard.SummaryWriter docs.
            max_queue_size: see tensorboard.SummaryWriter docs.
        """
        super().__init__(par_dir=par_dir)
        self._flush_secs = flush_secs
        self._max_queue_size = max_queue_size
        self._stashed_writers = {}
        self._writer = None
        return

    @property
    def writer(self) -> tensorboard.SummaryWriter:
        """The active SummaryWriter instance.

        Raises:
            AccessOutsideScopeError: if no run has been started yet.
        """
        if self._writer is None:
            raise exceptions.AccessOutsideScopeError()

        return self._writer

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        super().notify(event)
        if not self.record:
            return

        run_dir = self._get_run_dir()
        self._show_results(self.par_dir / self.folder_name)

        self._writer = tensorboard.SummaryWriter(
            log_dir=run_dir.as_posix(),
            max_queue=self._max_queue_size,
            flush_secs=self._flush_secs,
        )
        for i, tag in enumerate(event.tags):
            self.writer.add_text('tag ' + str(i), tag)

        return

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        if not self.record:
            return super().notify(event)

        for name, value in event.metrics.items():
            full_name = f'{event.model_name}/{event.source_name}-{name}'
            self.writer.add_scalar(full_name, value, global_step=event.epoch)

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.PauseExperimentEvent) -> None:
        if self._writer is not None:
            self.writer.flush()
            self._stashed_writers[event.run_id] = self._writer
            self._writer = None

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ContinueExperimentEvent) -> None:
        if not event.record:
            return super().notify(event)

        if event.run_id not in self._stashed_writers:
            raise exceptions.NoStashedStateError(self, event.run_id)

        self._writer = self._stashed_writers.pop(event.run_id)
        return super().notify(event)

    @override
    def clean_up(self) -> None:
        if self._writer is not None:
            self.writer.close()

        self._writer = None
        return super().clean_up()

    @override
    def close(self) -> None:
        for writer in self._stashed_writers.values():
            writer.close()

        self._stashed_writers.clear()
        return super().close()

    def _show_results(self, logdir: pathlib.Path) -> None:
        if self._is_notebook():
            tb_notebook.start(f'--logdir {logdir}')
        else:
            _logger.log(
                SERVER_LOG_LEVEL,
                'TensorBoard (%s):\ntensorboard --logdir %s',
                logdir.parent.name,
                logdir.resolve(),
            )

        return

    @staticmethod
    def _is_notebook() -> bool:
        """Detect a Jupyter notebook or similar environment."""
        try:
            from IPython.core import getipython
        except ImportError:
            return False
        else:
            shell = getipython.get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole

            if 'colab' in str(getipython.get_ipython().__class__):
                return True

        return False
