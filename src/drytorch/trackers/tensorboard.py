"""TensorBoard tracker."""

import functools
import importlib.util
import logging
import pathlib
import shutil
import socket
import subprocess

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

    The server it starts lasts as long as the trackers of its experiment: it
    stops when they are released, and at the latest when the process ends. To
    read the results later, start a server on the same folder instead:
    ``tensorboard --logdir <par_dir>/tensorboard``.

    Attributes:
        base_port: starting port number for TensorBoard.
        instance_count: counter for TensorBoard instances started.
    """

    base_port: ClassVar[int] = 6006
    folder_name: ClassVar[str] = 'tensorboard'
    instance_count: ClassVar[int] = 0

    _current_logdir: pathlib.Path | None
    _flush_secs: int
    _instance_number: int
    _max_queue_size: int
    _port: int | None
    _process: subprocess.Popen | None
    _start_server: bool
    _stashed_writers: dict[str, tensorboard.SummaryWriter]
    _writer: tensorboard.SummaryWriter | None

    def __init__(
        self,
        *,
        par_dir: pathlib.Path | None = None,
        flush_secs: int = 120,
        max_queue_size: int = 10,
        start_server: bool = True,
    ) -> None:
        """Initialize.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            flush_secs: tensorboard.SummaryWriter docs.
            max_queue_size: see tensorboard.SummaryWriter docs.
            start_server: if ``True``, start a local TensorBoard server.
        """
        super().__init__(par_dir=par_dir)
        self._current_logdir = None
        self._flush_secs = flush_secs
        self.__class__.instance_count += 1
        self._instance_number = self.__class__.instance_count
        self._max_queue_size = max_queue_size
        self._port = None
        self._process = None
        self._start_server = start_server
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
        run_dir = self._get_run_dir()
        if self._start_server:
            self._start_tensorboard(self.par_dir / self.folder_name)

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
        self._terminate_server()
        return super().close()

    def _start_tensorboard(self, logdir: pathlib.Path) -> None:
        """Start a TensorBoard server, shown inline in a notebook."""
        if self._is_notebook():
            tb_notebook.start(f'--logdir {logdir}')
        else:
            self._start_tensorboard_server(logdir)
        return

    def _start_tensorboard_server(self, logdir: pathlib.Path) -> None:
        """Start a TensorBoard server titled after the experiment folder."""
        if self._process is not None:
            if self._process.poll() is None and self._current_logdir == logdir:
                return

            self._terminate_server()

        instance_port = self.base_port + self._instance_number
        port = self._find_free_port(start=instance_port)
        self._port = port
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise exceptions.TrackerError(self, 'Invalid port')

        tensorboard_executable_path = shutil.which('tensorboard')
        if tensorboard_executable_path is None:
            msg = 'TensorBoard executable not found.'
            raise exceptions.TrackerError(self, msg)

        title = logdir.parent.name
        try:
            self._process = subprocess.Popen(  # noqa: S603
                [
                    tensorboard_executable_path,
                    'serve',
                    '--logdir',
                    str(logdir),
                    '--port',
                    str(port),
                    '--reload_multifile',
                    'true',
                    '--window_title',
                    title,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as ose:
            msg = 'TensorBoard failed to start'
            raise exceptions.TrackerError(self, msg) from ose

        self._current_logdir = logdir
        _logger.log(
            SERVER_LOG_LEVEL,
            'TensorBoard (%s): http://localhost:%d',
            title,
            port,
        )
        return

    def _terminate_server(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._current_logdir = None
            self._port = None

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

    @staticmethod
    def _find_free_port(start: int = 6006, max_tries: int = 100) -> int:
        """Find a free port starting from the given one."""
        for port in range(start, start + max_tries):
            if TensorBoard._port_available(port):
                return port

        msg = f'No free ports available after {max_tries} tries.'
        raise exceptions.TrackerError(TensorBoard, msg)

    @staticmethod
    def _port_available(port: int) -> bool:
        """Check if the given port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) != 0
