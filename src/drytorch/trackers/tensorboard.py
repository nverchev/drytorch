"""Module containing a TensorBoard tracker."""

import functools
import pathlib
import shutil
import socket
import subprocess

from importlib.util import find_spec
from typing import ClassVar

from tensorboard import notebook as tb_notebook
from torch.utils import tensorboard
from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.trackers import base_classes


__all__ = [
    'TensorBoard',
]


if find_spec('tensorboard') is None:
    _MSG = 'TensorBoard is not installed. Run `pip install tensorboard`.'
    raise ImportError(_MSG)


class TensorBoard(base_classes.Dumper):
    """Tracker that wraps the TensorBoard SummaryWriter.

    Attributes:
        base_port: starting port number for TensorBoard.
        instance_count: counter for TensorBoard instances started.
    """

    folder_name = 'tensorboard'
    base_port: ClassVar[int] = 6006
    instance_count: ClassVar[int] = 0

    _writer: tensorboard.SummaryWriter | None
    _process: subprocess.Popen | None
    _port: int | None
    _instance_number: int
    _start_server: bool
    _max_queue_size: int
    _flush_secs: int

    def __init__(
        self,
        par_dir: pathlib.Path | None = None,
        start_server: bool = True,
        max_queue_size: int = 10,
        flush_secs: int = 120,
    ) -> None:
        """Constructor.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            start_server: if True, start a local TensorBoard server.
            max_queue_size: see tensorboard.SummaryWriter docs.
            flush_secs: tensorboard.SummaryWriter docs.
        """
        super().__init__(par_dir)
        self._writer = None
        self._process = None
        self._port = None
        self.__class__.instance_count += 1
        self._instance_number = self.__class__.instance_count
        self._start_server = start_server
        self._max_queue_size = max_queue_size
        self._flush_secs = flush_secs
        return

    @property
    def writer(self) -> tensorboard.SummaryWriter:
        """The active SummaryWriter instance."""
        if self._writer is None:
            raise exceptions.AccessOutsideScopeError()

        return self._writer

    @override
    def clean_up(self) -> None:
        if self._writer is not None:
            self.writer.close()

        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None

        self._writer = None
        return super().clean_up()

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

        self.writer.flush()
        return super().notify(event)

    def _start_tensorboard(self, logdir: pathlib.Path) -> None:
        """Start a TensorBoard server and open it in the default browser."""
        if self._is_notebook():
            tb_notebook.start(f'--logdir {logdir}')
        else:
            self._start_tensorboard_server(logdir)
        return

    def _start_tensorboard_server(self, logdir: pathlib.Path) -> None:
        """Start a TensorBoard server process."""
        instance_port = self.base_port + self._instance_number
        port = self._find_free_port(start=instance_port)
        self._port = port
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise exceptions.TrackerError(self, 'Invalid port')

        tensorboard_executable_path = shutil.which('tensorboard')
        if tensorboard_executable_path is None:
            msg = 'TensorBoard executable not found.'
            raise exceptions.TrackerError(self, msg)

        try:
            self._process = subprocess.Popen(  # noqa: S603
                [  # noqa: S603
                    tensorboard_executable_path,
                    'serve',
                    '--logdir',
                    str(logdir),
                    '--port',
                    str(port),
                    '--reload_multifile',
                    'true',
                ],
            )
        except subprocess.CalledProcessError as cpe:
            msg = 'TensorBoard failed to start'
            raise exceptions.TrackerError(self, msg) from cpe

        return

    @staticmethod
    def _is_notebook() -> bool:
        """Detect a Jupyter notebook or similar environment."""
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:
            return False
        else:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole

            if 'colab' in str(get_ipython().__class__):
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
