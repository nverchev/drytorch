"""Module containing a TensorBoard tracker."""

import dataclasses
import functools
import pathlib
import shutil
import socket
import subprocess
import warnings
import webbrowser

from importlib.util import find_spec

from torch.utils import tensorboard
from typing_extensions import override

from drytorch import exceptions, log_events
from drytorch.trackers import base_classes
from drytorch.utils import repr_utils


if find_spec('tensorboard') is None:
    _MSG = 'TensorBoard is not installed. Run `pip install tensorboard`.'
    raise ImportError(_MSG)


class TensorBoard(base_classes.Dumper):
    """Tracker that wraps the TensorBoard SummaryWriter.

    Class Attributes:
        folder_name: name of the folder containing the output.
        base_port: starting port number for TensorBoard.
        instance_count: counter for TensorBoard instances started.
    """

    folder_name = 'tensorboard_runs'
    base_port = 6006
    instance_count = 0

    def __init__(
        self,
        par_dir: pathlib.Path | None = None,
        resume_run: bool = False,
        start_server: bool = True,
        open_browser: bool = False,
        max_queue_size: int = 10,
        flush_secs: int = 120,
    ) -> None:
        """Constructor.

        Args:
            par_dir: Directory to store metadata and logs. Defaults to the
                current experiment's one.
            resume_run: if True, resume from the latest run in the same folder.
            start_server: if True, start a local TensorBoard server.
            open_browser: if True, open TensorBoard in the browser.
            max_queue_size: see tensorboard.SummaryWriter docs.
            flush_secs: tensorboard.SummaryWriter docs.
        """
        super().__init__(par_dir)
        self.resume_run = resume_run
        self._writer: tensorboard.SummaryWriter | None = None
        self._port: int | None = None
        self.__class__.instance_count += 1
        self._instance_number = self.__class__.instance_count
        self._start_server = start_server
        self._open_browser = open_browser
        self._max_queue_size = max_queue_size
        self._flush_secs = flush_secs

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
        self._writer = None
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        super().notify(event)
        # determine the root directory
        if event.variation is None or self._par_dir is None:
            par_dir = self.par_dir
        else:
            par_dir = self.par_dir.parent  # to compare variations

        run_dir = par_dir / self.folder_name
        retrieved = self._get_last_run(run_dir) if self.resume_run else None
        if retrieved is None:
            # prevent nested folders
            id_list = [event.exp_name]
            if event.variation is not None:
                id_list.append(event.variation)
            id_list.append(event.exp_ts)
            exp_id = '_'.join(id_list)
            root_dir = run_dir / exp_id
        else:
            root_dir = retrieved

        # start the TensorBoard server
        if self._start_server:
            self._start_tensorboard(run_dir)

        # initialize writer
        self._writer = tensorboard.SummaryWriter(
            log_dir=root_dir.as_posix(),
            max_queue=self._max_queue_size,
            flush_secs=self._flush_secs,
        )
        for i, tag in enumerate(event.tags):
            self.writer.add_text('tag ' + str(i), tag)

        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        for name, value in event.metrics.items():
            full_name = f'{event.model_name}/{event.source_name}-{name}'
            self.writer.add_scalar(full_name, value, global_step=event.epoch)
        self.writer.flush()

        return super().notify(event)

    def _start_tensorboard(self, logdir: pathlib.Path) -> None:
        """Start a TensorBoard server and open it in the default browser."""
        # allocate a port
        instance_port = self.base_port + self._instance_number
        port = self._find_free_port(start=instance_port)
        self._port = port
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise exceptions.TrackerError(self, 'Invalid port')

        tensorboard_path = shutil.which('tensorboard')
        if tensorboard_path is None:
            msg = 'TensorBoard executable not found.'
            raise exceptions.TrackerError(self, msg)

        try:
            subprocess.Popen(  # noqa: S603
                [
                    tensorboard_path,
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

        if self._open_browser:
            try:
                webbrowser.open(f'http://localhost:{port}')
            except webbrowser.Error as we:
                msg = f'Failed to open web browser: {we}'
                warnings.warn(msg, exceptions.DryTorchWarning, stacklevel=2)
            except OSError as ose:
                msg = f'OS-level error while opening browser: {ose}'
                warnings.warn(msg, exceptions.DryTorchWarning, stacklevel=2)

    @staticmethod
    def _find_free_port(start: int = 6006, max_tries: int = 100) -> int:
        """Find a free port starting from the given one."""
        for port in range(start, start + max_tries):
            if TensorBoard._port_available(port):
                return port
        msg = f'No free ports available after {max_tries} tries.'
        raise exceptions.TrackerError(TensorBoard, msg)

    @staticmethod
    def _get_last_run(main_dir: pathlib.Path) -> pathlib.Path | None:
        all_dirs = [d for d in main_dir.iterdir() if d.is_dir()]
        if not all_dirs:
            msg = 'TensorBoard: No previous runs. Starting a new one.'
            warnings.warn(msg, exceptions.DryTorchWarning, stacklevel=2)
            return None
        return max(all_dirs, key=lambda d: d.stat().st_ctime)

    @staticmethod
    def _port_available(port: int) -> bool:
        """Check if the given port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) != 0
