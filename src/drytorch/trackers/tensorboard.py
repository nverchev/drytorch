"""Module containing a TensorBoard tracker."""

import functools
import pathlib
import subprocess
import warnings
import socket
import webbrowser
import datetime

from typing import Optional
from typing_extensions import override

from torch.utils import tensorboard

try:
    import tensorboard as _tensorboard  # type: ignore
except ImportError as ie:
    err_msg = 'TensorBoard is not installed. Run `pip install tensorboard`.'
    raise ImportError(err_msg) from ie

from drytorch import exceptions
from drytorch import log_events
from drytorch.trackers import base_classes


class TensorBoard(base_classes.Dumper):
    """
    Tracker that wraps the TensorBoard SummaryWriter.

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
            par_dir: Optional[pathlib.Path] = None,
            resume_run: bool = False
    ) -> None:
        """
        Args:
            par_dir: the directory where to dump metadata. Defaults to the
                one for the current experiment.
            resume_run: load previous sessions having the same directory.
        """
        super().__init__(par_dir)
        self.resume_run = resume_run
        self._writer: Optional[tensorboard.SummaryWriter] = None
        self._port: Optional[int] = None
        self.__class__.instance_count += 1
        self._instance_number = self.__class__.instance_count

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

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        super().notify(event)

        # determine the root directory
        run_dir = self.par_dir / self.folder_name
        retrieved = self._get_last_run(run_dir) if self.resume_run else None
        if retrieved is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            root_dir = run_dir / timestamp
        else:
            root_dir = retrieved

        # start the TensorBoard server
        self._start_tensorboard(run_dir)

        # initialize writer
        self._writer = tensorboard.SummaryWriter(log_dir=root_dir.as_posix())
        if event.config:
            try:
                self.writer.add_hparams(hparam_dict=event.config,
                                        metric_dict={})
            except TypeError:
                pass

        return

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
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
        try:
            subprocess.Popen([
                "tensorboard",
                "serve",  # required with modern TensorBoard CLI
                "--logdir", str(logdir),
                "--port", str(port),
                "--reload_multifile", "true",
            ])
        except FileNotFoundError:
            msg = 'TensorBoard executable not found.'
            raise exceptions.TrackerException(self, msg)

        try:
            webbrowser.open(f'http://localhost:{port}')
        except Exception as e:
            msg = f'Could not open browser for TensorBoard: {e}'
            warnings.warn(msg, exceptions.DryTorchWarning)

    @staticmethod
    def _find_free_port(start: int = 6006, max_tries: int = 100) -> int:
        """Find a free port starting from the given one."""
        for port in range(start, start + max_tries):
            if TensorBoard._port_available(port):
                return port
        msg = f'No free ports available after {max_tries} tries.'
        raise exceptions.TrackerException(TensorBoard, msg)

    @staticmethod
    def _get_last_run(main_dir: pathlib.Path) -> Optional[pathlib.Path]:
        all_dirs = [d for d in main_dir.iterdir() if d.is_dir()]
        if not all_dirs:
            msg = 'TensorBoard: No previous runs. Starting a new one.'
            warnings.warn(msg, exceptions.DryTorchWarning)
            return None
        return max(all_dirs, key=lambda d: d.stat().st_ctime)

    @staticmethod
    def _port_available(port: int) -> bool:
        """Check if the given port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) != 0
