"""
This module sets up custom logging configurations for the 'dry_torch' logger.

It defines and implements a formatter that formats log messages according to
the levels defined in the INFO_LEVELS variable.
By default, it prints to stdout and does not propagate to the main root.

Attributes:
    INFO_LEVELS: InfoLevels object for global settings.
"""
import functools
import logging
import sys
from typing import NamedTuple, Optional
from typing_extensions import override

from src.dry_torch import events


class InfoLevels(NamedTuple):
    """NamedTuple that defines different levels of information for logging."""
    tqdm_bar: int
    metrics: int
    epoch: int
    io: int
    training: int
    experiment: int


class InfoFormatter(logging.Formatter):
    """
    Custom formatter to format log messages based on their level.

    It adds a timestamp to logs of level training. When the logger level is
    set to a value higher than metrics, it overwrites the current epoch's log.
    """

    default_msec_format = ''

    @override
    def format(self, record: logging.LogRecord) -> str:
        self._style._fmt = self._info_fmt(record.levelno)
        return super().format(record)

    @staticmethod
    def _info_fmt(level_no: Optional[int] = None) -> str:
        if level_no == INFO_LEVELS.training:
            return '[%(asctime)s] - %(message)s\n'
        if level_no == INFO_LEVELS.epoch:
            if logger.level > INFO_LEVELS.metrics:
                return '%(message)s ...\r'
        return '%(message)s\n'


class DryTorchFilter(logging.Filter):
    """Filter that excludes logs from 'dry_torch'."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return 'dry_torch' not in record.name


def get_verbosity() -> int:
    """This function gets the verbosity level of the 'dry_torch' logger."""
    return logger.level


def set_verbosity(level_no: int):
    """This function sets the verbosity level of the 'dry_torch' logger."""
    global logger
    logger.setLevel(level_no)


def disable_default_handler() -> None:
    """This function disable the handler and filter of the local logger."""
    logger.setLevel(logging.NOTSET)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())


def enable_default_handler() -> None:
    """This function sets up the default logging configuration."""
    global logger
    logger.handlers.clear()

    formatter = InfoFormatter()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.terminator = ''
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(INFO_LEVELS.tqdm_bar)
    logger.propagate = False


def disable_propagation() -> None:
    """
    This function reverts the changes made by enable_propagation
    """
    global logger

    logger.propagate = False
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        for log_filter in handler.filters:
            if isinstance(log_filter, DryTorchFilter):
                handler.removeFilter(log_filter)
                break


def enable_propagation(deduplicate_stdout: bool = True) -> None:
    """
    This function allows logs to propagate to the root logger.

    Args:
        deduplicate_stdout: whether to remove local messages from stdout.
    """
    global logger

    logger.propagate = True
    if deduplicate_stdout:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if hasattr(handler.stream, 'name'):
                    if handler.stream.name == '<stdout>':
                        handler.addFilter(DryTorchFilter())


logger = logging.getLogger('dry_torch')
INFO_LEVELS = InfoLevels(tqdm_bar=17,
                         metrics=21,
                         epoch=23,
                         io=25,
                         training=27,
                         experiment=28)

for name, level in INFO_LEVELS._asdict().items():
    logging.addLevelName(level, name.center(10))

enable_default_handler()


class BuiltinLogger(events.Subscriber):

    @functools.singledispatchmethod
    def log(self, event: events.Event) -> None:
        return

    @log.register
    def _(self, event: events.TrainingStart) -> None:
        logger.log(INFO_LEVELS.training,
                   'Training %(model_name)s.',
                   {'model_name': event.model_name})
        return

    @log.register
    def _(self, event: events.TrainingEnd) -> None:
        logger.log(INFO_LEVELS.training, 'End of training.')
        return

