"""
This module sets up custom logging configurations for the 'dry_torch' logger.

It defines and implements a formatter that formats log messages according to
the levels defined in the INFO_LEVELS variable.
By default, it prints to stdout and does not propagate to the main root.

Attributes:
    INFO_LEVELS: InfoLevels object for global settings.
"""

import logging
import sys
from typing import NamedTuple, Optional

from typing_extensions import override


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


def set_default_logging() -> None:
    """ This function sets up the default logging configuration."""
    logger.handlers.clear()
    formatter = InfoFormatter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.terminator = ''
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(INFO_LEVELS.tqdm_bar)
    logger.propagate = False


def propagate_to_root_logger(deduplicate_stdout: bool = True,
                             remove_handlers: bool = False) -> None:
    """
    This function allow logs to propagate to the root logger.

    Args:
        deduplicate_stdout: if True, filters dry_torch logs from root stdout.
        remove_handlers: if True, removes all handlers from the module logger.
    """
    logger.propagate = True
    if remove_handlers:
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.NOTSET)

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

set_default_logging()
