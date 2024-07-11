import sys
import logging
from typing import NamedTuple, Optional


class InfoLevels(NamedTuple):
    tqdm_bar: int
    metrics: int
    epoch: int
    checkpoint: int
    training: int
    experiment: int


class InfoFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        self._style._fmt = self._info_fmt(record.levelno)
        return super().format(record)

    @staticmethod
    def _info_fmt(level_no: Optional[int] = None) -> str:
        if level_no == INFO_LEVELS.training:
            return '%(asctime)s\n%(message)s\n'
        if level_no == INFO_LEVELS.epoch:
            if logger.level > INFO_LEVELS.metrics:
                return '%(message)s ...\r'
        return '%(message)s\n'


def set_default_logging() -> None:
    logger.handlers.clear()
    formatter = InfoFormatter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.terminator = ''
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(INFO_LEVELS.checkpoint)
    logger.propagate = False


def propagate_to_main_logger() -> None:
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


logger = logging.getLogger('dry_torch')
INFO_LEVELS = InfoLevels(17, 21, 23, 25, 27, 28)
set_default_logging()
