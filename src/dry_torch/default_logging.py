import sys
import logging
from typing import NamedTuple, Optional, Final


class InfoLevels(NamedTuple):
    checkpoint: int
    experiment: int
    metrics: int
    epoch: int
    training: int


class InfoFilter(logging.Filter):
    def __init__(self, level_no: Optional[int] = None, name='') -> None:
        super().__init__(name)
        self.level_no: Final = level_no

    def filter(self, record) -> bool:
        if self.level_no is None:
            return True
        return record.levelno == self.level_no


class InfoFormatter(logging.Formatter):

    def __init__(self,
                 fmt: Optional[str] = None,
                 datefmt='%H:%M:%S %d-%m-%Y',
                 *args,
                 level_no: Optional[int] = None,
                 defaults=None) -> None:
        self.default_fmt: Final = fmt
        fmt = self._info_fmt(level_no)
        super().__init__(fmt, datefmt, *args, defaults=defaults)

    def _info_fmt(self, level_no: Optional[int] = None) -> Optional[str]:
        if level_no == INFO_LEVELS.training:
            return '%(asctime)s\n%(message)s'
        return self.default_fmt


def set_default_logging() -> None:
    logger.handlers.clear()
    for level_name, level_no in INFO_LEVELS._asdict().items():
        formatter = InfoFormatter(level_no=level_no)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(InfoFilter(level_no, level_name))
        logger.addHandler(stdout_handler)
    logger.setLevel(INFO_LEVELS.checkpoint)
    logger.propagate = False


def propagate_to_main_logger() -> None:
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


logger = logging.getLogger('dry_torch')
INFO_LEVELS = InfoLevels(17, 21, 23, 25, 27)
set_default_logging()
