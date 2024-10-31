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
from src.dry_torch import tracking

logger = logging.getLogger('dry_torch')


class InfoLevels(NamedTuple):
    """NamedTuple that defines different levels of information for logging."""
    internal: int
    metrics: int
    epoch: int
    checkpoint: int
    training: int
    experiment: int


INFO_LEVELS = InfoLevels(internal=19,
                         metrics=21,
                         epoch=23,
                         checkpoint=25,
                         training=27,
                         experiment=28)

for name, level in INFO_LEVELS._asdict().items():
    logging.addLevelName(level, name.center(10))


class InfoFormatter(logging.Formatter):
    """
    Custom formatter to format log messages based on their level.

    It adds a timestamp to logs of level training. When the logger level is
    set to a value higher than metrics, it overwrites the current epoch's log.
    """

    default_msec_format = ''

    @override
    def format(self, record: logging.LogRecord) -> str:
        # if record.msg[-1:] != '\r':
        #     record.msg += '\n'
        self._style._fmt = self._info_fmt(record.levelno)
        return super().format(record)

    @staticmethod
    def _info_fmt(level_no: Optional[int] = None) -> str:
        if level_no == INFO_LEVELS.training:
            return '[%(asctime)s] - %(message)s\n'
        if level_no == INFO_LEVELS.epoch:
            return '%(message)s'
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
    logger.setLevel(INFO_LEVELS.metrics)
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


enable_default_handler()


class BuiltinLogger(tracking.Logger):

    def __init__(self) -> None:
        super().__init__()

    @functools.singledispatchmethod
    def notify(self, event: events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: events.StartTraining) -> None:
        logger.log(INFO_LEVELS.training,
                   'Training %(model_name)s started.',
                   {'model_name': event.model_name})
        return

    @notify.register
    def _(self, _event: events.EndTraining) -> None:
        logger.log(INFO_LEVELS.training, 'Training ended.')
        return

    @notify.register
    def _(self, event: events.StartEpoch) -> None:
        final_epoch = str(event.final_epoch)
        if final_epoch is not None:
            fix_len = len(final_epoch)
            final_epoch = '/' + final_epoch
        else:
            fix_len = 1

        epoch_msg = f'====> Epoch %(epoch){fix_len}d%(final_epoch)s: \r'

        logger.log(INFO_LEVELS.epoch,
                   epoch_msg,
                   {'epoch': event.epoch, 'final_epoch': final_epoch})
        logger.log(INFO_LEVELS.metrics, '')
        return

    # @notify.register
    # def _(self, _event: events.EndEpoch) -> None:
    #     logger.log(INFO_LEVELS.metrics, '')
    #     return

    @notify.register
    def _(self, event: events.SaveCheckpoint) -> None:
        logger.log(INFO_LEVELS.checkpoint,
                   f'%(definition)s saved in: %(location)s.',
                   {'definition': event.definition.capitalize(),
                    'location': event.location}
                   )
        return

    @notify.register
    def _(self, event: events.LoadCheckpoint) -> None:
        logger.log(INFO_LEVELS.checkpoint,
                   f'Loaded %(definition)s at epoch %(epoch)d.',
                   {'definition': event.definition.capitalize(),
                    'epoch': event.epoch}
                   )
        return

    @notify.register
    def _(self, event: events.MetricsCreation) -> None:
        log_msg_list: list[str] = ['%(desc)-24s']
        desc = event.source.rjust(15) + ': '
        log_args: dict[str, str | float] = {'desc': desc}
        for metric, value in event.metrics.items():
            log_msg_list.append(f'%({metric})16s=%({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        return

    @notify.register
    def _(self, event: events.StartTest) -> None:
        logger.log(INFO_LEVELS.experiment,
                   'Testing %(model_name)s started.',
                   {'model_name': event.model_name})

    @notify.register
    def _(self, event: events.TerminatedTraining) -> None:
        explicit_cause = '' if event.cause is None else 'by ' + event.cause
        logger.log(INFO_LEVELS.training,
                   'Training terminated at epoch %(epoch)d %(explicit_cause)s.',
                   {'epoch': event.epoch, 'explicit_cause': explicit_cause})

    @notify.register
    def _(self, event: events.StartExperiment) -> None:
        logger.log(INFO_LEVELS.experiment,
                   'Running experiment: %(name)s.',
                   {'name': event.exp_name})

    @notify.register
    def _(self, event: events.StartExperiment) -> None:
        logger.log(INFO_LEVELS.experiment,
                   'Running experiment: %(name)s.',
                   {'name': event.exp_name})

    @notify.register
    def _(self, event: events.ModelDidNotConverge) -> None:
        logger.error(event.exception)
