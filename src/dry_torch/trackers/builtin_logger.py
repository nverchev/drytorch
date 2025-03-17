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
from typing import NamedTuple
from typing_extensions import override, Literal

from dry_torch import log_events
from dry_torch import tracking

logger = logging.getLogger('dry_torch')


class InfoLevels(NamedTuple):
    """NamedTuple that defines different levels of information for logging."""
    internal: int
    metrics: int
    epoch: int
    param_update: int
    checkpoint: int
    experiment: int
    training: int
    test: int


INFO_LEVELS = InfoLevels(internal=19,
                         metrics=21,
                         epoch=23,
                         param_update=25,
                         checkpoint=26,
                         experiment=27,
                         training=28,
                         test=29,
                         )

for name, level in INFO_LEVELS._asdict().items():
    logging.addLevelName(level, name.center(10))


class DryTorchFormatter(logging.Formatter):
    """Default formatter for the dry_torch logger."""
    default_msec_format = ''

    @override
    def format(self, record: logging.LogRecord) -> str:
        self._style._fmt = self._info_fmt(record.levelno)
        return super().format(record)

    @staticmethod
    def _info_fmt(level_no: int) -> str:
        if level_no >= INFO_LEVELS.experiment:
            return '[%(asctime)s] - %(message)s\n'
        return '%(message)s\n'


class ProgressFormatter(DryTorchFormatter):
    """Formatter that dynamically overwrites metrics and epoch logs."""

    @staticmethod
    def _info_fmt(level_no: int) -> str:
        if level_no == INFO_LEVELS.metrics:
            return '[%(asctime)s] - %(message)s\r'
        if level_no == INFO_LEVELS.epoch:
            return '[%(asctime)s] - %(message)s'
        else:
            return DryTorchFormatter._info_fmt(level_no)


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
    return


def disable_default_handler() -> None:
    """This function disable the handler and filter of the local logger."""
    logger.setLevel(logging.NOTSET)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return


def enable_default_handler() -> None:
    """This function sets up the default logging configuration."""
    global logger
    logger.handlers.clear()

    formatter = DryTorchFormatter()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.terminator = ''
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(INFO_LEVELS.metrics)
    logger.propagate = False
    return


def disable_propagation() -> None:
    """This function reverts the changes made by enable_propagation."""
    global logger

    logger.propagate = False
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        for log_filter in handler.filters:
            if isinstance(log_filter, DryTorchFilter):
                handler.removeFilter(log_filter)
                break
    return


def set_formatter(style: Literal['progress', 'dry_torch']) -> None:
    """Set the formatter for the stdout handler of the dry_torch logger."""
    global logger

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if hasattr(handler.stream, 'name'):
                if handler.stream.name == '<stdout>':
                    if style == 'progress':
                        handler.formatter = ProgressFormatter()
                    elif style == 'dry_torch':
                        handler.formatter = DryTorchFormatter()
                    else:
                        raise ValueError('Invalid formatter style.')
    return


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
    return


enable_default_handler()


class BuiltinLogger(tracking.Tracker):

    def __init__(self) -> None:
        super().__init__()

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartTraining) -> None:
        logger.log(INFO_LEVELS.training,
                   'Training %(model_name)s started.',
                   {'model_name': format(event.model_name, 's')})
        return

    @notify.register
    def _(self, _event: log_events.EndTraining) -> None:
        logger.log(INFO_LEVELS.training, 'Training ended.')
        return

    @notify.register
    def _(self, event: log_events.StartEpoch) -> None:
        final_epoch = event.final_epoch
        if final_epoch is not None:
            final_epoch_str = str(final_epoch)
            fix_len = len(final_epoch_str)
            final_epoch_str = '/' + final_epoch_str
        else:
            fix_len = 1
            final_epoch_str = ''
        epoch_msg = f'====> Epoch %(epoch){fix_len}d%(final_epoch_str)s:'

        logger.log(INFO_LEVELS.epoch,
                   epoch_msg,
                   {'epoch': event.epoch, 'final_epoch_str': final_epoch_str})
        return

    @notify.register
    def _(self, event: log_events.SaveModel) -> None:
        logger.log(INFO_LEVELS.checkpoint,
                   f'Saving %(name)s %(definition)s in: %(location)s.',
                   {'name': format(event.model_name, 's'),
                    'definition': event.definition.capitalize(),
                    'location': event.location}
                   )
        return

    @notify.register
    def _(self, event: log_events.LoadModel) -> None:
        logger.log(INFO_LEVELS.checkpoint,
                   f'Loading %(name)s %(definition)s at epoch %(epoch)d.',
                   {'name': format(event.model_name, 's'),
                    'definition': event.definition.capitalize(),
                    'epoch': event.epoch}
                   )
        return

    @notify.register
    def _(self, event: log_events.EpochMetrics) -> None:
        log_msg_list: list[str] = ['%(desc)s']
        desc = format(event.source, 's').rjust(15) + ': '
        log_args: dict[str, str | float] = {'desc': desc}
        for metric, value in event.metrics.items():
            log_msg_list.append(f'%({metric})s=%({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        return

    @notify.register
    def _(self, event: log_events.Test) -> None:
        logger.log(INFO_LEVELS.test,
                   'Testing %(model_name)s started.',
                   {'model_name': format(event.model_name, 's')})
        return

    @notify.register
    def _(self, event: log_events.TerminatedTraining) -> None:

        msg = '. '.join([
            '%(source)s: Training %(model_name)s terminated at epoch %(epoch)d',
            'Reason: %(reason)s',
        ])
        log_args = {'source': format(event.source, 's'),
                    'model_name': format(event.model_name, 's'),
                    'reason': event.reason,
                    'epoch': event.epoch}
        logger.log(INFO_LEVELS.training, msg, log_args)
        return

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        logger.log(INFO_LEVELS.experiment,
                   'Running experiment: %(name)s.',
                   {'name': format(event.exp_name, 's')})
        return

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        logger.log(INFO_LEVELS.internal,
                   'Experiment: %(name)s stopped.',
                   {'name': format(event.exp_name, 's')})
        return

    @notify.register
    def _(self, event: log_events.UpdateLearningRate) -> None:

        message_parts = [
            '%(source)s: Updated %(model_name)s optimizer at epoch %(epoch)d',
        ]
        if event.base_lr is not None:
            message_parts.append('New learning rate: %(learning_rate)s')
        if event.scheduler_name is not None:
            message_parts.append('New scheduler: %(scheduler_name)s')

        msg = '.\n'.join(message_parts) + '.'

        log_args = {'source': event.source,
                    'model_name': format(event.model_name, 's'),
                    'epoch': event.epoch,
                    'learning_rate': event.base_lr,
                    'scheduler_name': event.scheduler_name}
        logger.log(INFO_LEVELS.param_update, msg, log_args)
        return
