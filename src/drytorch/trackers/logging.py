"""Module containing custom logging configurations for the 'drytorch' logger.

It defines and implements a formatter that formats log messages according to
the levels defined in the INFO_LEVELS variable. By default, it prints to
stream and does not propagate to the main root.

Attributes:
    INFO_LEVELS: InfoLevels object for global settings.
"""

from __future__ import annotations

import functools
import logging
import sys

from typing import TYPE_CHECKING, Literal, NamedTuple

from typing_extensions import override

from drytorch.core import log_events, track


if TYPE_CHECKING:
    from _typeshed import SupportsWrite

logger: logging.Logger = logging.getLogger('drytorch')


class InfoLevels(NamedTuple):
    """NamedTuple that defines different levels of information for logging.

    Attributes:
        internal: level for internal logging messages.
        metrics: level for metric reporting.
        epoch: level for epoch-related messages.
        model_state: level for model state changes.
        experiment: level for experiment-related messages.
        training: level for training-related messages.
        test: level for test-related messages.
    """

    internal: int
    metrics: int
    epoch: int
    model_state: int
    experiment: int
    training: int
    test: int


class BuiltinLogger(track.Tracker):
    """Tracker that streams logging messages through the built-in logger."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartTrainingEvent) -> None:
        logger.log(
            INFO_LEVELS.training,
            'Training %(model_name)s started.',
            {'model_name': event.model_name},
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndTrainingEvent) -> None:
        logger.log(INFO_LEVELS.training, 'Training ended.')
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartEpochEvent) -> None:
        final_epoch = event.end_epoch
        if final_epoch is not None:
            final_epoch_str = str(final_epoch)
            fix_len = len(final_epoch_str)
            final_epoch_str = '/' + final_epoch_str
        else:
            fix_len = 1
            final_epoch_str = ''

        epoch_msg = f'====> Epoch %(epoch){fix_len}d%(final_epoch)s:'
        logger.log(
            INFO_LEVELS.epoch,
            epoch_msg,
            {'epoch': event.epoch, 'final_epoch': final_epoch_str},
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndEpochEvent) -> None:
        logger.log(INFO_LEVELS.internal, 'Epoch completed.')
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.SaveModelEvent) -> None:
        logger.log(
            INFO_LEVELS.model_state,
            'Saving %(name)s %(definition)s in: %(location)s.',
            {
                'name': event.model_name,
                'definition': event.definition,
                'location': event.location,
            },
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.LoadModelEvent) -> None:
        logger.log(
            INFO_LEVELS.model_state,
            'Loading %(name)s %(definition)s at epoch %(epoch)d.',
            {
                'name': event.model_name,
                'definition': event.definition,
                'epoch': event.epoch,
            },
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        log_msg_list: list[str] = ['%(desc)s']
        desc = _to_desc(event.source_name)
        log_args: dict[str, str | float] = {'desc': desc}
        for metric, value in event.metrics.items():
            log_msg_list.append(f'%({metric})s=%({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})

        logger.log(INFO_LEVELS.metrics, '\t'.join(log_msg_list), log_args)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartTestEvent) -> None:
        logger.log(
            INFO_LEVELS.test,
            'Testing %(model_name)s started.',
            {'model_name': event.model_name},
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndTestEvent) -> None:
        logger.log(INFO_LEVELS.internal, 'Test executed without errors.')
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.TerminatedTrainingEvent) -> None:
        msg = '. '.join(
            [
                'Training %(model_name)s terminated at epoch %(epoch)d',
                'Reason: %(reason)s',
            ]
        )
        log_args = {
            'model_name': event.model_name,
            'reason': event.reason,
            'epoch': event.epoch,
        }
        logger.log(INFO_LEVELS.training, msg, log_args)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        msg = 'Running experiment: %(name)s'
        args = {'name': event.exp_name}
        logger.log(INFO_LEVELS.experiment, msg + '.', args)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        logger.log(
            INFO_LEVELS.internal,
            'Experiment: %(name)s stopped.',
            {'name': event.exp_name},
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.LearningRateEvent) -> None:
        message_parts = [
            'Updated %(model_name)s optimizer at epoch %(epoch)d',
        ]
        if event.base_lr is not None:
            message_parts.append('New learning rate: %(learning_rate)s')

        if event.scheduler_name is not None:
            message_parts.append('New scheduler: %(scheduler_name)s')

        msg = '. '.join(message_parts) + '.'

        log_args = {
            'model_name': event.model_name,
            'epoch': event.epoch,
            'learning_rate': event.base_lr,
            'scheduler_name': event.scheduler_name,
        }
        logger.log(INFO_LEVELS.model_state, msg, log_args)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ModelRegistrationEvent) -> None:
        msg = 'Model %(model_name)s has been registered.'
        logger.log(INFO_LEVELS.internal, msg, {'model_name': event.model_name})
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ActorRegistrationEvent) -> None:
        msg = 'Source %(source_name)s %(model_name)s has been registered.'
        args = {
            'model_name': event.model_name,
            'source_name': event.actor_name,
        }
        logger.log(INFO_LEVELS.internal, msg, args)
        return super().notify(event)


class DryTorchFilter(logging.Filter):
    """Filter that excludes logs from 'drytorch'."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return 'drytorch' not in record.name


class DryTorchFormatter(logging.Formatter):
    """Default formatter for the drytorch logger."""

    default_msec_format = ''

    @override
    def format(self, record: logging.LogRecord) -> str:
        self._style._fmt = self._info_fmt(record.levelno)
        return super().format(record)

    @staticmethod
    def _info_fmt(level_no: int) -> str:
        if level_no >= INFO_LEVELS.experiment:
            return '\r[%(asctime)s] - %(message)s\n'

        return '\r%(message)s\n'


class ProgressFormatter(DryTorchFormatter):
    """Formatter that dynamically overwrites metrics and epoch logs."""

    @staticmethod
    def _info_fmt(level_no: int) -> str:
        if level_no == INFO_LEVELS.epoch:
            return '%(message)s ...\r'
        elif level_no == INFO_LEVELS.model_state:
            return '%(message)s\r'
        else:
            return DryTorchFormatter._info_fmt(level_no)


def disable_default_handler() -> None:
    """Disable the handler and filter of the local logger."""
    logger.setLevel(logging.NOTSET)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return


def enable_default_handler(stream: SupportsWrite[str] = sys.stderr) -> None:
    """Set up the default logging configuration."""
    logger.handlers.clear()
    formatter = DryTorchFormatter()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.terminator = ''
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = False
    return


def disable_propagation() -> None:
    """Revert the changes made by enable_propagation."""
    logger.propagate = False
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        for log_filter in handler.filters:
            if isinstance(log_filter, DryTorchFilter):
                handler.removeFilter(log_filter)
                break

    return


def enable_propagation(deduplicate_stream: bool = True) -> None:
    """Propagate to the root logger.

    Args:
        deduplicate_stream: whether to remove local messages from the stream.
    """
    logger.propagate = True
    if deduplicate_stream:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream in (
                    h.stream
                    for h in logger.handlers
                    if isinstance(h, logging.StreamHandler)
                ):
                    handler.addFilter(DryTorchFilter())

    return


def set_formatter(style: Literal['drytorch', 'progress']) -> None:
    """Set the formatter for the stream handler of the drytorch logger."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if style == 'progress':
                handler.formatter = ProgressFormatter()
            elif style == 'drytorch':
                handler.formatter = DryTorchFormatter()

            else:
                raise ValueError('Invalid formatter style.')

    return


def get_verbosity() -> int:
    """Get the verbosity level of the 'drytorch' logger."""
    return logger.level


def set_verbosity(level_no: int):
    """Set the verbosity level of the 'drytorch' logger."""
    logger.setLevel(level_no)
    return


def _to_desc(text: str) -> str:
    return text.rjust(15) + ': '


INFO_LEVELS = InfoLevels(
    internal=19,
    metrics=21,
    epoch=23,
    model_state=25,
    experiment=27,
    training=28,
    test=29,
)
for name, level in INFO_LEVELS._asdict().items():
    logging.addLevelName(level, name.center(10))

enable_default_handler()
