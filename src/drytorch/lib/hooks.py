"""Hooks and callbacks for a Trainer."""

from __future__ import annotations

import abc

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Generic, final

from typing_extensions import TypeVar, override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import monitoring, schedulers
from drytorch.lib.schedulers import RestartScheduler


__all__ = [
    'ChangeScheduler',
    'EarlyStopping',
    'EvaluationHook',
    'HookChain',
    'Pruning',
    'ReduceLROnPlateau',
    'RestartScheduleOnPlateau',
    'SavingHook',
    'TerminateTraining',
    'TrainerHook',
]

_C_co = TypeVar(
    '_C_co',
    bound=monitoring.Condition,
    covariant=True,
    default=monitoring.Always,
)


class HookChain:
    """Execute a fixed chain of hooks in order.

    Attributes:
        hooks: the hooks in order of execution.
    """

    hooks: Final[tuple[p.HookProtocol, ...]]

    def __init__(self, hooks: Sequence[p.HookProtocol] = ()) -> None:
        """Initialize.

        Args:
            hooks: the hooks to execute, in order.
        """
        self.hooks = tuple(hooks)
        return

    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind every hook of the chain to the trainer that calls them.

        Args:
            trainer: the trainer calling the hooks.
        """
        for hook in self.hooks:
            hook.bind_trainer(trainer)

        return

    def execute(self) -> None:
        """Execute the hooks in order."""
        for hook in self.hooks:
            hook()

        return


class TrainerHook(abc.ABC, Generic[_C_co]):
    """Hook bound to a trainer, running when its condition holds.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    when: Final[_C_co]
    _last_run_epoch: int
    _trainer: p.TrainerProtocol[Any, Any, Any] | None

    def __init__(self, when: _C_co = monitoring.ALWAYS) -> None:
        """Initialize.

        Args:
            when: decides whether the hook runs. Default runs every time.
        """
        self.when = when
        self._last_run_epoch = 0
        self._trainer = None
        return

    @final
    def __call__(self) -> None:
        """Observe what the condition follows, then run when it holds."""
        self.when.observe(self)
        if self.when(self):
            self.run()
            self._last_run_epoch = self.trainer.model.epoch

        return

    @property
    def last_run_epoch(self) -> int:
        """The epoch when the hook last ran, or 0 if never run."""
        return self._last_run_epoch

    @property
    def trainer(self) -> p.TrainerProtocol[Any, Any, Any]:
        """The trainer the hook is bound to.

        Raises:
            HookNotBoundError: if the trainer has not bound the hook yet.
        """
        if self._trainer is None:
            raise exceptions.HookNotBoundError(self._get_name())

        return self._trainer

    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind the hook and its condition to the trainer that calls it.

        Args:
            trainer: the trainer calling the hook.

        Raises:
            HookAlreadyBoundError: if another trainer already bound the hook.
        """
        if self._trainer is not None and self._trainer is not trainer:
            raise exceptions.HookAlreadyBoundError(self._get_name())

        self._trainer = trainer
        self.when.bind_trainer(trainer)
        return

    @abc.abstractmethod
    def run(self) -> None:
        """Execute what the hook does."""

    def _get_name(self) -> str:
        return self.__class__.__name__

    def _get_parameters(self) -> Mapping[str, Any]:
        return {}


class EvaluationHook(TrainerHook[_C_co]):
    """Run an evaluation during the training.

    Attributes:
        evaluation: the evaluation to run, for example a validation.
        when: decides whether the hook runs when the trainer calls it.
    """

    evaluation: Final[p.SourceProtocol]

    def __init__(
        self,
        evaluation: p.SourceProtocol,
        when: _C_co = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            evaluation: the evaluation to run.
            when: decides whether the hook runs. Default runs every time.
        """
        super().__init__(when)
        self.evaluation = evaluation
        return

    @override
    def run(self) -> None:
        """Run the evaluation."""
        self.evaluation()
        return


class SavingHook(TrainerHook[_C_co]):
    """Save the checkpoint of the model of its trainer.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    @override
    def run(self) -> None:
        """Save the checkpoint."""
        self.trainer.save_checkpoint()
        return


class TerminateTraining(TrainerHook[_C_co]):
    """Terminate the training of its trainer.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    _reason: ClassVar[str] = 'a hook terminated the training'

    def __init__(self, when: _C_co = monitoring.ALWAYS) -> None:
        """Initialize.

        Args:
            when: decides whether the hook runs. Default runs every time.
        """
        super().__init__(when)
        return

    @override
    def run(self) -> None:
        """Terminate the training."""
        self.trainer.terminate_training(self._reason)
        return


class EarlyStopping(
    TerminateTraining[
        monitoring.AndCondition[monitoring.After, monitoring.Plateau]
    ]
):
    """Terminate the training when a metric stops improving.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    _reason: ClassVar[str] = 'the metric stopped improving'

    def __init__(
        self,
        patience: int = 10,
        start_from_epoch: int = 2,
        min_delta: float = 1e-8,
        watch: monitoring.Watch | None = None,
    ) -> None:
        """Initialize.

        Args:
            patience: the number of records to wait for an improvement,
                as many as the runs of the source rather than epochs.
            start_from_epoch: the first epoch when the training may stop.
            min_delta: the minimum change counting as an improvement.
            watch: what to follow. Default follows the loss of the trainer.
        """
        plateau = monitoring.Plateau(
            patience=patience, min_delta=min_delta, watch=watch
        )
        super().__init__(when=monitoring.After(start_from_epoch) & plateau)
        return

    @property
    def plateau(self) -> monitoring.Plateau:
        """The condition following the metric."""
        return self.when.right

    @override
    def run(self) -> None:
        """Terminate the training, reporting the best value so far."""
        monitor = self.plateau.monitor
        best = self.plateau.best_value
        msg = f'{self._reason}: best {monitor.metric_name}={best}'
        self.trainer.terminate_training(msg)
        return


class Pruning(TerminateTraining[monitoring.WorseThan]):
    """Terminate the training when a metric misses a threshold.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    _reason: ClassVar[str] = 'the metric missed a threshold'

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        watch: monitoring.Watch | None = None,
    ) -> None:
        """Initialize.

        Args:
            thresholds: the value to reach at each given epoch.
            watch: what to follow. Default follows the loss of the trainer.
        """
        super().__init__(when=monitoring.WorseThan(thresholds, watch=watch))
        return

    @override
    def run(self) -> None:
        """Terminate the training, reporting the threshold it missed."""
        epoch = self.trainer.model.epoch
        threshold = self.when.thresholds[epoch]
        metric_name = self.when.monitor.metric_name
        msg = f'{self._reason}: {metric_name} did not reach {threshold}'
        self.trainer.terminate_training(msg)
        return


class ChangeScheduler(TrainerHook[_C_co], abc.ABC):
    """Change the learning rate schedule of its trainer.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    @abc.abstractmethod
    def get_scheduler(
        self, epoch: int, scheduler: p.SchedulerProtocol
    ) -> p.SchedulerProtocol:
        """Modify the scheduler of the trainer.

        Args:
            epoch: the current epoch.
            scheduler: the scheduler to modify.

        Returns:
            The modified scheduler.
        """

    @override
    def run(self) -> None:
        """Give the trainer the modified scheduler."""
        scheduler = self.get_scheduler(
            self.trainer.model.epoch, self.trainer.learning_schema.scheduler
        )
        self.trainer.update_learning_rate(base_lr=None, scheduler=scheduler)
        return


class ReduceLROnPlateau(
    ChangeScheduler[
        monitoring.AndCondition[monitoring.Plateau, monitoring.Cooldown]
    ]
):
    """Reduce the learning rate when a metric stops improving.

    Attributes:
        factor: the factor scaling the learning rate down.
        when: decides whether the hook runs when the trainer calls it.
    """

    factor: Final[float]
    _n_reductions: int

    def __init__(
        self,
        factor: float = 0.1,
        patience: int = 0,
        cooldown: int = 0,
        min_delta: float = 1e-8,
        watch: monitoring.Watch | None = None,
    ) -> None:
        """Initialize.

        Args:
            factor: the factor scaling the learning rate down.
            patience: the number of records to wait for an improvement,
                as many as the runs of the source rather than epochs.
            cooldown: the epochs to skip after the last reduction.
            min_delta: the minimum change counting as an improvement.
            watch: what to follow. Default follows the loss of the trainer.
        """
        super().__init__(when=_on_plateau(patience, cooldown, min_delta, watch))
        self.factor = factor
        self._n_reductions = 0
        return

    @property
    def plateau(self) -> monitoring.Plateau:
        """The condition following the metric."""
        return self.when.left

    @override
    def get_scheduler(
        self, epoch: int, scheduler: p.SchedulerProtocol
    ) -> p.SchedulerProtocol:
        """Scale down the learning rate of the scheduler."""
        _not_used = epoch
        self._n_reductions += 1
        return schedulers.RescaleScheduler(
            scheduler, self.factor**self._n_reductions
        )


class RestartScheduleOnPlateau(
    ChangeScheduler[
        monitoring.AndCondition[monitoring.Plateau, monitoring.Cooldown]
    ]
):
    """Restart the schedule when a metric stops improving.

    Attributes:
        when: decides whether the hook runs when the trainer calls it.
    """

    def __init__(
        self,
        patience: int = 0,
        cooldown: int = 0,
        min_delta: float = 1e-8,
        watch: monitoring.Watch | None = None,
    ) -> None:
        """Initialize.

        Args:
            patience: the number of records to wait for an improvement,
                as many as the runs of the source rather than epochs.
            cooldown: the epochs to skip after the last restart.
            min_delta: the minimum change counting as an improvement.
            watch: what to follow. Default follows the loss of the trainer.
        """
        super().__init__(when=_on_plateau(patience, cooldown, min_delta, watch))
        return

    @property
    def plateau(self) -> monitoring.Plateau:
        """The condition following the metric."""
        return self.when.left

    @override
    def get_scheduler(
        self, epoch: int, scheduler: p.SchedulerProtocol
    ) -> p.SchedulerProtocol:
        """Consider the training until now a warm-up and restart it."""
        return RestartScheduler(scheduler, epoch)


def _on_plateau(
    patience: int,
    cooldown: int,
    min_delta: float,
    watch: monitoring.Watch | None,
) -> monitoring.AndCondition[monitoring.Plateau, monitoring.Cooldown]:
    plateau = monitoring.Plateau(
        patience=patience, min_delta=min_delta, watch=watch
    )
    return plateau & monitoring.Cooldown(cooldown)
