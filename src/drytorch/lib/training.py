"""Classes for training a model."""

from __future__ import annotations

import warnings

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Final, Generic, TypedDict, TypeVar, cast

import torch

from torch import amp
from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.core import protocols as p
from drytorch.lib import evaluations, hooks, runners


__all__ = [
    'ModelOptimizer',
    'Trainer',
    'TrainerBuilder',
]


class _OptParams(TypedDict):
    params: Iterator[torch.nn.Parameter]
    lr: float


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)


class ModelOptimizer:
    """Bundle the module and its optimizer.

    It supports different learning rates to separate parameters' groups.
    """

    _gradient_op: p.GradientOpProtocol
    _lr: float | Mapping[str, float]
    _model: Final[p.ModelProtocol]
    _module: Final[torch.nn.Module]
    _optimizer: torch.optim.Optimizer
    _params_lr: list[_OptParams]
    _scaler: amp.grad_scaler.GradScaler
    _scheduler: p.SchedulerProtocol

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        learning_schema: p.LearningProtocol,
    ) -> None:
        """Initialize.

        Args:
            model: the model to be optimized.
            learning_schema: the learning scheme for the optimizer.
        """
        self._gradient_op = learning_schema.gradient_op
        self._model = model
        self._module = model.module
        self._lr = {}
        self._params_lr = []
        self._scheduler = learning_schema.scheduler
        self.base_lr = learning_schema.base_lr
        self._optimizer = learning_schema.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_opt_params()),
            **learning_schema.optimizer_defaults,
        )
        self._model.bind_optimizer(self._optimizer)
        self._scaler = amp.grad_scaler.GradScaler(
            model.device.type,
            enabled=model.mixed_precision,
        )
        return

    @property
    def base_lr(self) -> float | Mapping[str, float]:
        """Learning rate(s) for the module parameters.

        Raises:
            MissingParamError: if parameters are missing from the dictionary.
        """
        return self._lr

    @base_lr.setter
    def base_lr(self, lr: float | Mapping[str, float]) -> None:
        self._lr = dict(lr) if isinstance(lr, Mapping) else lr
        if isinstance(lr, float | int):
            self._params_lr = [
                {'params': self._module.parameters(), 'lr': lr},
            ]
        else:
            self._params_lr = [
                {'params': getattr(self._module, k).parameters(), 'lr': v}
                for k, v in lr.items()
            ]
            if not self._params_lr_contains_all_params():
                module_names: list[str] = [
                    named_elem[0] for named_elem in self._module.named_modules()
                ]
                raise exceptions.MissingParamError(module_names, list(lr))

        return

    @override
    def __repr__(self) -> str:
        desc = '{}(module={}, optimizer={})'
        return desc.format(
            self.__class__.__name__,
            self._model.name,
            self._optimizer.__class__.__name__,
        )

    def get_opt_params(self) -> list[_OptParams]:
        """Actual learning rates for each parameter updated according.

        Returns:
            The parameter groups with their scheduled learning rates.
        """
        return [
            _OptParams(params=g['params'], lr=self.get_scheduled_lr(g['lr']))
            for g in self._params_lr
        ]

    def get_scheduled_lr(self, lr: float) -> float:
        """Update the base learning rate according to the scheduler.

        Args:
            lr: base learning rate.

        Returns:
            The scheduled learning rate.
        """
        return self._scheduler(lr, self._model.epoch)

    def update_learning_rate(
        self,
        base_lr: float | Mapping[str, float] | None = None,
        scheduler: p.SchedulerProtocol | None = None,
    ) -> None:
        """Recalculate the learning rates for the current epoch.

        It updates the learning rates for each parameter's group in the
        optimizer based on input learning rate(s) and scheduler.

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """
        if scheduler is not None:
            self._scheduler = scheduler

        if base_lr is not None:
            self.base_lr = base_lr

        for g, up_g in zip(
            self._optimizer.param_groups, self.get_opt_params(), strict=False
        ):
            g['lr'] = up_g['lr']

        return

    def optimize(self, loss_value: torch.Tensor):
        """Optimize the model backpropagating the loss value.

        Args:
            loss_value: the output tensor for the loss.
        """
        self._optimizer.zero_grad()
        self._scaler.scale(loss_value).backward()
        self._scaler.unscale_(self._optimizer)
        self._gradient_op(self._model.module.parameters())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        return

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(
            self._count_params(elem['params']) for elem in self._params_lr
        )
        total_params_model = self._count_params(self._module.parameters())
        return total_params_lr == total_params_model

    @staticmethod
    def _count_params(params: Iterator[Any]) -> int:
        """Count the number of parameters."""
        return sum(1 for _ in params)


class Trainer(
    runners.ModelRunnerWithLogs[
        Input, Target, Output, p.LossProtocol[Output, Target]
    ],
):
    """Implement the standard Pytorch training loop.

    Attributes:
        learning_schema: optimizer settings and scheduling given at creation.
        loader: provides inputs and targets in batches.
        model: the model to train.
        objective: determines the optimization's criterion.
        post_epoch_hooks: hooks executed after each epoch, validation first.
        pre_epoch_hooks: hooks executed before each epoch.
        validation: the validation run during training, if any.
    """

    learning_schema: Final[p.LearningProtocol]
    post_epoch_hooks: Final[hooks.HookChain[Trainer[Input, Target, Output]]]
    pre_epoch_hooks: Final[hooks.HookChain[Trainer[Input, Target, Output]]]
    validation: Final[evaluations.Validation[Input, Target, Output] | None]
    _model_optimizer: Final[ModelOptimizer]
    _terminated: bool

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        learning_schema: p.LearningProtocol,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        loss: p.LossProtocol[Output, Target],
        post_epoch_hooks: Sequence[
            Callable[[Trainer[Input, Target, Output]], None]
        ] = (),
        pre_epoch_hooks: Sequence[
            Callable[[Trainer[Input, Target, Output]], None]
        ] = (),
        val_loader: p.LoaderProtocol[tuple[Input, Target]] | None = None,
        val_name: str = '',
        validate_every: int = 1,
    ) -> None:
        """Initialize.

        Args:
            model: the model containing the weights to evaluate.
            name: the base name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            learning_schema: contains optimizer settings and scheduling.
            loader: provides inputs and targets in batches.
            loss: determines the optimization's criterion.
            post_epoch_hooks: hooks executed after each epoch and validation.
            pre_epoch_hooks: hooks executed before each epoch.
            val_loader: the loader for validation with the same metrics as
                for training. Default does not validate.
            val_name: the name for the validation.
            validate_every: the number of epochs between validations.

        Raises:
            ValueError: if validate_every is not strictly positive when
                val_loader is given.
        """
        super().__init__(model, loader=loader, objective=loss, name=name)
        self.validation = self._create_validation(val_loader, val_name)
        if self.validation is None:
            post_epoch_chain = hooks.HookChain(post_epoch_hooks)
        else:
            val_hook = hooks.evaluation_hook(self.validation, validate_every)
            post_epoch_chain = hooks.HookChain([val_hook, *post_epoch_hooks])

        self.post_epoch_hooks = post_epoch_chain
        self.pre_epoch_hooks = hooks.HookChain(pre_epoch_hooks)
        self.learning_schema = learning_schema
        self._model_optimizer = ModelOptimizer(model, learning_schema)
        self._terminated = False
        return

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        """Train the module for one epoch.

        Args:
            store_outputs: whether to store model outputs.

        Raises:
            ConvergenceError: if the loss is infinite or NaN, after terminating
                the training.
        """
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning(), stacklevel=1)
            return

        self.model.module.train()
        self.model.increment_epoch()
        self._model_optimizer.update_learning_rate()
        try:
            super().__call__()
        except exceptions.ConvergenceError as ce:
            self.terminate_training(reason=str(ce))
            raise ce

        self.model.post_epoch_update()
        return

    @property
    def terminated(self) -> bool:
        """If true, this trainer should not be used for training anymore."""
        return self._terminated

    def load_checkpoint(self, epoch: int = -1) -> None:
        """Load model and optimizer state from a checkpoint.

        Args:
            epoch: the epoch from which to load the checkpoint.
                Defaults to the last saved epoch.
        """
        self.model.load_state(epoch=epoch)
        return

    def save_checkpoint(self) -> None:
        """Save model and optimizer state in a checkpoint."""
        self.model.save_state()
        return

    def terminate_training(self, reason: str) -> None:
        """Prevent the trainer from continuing the training.

        Args:
            reason: why the training was terminated.

        Side Effects:
            Emits :class:`~drytorch.core.log_events.TerminatedTrainingEvent`.
        """
        self._terminated = True
        log_events.TerminatedTrainingEvent(
            source_name=self.name,
            model_name=self.model.name,
            epoch=self.model.epoch,
            reason=reason,
        )
        return

    def train(self, n_epochs: int) -> None:
        """Train the module for the specified number of epochs.

        Args:
            n_epochs: the number of epochs for which train the module.
        """
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning(), stacklevel=1)
            return
        final_epoch = self.model.epoch + n_epochs
        log_events.StartTrainingEvent(
            source_name=self.name,
            model_name=self.model.name,
            start_epoch=self.model.epoch,
            end_epoch=final_epoch,
        )
        for _ in range(n_epochs):
            self.pre_epoch_hooks.execute(self)
            if self.terminated:
                return

            log_events.StartEpochEvent(
                source_name=self.name,
                model_name=self.model.name,
                epoch=self.model.epoch + 1,
                end_epoch=final_epoch,
            )
            self()
            self.post_epoch_hooks.execute(self)
            log_events.EndEpochEvent(
                source_name=self.name,
                model_name=self.model.name,
                epoch=self.model.epoch,
            )
            if self.terminated:
                return

        log_events.EndTrainingEvent(self.name)
        return

    def train_until(self, epoch: int) -> None:
        """Train the module until the specified epoch.

        Args:
            epoch: the final epoch in the training.

        """
        remaining_epochs = epoch - self.model.epoch
        if remaining_epochs > 0:
            self.train(remaining_epochs)

        if remaining_epochs < 0:
            warnings.warn(
                exceptions.PastEpochWarning(epoch, self.model.epoch),
                stacklevel=1,
            )
        return

    def update_learning_rate(
        self,
        base_lr: float | Mapping[str, float] | None = None,
        scheduler: p.SchedulerProtocol | None = None,
    ) -> None:
        """Update the learning rate(s).

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.

        Raises:
            MissingParamError: if ``base_lr`` misses parameters of the module.
        """
        self._model_optimizer.update_learning_rate(base_lr, scheduler)
        scheduler_name = None if scheduler is None else repr(scheduler)
        log_events.LearningRateEvent(
            model_name=self.model.name,
            source_name=self.name,
            epoch=self.model.epoch,
            base_lr=base_lr,
            scheduler_name=scheduler_name,
        )
        return

    def _create_validation(
        self,
        val_loader: p.LoaderProtocol[tuple[Input, Target]] | None,
        val_name: str,
    ) -> evaluations.Validation[Input, Target, Output] | None:
        if val_loader is None:
            return None

        return evaluations.Validation(
            self.model, name=val_name, loader=val_loader, metric=self.objective
        )

    @override
    def _run_backward(self, outputs: Output, targets: Target) -> None:
        # replace super call
        loss_value = self.objective.forward(outputs, targets)
        try:
            if torch.isinf(loss_value) or torch.isnan(loss_value):
                raise exceptions.ConvergenceError(loss_value.item())

        except RuntimeError as re:
            if loss_value.numel() != 1:
                raise exceptions.LossNotScalarError(loss_value.shape) from re

            raise re

        self._model_optimizer.optimize(loss_value)
        self.model.post_batch_update()
        return

    @staticmethod
    def builder(
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        learning_schema: p.LearningProtocol,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        loss: p.LossProtocol[Output, Target],
    ) -> TrainerBuilder[Input, Target, Output]:
        """Collect the validation and the hooks before creating a trainer.

        Args:
            model: the model containing the weights to evaluate.
            name: the base name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            learning_schema: contains optimizer settings and scheduling.
            loader: provides inputs and targets in batches.
            loss: determines the optimization's criterion.

        Returns:
            A builder for a trainer with these settings.
        """
        return TrainerBuilder(
            model,
            name,
            learning_schema=learning_schema,
            loader=loader,
            loss=loss,
        )


class TrainerBuilder(Generic[Input, Target, Output]):
    """Collect the validation and the hooks before creating a trainer."""

    _learning_schema: Final[p.LearningProtocol]
    _loader: Final[p.LoaderProtocol[tuple[Input, Target]]]
    _loss: Final[p.LossProtocol[Output, Target]]
    _model: Final[p.ModelProtocol[Input, Output]]
    _name: Final[str]
    _post_epoch_hooks: Final[
        list[Callable[[Trainer[Input, Target, Output]], None]]
    ]
    _pre_epoch_hooks: Final[
        list[Callable[[Trainer[Input, Target, Output]], None]]
    ]
    _val_loader: p.LoaderProtocol[tuple[Input, Target]] | None
    _val_name: str
    _validate_every: int

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        learning_schema: p.LearningProtocol,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        loss: p.LossProtocol[Output, Target],
    ) -> None:
        """Initialize.

        Args:
            model: the model containing the weights to evaluate.
            name: the base name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            learning_schema: contains optimizer settings and scheduling.
            loader: provides inputs and targets in batches.
            loss: determines the optimization's criterion.
        """
        self._learning_schema = learning_schema
        self._loader = loader
        self._loss = loss
        self._model = model
        self._name = name
        self._post_epoch_hooks = []
        self._pre_epoch_hooks = []
        self._val_loader = None
        self._val_name = ''
        self._validate_every = 1
        return

    def add_post_epoch_hook(
        self, hook: Callable[[Trainer[Input, Target, Output]], None]
    ) -> None:
        """Add a hook to execute after each epoch and validation.

        Args:
            hook: the hook to execute after the ones already added.
        """
        self._post_epoch_hooks.append(hook)
        return

    def add_pre_epoch_hook(
        self, hook: Callable[[Trainer[Input, Target, Output]], None]
    ) -> None:
        """Add a hook to execute before each epoch.

        Args:
            hook: the hook to execute after the ones already added.
        """
        self._pre_epoch_hooks.append(hook)
        return

    def add_validation(
        self,
        val_loader: p.LoaderProtocol[tuple[Input, Target]],
        val_name: str = '',
        validate_every: int = 1,
    ) -> None:
        """Validate with the same metrics as for training.

        Calling it again replaces the previous validation.

        Args:
            val_loader: the loader for validation.
            val_name: the name for the validation.
            validate_every: the number of epochs between validations.
        """
        self._val_loader = val_loader
        self._val_name = val_name
        self._validate_every = validate_every
        return

    def build(self) -> Trainer[Input, Target, Output]:
        """Create the trainer with the collected validation and hooks.

        Returns:
            The trainer.
        """
        return Trainer(
            self._model,
            self._name,
            loader=self._loader,
            loss=self._loss,
            learning_schema=self._learning_schema,
            val_loader=self._val_loader,
            val_name=self._val_name,
            validate_every=self._validate_every,
            pre_epoch_hooks=self._pre_epoch_hooks,
            post_epoch_hooks=self._post_epoch_hooks,
        )
