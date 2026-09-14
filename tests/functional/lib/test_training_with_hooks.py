"""Functional tests for training hooks."""

from collections.abc import Callable, Generator
from typing import Any

import pytest

from drytorch import Trainer
from drytorch.core.experimenting import Run
from drytorch.lib import hooks, schedulers


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


@pytest.fixture
def create_trainer(
    linear_model, standard_learning_schema, square_loss_calc, identity_loader
) -> Callable[..., Trainer]:
    """Provide a function creating a trainer for the linear model."""

    def _create(**kwargs: Any) -> Trainer:
        return Trainer(
            linear_model,
            name='MyTrainer',
            loader=identity_loader,
            learning_schema=standard_learning_schema,
            loss=square_loss_calc,
            **kwargs,
        )

    return _create


def test_early_stopping(square_loss_calc, create_trainer) -> None:
    """Test early stopping when monitoring training."""
    hook = hooks.EarlyStoppingCallback(
        square_loss_calc, patience=2, min_delta=1
    )
    trainer = create_trainer(post_epoch_hooks=[hook])
    trainer.train(4)
    # 2 epochs of patience and terminate at 3 or 4
    assert trainer.model.epoch in {3, 4}


def test_early_stopping_on_val(
    linear_model,
    standard_learning_schema,
    identity_loader,
    square_loss_calc,
) -> None:
    """Test early stopping when monitoring validation, using the builder."""
    hook = hooks.EarlyStoppingCallback(
        square_loss_calc, patience=2, min_delta=1
    )
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_validation(val_loader=identity_loader)
    builder.add_post_epoch_hook(hook)
    trainer = builder.build()
    trainer.train(4)
    # 5 epochs of patience and terminate at 3
    assert trainer.model.epoch == 3


def test_pruning_callback(square_loss_calc, create_trainer) -> None:
    """Test pruning based on metric thresholds."""
    pruning_thresholds = {2: 1.0, 3: 0.0}
    hook = hooks.PruneCallback(
        thresholds=pruning_thresholds,
        metric=square_loss_calc,
    )
    trainer = create_trainer(post_epoch_hooks=[hook])
    trainer.train(4)
    # stop at epoch 3 because the loss is always greater than 0
    assert trainer.model.epoch == 3


def test_reduce_lr_on_plateau(square_loss_calc, create_trainer) -> None:
    """Test learning rate reduction on plateau."""
    factor = 0.1
    epochs = 5
    hook = hooks.ReduceLROnPlateau(
        metric=square_loss_calc,
        factor=factor,
        min_delta=1,
    )
    trainer = create_trainer(post_epoch_hooks=[hook])
    initial_lr = trainer._model_optimizer.base_lr
    trainer.train(epochs)
    final_lr = trainer._model_optimizer.get_scheduled_lr(initial_lr)

    # Due to high min_delta, the hook is triggered every epoch except first
    assert final_lr == pytest.approx(factor ** (epochs - 1) * initial_lr)


def test_restart_schedule_on_plateau(square_loss_calc, create_trainer) -> None:
    """Test learning rate schedule restart on plateau."""
    exp_scheduler = schedulers.ExponentialScheduler()
    hook = hooks.RestartScheduleOnPlateau(metric=square_loss_calc)
    trainer = create_trainer(post_epoch_hooks=[hook])
    initial_lr = trainer._model_optimizer.base_lr
    trainer.update_learning_rate(scheduler=exp_scheduler)
    trainer.train(5)
    # training should complete with schedule restarts
    final_lr = trainer._model_optimizer.get_scheduled_lr(initial_lr)
    assert not final_lr == pytest.approx(exp_scheduler(initial_lr, 4))
