"""Functional tests for training hooks."""

from collections.abc import Generator

import pytest

from drytorch import Trainer
from drytorch.core import exceptions
from drytorch.core.experimenting import Run
from drytorch.lib import hooks, schedulers


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


def test_early_stopping(
    linear_model,
    standard_learning_schema,
    identity_loader,
    square_loss_calc,
) -> None:
    """Test early stopping when monitoring training."""
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_early_stopping(
        patience=2,
        min_delta=1,
        watch=builder.watch(training=True),
    )
    trainer = builder.build()
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
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_validation(val_loader=identity_loader)
    builder.add_early_stopping(patience=2, min_delta=1)
    trainer = builder.build()
    trainer.train(4)

    # 5 epochs of patience and terminate at 3
    assert trainer.model.epoch == 3


def test_pruning_callback(
    linear_model,
    standard_learning_schema,
    identity_loader,
    square_loss_calc,
) -> None:
    """Test pruning based on metric thresholds."""
    pruning_thresholds = {2: 1.0, 3: 0.0}
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_pruning(
        pruning_thresholds,
        watch=builder.watch(training=True),
    )
    trainer = builder.build()
    trainer.train(4)

    # stop at epoch 3 because the loss is always greater than 0
    assert trainer.model.epoch == 3


def test_reduce_lr_on_plateau(
    linear_model,
    standard_learning_schema,
    identity_loader,
    square_loss_calc,
) -> None:
    """Test learning rate reduction on plateau."""
    factor = 0.1
    epochs = 5
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_reduce_lr_on_plateau(
        factor=factor,
        min_delta=1,
        watch=builder.watch(training=True),
    )
    trainer = builder.build()
    initial_lr = trainer._model_optimizer.base_lr
    trainer.train(epochs)
    final_lr = trainer._model_optimizer.get_scheduled_lr(initial_lr)

    # Due to high min_delta, the hook is triggered every epoch except first
    assert final_lr == pytest.approx(factor ** (epochs - 1) * initial_lr)


def test_restart_schedule_on_plateau(
    linear_model,
    standard_learning_schema,
    identity_loader,
    square_loss_calc,
) -> None:
    """Test learning rate schedule restart on plateau."""
    exp_scheduler = schedulers.ExponentialScheduler()
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_restart_schedule_on_plateau(
        watch=builder.watch(training=True),
    )
    trainer = builder.build()
    initial_lr = trainer._model_optimizer.base_lr
    trainer.update_learning_rate(scheduler=exp_scheduler)
    trainer.train(5)
    final_lr = trainer._model_optimizer.get_scheduled_lr(initial_lr)

    # training should complete with schedule restarts
    assert not final_lr == pytest.approx(exp_scheduler(initial_lr, 4))


def test_hook_placed_before_validation_raises(
    linear_model,
    standard_learning_schema,
    identity_loader,
    square_loss_calc,
) -> None:
    """Test hook placed before its validation raises SourceRunAfterHookError."""
    builder = Trainer.builder(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    builder.add_validation(val_loader=identity_loader, val_name='Val')
    val_hook = builder._post_epoch_hooks.pop()
    watch = builder.watch()
    early_stopping = hooks.EarlyStopping(patience=2, watch=watch)
    builder.add_post_epoch_hook(early_stopping)
    builder.add_post_epoch_hook(val_hook)
    trainer = builder.build()

    with pytest.raises(exceptions.SourceRunAfterHookError):
        trainer.train(2)
