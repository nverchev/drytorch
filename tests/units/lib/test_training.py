"""Tests for the ``training`` module."""

from collections.abc import Callable
from typing import Any

import torch

import pytest

from drytorch.core import exceptions
from drytorch.lib.training import ModelOptimizer, Trainer, TrainerBuilder
from tests.simple_classes import MLP


@pytest.fixture(autouse=True, scope='module')
def setup_module(module_mocker) -> None:
    """Fixture for a mock experiment."""
    module_mocker.patch('drytorch.core.registering.register_runner')
    return


class TestModelOptimizer:
    """Tests for ModelOptimizer."""

    @pytest.fixture()
    def model_optimizer(
        self, mock_model, mock_learning_schema
    ) -> ModelOptimizer:
        """Set up a test instance."""
        mock_model.module = MLP()
        return ModelOptimizer(
            model=mock_model,
            learning_schema=mock_learning_schema,
        )

    def test_update_learning_rate_global(self, model_optimizer) -> None:
        """Test it correctly updates global learning rate."""
        model_optimizer.update_learning_rate(base_lr=0.02)

        for param_group in model_optimizer._optimizer.param_groups:
            assert param_group['lr'] == 0.02

    def test_update_learning_rate_param(self, model_optimizer) -> None:
        """Test it correctly updates parameter-specific learning rates."""
        dict_lr: dict[str, float] = {'linear': 0.01, 'linear2': 0.001}
        model_optimizer.update_learning_rate(base_lr=dict_lr)

        param_groups = model_optimizer._optimizer.param_groups
        for param_group, lr in zip(
            param_groups, dict_lr.values(), strict=False
        ):
            assert param_group['lr'] == lr

    def test_missing_param_error(self, model_optimizer) -> None:
        """Test that MissingParamError is raised when params are missing."""
        with pytest.raises(exceptions.MissingParamError):
            model_optimizer.update_learning_rate(base_lr={'linear': 0.1})

    def test_base_lr_dict_copied(self, model_optimizer) -> None:
        """Test that the base_lr dictionary is copied when updated."""
        dict_lr: dict[str, float] = {'linear': 0.01, 'linear2': 0.001}
        model_optimizer.update_learning_rate(base_lr=dict_lr)

        dict_lr['linear'] = 0.999

        assert model_optimizer.base_lr == {'linear': 0.01, 'linear2': 0.001}


class TestTrainer:
    """Tests for Trainer."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test class."""
        self.model_optimizer = mocker.patch(
            'drytorch.lib.training.ModelOptimizer'
        )
        self.start_training_event = mocker.patch(
            'drytorch.core.log_events.StartTrainingEvent'
        )
        self.end_training_event = mocker.patch(
            'drytorch.core.log_events.EndTrainingEvent'
        )
        self.start_epoch_event = mocker.patch(
            'drytorch.core.log_events.StartEpochEvent'
        )
        self.end_epoch_event = mocker.patch(
            'drytorch.core.log_events.EndEpochEvent'
        )
        self.iterate_event = mocker.patch(
            'drytorch.core.log_events.IterateBatchEvent'
        )
        self.metrics_event = mocker.patch(
            'drytorch.core.log_events.MetricEvent'
        )
        self.terminated_event = mocker.patch(
            'drytorch.core.log_events.TerminatedTrainingEvent'
        )
        self.learning_rate_event = mocker.patch(
            'drytorch.core.log_events.LearningRateEvent'
        )
        return

    @pytest.fixture()
    def create_trainer(
        self, mock_model, mock_learning_schema, mock_loss, mock_loader
    ) -> Callable[..., Trainer]:
        """Provide a function creating a Trainer with mock components."""

        def _create(**kwargs: Any) -> Trainer:
            return Trainer(
                mock_model,
                learning_schema=mock_learning_schema,
                loss=mock_loss,
                loader=mock_loader,
                name='TestTrainer',
                **kwargs,
            )

        return _create

    @pytest.fixture()
    def trainer(self, create_trainer) -> Trainer:
        """Set up a Trainer instance with mock components."""
        return create_trainer()

    def test_update_learning_rate(self, trainer) -> None:
        """Test the learning rate is updated and the event is emitted."""
        trainer.update_learning_rate(base_lr=0.01)

        update = self.model_optimizer.return_value.update_learning_rate
        update.assert_called_once_with(0.01, None)
        self.learning_rate_event.assert_called_once()

    def test_failed_update_learning_rate_emits_no_event(self, trainer) -> None:
        """Test a failed learning rate update emits no event."""
        update = self.model_optimizer.return_value.update_learning_rate
        update.side_effect = exceptions.MissingParamError(['module'], ['other'])

        with pytest.raises(exceptions.MissingParamError):
            trainer.update_learning_rate(base_lr={'other': 0.01})

        self.learning_rate_event.assert_not_called()

    def test_loss_not_a_scalar(self, mocker, trainer) -> None:
        """Test that convergence error correctly terminates training."""
        loss_value = torch.ones(2, 1)
        loss_value.requires_grad = True
        mock = mocker.MagicMock(return_value=loss_value)
        trainer.objective.forward = mock

        with pytest.raises(exceptions.LossNotScalarError):
            trainer.train(1)

    def test_no_validation(self, trainer) -> None:
        """Test a trainer without validation loader has no validation."""
        assert trainer.validation is None
        assert trainer.post_epoch_hooks.hooks == ()

    def test_validation_before_post_epoch_hooks(
        self, mocker, create_trainer
    ) -> None:
        """Test the validation runs before the other post-epoch hooks."""
        calls: list[str] = []
        validation = mocker.Mock(side_effect=lambda: calls.append('val'))
        mocker.patch(
            'drytorch.lib.evaluations.Validation', return_value=validation
        )
        post_hook = mocker.Mock(side_effect=lambda _: calls.append('post'))
        trainer = create_trainer(
            val_loader=mocker.Mock(), post_epoch_hooks=[post_hook]
        )

        trainer.post_epoch_hooks.execute(trainer)

        assert trainer.validation is validation
        assert calls == ['val', 'post']

    def test_validate_every(self, mocker, create_trainer, mock_model) -> None:
        """Test the validation runs every given number of epochs."""
        validation = mocker.Mock()
        mocker.patch(
            'drytorch.lib.evaluations.Validation', return_value=validation
        )
        trainer = create_trainer(val_loader=mocker.Mock(), validate_every=2)

        for epoch in (1, 2, 3):
            mock_model.epoch = epoch
            trainer.post_epoch_hooks.execute(trainer)

        validation.assert_called_once()

    @pytest.mark.parametrize('validate_every', [0, -1])
    def test_validate_every_not_positive(
        self, mocker, create_trainer, validate_every
    ) -> None:
        """Test a non-positive validate_every raises ValueError."""
        with pytest.raises(ValueError):
            create_trainer(
                val_loader=mocker.Mock(), validate_every=validate_every
            )

    def test_loss_higher_is_better(self, create_trainer, mock_loss) -> None:
        """Test a loss that is better when higher raises ValueError."""
        mock_loss.higher_is_better = True

        with pytest.raises(ValueError):
            create_trainer()

    @pytest.mark.parametrize('higher_is_better', [False, None])
    def test_loss_lower_is_better(
        self, create_trainer, mock_loss, higher_is_better
    ) -> None:
        """Test a loss that is not better when higher is accepted."""
        mock_loss.higher_is_better = higher_is_better

        trainer = create_trainer()

        assert trainer.objective is mock_loss

    def test_builder(
        self, mock_model, mock_learning_schema, mock_loss, mock_loader
    ) -> None:
        """Test builder returns a TrainerBuilder."""
        builder = Trainer.builder(
            mock_model,
            loader=mock_loader,
            loss=mock_loss,
            learning_schema=mock_learning_schema,
        )

        assert isinstance(builder, TrainerBuilder)

    def test_post_epoch_hook_terminates_training(self, create_trainer) -> None:
        """Test that post-epoch hook termination prevents end_training_event."""

        def terminate_hook(instance):
            instance.terminate_training('test termination')

        trainer = create_trainer(post_epoch_hooks=[terminate_hook])

        trainer.train(3)

        self.terminated_event.assert_called_once()
        self.start_epoch_event.assert_called_once()
        self.end_epoch_event.assert_called_once()
        self.end_training_event.assert_not_called()

    def test_pre_epoch_hook_terminates_training(self, create_trainer) -> None:
        """Test that pre-epoch hook termination prevents end_training_event."""

        def terminate_hook(instance):
            instance.terminate_training('test termination')

        trainer = create_trainer(pre_epoch_hooks=[terminate_hook])

        trainer.train(3)

        self.terminated_event.assert_called_once()
        self.start_epoch_event.assert_not_called()
        self.end_epoch_event.assert_not_called()
        self.end_training_event.assert_not_called()

    def test_normal_training_emits_end_event(self, trainer) -> None:
        """Test that normal training emits end_training_event."""
        trainer.train(1)

        self.end_epoch_event.assert_called_once()
        self.end_training_event.assert_called_once()
        self.terminated_event.assert_not_called()

    def test_terminate_training(self, trainer) -> None:
        """Test that terminated correctly stop training."""
        trainer.terminate_training(reason='This is a test.')

        with pytest.warns(exceptions.TerminatedTrainingWarning):
            trainer()

        self.terminated_event.assert_called_once()

    def test_train_until(self, mocker, trainer) -> None:
        """Test train_until correctly calculates the remaining epochs."""
        trainer.model.epoch = 2
        mock_train = mocker.MagicMock()
        trainer.train = mock_train

        trainer.train_until(4)

        mock_train.assert_called_once_with(2)

    def test_past_epoch_warning(self, trainer) -> None:
        """Test a warning is raised when trying to train to a past epoch."""
        trainer.model.epoch = 4

        with pytest.warns(exceptions.PastEpochWarning):
            trainer.train_until(3)

    def test_hook_execution_order(self, mocker, create_trainer) -> None:
        """Test that hooks are executed in the correct order."""
        pre_hook_list = [mocker.MagicMock(), mocker.MagicMock()]
        post_hook_list = [mocker.MagicMock(), mocker.MagicMock()]
        trainer = create_trainer(
            pre_epoch_hooks=pre_hook_list, post_epoch_hooks=post_hook_list
        )

        trainer.train(1)

        # Verify pre-hooks are called before post-hooks within the epoch
        hook_list = pre_hook_list + post_hook_list
        ordered_list = []
        for hook in hook_list:
            hook.assert_called_once()
            ordered_list.append(hook.call_args_list[0])

        assert ordered_list == sorted(ordered_list)


class TestTrainerBuilder:
    """Tests for TrainerBuilder."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test class."""
        self.mock_trainer_cls = mocker.patch('drytorch.lib.training.Trainer')
        return

    @pytest.fixture()
    def builder(
        self, mock_model, mock_learning_schema, mock_loss, mock_loader
    ) -> TrainerBuilder:
        """Set up a TrainerBuilder instance with mock components."""
        return TrainerBuilder(
            mock_model,
            'TestTrainer',
            loader=mock_loader,
            loss=mock_loss,
            learning_schema=mock_learning_schema,
        )

    def test_build(
        self,
        mocker,
        builder,
        mock_model,
        mock_learning_schema,
        mock_loss,
        mock_loader,
    ) -> None:
        """Test build passes the collected validation and hooks."""
        val_loader = mocker.Mock()
        pre_hook = mocker.Mock()
        post_hook = mocker.Mock()
        builder.add_validation(val_loader, 'Val', validate_every=2)
        builder.add_pre_epoch_hook(pre_hook)
        builder.add_post_epoch_hook(post_hook)

        trainer = builder.build()

        self.mock_trainer_cls.assert_called_once_with(
            mock_model,
            'TestTrainer',
            loader=mock_loader,
            loss=mock_loss,
            learning_schema=mock_learning_schema,
            val_loader=val_loader,
            val_name='Val',
            validate_every=2,
            pre_epoch_hooks=[pre_hook],
            post_epoch_hooks=[post_hook],
        )
        assert trainer is self.mock_trainer_cls.return_value
