"""Tests for the "training" module."""

import pytest

import torch

from drytorch import Trainer
from drytorch import exceptions


class TestTrainer:
    @pytest.fixture(autouse=True)
    def setup(self,
              mock_model,
              mock_learning_scheme,
              mock_loss,
              mock_loader,
              mocker):
        """Set up a Trainer instance with mock components."""
        self.model_optimizer = mocker.patch(
            'drytorch.learning.ModelOptimizer'
        )
        self.trainer = Trainer(
            mock_model,
            learning_scheme=mock_learning_scheme,
            loss=mock_loss,
            loader=mock_loader,
            name='TestTrainer'
        )
        self.start_training_event = mocker.patch(
            'drytorch.log_events.StartTraining')
        self.end_training_event = mocker.patch(
            'drytorch.log_events.EndTraining')
        self.start_epoch_event = mocker.patch(
            'drytorch.log_events.StartEpoch'
        )
        self.end_epoch_event = mocker.patch(
            'drytorch.log_events.EndEpoch'
        )
        self.iterate_event = mocker.patch(
            'drytorch.log_events.IterateBatch'
        )
        self.metrics_event = mocker.patch(
            'drytorch.log_events.Metrics'
        )
        self.terminated_event = mocker.patch(
            'drytorch.log_events.TerminatedTraining'
        )

    def test_call_events(self) -> None:
        """Test train method invokes the necessary hooks and events."""

        self.trainer.train(2)

        self.start_training_event.assert_called_once_with(
            source_name=self.trainer.name,
            model_name=self.trainer.model.name,
            start_epoch=self.trainer.model.epoch,
            end_epoch=self.trainer.model.epoch + 2)
        self.end_training_event.assert_called_once()
        self.start_epoch_event.assert_called()
        self.end_epoch_event.assert_called()
        return

    def test_call_validation(self, mocker, mock_loader) -> None:
        """Test that validation loader is called when validation is added."""
        # Spy on the validation loader to check calls
        spy_loader = mocker.spy(mock_loader, '__iter__')

        # Set up validation and train
        self.trainer.add_validation(spy_loader)
        self.trainer()

        # Ensure the validation loader was used during training
        spy_loader.assert_called_once()
        return

    @pytest.mark.parametrize('invalid_value', [torch.inf, torch.nan])
    def test_convergence_error(self, mocker, invalid_value) -> None:
        """Test that convergence error is called and terminates training."""
        loss_value = torch.FloatTensor([invalid_value])
        loss_value.requires_grad = True
        mock = mocker.MagicMock(return_value=loss_value)
        self.trainer.objective.forward = mock
        self.trainer.train(3)
        mock.assert_called_once()

    def test_loss_not_a_scalar(self, mocker) -> None:
        """Test that convergence error correctly terminates training."""
        loss_value = torch.ones(2, 1)
        loss_value.requires_grad = True
        mock = mocker.MagicMock(return_value=loss_value)
        self.trainer.objective.forward = mock
        with pytest.raises(exceptions.LossNotScalarError):
            self.trainer.train(1)

    def test_terminate_training(self) -> None:
        """Test that terminated correctly stop training."""
        self.trainer.terminate_training(reason='This is a test.')
        with pytest.warns(exceptions.TerminatedTrainingWarning):
            self.trainer()
        self.terminated_event.assert_called_once()

    def test_train_until(self, mocker) -> None:
        """Test train_until correctly calculates the remaining epochs."""
        self.trainer.model.epoch = 2
        mock_train = mocker.MagicMock()
        self.trainer.train = mock_train
        self.trainer.train_until(4)
        mock_train.assert_called_once_with(2)

    def test_past_epoch_warning(self) -> None:
        """Test a warning is raised when trying to train to a past epoch."""
        self.trainer.model.epoch = 4

        with pytest.warns(exceptions.PastEpochWarning):
            self.trainer.train_until(3)

    def test_hook_execution_order(self, mocker) -> None:
        """Test that hooks are executed in the correct order."""
        # Mock the hooks to track their order of execution
        pre_hook_list = [mocker.MagicMock(), mocker.MagicMock()]
        post_hook_list = [mocker.MagicMock(), mocker.MagicMock()]
        self.trainer.pre_epoch_hooks.register_all(pre_hook_list)
        self.trainer.post_epoch_hooks.register_all(post_hook_list)

        self.trainer.train(1)

        # Verify pre hooks are called before post hooks within the epoch
        hook_list = pre_hook_list + post_hook_list
        ordered_list = []
        for hook in hook_list:
            hook.assert_called_once()
            ordered_list.append(hook.call_args_list[0])

        assert ordered_list == sorted(ordered_list)
