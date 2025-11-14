"""Tests for the "train" module."""

import torch

import pytest

from drytorch import Trainer
from drytorch.core import exceptions


@pytest.fixture(autouse=True)
def setup_module(session_mocker) -> None:
    """Fixture for a mock experiment."""
    # Patch at the point where register_source is called
    session_mocker.patch('drytorch.core.register.register_actor')
    return


class TestTrainer:
    """Tests for Trainer."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Set up the test class."""
        self.model_optimizer = mocker.patch(
            'drytorch.lib.models.ModelOptimizer'
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
        return

    @pytest.fixture()
    def trainer(
        self, mock_model, mock_learning_schema, mock_loss, mock_loader
    ) -> Trainer:
        """Set up a Trainer instance with mock components."""
        return Trainer(
            mock_model,
            learning_schema=mock_learning_schema,
            loss=mock_loss,
            loader=mock_loader,
            name='TestTrainer',
        )

    def test_loss_not_a_scalar(self, mocker, trainer) -> None:
        """Test that convergence error correctly terminates training."""
        loss_value = torch.ones(2, 1)
        loss_value.requires_grad = True
        mock = mocker.MagicMock(return_value=loss_value)
        trainer.objective.forward = mock
        with pytest.raises(exceptions.LossNotScalarError):
            trainer.train(1)

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

    def test_hook_execution_order(self, mocker, trainer) -> None:
        """Test that hooks are executed in the correct order."""
        # Mock the hooks to track their order of execution
        mocker.patch('drytorch.lib.runners.register.register_actor')
        pre_hook_list = [mocker.MagicMock(), mocker.MagicMock()]
        post_hook_list = [mocker.MagicMock(), mocker.MagicMock()]
        trainer.pre_epoch_hooks.register_all(pre_hook_list)
        trainer.post_epoch_hooks.register_all(post_hook_list)

        trainer.train(1)

        # Verify pre-hooks are called before post-hooks within the epoch
        hook_list = pre_hook_list + post_hook_list
        ordered_list = []
        for hook in hook_list:
            hook.assert_called_once()
            ordered_list.append(hook.call_args_list[0])

        assert ordered_list == sorted(ordered_list)
