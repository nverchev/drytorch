"""Tests for the "checkpoint" module."""

import pathlib
import time

import torch

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.experiment import Experiment
from drytorch.lib import checkpoints


@pytest.fixture(autouse=True, scope='module')
def setup_module(session_mocker, tmpdir_factory) -> None:
    """Fixture for a mock experiment."""
    mock_experiment = session_mocker.create_autospec(Experiment, instance=True)
    mock_experiment.name = 'mock_experiment'
    mock_experiment.run_dir = pathlib.Path(tmpdir_factory.mktemp('experiments'))
    session_mocker.patch(
        'drytorch.Experiment.get_current', return_value=mock_experiment
    )


class TestPathManager:
    """Tests for PathManager."""

    @pytest.fixture()
    def manager(
        self, mock_model, tmp_path
    ) -> checkpoints.CheckpointPathManager:
        """Set up the path manager."""
        return checkpoints.CheckpointPathManager(mock_model, tmp_path)

    def test_paths(self, manager):
        """Test that the paths have the correct name."""
        epoch_dir = manager.epoch_dir
        paths = [manager.model_state_path, manager.optimizer_state_path]
        expected_paths = [
            epoch_dir / 'model_state.pt',
            epoch_dir / 'optimizer_state.pt',
        ]

        for path, expected_path in zip(paths, expected_paths, strict=False):
            assert path == expected_path


class TestLocalCheckpoint:
    """Tests for LocalCheckpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Set up the model state class."""
        self.mock_save_event = mocker.patch.object(log_events, 'SaveModelEvent')
        self.mock_load_event = mocker.patch.object(log_events, 'LoadModelEvent')

    @pytest.fixture()
    def optimizer(self, mock_model) -> torch.optim.Optimizer:
        """Set up the optimizer."""
        return torch.optim.SGD(mock_model.module.parameters())

    @pytest.fixture()
    def checkpoint(
        self, mock_model, optimizer, tmp_path
    ) -> checkpoints.LocalCheckpoint:
        """Set up the checkpoint."""
        checkpoint = checkpoints.LocalCheckpoint(tmp_path)
        checkpoint.bind_model(mock_model)
        checkpoint.bind_optimizer(optimizer)
        return checkpoint

    def test_checkpoint_not_initialized(self, tmp_path) -> None:
        """Test it raises an error if no model was registered."""
        checkpoint = checkpoints.LocalCheckpoint(tmp_path)
        with pytest.raises(exceptions.CheckpointNotInitializedError):
            checkpoint.load()

    def test_get_last_saved_epoch_no_checkpoints(self, checkpoint) -> None:
        """Test it raises an error if it cannot find any folder."""
        with pytest.raises(exceptions.ModelNotFoundError):
            checkpoint.load()

    def test_get_last_saved_epoch_wrong_checkpoint(self, checkpoint) -> None:
        """Test it raises an error if it cannot find any folder."""
        checkpoint.save()
        with pytest.raises(exceptions.EpochNotFoundError):
            checkpoint.load(1000)

    def test_save_and_load(self, checkpoint) -> None:
        """Test it saves the model's state."""
        checkpoint.save()
        self.mock_save_event.assert_called_once()
        old_weight = checkpoint.model.module.weight.clone()
        new_weight = torch.FloatTensor([[0.0]])
        checkpoint.model.module.weight = torch.nn.Parameter(new_weight)
        assert old_weight != checkpoint.model.module.weight
        checkpoint.load(checkpoint.model.epoch)
        self.mock_load_event.assert_called_once()
        assert old_weight == checkpoint.model.module.weight
        old_lr = checkpoint.optimizer.param_groups[0]['lr']
        new_lr = 0.01
        checkpoint.optimizer.param_groups[0]['lr'] = new_lr
        assert old_lr != checkpoint.optimizer.param_groups[0]['lr']
        checkpoint.load(checkpoint.model.epoch)
        assert old_lr == checkpoint.optimizer.param_groups[0]['lr']

    def test_get_last_saved_epoch(self, checkpoint, mock_model) -> None:
        """Test it recovers the epoch of the longest trained model."""
        checkpoint.save()
        old_epoch = checkpoint.model.epoch
        assert checkpoint._get_last_saved_epoch() == old_epoch
        new_epoch = 15
        time.sleep(0.01)
        checkpoint.model.epoch = new_epoch
        checkpoint.save()
        assert checkpoint._get_last_saved_epoch() == new_epoch
        model_with_no_bias = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model_with_no_bias.parameters())
        checkpoint.remove_model()
        checkpoint.bind_model(mock_model)
        checkpoint.bind_optimizer(optimizer)
        with pytest.warns(exceptions.OptimizerNotLoadedWarning):
            checkpoint.load()
