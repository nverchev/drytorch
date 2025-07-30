"""Tests for the "checkpoint" module."""

import time

import torch

import pytest

from drytorch import checkpointing, exceptions, log_events


class TestPathManager:
    """Tests for PathManager."""

    @pytest.fixture()
    def manager(self,
                mock_model,
                tmp_path) -> checkpointing.CheckpointPathManager:
        """Set up the path manager."""
        return checkpointing.CheckpointPathManager(mock_model, tmp_path)

    def test_dirs_creation(self, manager, mock_model):
        """Test that the directories are created when called."""
        checkpoint_dir = manager._root_dir / mock_model.name / 'checkpoints'
        epoch_dir = checkpoint_dir / f'epoch_{mock_model.epoch}'
        expected_dirs = [checkpoint_dir, epoch_dir]

        for expected_dir in expected_dirs:
            assert not expected_dir.exists()

        dirs = [manager.checkpoint_dir, manager.epoch_dir]

        for dir_, expected_dir in zip(dirs, expected_dirs, strict=False):
            assert dir_ == expected_dir
            assert dir_.exists()
            assert dir_.is_dir()

        return

    def test_paths(self, manager):
        """Test that the paths have the correct name."""
        epoch_dir = manager.epoch_dir
        paths = [manager.state_path, manager.optimizer_path]
        expected_paths = [epoch_dir / 'state.pt', epoch_dir / 'optimizer.pt']

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
    def checkpoint(self,
                   mock_model,
                   optimizer) -> checkpointing.LocalCheckpoint:
        """Set up the checkpoint."""
        checkpoint = checkpointing.LocalCheckpoint()
        checkpoint.register_model(mock_model)
        checkpoint.register_optimizer(optimizer)
        return checkpoint

    def test_get_last_saved_epoch_no_checkpoints(self, checkpoint) -> None:
        """Test it raises an error if it cannot find any folder."""
        with pytest.raises(exceptions.ModelNotFoundError):
            checkpoint.load()

    def test_save_and_load(self, checkpoint) -> None:
        """Test it saves the model's state."""
        checkpoint.save()
        self.mock_save_event.assert_called_once()
        old_weight = checkpoint.model.module.weight.clone()
        new_weight = torch.FloatTensor([[0.]])
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
        checkpoint.register_model(mock_model)
        checkpoint.register_optimizer(optimizer)
        with pytest.warns(exceptions.OptimizerNotLoadedWarning):
            checkpoint.load()
