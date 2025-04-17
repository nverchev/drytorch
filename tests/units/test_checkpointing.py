"""Tests for the checkpoint module."""

import pytest
import time
import torch

from dry_torch import checkpointing, log_events
from dry_torch import exceptions


class TestPathManager:

    @pytest.fixture(autouse=True)
    def setup(self, mock_model, tmp_path) -> None:
        """Set up the path manager."""
        self.par_dir = tmp_path
        self.manager = checkpointing.CheckpointPathManager(mock_model, tmp_path)
        return

    def test_dirs_creation(self, mock_model):
        """Test that the directories are created when called."""
        checkpoint_dir = self.par_dir / mock_model.name / 'checkpoints'
        epoch_dir = checkpoint_dir / f'epoch_{mock_model.epoch}'
        expected_dirs = [checkpoint_dir, epoch_dir]

        for expected_dir in expected_dirs:
            assert not expected_dir.exists()

        dirs = [self.manager.checkpoint_dir, self.manager.epoch_dir]

        for dir_, expected_dir in zip(dirs, expected_dirs):
            assert dir_ == expected_dir
            assert dir_.exists()
            assert dir_.is_dir()

        return

    def test_paths(self):
        """Test that the paths have the correct name."""
        epoch_dir = self.manager.epoch_dir
        paths = [self.manager.state_path, self.manager.optimizer_path]
        expected_paths = [epoch_dir / 'state.pt', epoch_dir / 'optimizer.pt']

        for path, expected_path in zip(paths, expected_paths):
            assert path == expected_path


class TestLocalCheckpoint:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mocker):
        """Set up the model state class."""
        optimizer = torch.optim.SGD(mock_model.module.parameters())
        self.checkpoint = checkpointing.LocalCheckpoint()
        self.checkpoint.register_model(mock_model)
        self.checkpoint.register_optimizer(optimizer)
        self.mock_save_event = mocker.patch.object(log_events, 'SaveModel')
        self.mock_load_event = mocker.patch.object(log_events, 'LoadModel')

    def test_get_last_saved_epoch_no_checkpoints(self):
        """Test it raises error if it cannot find any folder."""
        with pytest.raises(exceptions.ModelNotFoundError):
            self.checkpoint.load()

    def test_save_and_load(self):
        """Test it saves the model's state."""
        self.checkpoint.save()
        self.mock_save_event.assert_called_once()
        old_weight = self.checkpoint.model.module.weight.clone()
        new_weight = torch.FloatTensor([[0.]])
        self.checkpoint.model.module.weight = torch.nn.Parameter(new_weight)
        assert old_weight != self.checkpoint.model.module.weight
        self.checkpoint.load(self.checkpoint.model.epoch)
        self.mock_load_event.assert_called_once()
        assert old_weight == self.checkpoint.model.module.weight
        old_lr = self.checkpoint.optimizer.param_groups[0]['lr']
        new_lr = 0.01
        self.checkpoint.optimizer.param_groups[0]['lr'] = new_lr
        assert old_lr != self.checkpoint.optimizer.param_groups[0]['lr']
        self.checkpoint.load(self.checkpoint.model.epoch)
        assert old_lr == self.checkpoint.optimizer.param_groups[0]['lr']

    def test_get_last_saved_epoch(self, mock_model):
        """Test it recovers the epoch of the longest trained model."""
        self.checkpoint.save()
        old_epoch = self.checkpoint.model.epoch
        assert self.checkpoint._get_last_saved_epoch() == old_epoch
        new_epoch = 15
        time.sleep(0.01)
        self.checkpoint.model.epoch = new_epoch
        self.checkpoint.save()
        assert self.checkpoint._get_last_saved_epoch() == new_epoch
        model_with_no_bias = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model_with_no_bias.parameters())
        self.checkpoint2 = checkpointing.LocalCheckpoint()
        self.checkpoint2.register_model(mock_model)
        self.checkpoint2.register_optimizer(optimizer)
        with pytest.warns(exceptions.OptimizerNotLoadedWarning):
            self.checkpoint2.load()
