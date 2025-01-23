"""Tests for the checkpoint module."""

import pytest

import torch

from src.dry_torch import checkpoint
from src.dry_torch import exceptions


class TestPathManager:

    @pytest.fixture(autouse=True)
    def setup(self, mock_model) -> None:
        """Set up the path manager."""
        mock_model.name = 'mock_1'
        self.manager = checkpoint.PathManager(mock_model)
        return

    def test_dirs_creation(self, mock_model, experiment):
        """Test that the directories are created when called."""
        checkpoint_dir = experiment.dir / mock_model.name / 'checkpoints'
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


class TestModelStateIO:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model):
        """Set up the model state class."""
        mock_model.name = 'mock_2'
        self.model_io = checkpoint.ModelStateIO(mock_model)

    def test_get_last_saved_epoch_no_checkpoints(self, mocker):
        """Test it raises error if it cannot find any folder."""
        with pytest.raises(exceptions.ModelNotFoundError):
            self.model_io.load()

    def test_save_and_load(self):
        """Test it saves the model's state."""
        self.model_io.save()
        old_weight = self.model_io.model.module.weight.clone()
        new_weight = torch.FloatTensor([[0.]])
        self.model_io.model.module.weight = torch.nn.Parameter(new_weight)
        assert old_weight != self.model_io.model.module.weight
        self.model_io.load(self.model_io.model.epoch)
        assert old_weight == self.model_io.model.module.weight

    def test_get_last_saved_epoch(self, mocker):
        """Test it recovers the epoch of the longest trained model."""
        self.model_io.save()
        old_epoch = self.model_io.model.epoch
        assert self.model_io._get_last_saved_epoch() == old_epoch
        new_epoch = 15
        self.model_io.model.epoch = new_epoch
        self.model_io.save()
        assert self.model_io._get_last_saved_epoch() == new_epoch


class TestCheckpointIO:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mocker):
        """Set up the Checkpoint class."""
        mock_model.name = 'mock_3'
        optimizer = torch.optim.SGD(mock_model.module.parameters())
        self.checkpoint_io = checkpoint.CheckpointIO(mock_model, optimizer)

    def test_save_and_load(self):
        """Test it saves the model's state."""
        self.checkpoint_io.save()
        old_lr = self.checkpoint_io.optimizer.param_groups[0]['lr']
        new_lr = 0.01
        self.checkpoint_io.optimizer.param_groups[0]['lr'] = new_lr
        assert old_lr != self.checkpoint_io.optimizer.param_groups[0]['lr']
        self.checkpoint_io.load(self.checkpoint_io.model.epoch)
        assert old_lr == self.checkpoint_io.optimizer.param_groups[0]['lr']

    def test_checkpoint_load_optimizer_warning(self, mock_model):
        """Test it raises a warning if the optimizer state is incompatible."""
        self.checkpoint_io.save()
        model_with_no_bias = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model_with_no_bias.parameters())
        self.checkpoint_io = checkpoint.CheckpointIO(mock_model, optimizer)

        with pytest.warns(exceptions.OptimizerNotLoadedWarning):
            self.checkpoint_io.load()
