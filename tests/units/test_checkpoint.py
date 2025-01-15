"""Tests for the checkpoint module"""

import pathlib

import pytest

from src.dry_torch import ModelStateIO
from src.dry_torch import CheckpointIO
from src.dry_torch import exceptions


class TestModelStateIO:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model):
        """Set up the model state class."""
        self.model_io = ModelStateIO(mock_model)

    def test_get_last_saved_epoch(self, mocker):
        """Test it recovers the epoch of the longest trained model."""
        mock_path_1 = mocker.MagicMock()
        mock_path_1.stem = 'state_1'
        mock_path_1.is_dir.return_value = True
        mock_path_2 = mocker.MagicMock()
        mock_path_2.stem = 'state_2'
        mock_path_1.is_dir.return_value = True
        paths = mocker.MagicMock()
        paths.checkpoint_dir.iterdir.return_value = [mock_path_1, mock_path_2]
        self.model_io.paths = paths
        assert self.model_io._get_last_saved_epoch() == 2

    def test_get_last_saved_epoch_no_checkpoints(self, mocker):
        """Test it raises error if it cannot find any folder."""
        paths = mocker.MagicMock()
        paths.checkpoint_dir.iterdir.return_value = []
        self.model_io.paths = paths
        with pytest.raises(exceptions.ModelNotFoundError):
            self.model_io._get_last_saved_epoch()


class TestCheckpointIO:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mocker):
        """Set up the Checkpoint class."""
        optimizer = mocker.MagicMock()
        self.checkpoint_io = CheckpointIO(mock_model, optimizer)

    def test_checkpoint_load_optimizer_warning(self, mocker):
        """Test it raises a warning if it cannot find the optimizer."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.load_state_dict.side_effect = ValueError('Test error')
        mocker.patch('torch.load')
        ModelStateIO.load = mocker.MagicMock()
        self.checkpoint_io.optimizer = mock_optimizer
        paths = mocker.MagicMock()
        paths.epoch_dir = pathlib.Path()
        self.checkpoint_io.paths = paths
        with pytest.warns(exceptions.OptimizerNotLoadedWarning):
            self.checkpoint_io.load()
