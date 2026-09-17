"""Tests for the ``checkpoints`` module."""

import torch

import pytest

from drytorch.core import exceptions, log_events
from drytorch.lib.checkpoints import (
    SAFE_GLOBALS,
    CheckpointPathManager,
    LocalCheckpoint,
)


class TestPathManager:
    """Tests for PathManager."""

    @pytest.fixture()
    def manager(self, mock_model, tmp_path) -> CheckpointPathManager:
        """Set up the path manager."""
        return CheckpointPathManager(mock_model, tmp_path)

    def test_paths(self, manager) -> None:
        """Test that the paths have the correct name."""
        epoch_dir = manager.get_epoch_dir(3)
        paths = [
            manager.get_model_state_path('model', 3),
            manager.get_optimizer_state_path(3),
        ]
        expected_paths = [
            epoch_dir / 'model_state.pt',
            epoch_dir / 'optimizer_state.pt',
        ]

        for path, expected_path in zip(paths, expected_paths, strict=False):
            assert path == expected_path


class TestLocalCheckpoint:
    """Tests for LocalCheckpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the model state class."""
        self.mock_save_event = mocker.patch.object(log_events, 'SaveModelEvent')
        self.mock_load_event = mocker.patch.object(log_events, 'LoadModelEvent')
        self.mock_check_run = mocker.patch(
            'drytorch.core.registering.check_current_run'
        )

    @pytest.fixture()
    def optimizer(self, mock_model) -> torch.optim.Optimizer:
        """Set up the optimizer."""
        return torch.optim.SGD(mock_model.module.parameters())

    @pytest.fixture()
    def checkpoint(self, mock_model, optimizer, tmp_path) -> LocalCheckpoint:
        """Set up the checkpoint."""
        checkpoint = LocalCheckpoint(tmp_path)
        checkpoint.bind_model(mock_model)
        checkpoint.bind_optimizer(optimizer)
        return checkpoint

    def test_par_dir(self, checkpoint, tmp_path) -> None:
        """Test par_dir attribute on LocalCheckpoint."""
        default_checkpoint = LocalCheckpoint()

        assert checkpoint.par_dir == tmp_path
        assert default_checkpoint.par_dir is None

    def test_checkpoint_not_initialized(self, tmp_path) -> None:
        """Test it raises an error if no model was registered."""
        checkpoint = LocalCheckpoint(tmp_path)

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

    def test_save_and_load_module_state(self, checkpoint) -> None:
        """Test saving and restoring module parameters."""
        checkpoint.save()
        old_weight = checkpoint.model.module.weight.clone()
        new_weight = torch.FloatTensor([[0.0]])
        checkpoint.model.module.weight = torch.nn.Parameter(new_weight)

        checkpoint.load(checkpoint.model.epoch)

        self.mock_save_event.assert_called_once()
        self.mock_load_event.assert_called_once()
        assert old_weight == checkpoint.model.module.weight

    def test_save_and_load_optimizer_state(self, checkpoint) -> None:
        """Test saving and restoring optimizer parameters."""
        checkpoint.save()
        old_lr = checkpoint.optimizer.param_groups[0]['lr']
        checkpoint.optimizer.param_groups[0]['lr'] = 0.01

        checkpoint.load(checkpoint.model.epoch)

        assert old_lr == checkpoint.optimizer.param_groups[0]['lr']

    def test_failed_save_emits_no_event(self, mocker, checkpoint) -> None:
        """Test a failed save emits no SaveModelEvent."""
        mocker.patch.object(checkpoint, '_save_module', side_effect=OSError)

        with pytest.raises(OSError):
            checkpoint.save()

        self.mock_save_event.assert_not_called()

    def test_failed_load_emits_no_event(self, mocker, checkpoint) -> None:
        """Test a failed load emits no LoadModelEvent."""
        checkpoint.save()
        mocker.patch.object(
            checkpoint, '_load_module', side_effect=RuntimeError
        )

        with pytest.raises(RuntimeError):
            checkpoint.load(checkpoint.model.epoch)

        self.mock_load_event.assert_not_called()

    def test_load_returns_epoch(self, checkpoint) -> None:
        """Test load returns the loaded epoch."""
        checkpoint.save()

        epoch = checkpoint.load()

        assert epoch == checkpoint.model.epoch

    def test_get_last_saved_epoch(self, checkpoint) -> None:
        """Test it recovers the epoch of the longest trained model."""
        checkpoint.save()
        checkpoint.model.epoch = 15
        checkpoint.save()

        last_epoch = checkpoint._get_last_saved_epoch()

        assert last_epoch == 15

    def test_optimizer_not_loaded_warning(self, checkpoint, mock_model) -> None:
        """Test warning when optimizer state cannot be loaded."""
        checkpoint.save()
        model_with_no_bias = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model_with_no_bias.parameters())
        checkpoint.remove_model()
        checkpoint.bind_model(mock_model)
        checkpoint.bind_optimizer(optimizer)

        with pytest.warns(exceptions.OptimizerNotLoadedWarning):
            checkpoint.load()

    def test_save_multiple_modules(self, checkpoint, mock_model) -> None:
        """Test it saves multiple modules."""
        epoch_dir = checkpoint.paths.get_epoch_dir(mock_model.epoch)
        aux_module = torch.nn.Linear(10, 5)
        checkpoint.bind_model(mock_model)
        checkpoint.bind_module('aux', aux_module)

        checkpoint.save()

        assert (epoch_dir / 'model_state.pt').exists()
        assert (epoch_dir / 'aux_state.pt').exists()
        assert (epoch_dir / 'optimizer_state.pt').exists()

    def test_load_multiple_modules(self, checkpoint, mock_model) -> None:
        """Test it loads multiple modules correctly."""
        aux_module = torch.nn.Linear(10, 5)
        checkpoint.bind_model(mock_model)
        checkpoint.bind_module('aux', aux_module)
        checkpoint.save()
        for p in mock_model.module.parameters():
            p.data.zero_()

        for p in aux_module.parameters():
            p.data.zero_()

        checkpoint.load()

        for p in mock_model.module.parameters():
            assert not torch.allclose(p.data, torch.zeros_like(p))

        for p in aux_module.parameters():
            assert not torch.allclose(p.data, torch.zeros_like(p))

    def test_bind_optimizer_already_bound(self, checkpoint, mock_model) -> None:
        """Test it raises an error when binding a second optimizer."""
        optimizer2 = torch.optim.SGD(mock_model.module.parameters(), lr=0.01)

        with pytest.raises(exceptions.OptimizerAlreadyBoundError):
            checkpoint.bind_optimizer(optimizer2)

    def test_bind_model_already_bound(
        self, checkpoint, mocker, mock_model
    ) -> None:
        """Test it raises an error when binding a second model."""
        mock_model2 = mocker.MagicMock(spec=type(mock_model))
        mock_model2.name = 'MockModel2'

        with pytest.raises(exceptions.ModelAlreadyBoundError):
            checkpoint.bind_model(mock_model2)


def test_safe_globals_is_tuple() -> None:
    """Test that SAFE_GLOBALS is an immutable tuple."""
    assert isinstance(SAFE_GLOBALS, tuple)
    assert len(SAFE_GLOBALS) > 0
