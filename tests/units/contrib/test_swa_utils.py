"""Tests for the "swa_utils" module."""

import pytest

import torch

from drytorch.contrib.swa_utils import ModelMomentaUpdater


class TestModelMomentaUpdater:
    """Tests for the ModelMomentaUpdater class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, mock_model) -> None:
        """Set up the tests."""
        self.mock_bn_module = mocker.Mock(spec=torch.nn.BatchNorm2d)
        self.mock_bn_module.momentum = 0.1
        mock_model.module = mocker.Mock(spec=torch.nn.Module)
        mock_model.module.modules.return_value = [self.mock_bn_module]
        mock_model.module.training = True
        mocker.patch('drytorch.running.ModelCaller.__call__')

    @pytest.fixture
    def updater(self, mock_model) -> ModelMomentaUpdater:
        """Set up a test instance."""
        return ModelMomentaUpdater(mock_model)

    def test_call_resets_bn_stats_and_momentum(self,
                                               updater,
                                               mock_model) -> None:
        """Test that BatchNorm stats are reset."""
        updater.model = mock_model
        updater()
        self.mock_bn_module.reset_running_stats.assert_called_once()

    def test_call_restores_original_momentum(self, updater, mock_model) -> None:
        """Test that original momentum values are restored."""
        updater.model = mock_model
        original_momentum = 0.1
        self.mock_bn_module.momentum = original_momentum
        updater()
        assert self.mock_bn_module.momentum == original_momentum

    def test_call_preserves_training_mode(self, updater, mock_model) -> None:
        """Test that original training mode is preserved."""
        updater.model = mock_model
        mock_model.module.training = False
        updater()
        mock_model.module.train.assert_called_with(False)

    def test_call_with_no_batch_norm_modules(self,
                                             mocker,
                                             updater,
                                             mock_model) -> None:
        """Test early return when no BatchNorm modules are found."""
        updater.model = mock_model
        mock_model.module.modules.return_value = []
        mock_super_call = mocker.patch('drytorch.running.ModelCaller.__call__')
        updater()
        mock_super_call.assert_not_called()
