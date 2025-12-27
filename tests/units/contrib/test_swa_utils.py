"""Tests for the "swa_utils" module."""

from torch import nn

import pytest

from drytorch.contrib import swa_utils
from drytorch.core import protocols as p
from drytorch.lib import runners


class TestBatchNormUpdater:
    """Tests for the BatchNormUpdater class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Patch ModelRunner' call method."""
        self.patch_super_call = mocker.patch(
            'drytorch.lib.runners.ModelRunner.__call__'
        )

    @pytest.fixture
    def bn_network(self) -> nn.Sequential:
        """Create a mock module with BatchNorm layers."""
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.BatchNorm1d(10, momentum=0.2),  # Specific momentum
        )

    @pytest.fixture
    def mock_bn_model(self, bn_network, mock_model) -> p.ModelProtocol:
        """Create a mock model with a BN module."""
        mock_model.module = bn_network
        return mock_model

    @pytest.fixture
    def updater(self, mock_bn_model, mock_loader):
        """Create an updater instance."""
        return swa_utils.BatchNormUpdater(mock_bn_model, loader=mock_loader)

    def test_initialization(self, updater, mock_model, mock_loader):
        """Test initialization of BatchNormUpdater."""
        assert isinstance(updater, runners.ModelRunner)
        assert updater.model == mock_model
        assert updater.loader == mock_loader

    def test_call_no_bn(self, mock_model, mock_loader):
        """Test graceful handling when no method has BN."""
        updater = swa_utils.BatchNormUpdater(mock_model, loader=mock_loader)

        updater()

        self.patch_super_call.assert_not_called()

    def test_call_logic_with_bn(self, updater, bn_network):
        """Test that __call__ performs the correct sequence of operations."""
        bn_layers = [
            m for m in bn_network.modules() if isinstance(m, nn.BatchNorm1d)
        ]
        bn1, bn2 = bn_layers
        bn1.momentum = 0.5
        bn2.momentum = 0.1
        bn_network.eval()

        def _assert_reset(*args, **kwargs):
            for bn in bn_layers:
                assert bn.momentum is None

            assert bn_network.training is True

        self.patch_super_call.side_effect = _assert_reset

        updater()

        self.patch_super_call.assert_called_once()
        assert bn1.momentum == 0.5
        assert bn2.momentum == 0.1
        assert not bn_network.training
