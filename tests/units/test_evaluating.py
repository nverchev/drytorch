"""Tests for the evaluating module."""

import pytest

from src.dry_torch import Diagnostic


class TestDiagnostic:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_metric, mock_loader):

        """Set up the Diagnostic."""
        self.diagnostic = Diagnostic(
            mock_model,
            loader=mock_loader,
            calculator=mock_metric,
            mixed_precision=True,
        )

    def test_store_outputs(self, mocker):
        """Test outputs are correctly stored if store_outputs flag is active."""
        mock_output = mocker.Mock()
        mock_apply_ops = mocker.patch(
            'src.dry_torch.apply_ops.apply_cpu_detach',
            return_value=mock_output)

        self.diagnostic._store(mock_output)

        mock_apply_ops.assert_called_once_with(mock_output)
        assert self.diagnostic.outputs_list == [mock_output]

    def test_str_repr(self):
        """Test name is process correctly."""
        self.diagnostic.name = 'Test Diagnostic.123'
        assert str(self.diagnostic) == 'Test Diagnostic'
