"""Tests for the evaluating module"""

import pytest
from src.dry_torch import Diagnostic


class TestDiagnostic:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_metrics_calc, mock_loader):
        """Set up the Diagnostic."""
        self.diagnostic = Diagnostic(
            mock_model,
            loader=mock_loader,
            metrics_calc=mock_metrics_calc,
            mixed_precision=True,
        )

    def test_clear_metrics(self, mocker):
        """Test that clear_metrics clears metrics and resets last metrics."""
        mock_results = {'accuracy': 0.9}
        mock_metrics = mocker.MagicMock()
        mock_metrics.reduce_all.return_value = mock_results
        self.diagnostic._metrics = mock_metrics

        assert self.diagnostic.metrics == mock_results
        assert self.diagnostic.metrics == mock_results
        mock_metrics.reduce_all.assert_called_once()

        self.diagnostic.clear_metrics()

        mock_metrics.reset_mock()
        assert self.diagnostic.metrics == mock_results
        mock_metrics.reduce_all.assert_called_once()

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
