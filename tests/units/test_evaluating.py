"""Tests for the "evaluating" module."""

import pytest

from dry_torch import exceptions
from dry_torch import Diagnostic


class TestDiagnostic:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_metric, mock_loader) -> None:
        """Set up a diagnostic for testing. """
        self.name = 'test_diagnostic'
        self.diagnostic = Diagnostic(
            mock_model,
            name=self.name,
            loader=mock_loader,
            metric=mock_metric,
            mixed_precision=True,
        )

    def test_store_outputs(self, mocker) -> None:
        """Test outputs are correctly stored if store_outputs flag is active."""
        mock_output = mocker.Mock()
        mock_apply_ops = mocker.patch(
            'dry_torch.utils.apply_ops.apply_cpu_detach',
            return_value=mock_output)

        self.diagnostic._store(mock_output)

        mock_apply_ops.assert_called_once_with(mock_output)
        assert self.diagnostic.outputs_list == [mock_output]

    @pytest.mark.parametrize('error', [
        exceptions.FuncNotApplicableError('wrong_func', 'wrong_type'),
        exceptions.NamedTupleOnlyError('wrong_type')

    ])
    def test_store_outputs_warning(self, mocker, error) -> None:
        """Test warning is raised if output cannot be stored."""
        mock_output = mocker.Mock()
        mock_apply_ops = mocker.patch(
            'dry_torch.utils.apply_ops.apply_cpu_detach',
            side_effect=error)

        with pytest.warns(exceptions.CannotStoreOutputWarning):
            self.diagnostic._store(mock_output)

        mock_apply_ops.assert_called_once_with(mock_output)
        assert self.diagnostic.outputs_list == []

    def test_str(self):
        """Test string representation of the diagnostic."""
        assert str(self.diagnostic).startswith(self.name)
