"""Tests for the "evaluations" module."""

import pytest

from drytorch.lib.evaluations import Diagnostic, Validation
from drytorch.lib.evaluations import Test as _Test


@pytest.fixture(autouse=True)
def setup_module(session_mocker) -> None:
    """Fixture for a mock experiment."""
    session_mocker.patch('drytorch.core.register.register_actor')
    return


class TestDiagnostic:
    """Tests for the ModelRunner class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, example_named_metrics) -> None:
        """Set up the tests."""
        self.mock_super_call = mocker.patch(
            'drytorch.lib.runners.ModelRunnerWithObjective.__call__'
        )
        return

    @pytest.fixture
    def diagnostic(self, mock_model, mock_loader, mock_loss) -> Diagnostic:
        """Set up a test instance."""
        return Diagnostic(
            mock_model,
            loader=mock_loader,
            objective=mock_loss,
        )

    def test_call_method(self, mocker, mock_model, diagnostic) -> None:
        """Test __call__ method sets the model to eval mode and runs epoch."""
        mock = mocker.Mock()
        mock_model.module = mock
        diagnostic(store_outputs=True)
        self.mock_super_call.assert_called_once()
        mock.eval.assert_called_once()


class TestValidation:
    """Tests for the Validation class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, example_named_metrics) -> None:
        """Set up the tests."""
        self.mock_super_init = mocker.patch(
            'drytorch.lib.runners.ModelRunnerWithLogs.__init__'
        )
        return

    @pytest.fixture
    def validation(self, mock_model, mock_metric, mock_loader) -> Validation:
        """Set up a test instance."""
        return Validation(
            mock_model,
            name='test_evaluation',
            loader=mock_loader,
            metric=mock_metric,
        )

    def test_initialization(self, validation) -> None:
        """Test parent __init__ is called."""
        self.mock_super_init.assert_called_once()


class TestTest:
    """Tests for the Test class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_start_test = mocker.patch(
            'drytorch.core.log_events.StartTestEvent'
        )
        self.mock_end_test = mocker.patch(
            'drytorch.core.log_events.EndTestEvent'
        )
        self.mock_super_call = mocker.patch(
            'drytorch.lib.evaluations.Validation.__call__'
        )
        return

    @pytest.fixture
    def test_instance(self, mock_model, mock_metric, mock_loader) -> _Test:
        """Set up a test instance."""
        return _Test(
            mock_model,
            name='test_instance',
            loader=mock_loader,
            metric=mock_metric,
        )

    def test_call_logging(self, test_instance) -> None:
        """Test __call__ method logs start and end test events."""
        test_instance(store_outputs=True)

        self.mock_start_test.assert_called_once_with(
            test_instance.name, test_instance.model.name
        )
        self.mock_super_call.assert_called_once_with(True)
        self.mock_end_test.assert_called_once_with(
            test_instance.name, test_instance.model.name
        )
