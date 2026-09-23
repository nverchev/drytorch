"""Tests for the ``evaluations`` module."""

from typing import Any

import pytest

from drytorch.lib.evaluations import Evaluation, Validation
from drytorch.lib.evaluations import Test as _Test
from drytorch.lib.objectives import Metric


@pytest.fixture(autouse=True, scope='module')
def setup_module(module_mocker) -> None:
    """Fixture for a mock experiment."""
    module_mocker.patch('drytorch.core.registering.register_runner')
    return


class TestEvaluation:
    """Tests for the Evaluation class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, example_named_metrics) -> None:
        """Set up the tests."""
        self.mock_super_call = mocker.patch(
            'drytorch.lib.runners.Source.__call__'
        )
        return

    @pytest.fixture
    def evaluation(self, mock_model, mock_loader, mock_loss) -> Evaluation:
        """Set up a test instance."""
        return Evaluation(
            mock_model,
            loader=mock_loader,
            metric=mock_loss,
        )

    def test_call_method(self, mocker, mock_model, evaluation) -> None:
        """Test __call__ method sets the model to eval mode and runs epoch."""
        mock = mocker.Mock()
        mock_model.module = mock

        evaluation(store_outputs=True)

        self.mock_super_call.assert_called_once()
        mock.eval.assert_called_once()


class TestValidation:
    """Tests for the Validation class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, example_named_metrics) -> None:
        """Set up the tests."""

        def _set_model(instance, model, *_, objective=None, **__) -> None:
            instance.model = model  # read by the registering metaclass
            instance.objective = objective
            return

        self.mock_super_init = mocker.patch(
            'drytorch.lib.runners.Source.__init__',
            autospec=True,
            side_effect=_set_model,
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

    def test_metric_type_survives(self, mock_model, mock_loader) -> None:
        """Test that a metric subclass preserves its concrete type."""

        class CustomMetric(Metric[Any, Any]):
            custom_attr: str = 'reachable'

        custom_metric = CustomMetric(
            lambda outputs, targets: outputs.sum(), name='custom'
        )

        validation = Validation(
            mock_model,
            loader=mock_loader,
            metric=custom_metric,
        )

        assert validation.objective.custom_attr == 'reachable'


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
            'drytorch.lib.evaluations.Evaluation.__call__'
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
        self.mock_super_call.assert_called_once_with(store_outputs=True)
        self.mock_end_test.assert_called_once_with(
            test_instance.name, test_instance.model.name
        )
