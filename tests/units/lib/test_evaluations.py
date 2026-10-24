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
    module_mocker.patch('drytorch.core.registering.register_test')
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

        def _set_model(
            instance, model, *_, name='', objective=None, **__
        ) -> None:
            instance.model = model  # read by the registering metaclass
            instance._name = name
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

    def test_publish_metrics(self, mocker, validation) -> None:
        """Test that Validation still publishes a plain MetricEvent."""
        mock_metric_event = mocker.patch('drytorch.core.log_events.MetricEvent')
        metrics = {'loss': 0.5}
        validation._publish_metrics(metrics)

        mock_metric_event.assert_called_once_with(
            model_name=validation.model.name,
            source_name=validation.name,
            epoch=validation.model.epoch,
            metrics=metrics,
        )

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

    def test_registration(
        self, mocker, mock_model, mock_metric, mock_loader
    ) -> None:
        """Test creating a Validation registers as runner."""
        mock_runner_reg = mocker.patch(
            'drytorch.core.registering.register_runner'
        )
        mock_test_reg = mocker.patch('drytorch.core.registering.register_test')
        val = Validation(mock_model, loader=mock_loader, metric=mock_metric)
        mock_runner_reg.assert_called_once_with(val, mock_model)
        mock_test_reg.assert_not_called()


class TestTest:
    """Tests for the Test class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_test_metric = mocker.patch(
            'drytorch.core.log_events.TestMetricEvent'
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

    def test_publish_metrics(self, test_instance) -> None:
        """Test that Test publishes TestMetricEvent."""
        metrics = {'loss': 0.5}
        test_instance._publish_metrics(metrics)

        self.mock_test_metric.assert_called_once_with(
            model_name=test_instance.model.name,
            source_name=test_instance.name,
            epoch=test_instance.model.epoch,
            metrics=metrics,
        )

    def test_registration(
        self, mocker, mock_model, mock_metric, mock_loader
    ) -> None:
        """Test creating a Test registers as test runner."""
        mock_runner_reg = mocker.patch(
            'drytorch.core.registering.register_runner'
        )
        mock_test_reg = mocker.patch('drytorch.core.registering.register_test')
        test = _Test(mock_model, loader=mock_loader, metric=mock_metric)
        mock_test_reg.assert_called_once_with(test, mock_model)
        mock_runner_reg.assert_not_called()
