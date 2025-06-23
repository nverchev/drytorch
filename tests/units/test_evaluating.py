"""Tests for the "evaluating" module."""

import pytest

from drytorch.evaluating import exceptions
from drytorch.evaluating import Source
from drytorch.evaluating import ModelRunner
from drytorch.evaluating import Diagnostic
from drytorch.evaluating import Test


class TestSource:
    """Tests for the Source class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """ Set up the tests. """
        self.mock_record_model_call = mocker.patch(
            'drytorch.registering.record_model_call'
        )
        return

    @pytest.fixture
    def source(self, mock_model) -> Source:
        """Set up a test instance."""
        return Source(mock_model)

    def test_call_registration(self, source) -> None:
        """Test __call__ method registers model on first call."""
        assert source._registered is False

        source()
        assert source._registered is True
        self.mock_record_model_call.assert_called_once_with(source,
                                                            source.model)

        # Second call should not register again
        self.mock_record_model_call.reset_mock()

        source()
        self.mock_record_model_call.assert_not_called()


class TestModelRunner:
    """Tests for the ModelRunner class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_output = mocker.Mock()
        self.mock_apply_ops = mocker.patch(
            'drytorch.utils.apply_ops.apply_cpu_detach',
            return_value=self.mock_output)
        self.mock_apply_to = mocker.patch('drytorch.utils.apply_ops.apply_to')
        self.mock_apply_to.side_effect = lambda x, device: x
        self.mock_autocast = mocker.patch('torch.autocast')
        self.mock_context = mocker.Mock()
        self.mock_autocast.return_value.__enter__ = mocker.Mock(
            return_value=self.mock_context
        )
        self.mock_autocast.return_value.__exit__ = mocker.Mock(
            return_value=None
        )
        self.mock_log_events_metrics = mocker.patch(
            'drytorch.log_events.Metrics'
        )
        self.mock_loading = mocker.patch(
            'drytorch.loading.check_dataset_length',
            return_value=100)
        self.mock_iterate_batch = mocker.patch(
            'drytorch.log_events.IterateBatch'
        )
        self.mock_repr_metrics = mocker.patch('drytorch.metrics.repr_metrics',
                                              return_value={'loss': 0.1})
        return

    @pytest.fixture
    def runner(self, mock_model, mock_metric, mock_loader) -> ModelRunner:
        """Set up a test instance."""
        return Diagnostic(
            mock_model,
            name='test_runner',
            loader=mock_loader,
            metric=mock_metric,
        )

    def test_initialization(self, mock_model, mock_loader, runner) -> None:
        """Test evaluation initialization with all parameters."""
        assert runner.model == mock_model
        assert runner.name == 'test_runner'
        assert runner.loader == mock_loader
        assert runner.outputs_list == []

    def test_get_batches(self, mocker, runner) -> None:
        """Test batch generation applies operations to device."""
        mock_batch1 = mocker.Mock()
        mock_batch2 = mocker.Mock()
        runner.loader.__iter__ = mocker.Mock(
            return_value=iter([mock_batch1, mock_batch2])
        )
        batches = list(runner._get_batches())
        assert len(batches) == 2
        assert self.mock_apply_to.call_count == 2
        self.mock_apply_to.assert_any_call(mock_batch1, runner.model.device)
        self.mock_apply_to.assert_any_call(mock_batch2, runner.model.device)

    def test_run_backwards(self, mocker, runner) -> None:
        """Test backwards pass updates objective."""
        mock_outputs = mocker.Mock()
        mock_targets = mocker.Mock()
        runner._run_backwards(mock_outputs, mock_targets)
        runner.objective.update.assert_called_once_with(mock_outputs,
                                                        mock_targets)

    def test_run_batch(self, mocker, runner) -> None:
        """Test batch processing runs forward and backwards."""
        mock_inputs = mocker.Mock()
        mock_targets = mocker.Mock()
        mock_outputs = mocker.Mock()
        mock_batch = (mock_inputs, mock_targets)
        runner._run_forward = mocker.Mock(return_value=mock_outputs)
        runner._run_backwards = mocker.Mock()
        result = runner._run_batch(mock_batch)
        assert result == mock_outputs
        runner._run_forward.assert_called_once_with(mock_inputs)
        runner._run_backwards.assert_called_once_with(mock_outputs,
                                                      mock_targets)

    def test_run_forward(self,
                         mocker,
                         runner) -> None:
        """Test forward pass."""
        mock_inputs = mocker.Mock()
        mock_outputs = mocker.Mock()
        runner.model.return_value = mock_outputs
        runner.mixed_precision = False
        result = runner._run_forward(mock_inputs)
        assert result == mock_outputs
        runner.model.assert_called_once()

    def test_log_metrics(self,
                         mocker,
                         runner,
                         example_named_metrics) -> None:
        """Test metrics logging."""
        runner._log_metrics(example_named_metrics)
        self.mock_log_events_metrics.assert_called_once_with(
            model_name=runner.model.name,
            source_name=runner.name,
            epoch=runner.model.epoch,
            metrics=example_named_metrics
        )

    def test_run_epoch_without_storing_outputs(self,
                                               mocker,
                                               runner) -> None:
        """Test epoch run without storing outputs."""
        mock_batch1 = (mocker.Mock(), mocker.Mock())
        mock_batch2 = (mocker.Mock(), mocker.Mock())
        mock_outputs1 = mocker.Mock()
        mock_outputs2 = mocker.Mock()
        runner._get_batches = mocker.Mock(
            return_value=[mock_batch1, mock_batch2]
        )
        runner._run_batch = mocker.Mock(
            side_effect=[mock_outputs1, mock_outputs2]
        )
        runner._log_metrics = mocker.Mock()
        runner._store = mocker.Mock()
        mock_pbar = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_pbar

        # Reset the objective mock to avoid counting the initialization call
        runner.objective.reset.reset_mock()
        runner._run_epoch(store_outputs=False)
        assert runner.outputs_list == []
        runner.objective.reset.assert_called_once()
        self.mock_loading.assert_called_once_with(runner.loader.dataset)
        self.mock_iterate_batch.assert_called_once_with(
            runner.name,
            runner.loader.batch_size,
            len(runner.loader),
            100
        )
        assert runner._run_batch.call_count == 2
        assert mock_pbar.update.call_count == 2
        runner._store.assert_not_called()
        runner._log_metrics.assert_called_once()

    def test_run_epoch_with_storing_outputs(self, mocker, runner) -> None:
        """Test epoch run with storing outputs."""
        mock_batch = (mocker.Mock(), mocker.Mock())
        mock_outputs = mocker.Mock()
        runner._get_batches = mocker.Mock(return_value=[mock_batch])
        runner._run_batch = mocker.Mock(return_value=mock_outputs)
        runner._log_metrics = mocker.Mock()
        runner._store = mocker.Mock()
        mock_pbar = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_pbar
        runner._run_epoch(store_outputs=True)
        runner._store.assert_called_once_with(mock_outputs)

    def test_outputs_list_cleared_on_epoch_run(self,
                                               mocker,
                                               runner) -> None:
        """Test that outputs list is cleared at the start of each epoch."""
        runner.outputs_list = [mocker.Mock(), mocker.Mock()]
        runner._get_batches = mocker.Mock(return_value=[])
        runner._log_metrics = mocker.Mock()
        mock_pbar = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_pbar
        runner._run_epoch(store_outputs=False)
        assert runner.outputs_list == []

    def test_objective_reset_on_epoch_run(self, mocker, runner) -> None:
        """Test that objective is reset at the start of each epoch."""
        runner._get_batches = mocker.Mock(return_value=[])
        runner._log_metrics = mocker.Mock()
        mock_pbar = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_pbar

        # Reset the objective mock to avoid counting the initialization call
        runner.objective.reset.reset_mock()
        runner._run_epoch(store_outputs=False)
        runner.objective.reset.assert_called_once()

    def test_repr_method(self, runner) -> None:
        """Test __repr__ method returns expected format."""
        expected = f"{runner.name} for model {runner.model.name}"
        assert repr(runner) == expected

    def test_name_property(self, runner) -> None:
        """Test name property starts with expected value."""
        assert runner.name.startswith('test_runner')

    def test_store_outputs(self, runner) -> None:
        """Test outputs are correctly stored if store_outputs flag is active."""
        test_object = object()
        runner._store(test_object)
        self.mock_apply_ops.assert_called_once_with(test_object)
        assert runner.outputs_list == [self.mock_output]

    @pytest.mark.parametrize('warning', [
        exceptions.FuncNotApplicableError('wrong_func', 'wrong_type'),
        exceptions.NamedTupleOnlyError('wrong_type')

    ])
    def test_store_outputs_warning(self, mocker, runner,
                                   warning) -> None:
        """Test warning is raised if output cannot be stored."""
        mock_output = mocker.Mock()
        mock_apply_ops = mocker.patch(
            'drytorch.utils.apply_ops.apply_cpu_detach',
            side_effect=warning)

        with pytest.warns(exceptions.CannotStoreOutputWarning):
            runner._store(mock_output)

        mock_apply_ops.assert_called_once_with(mock_output)
        assert runner.outputs_list == []

    def test_str(self, runner):
        """Test string representation of the evaluation."""
        assert str(runner).startswith(runner.name)


class TestDiagnostic:
    """Tests for the Diagnostic class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_super_call = mocker.patch(
            'drytorch.evaluating.Source.__call__')
        return

    @pytest.fixture
    def diagnostic(self, mock_model, mock_metric, mock_loader) -> Diagnostic:
        """Set up a test instance."""
        return Diagnostic(
            mock_model,
            name='test_diagnostic',
            loader=mock_loader,
            metric=mock_metric,
        )

    def test_call_method(self, mocker, diagnostic) -> None:
        """Test __call__ method sets model to eval mode and runs epoch."""
        diagnostic._run_epoch = mocker.Mock()
        diagnostic.model.module.eval = mocker.Mock()
        diagnostic(store_outputs=True)
        self.mock_super_call.assert_called_once()
        diagnostic.model.module.eval.assert_called_once()
        diagnostic._run_epoch.assert_called_once_with(True)


class TestTest:
    """Tests for the Test class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_start_test = mocker.patch('drytorch.log_events.StartTest')
        self.mock_end_test = mocker.patch('drytorch.log_events.EndTest')
        self.mock_super_call = mocker.patch(
            'drytorch.evaluating.Diagnostic.__call__')
        return

    @pytest.fixture
    def test_instance(self, mock_model, mock_metric, mock_loader) -> Test:
        """Set up a test instance."""
        return Test(
            mock_model,
            name='test_instance',
            loader=mock_loader,
            metric=mock_metric,
        )

    def test_call_logging(self, test_instance) -> None:
        """Test __call__ method logs start and end test events."""
        test_instance(store_outputs=True)

        self.mock_start_test.assert_called_once_with(test_instance.name,
                                                     test_instance.model.name)
        self.mock_super_call.assert_called_once_with(True)
        self.mock_end_test.assert_called_once_with(test_instance.name,
                                                   test_instance.model.name)
