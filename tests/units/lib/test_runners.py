"""Tests for the "runners" module."""

from typing_extensions import override

import pytest

from drytorch.core import exceptions
from drytorch.lib.runners import (
    ModelCaller,
    ModelRunner,
    ModelRunnerWithLogs,
    ModelRunnerWithObjective,
)


@pytest.fixture(autouse=True, scope='module')
def setup_module(session_mocker) -> None:
    """Fixture for a mock experiment."""
    session_mocker.patch('drytorch.core.register.register_actor')
    return


class SimpleCaller(ModelCaller):
    """Simplest concrete implementation for ModelCaller."""

    @override
    def __call__(self) -> None:
        super().__call__()
        return


class TestModelCaller:
    """Tests for the ModelCaller class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_register_actor = mocker.patch(
            'drytorch.core.register.register_actor'
        )
        return

    @pytest.fixture
    def caller_name(self) -> str:
        """Set up a test argument."""
        return 'Caller'

    @pytest.fixture
    def caller(self, mock_model, caller_name) -> ModelCaller:
        """Set up a test instance."""
        return SimpleCaller(mock_model, caller_name)

    def test_initialization(self, mock_model, caller, caller_name) -> None:
        """Test initialization with all parameters."""
        assert caller.model == mock_model
        assert caller.name == caller_name

    def test_name_property(self, caller, caller_name) -> None:
        """Test name property starts with the expected value."""
        assert caller.name.startswith(caller_name)

    def test_str(self, caller, mock_model, caller_name):
        """Test string representation of the class."""
        assert str(caller).startswith(caller_name)

    def test_call_registration(self, caller) -> None:
        """Test __call__ method registers model on first call."""
        caller()
        self.mock_register_actor.assert_called_once_with(caller, caller.model)


class TestModelRunner:
    """Tests for the ModelRunner class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_input = mocker.Mock()
        self.mock_target = mocker.Mock()
        self.mock_output = mocker.Mock()
        self.mock_apply_ops = mocker.patch(
            'drytorch.utils.apply_ops.apply_cpu_detach',
            return_value=self.mock_output,
        )
        self.mock_apply_to = mocker.patch('drytorch.utils.apply_ops.apply_to')
        self.mock_apply_to.side_effect = lambda x, device: x
        self.mock_iterate_batch = mocker.patch(
            'drytorch.core.log_events.IterateBatchEvent'
        )
        self.mock_repr_metrics = mocker.patch(
            'drytorch.lib.objectives.repr_metrics', return_value={'loss': 0.1}
        )
        return

    @pytest.fixture
    def runner(self, mock_model, mock_loader) -> ModelRunner:
        """Set up a test instance."""
        return ModelRunner(
            mock_model,
            loader=mock_loader,
        )

    def test_initialization(self, mock_model, mock_loader, runner) -> None:
        """Test initialization with all parameters."""
        assert runner.loader == mock_loader
        assert runner.outputs_list == []

    def test_get_batches(self, runner, mock_loader) -> None:
        """Test batch generation applies operations to the device."""
        batches = list(runner._get_batches())
        loaded = list(mock_loader)
        assert len(batches) == len(loaded)
        for batch in loaded:
            self.mock_apply_to.assert_any_call(batch, runner.model.device)

    def test_run_batch(self, mocker, runner) -> None:
        """Test batch processing runs forward and backwards."""
        mock_batch = (self.mock_input, self.mock_target)
        fwd = mocker.patch.object(
            runner, '_run_forward', return_value=self.mock_output
        )
        bwd = mocker.patch.object(runner, '_run_backward')
        result = runner._run_batch(mock_batch)
        assert result == self.mock_output
        fwd.assert_called_once_with(self.mock_input)
        bwd.assert_called_once_with(self.mock_output, self.mock_target)

    def test_run_forward(self, runner, mock_model) -> None:
        """Test forward pass."""
        mock_model.return_value = self.mock_output
        runner.mixed_precision = False
        result = runner._run_forward(self.mock_input)
        assert result == self.mock_output
        mock_model.assert_called_once()

    def test_run_epoch_without_storing_outputs(self, mocker, runner) -> None:
        """Test epoch run without storing outputs."""
        get_batch = mocker.patch.object(runner, '_get_batches')
        get_batch.return_value = 2 * [(self.mock_input, self.mock_target)]
        run_batch = mocker.patch.object(runner, '_run_batch')
        run_batch.return_value = [self.mock_output]
        _store = mocker.patch.object(runner, '_store')
        mock_pbar = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_pbar

        # Reset the objective mock to avoid counting the initialization call
        runner._run_epoch(store_outputs=False)
        assert runner.outputs_list == []

        self.mock_iterate_batch.assert_called_once()
        assert run_batch.call_count == 2
        assert mock_pbar.update.call_count == 2
        _store.assert_not_called()

    def test_run_epoch_with_storing_outputs(self, mocker, runner) -> None:
        """Test epoch run with storing outputs."""
        get_batch = mocker.patch.object(runner, '_get_batches')
        get_batch.return_value = [(self.mock_input, self.mock_target)]
        run_batch = mocker.patch.object(runner, '_run_batch')
        run_batch.return_value = [self.mock_output]
        _store = mocker.patch.object(runner, '_store')
        runner._run_epoch(store_outputs=True)
        _store.assert_called_once_with([self.mock_output])

    def test_outputs_list_cleared_on_epoch_run(self, mocker, runner) -> None:
        """Test that the output list is cleared at the start of each epoch."""
        runner.outputs_list = [self.mock_output]
        get_batch = mocker.patch.object(runner, '_get_batches')
        get_batch.return_value = [(self.mock_input, self.mock_target)]
        run_batch = mocker.patch.object(runner, '_run_batch')
        run_batch.return_value = [self.mock_output]
        _store = mocker.patch.object(runner, '_store')
        runner._run_epoch(store_outputs=False)
        assert runner.outputs_list == []

    def test_store_outputs(self, runner) -> None:
        """Test outputs are correctly stored if store_outputs = True."""
        test_object = object()
        runner._store(test_object)
        self.mock_apply_ops.assert_called_once_with(test_object)
        assert runner.outputs_list == [self.mock_output]

    @pytest.mark.parametrize(
        'warning',
        [
            exceptions.FuncNotApplicableError('wrong_func', 'wrong_type'),
            exceptions.NamedTupleOnlyError('wrong_type'),
        ],
    )
    def test_store_outputs_warning(self, mocker, runner, warning) -> None:
        """Test warning is raised if output cannot be stored."""
        mock_output = mocker.Mock()
        mock_apply_ops = mocker.patch(
            'drytorch.utils.apply_ops.apply_cpu_detach', side_effect=warning
        )

        with pytest.warns(exceptions.CannotStoreOutputWarning):
            runner._store(mock_output)

        mock_apply_ops.assert_called_once_with(mock_output)
        assert runner.outputs_list == []


class TestModelRunnerWithObjective:
    """Tests for the ModelRunner class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, mock_loss) -> None:
        """Set up the tests."""
        self.mock_target = mocker.Mock()
        self.mock_output = mocker.Mock()
        mocker.patch('drytorch.lib.runners.ModelRunner.__init__')
        mocker.patch('drytorch.lib.runners.ModelRunner._run_epoch')
        mocker.patch('drytorch.lib.runners.ModelRunner._run_backward')
        self.mock_repr_metrics = mocker.patch(
            'drytorch.lib.objectives.repr_metrics', return_value={'loss': 0.1}
        )
        self.mock_deepcopy = mocker.patch(
            'copy.deepcopy', return_value=mock_loss
        )
        return

    @pytest.fixture
    def runner(
        self, mock_model, mock_loader, mock_loss
    ) -> ModelRunnerWithObjective:
        """Set up a test instance."""
        return ModelRunnerWithObjective(
            mock_model,
            loader=mock_loader,
            objective=mock_loss,
        )

    def test_initialization(self, mock_loss, runner) -> None:
        """Test initialization with all parameters."""
        assert runner.objective == mock_loss
        self.mock_deepcopy.assert_called_once()

    def test_run_backward(self, runner) -> None:
        """Test backward pass updates the objective."""
        runner._run_backward(self.mock_output, self.mock_target)
        runner.objective.update.assert_called_once_with(
            self.mock_output, self.mock_target
        )

    def test_objective_reset_on_epoch_run(self, runner) -> None:
        """Test that the objective is reset at the start of each epoch."""
        runner.objective.reset.reset_mock()
        runner._run_epoch(store_outputs=False)
        runner.objective.reset.assert_called_once()


class TestModelRunnerWithLogs:
    """Tests for the ModelRunner class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, example_named_metrics) -> None:
        """Set up the tests."""
        mocker.patch('drytorch.lib.runners.ModelRunnerWithObjective._run_epoch')
        mocker.patch.object(
            ModelRunnerWithObjective,
            'computed_metrics',
            new_callable=mocker.PropertyMock,
            return_value=example_named_metrics,
        )
        self.mock_log_events_metrics = mocker.patch(
            'drytorch.core.log_events.MetricEvent'
        )
        return

    @pytest.fixture
    def runner(self, mock_model, mock_loader, mock_loss) -> ModelRunner:
        """Set up a test instance."""
        return ModelRunnerWithLogs(
            mock_model,
            loader=mock_loader,
            objective=mock_loss,
        )

    def test_run_epoch_without_storing_outputs(
        self, runner, example_named_metrics
    ) -> None:
        """Test epoch run logs metrics."""
        runner._run_epoch(False)
        self.mock_log_events_metrics.assert_called_once_with(
            model_name=runner.model.name,
            source_name=runner.name,
            epoch=runner.model.epoch,
            metrics=example_named_metrics,
        )
