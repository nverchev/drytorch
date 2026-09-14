"""Tests for the "runners" module."""

from typing import Protocol

import typing_extensions

from typing_extensions import override

import pytest

from drytorch.core import exceptions
from drytorch.lib.runners import (
    ModelRunner,
    ModelRunnerWithLogs,
    ModelRunnerWithObjective,
)


@pytest.fixture(autouse=True, scope='module')
def setup_module(module_mocker) -> None:
    """Fixture for a mock experiment."""
    module_mocker.patch('drytorch.core.registering.register_actor')
    return


class _TypingProtocol(Protocol):
    """Protocol from typing."""

    def first(self) -> None:
        """First method."""


class _ExtensionsProtocol(typing_extensions.Protocol):
    """Protocol from typing_extensions."""

    def second(self) -> None:
        """Second method."""


class _ExtendedRunner(ModelRunner):
    """Runner setting an attribute after the parent initialization."""

    extra: str

    def __init__(self, model, loader) -> None:
        """Initialize."""
        super().__init__(model, loader=loader)
        self.extra = 'extra'
        return


class _InheritedRunner(_ExtendedRunner):
    """Runner without its own initialization."""


class _FailingRunner(ModelRunner):
    """Runner whose initialization fails after the parent initialization."""

    def __init__(self, model, loader) -> None:
        """Initialize."""
        super().__init__(model, loader=loader)
        raise ValueError('failing initialization')


class _ProtocolRunner(ModelRunner, _TypingProtocol, _ExtensionsProtocol):
    """Runner inheriting protocols from typing and typing_extensions."""

    @override
    def first(self) -> None:
        return

    @override
    def second(self) -> None:
        return


class TestRegisteringMeta:
    """Tests for the metaclass registering the runners."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.registered_attributes: list[set[str]] = []
        self.mock_register_actor = mocker.patch(
            'drytorch.core.registering.register_actor',
            side_effect=lambda actor, _: self.registered_attributes.append(
                set(vars(actor))
            ),
        )
        return

    def test_registers_after_subclass_init(
        self, mock_model, mock_loader
    ) -> None:
        """Test the runner registers once, after the subclass initialization."""
        runner = _ExtendedRunner(mock_model, mock_loader)

        self.mock_register_actor.assert_called_once_with(runner, mock_model)
        assert 'extra' in self.registered_attributes[0]

    def test_registers_inherited_init(self, mock_model, mock_loader) -> None:
        """Test a subclass without initialization registers once."""
        runner = _InheritedRunner(mock_model, mock_loader)

        self.mock_register_actor.assert_called_once_with(runner, mock_model)

    def test_failed_init_does_not_register(
        self, mock_model, mock_loader
    ) -> None:
        """Test a runner is not registered when its initialization fails."""
        with pytest.raises(ValueError):
            _FailingRunner(mock_model, mock_loader)

        self.mock_register_actor.assert_not_called()

    def test_protocol_inheritance(self, mock_model, mock_loader) -> None:
        """Test a runner inheriting typing and typing_extensions protocols."""
        runner = _ProtocolRunner(mock_model, loader=mock_loader)

        self.mock_register_actor.assert_called_once_with(runner, mock_model)


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
            'drytorch.lib.objectives.compute_metrics',
            return_value={'loss': 0.1},
        )
        self.mock_register_actor = mocker.patch(
            'drytorch.core.registering.register_actor'
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
        assert runner.model == mock_model
        assert runner.loader == mock_loader
        assert runner.outputs_list == []
        assert runner.outputs_list is runner._outputs_list

    def test_registration_on_creation(self, mock_model, runner) -> None:
        """Test the runner registers as an actor of its model when created."""
        self.mock_register_actor.assert_called_once_with(runner, mock_model)

    def test_no_registration_on_call(self, mocker, runner) -> None:
        """Test calling the runner does not register it again."""
        mocker.patch.object(runner, '_run_epoch')

        runner()

        self.mock_register_actor.assert_called_once()

    def test_name_property(self, mock_model, mock_loader) -> None:
        """Test name property starts with the given name."""
        runner = ModelRunner(mock_model, 'Runner', loader=mock_loader)

        assert runner.name.startswith('Runner')

    def test_repr(self, mock_model, runner) -> None:
        """Test the representation contains the runner and model names."""
        assert repr(runner) == f'{runner.name}({mock_model.name})'

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
        mock_batch_event = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_batch_event

        # Reset the objective mock to avoid counting the initialization call
        runner._run_epoch(store_outputs=False)
        assert runner.outputs_list == []

        self.mock_iterate_batch.assert_called_once()
        assert run_batch.call_count == 2
        assert mock_batch_event.update.call_count == 2
        _store.assert_not_called()

    def test_run_epoch_distributed_refreshes_metrics(
        self, mocker, runner
    ) -> None:
        """Test distributed epoch run refreshes metrics after sync."""
        runner._is_distributed = True
        mocker.patch('torch.distributed.get_world_size', return_value=2)
        get_batch = mocker.patch.object(runner, '_get_batches')
        get_batch.return_value = [(self.mock_input, self.mock_target)]
        mocker.patch.object(
            runner, '_run_batch', return_value=[self.mock_output]
        )
        mocker.patch.object(runner, '_sync')
        mocker.patch.object(runner, '_gather_stored_outputs')
        mocker.patch.object(
            runner, '_compute_metrics', return_value={'loss': 0.1}
        )
        mock_batch_event = mocker.Mock()
        self.mock_iterate_batch.return_value = mock_batch_event

        runner._run_epoch(store_outputs=False)

        mock_batch_event.update.assert_called_once_with({'loss': 0.1}, 2)
        mock_batch_event.refresh.assert_called_once_with({'loss': 0.1})

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
        runner._outputs_list.append(self.mock_output)
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

        def _mock_init(instance, model, *_, **__):
            instance.model = model
            instance._is_distributed = True
            return

        self.mock_target = mocker.Mock()
        self.mock_output = mocker.Mock()
        mocker.patch(
            'drytorch.lib.runners.ModelRunner.__init__', new=_mock_init
        )
        mocker.patch('drytorch.lib.runners.ModelRunner._run_epoch')
        mocker.patch('drytorch.lib.runners.ModelRunner._run_backward')
        mocker.patch('drytorch.lib.runners.ModelRunner._run_backward')
        self.mock_repr_metrics = mocker.patch(
            'drytorch.lib.objectives.compute_metrics',
            return_value={'loss': 0.1},
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
            '_compute_metrics',
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
