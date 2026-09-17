"""Tests for the ``objectives`` module."""

from collections.abc import Callable

import torch

import pytest

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import aggregators
from drytorch.lib.objectives import (
    DictLoss,
    JoinedLoss,
    JoinedMetrics,
    Loss,
    LossWithMetrics,
    Metric,
    MetricCollection,
    check_device,
    compute_metrics,
    join,
)


_Tensor = torch.Tensor


@pytest.fixture(scope='module')
def metric_1() -> str:
    """Simple metric."""
    return 'Metric_1'


@pytest.fixture(scope='module')
def metric_2() -> str:
    """Another simple metric."""
    return 'Metric_2'


@pytest.fixture(scope='module')
def metric_fun_1(
    metric_1: str,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Simple metric fun."""
    return {metric_1: lambda x, y: x}


@pytest.fixture(scope='module')
def metric_fun_2(
    metric_2: str,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Another simple metric fun."""
    return {metric_2: lambda x, y: y}


@pytest.fixture(scope='module')
def extra_fn() -> Callable[
    [torch.Tensor, torch.Tensor], dict[str, torch.Tensor]
]:
    """A function returning multiple named metric values."""
    return lambda x, y: {'extra': x + y}


@pytest.fixture(scope='module')
def multi_fn() -> Callable[
    [torch.Tensor, torch.Tensor], dict[str, torch.Tensor]
]:
    """A function returning multiple named metric values."""
    return lambda x, y: {'primary': x, 'auxiliary': y}


class TestMetricCollection:
    """Tests for MetricCollection."""

    @pytest.fixture(scope='class')
    def metrics(self, metric_fun_1, metric_fun_2) -> MetricCollection:
        """Set up a MetricCollection instance with simple metric functions."""
        metric_fun_dict = metric_fun_1 | metric_fun_2
        return MetricCollection(**metric_fun_dict)

    def test_calculate(self, metric_1, metric_2, metrics) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}

        assert metrics.calculate(simple_outputs, simple_targets) == expected

    def test_update_compute_and_reset(
        self, metric_1, metric_2, metrics
    ) -> None:
        """Test it stores, reduces, and resets metrics correctly."""
        simple_outputs_1 = torch.tensor(1)
        simple_targets_1 = torch.tensor(0)
        simple_outputs_2 = torch.tensor(3)
        simple_targets_2 = torch.tensor(2)

        metrics.update(simple_outputs_1, simple_targets_1)
        metrics.update(simple_outputs_2, simple_targets_2)
        expected = {metric_1: torch.tensor(2), metric_2: torch.tensor(1)}

        assert metrics.compute() == expected

        metrics.reset()
        with pytest.warns(exceptions.ComputedBeforeUpdatedWarning):
            assert {} == metrics.compute()

        metrics.update(simple_outputs_1, simple_targets_1)
        expected = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}

        assert metrics.compute() == expected

    def test_compute_uses_cache(self, mocker, metric_1, metric_fun_1) -> None:
        """Test compute reuses the values until the state changes."""
        metrics = MetricCollection(**metric_fun_1)
        metrics.update(torch.tensor(1.0), torch.tensor(0.0))
        spy = mocker.spy(metrics, '_compute')

        metrics.compute()
        metrics.compute()
        metrics.update(torch.tensor(3.0), torch.tensor(0.0))
        after_update = metrics.compute()

        assert spy.call_count == 2
        assert after_update == {metric_1: torch.tensor(2.0)}

    def test_merge_state(self, metric_1, metric_fun_1) -> None:
        """Test merging states averages over the updates of both objectives."""
        metrics = MetricCollection(**metric_fun_1)
        other = MetricCollection(**metric_fun_1)
        metrics.update(torch.tensor(1.0), torch.tensor(0.0))
        other.update(torch.tensor(3.0), torch.tensor(0.0))
        metrics.compute()  # fills the cache

        metrics.merge_state(other)

        assert metrics.compute() == {metric_1: torch.tensor(2.0)}
        assert other.compute() == {metric_1: torch.tensor(3.0)}

    def test_merge_state_different_accumulators(self, metric_fun_1) -> None:
        """Test merging objectives with different accumulators raises."""

        class OtherAccumulator(aggregators.TorchMeanAccumulator):
            pass

        class OtherCollection(MetricCollection):
            @classmethod
            def _get_accumulator_cls(
                cls,
            ) -> type[aggregators.TorchMeanAccumulator]:
                return OtherAccumulator

        metrics = MetricCollection(**metric_fun_1)
        other = OtherCollection(**metric_fun_1)

        with pytest.raises(exceptions.AccumulatorMismatchError):
            metrics.merge_state(other)

    def test_or_repeated_metrics(self, metric_1, metric_fun_1) -> None:
        """Test it raises RepeatedMetricsError for a name with two functions."""
        col1 = MetricCollection(**metric_fun_1)
        col2 = MetricCollection(**{metric_1: lambda x, y: y})

        with pytest.raises(exceptions.RepeatedMetricsError):
            _ = col1 | col2

    def test_or_same_function(self, metric_fun_1) -> None:
        """Test joining a collection with itself keeps one copy."""
        collection = MetricCollection(**metric_fun_1)

        joined = collection | collection

        assert joined.named_fn == collection.named_fn

    def test_or(self, metric_1, metric_2, metrics) -> None:
        """Test | works as a union operator."""
        new_metric_fun_dict = {'NewMetric': lambda x, y: torch.tensor(0.5)}
        new_metrics = MetricCollection(**new_metric_fun_dict)

        combined_metrics = metrics | new_metrics

        expected_keys = {metric_1, metric_2, 'NewMetric'}
        assert set(combined_metrics.named_fn.keys()) == expected_keys

        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)
        combined_metrics.update(simple_outputs, simple_targets)

        expected = {
            metric_1: torch.tensor(1),
            metric_2: torch.tensor(0),
            'NewMetric': torch.tensor(0.5),
        }
        assert combined_metrics.compute() == expected

    def test_calculate_with_fn(self, metric_1, multi_fn, metric_fun_1) -> None:
        """Test it calculates both multi-output and named-fn metrics."""
        mc = MetricCollection(multi_fn, **metric_fun_1)
        result = mc.calculate(torch.tensor(1.0), torch.tensor(2.0))
        expected = {
            metric_1: torch.tensor(1.0),
            'primary': torch.tensor(1.0),
            'auxiliary': torch.tensor(2.0),
        }
        assert result == expected

    def test_or_merges_fn(self, extra_fn, multi_fn) -> None:
        """Test | merges multi-output functions from both collections."""
        mc1 = MetricCollection(multi_fn)
        mc2 = MetricCollection(extra_fn)
        combined = mc1 | mc2

        assert len(combined.fn) == 2

        result = combined.calculate(torch.tensor(1.0), torch.tensor(2.0))
        expected = {
            'primary': torch.tensor(1.0),
            'auxiliary': torch.tensor(2.0),
            'extra': torch.tensor(3.0),
        }
        assert result == expected


class TestMetric:
    """Tests for Metric."""

    @pytest.fixture(scope='class')
    def metric(self, metric_1, metric_fun_1) -> Metric:
        """Set up a Metric instance with a simple metric function."""
        self.simple_fun = next(iter(metric_fun_1.values()))
        return Metric(self.simple_fun, name=metric_1, higher_is_better=True)

    def test_calculate(self, metric_1, metric) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)

        expected = {metric_1: torch.tensor(1)}

        assert metric.calculate(simple_outputs, simple_targets) == expected

    def test_or(self, metric_1, metric) -> None:
        """Test | works as a union operator."""
        new_metrics = Metric[_Tensor, _Tensor](
            lambda x, y: torch.tensor(0.5),
            name='NewMetric',
            higher_is_better=True,
        )

        combined_metrics = metric | new_metrics

        expected_keys = {metric_1, 'NewMetric'}
        assert set(combined_metrics.named_fn.keys()) == expected_keys

        simple_outputs = torch.tensor(1)
        simple_targets = torch.tensor(0)
        combined_metrics.update(simple_outputs, simple_targets)

        expected = {metric_1: torch.tensor(1), 'NewMetric': torch.tensor(0.5)}
        assert combined_metrics.compute() == expected


class TestLossWithMetrics:
    """Tests for LossWithMetrics."""

    @pytest.fixture(scope='class')
    def example_metric_results(
        self, metric_1, metric_2
    ) -> dict[str, torch.Tensor]:
        """A possible calculated value for metrics."""
        return {
            metric_1: torch.tensor(1),
            metric_2: torch.tensor(2),
        }

    @pytest.fixture(scope='class')
    def loss_1(self, metric_1, metric_fun_1) -> LossWithMetrics:
        """Set up a base instance (as defined by the Loss subclass)."""
        # formula corresponds to what the formula components should look like
        return LossWithMetrics(
            criterion=lambda x: x[metric_1],
            formula=f'[{metric_1}]',
            **metric_fun_1,
        )

    @pytest.fixture(scope='class')
    def loss_2(self, metric_2, metric_fun_2) -> LossWithMetrics:
        """Set up a second base instance (as defined by the Loss subclass)."""
        return LossWithMetrics(
            criterion=lambda x: x[metric_2],
            formula=f'[{metric_2}]',
            **metric_fun_2,
        )

    @pytest.fixture(scope='class')
    def composed_loss_1(self, loss_1) -> LossWithMetrics:
        """Set up a LossWithMetrics instance with simple arguments."""
        return 2 * loss_1

    @pytest.fixture(scope='class')
    def composed_loss_2(self, loss_2) -> LossWithMetrics:
        """Set up a LossWithMetrics instance with simple arguments."""
        return 3 * loss_2

    def test_calculate(self, metric_1, composed_loss_1) -> None:
        """Test it calculates metrics correctly."""
        simple_outputs = torch.tensor(1.0)
        simple_targets = torch.tensor(0.0)
        expected = {
            'Combined Loss': torch.tensor(2.0),
            metric_1: torch.tensor(1.0),
        }
        assert (
            composed_loss_1.calculate(simple_outputs, simple_targets)
            == expected
        )

    def test_negate_loss(self, composed_loss_1, example_metric_results) -> None:
        """Test negation of a loss."""
        neg_loss = -composed_loss_1
        assert neg_loss.criterion(example_metric_results) == -2
        assert neg_loss.formula == '-(2 x [Metric_1])'

    def test_add_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test addition of two losses."""
        combined_loss = composed_loss_1 + composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 + 6
        assert combined_loss.formula == '2 x [Metric_1] + 3 x [Metric_2]'

    def test_subtract_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test subtraction of two losses."""
        combined_loss = composed_loss_1 - -composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 + 6
        assert combined_loss.formula == '2 x [Metric_1] - -(3 x [Metric_2])'

    def test_subtract_losses_same_direction(
        self, composed_loss_1, composed_loss_2
    ) -> None:
        """Test subtracting a loss with the same direction raises."""
        with pytest.raises(ValueError):
            _ = composed_loss_1 - composed_loss_2

    def test_add_losses_opposite_direction(
        self, composed_loss_1, composed_loss_2
    ) -> None:
        """Test adding a loss with the opposite direction raises."""
        with pytest.raises(ValueError):
            _ = composed_loss_1 + -composed_loss_2

    def test_direction_flips(self, composed_loss_1) -> None:
        """Test operations reversing the order of values flip the direction."""
        flipped = (
            -composed_loss_1,
            composed_loss_1**-1,
            -2 * composed_loss_1,
            composed_loss_1 / -2,
            3 - composed_loss_1,
            3 / composed_loss_1,
        )

        assert all(loss.higher_is_better for loss in flipped)

    def test_direction_kept(self, composed_loss_1) -> None:
        """Test operations keeping the order of values keep the direction."""
        kept = (
            composed_loss_1**2,
            2 * composed_loss_1,
            composed_loss_1 - 3,
            -3 / composed_loss_1,
        )

        assert not any(loss.higher_is_better for loss in kept)

    def test_multiply_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test multiplication of two losses."""
        combined_loss = composed_loss_1 * composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 * 6
        assert combined_loss.formula == '2 x [Metric_1] x 3 x [Metric_2]'

    def test_divide_losses(
        self, composed_loss_1, composed_loss_2, example_metric_results
    ) -> None:
        """Test division of two losses."""
        combined_loss = composed_loss_1 / -composed_loss_2
        assert combined_loss.criterion(example_metric_results) == 2 / -6
        expected = '2 x [Metric_1] / -(3 x [Metric_2])'
        assert combined_loss.formula == expected

    def test_combine_repeated_name(self) -> None:
        """Test combining losses whose metrics share a name raises."""
        first = Loss(lambda x, y: (x - y).abs(), name='L')
        second = Loss(lambda x, y: (x - y) ** 2, name='L')

        with pytest.raises(exceptions.RepeatedMetricsError):
            _ = first + second

    def test_combine_same_metric(self) -> None:
        """Test a loss can be combined with a loss sharing its metric."""
        loss = Loss(lambda x, y: (x - y).abs(), name='L')

        combined = loss + 2 * loss

        assert combined.forward(torch.tensor(3.0), torch.tensor(1.0)) == 6.0

    def test_formula_keeps_groups(self) -> None:
        """Test the formula of a product of sums keeps its parentheses."""
        l1, l2, l3, l4 = (
            Loss(lambda x, y: x, name=f'L{i}') for i in range(1, 5)
        )

        product = (l1 + l2) * (l3 + l4)

        assert product.formula == '([L1] + [L2]) x ([L3] + [L4])'
        assert (product + l1).formula == '([L1] + [L2]) x ([L3] + [L4]) + [L1]'

    def test_formula_parentheses_where_needed(self) -> None:
        """Test the formula has parentheses only where the order needs them."""
        a = Loss(lambda x, y: x, name='A')
        b = Loss(lambda x, y: x * y, name='B')
        c = Loss(lambda x, y: x + y, name='C', higher_is_better=True)
        d = Loss(lambda x, y: x - y, name='D', higher_is_better=True)

        formulas = [
            (a + b).formula,
            (a - (c + d)).formula,
            (-(a + b)).formula,
            ((a + b) ** 2).formula,
            ((-a) ** 2).formula,
            (-(a**2)).formula,
            (a * b + a).formula,
        ]

        assert formulas == [
            '[A] + [B]',
            '[A] - ([C] + [D])',
            '-([A] + [B])',
            '([A] + [B])^2',
            '(-[A])^2',
            '-[A]^2',
            '[A] x [B] + [A]',
        ]

    def test_loss_value_wins_name_clash(self) -> None:
        """Test the loss value is reported when a metric shares its name."""
        squared = Loss(lambda x, y: (x - y).abs(), name='Loss') ** 2

        values = squared.calculate(torch.tensor(3.0), torch.tensor(1.0))

        assert values['Loss'] == torch.tensor(4.0)

    def test_default_formula(self) -> None:
        """Test the default formula is the loss name in brackets."""
        loss = LossWithMetrics(criterion=lambda x: x['A'], A=lambda x, y: x)

        assert loss.formula == '[Loss]'


class TestLoss:
    """Tests for Loss."""

    @pytest.fixture(scope='class')
    def metric(self, metric_1, metric_fun_1) -> Metric:
        """Set up a Metric instance with a simple metric function."""
        self.simple_fun = next(iter(metric_fun_1.values()))
        return Metric(self.simple_fun, name=metric_1, higher_is_better=True)

    @pytest.fixture(scope='class')
    def example_metric_results(self, metric_1) -> dict[str, torch.Tensor]:
        """A possible calculated value for metrics."""
        return {metric_1: torch.tensor(2.0)}

    @pytest.fixture(scope='class')
    def loss_1(self, metric_1, metric_fun_1) -> Loss:
        """Set up a Loss instance with simple arguments."""
        return Loss(next(iter(metric_fun_1.values())), metric_1)

    def test_add_float(self, loss_1, example_metric_results) -> None:
        """Test addition by float."""
        combined_loss = loss_1 + 3
        assert combined_loss.criterion(example_metric_results) == 2 + 3
        assert combined_loss.formula == '[Metric_1] + 3'

    def test_subtract_float(self, loss_1, example_metric_results) -> None:
        """Test subtraction by float."""
        combined_loss = loss_1 - 3
        assert combined_loss.criterion(example_metric_results) == 2 - 3
        assert combined_loss.formula == '[Metric_1] - 3'

    def test_multiply_float(self, loss_1, example_metric_results) -> None:
        """Test multiplication by float."""
        combined_loss = loss_1 * 3
        assert combined_loss.criterion(example_metric_results) == 2 * 3
        assert combined_loss.formula == '[Metric_1] x 3'

    def test_divide_float(self, loss_1, example_metric_results) -> None:
        """Test division by float."""
        combined_loss = loss_1 / 3
        assert combined_loss.criterion(example_metric_results) == 2 / 3
        assert combined_loss.formula == '[Metric_1] / 3'

    def test_float_add(self, loss_1, example_metric_results) -> None:
        """Test addition to float."""
        combined_loss = 3 + loss_1
        assert combined_loss.criterion(example_metric_results) == 3 + 2
        assert combined_loss.formula == '3 + [Metric_1]'

    def test_float_subtract(self, loss_1, example_metric_results) -> None:
        """Test subtraction to float."""
        combined_loss = 3 - loss_1
        assert combined_loss.criterion(example_metric_results) == 3 - 2
        assert combined_loss.formula == '3 - [Metric_1]'

    def test_float_multiply(self, loss_1, example_metric_results) -> None:
        """Test multiplication to float."""
        combined_loss = 3 * loss_1
        assert combined_loss.criterion(example_metric_results) == 3 * 2
        assert combined_loss.formula == '3 x [Metric_1]'

    def test_float_divide(self, loss_1, example_metric_results) -> None:
        """Test division to float."""
        combined_loss = 3 / loss_1
        assert combined_loss.criterion(example_metric_results) == 3 / 2
        assert combined_loss.formula == '3 / [Metric_1]'

    def test_positive_exp(self, loss_1, example_metric_results) -> None:
        """Test exponentiation by positive float."""
        combined_loss = loss_1**2
        assert combined_loss.criterion(example_metric_results) == 2**2
        assert combined_loss.formula == '[Metric_1]^2'

    def test_negative_exp(self, loss_1, example_metric_results) -> None:
        """Test exponentiation by negative float."""
        combined_loss = loss_1**-2
        assert combined_loss.criterion(example_metric_results) == 2 ** (-2)
        assert combined_loss.formula == '[Metric_1]^-2'


class TestDictLoss:
    """Tests for DictLoss."""

    @pytest.fixture(scope='class')
    def loss_name(self) -> str:
        """Name of the loss key in the multi-output fn dict."""
        return 'primary'

    @pytest.fixture(scope='class')
    def dict_loss(self, multi_fn, loss_name) -> DictLoss:
        """A DictLoss wrapping the module-level multi-output fn."""
        return DictLoss(multi_fn, name=loss_name)

    def test_calculate(self, loss_name, dict_loss) -> None:
        """Test it returns all dict metrics including the loss key."""
        outputs = torch.tensor(3.0)
        targets = torch.tensor(1.0)
        result = dict_loss.calculate(outputs, targets)
        expected = {
            loss_name: torch.tensor(3.0),
            'auxiliary': torch.tensor(1.0),
        }
        assert result == expected

    def test_forward(self, dict_loss) -> None:
        """Test it returns the loss tensor identified by name."""
        outputs = torch.tensor(3.0)
        targets = torch.tensor(1.0)
        assert dict_loss.forward(outputs, targets) == torch.tensor(3.0)

    def test_update_compute_and_reset(self, loss_name, dict_loss) -> None:
        """Test it stores, reduces, and resets all metrics correctly."""
        dict_loss.reset()
        dict_loss.update(torch.tensor(1.0), torch.tensor(3.0))
        dict_loss.update(torch.tensor(3.0), torch.tensor(1.0))
        expected = {
            loss_name: torch.tensor(2.0),
            'auxiliary': torch.tensor(2.0),
        }
        assert dict_loss.compute() == expected

        dict_loss.reset()
        with pytest.warns(exceptions.ComputedBeforeUpdatedWarning):
            assert {} == dict_loss.compute()

    def test_arithmetic_propagates_fn(self, multi_fn, loss_name) -> None:
        """Test that arithmetic operations preserve multi-output fn metrics."""
        dl = DictLoss(multi_fn, name=loss_name)
        scaled = 2 * dl
        result = scaled.calculate(torch.tensor(3.0), torch.tensor(1.0))
        assert 'auxiliary' in result
        assert result['auxiliary'] == torch.tensor(1.0)


class TestJoinedMetrics:
    """Tests for JoinedMetrics."""

    @pytest.fixture(scope='class')
    def joined(self, metric_fun_1, metric_fun_2) -> JoinedMetrics:
        """Set up a JoinedMetrics instance from two MetricCollections."""
        return JoinedMetrics(
            MetricCollection(**metric_fun_1),
            MetricCollection(**metric_fun_2),
        )

    def test_update_compute_and_reset(self, metric_1, metric_2, joined) -> None:
        """Test it stores, reduces, and resets metrics from both objectives."""
        joined.reset()
        joined.update(torch.tensor(1), torch.tensor(0))
        joined.update(torch.tensor(3), torch.tensor(2))
        expected_before = {metric_1: torch.tensor(2), metric_2: torch.tensor(1)}
        computed_before = joined.compute()

        joined.reset()
        with pytest.warns(exceptions.ComputedBeforeUpdatedWarning):
            computed_reset = joined.compute()

        joined.update(torch.tensor(1), torch.tensor(0))
        computed_after = joined.compute()
        expected_after = {metric_1: torch.tensor(1), metric_2: torch.tensor(0)}
        assert computed_before == expected_before
        assert computed_reset == {}
        assert computed_after == expected_after

    def test_single_value_named_after_class(
        self, mocker, metric_1, metric_fun_1
    ) -> None:
        """Test a single computed value is named after its objective class."""
        other = mocker.create_autospec(p.ObjectiveProtocol, instance=True)
        other.compute.return_value = torch.tensor(2.0)
        joined = JoinedMetrics(MetricCollection(**metric_fun_1), other)

        joined.update(torch.tensor(1), torch.tensor(0))
        computed = joined.compute()

        assert computed == {
            metric_1: torch.tensor(1),
            other.__class__.__name__: torch.tensor(2.0),
        }

    def test_repeated_names(self, metric_fun_1) -> None:
        """Test compute raises RepeatedMetricsError for shared names."""
        joined = JoinedMetrics(
            MetricCollection(**metric_fun_1), MetricCollection(**metric_fun_1)
        )
        joined.update(torch.tensor(1), torch.tensor(0))

        with pytest.raises(exceptions.RepeatedMetricsError):
            joined.compute()

    def test_sync(self, mocker, metric_fun_1) -> None:
        """Test sync syncs the objectives supporting it and warns otherwise."""
        syncing = MetricCollection(**metric_fun_1)
        mock_sync = mocker.patch.object(syncing, 'sync')
        not_syncing = mocker.create_autospec(p.ObjectiveProtocol, instance=True)
        joined = JoinedMetrics(syncing, not_syncing)

        with pytest.warns(exceptions.ObjectiveSyncWarning):
            joined.sync()

        mock_sync.assert_called_once()


class TestJoinedLoss:
    """Tests for JoinedLoss."""

    @pytest.fixture(scope='class')
    def loss_metric(self) -> str:
        """Name of the loss metric."""
        return 'Loss'

    @pytest.fixture
    def joined(self, loss_metric, metric_fun_2) -> JoinedLoss:
        """Set up a JoinedLoss from a Loss and a MetricCollection."""
        loss = Loss(lambda x, y: (x - y).abs(), name=loss_metric)
        metric = MetricCollection(**metric_fun_2)
        return JoinedLoss(loss, metric)

    def test_forward(self, joined) -> None:
        """Test it returns the loss value unchanged."""
        outputs = torch.tensor(3.0)
        targets = torch.tensor(1.0)
        assert joined.forward(outputs, targets) == torch.tensor(2.0)

    def test_forward_updates_metrics(
        self, loss_metric, metric_2, joined
    ) -> None:
        """Test forward also updates the joined metrics."""
        joined.forward(torch.tensor(3.0), torch.tensor(1.0))

        expected = {loss_metric: torch.tensor(2.0), metric_2: torch.tensor(1.0)}
        assert joined.compute() == expected

    def test_update_compute_and_reset(
        self, loss_metric, metric_2, joined
    ) -> None:
        """Test it stores, reduces, and resets metrics from both objectives."""
        joined.reset()

        joined.update(torch.tensor(3.0), torch.tensor(1.0))
        joined.update(torch.tensor(5.0), torch.tensor(1.0))
        expected = {loss_metric: torch.tensor(3.0), metric_2: torch.tensor(1.0)}
        assert joined.compute() == expected

        joined.reset()
        with pytest.warns(exceptions.ComputedBeforeUpdatedWarning):
            assert {} == joined.compute()

        joined.update(torch.tensor(3.0), torch.tensor(1.0))
        expected = {loss_metric: torch.tensor(2.0), metric_2: torch.tensor(1.0)}
        assert joined.compute() == expected

    def test_name(self, loss_metric, joined) -> None:
        """Test the name is the name of the loss."""
        assert joined.name == loss_metric

    def test_name_without_attribute(self, mocker) -> None:
        """Test the name is the loss class name when the loss has none."""
        loss = mocker.create_autospec(p.LossProtocol, instance=True)

        joined = JoinedLoss(loss)

        assert joined.name == loss.__class__.__name__


def test_join_merges_collections(metric_fun_1, metric_fun_2) -> None:
    """Test join merges metric collections into one collection."""
    first = MetricCollection(**metric_fun_1)
    second = MetricCollection(**metric_fun_2)

    joined = join(first, second)

    assert type(joined) is MetricCollection
    assert joined.named_fn == dict(first.named_fn) | dict(second.named_fn)


def test_join_later_loss_as_metrics(metric_fun_2) -> None:
    """Test a loss after the first objective joins as its metrics."""
    loss = 2 * Loss(lambda x, y: (x - y).abs(), name='Abs')

    joined = join(MetricCollection(**metric_fun_2), loss)

    assert type(joined) is MetricCollection
    assert set(joined.named_fn) == {'Abs', *metric_fun_2}


def test_join_wraps_other_objectives(mocker, metric_fun_1) -> None:
    """Test join wraps objectives that are not collections."""
    other = mocker.create_autospec(p.ObjectiveProtocol, instance=True)

    joined = join(MetricCollection(**metric_fun_1), other)

    assert isinstance(joined, JoinedMetrics)


def test_join_repeated_names(metric_1, metric_fun_1) -> None:
    """Test join raises RepeatedMetricsError for a name with two functions."""
    other = MetricCollection(**{metric_1: lambda x, y: y})

    with pytest.raises(exceptions.RepeatedMetricsError):
        join(MetricCollection(**metric_fun_1), other)


def test_join_loss_first(metric_fun_2) -> None:
    """Test a loss first gives a new loss with the added metrics."""
    loss = Loss(lambda x, y: (x - y).abs(), name='Abs')
    original_named_fn = dict(loss.named_fn)
    metric = MetricCollection(**metric_fun_2)

    joined = join(loss, metric)

    assert isinstance(joined, LossWithMetrics)
    assert metric.named_fn.items() <= joined.named_fn.items()
    assert loss.named_fn == original_named_fn
    assert (joined.name, joined.formula) == (loss.name, loss.formula)


def test_join_loss_first_wraps_other_objectives(mocker) -> None:
    """Test a loss first wraps objectives that are not collections."""
    loss = Loss(lambda x, y: (x - y).abs(), name='Abs')
    other = mocker.create_autospec(p.ObjectiveProtocol, instance=True)

    joined = join(loss, other)

    assert isinstance(joined, JoinedLoss)


def test_join_other_loss_first(mocker, metric_fun_1) -> None:
    """Test any objective following the loss protocol first gives a loss."""
    other_loss = mocker.create_autospec(p.LossProtocol, instance=True)

    joined = join(other_loss, MetricCollection(**metric_fun_1))

    assert isinstance(joined, JoinedLoss)


def test_join_loss_first_repeated_name() -> None:
    """Test a loss first raises RepeatedMetricsError for the loss name."""
    loss = Loss(lambda x, y: (x - y).abs(), name='Abs')
    metric = Metric(lambda x, y: x, name='Abs')

    with pytest.raises(exceptions.RepeatedMetricsError):
        join(loss, metric)


def test_check_device_passes(mocker) -> None:
    """Test check passes when metrics are on the correct device."""
    device = torch.device('cpu')
    mock_calculator = mocker.MagicMock(spec=p.ObjectiveProtocol)
    mock_calculator.compute.return_value = {
        'loss': torch.tensor([0.5], device=device),
        'accuracy': torch.tensor([0.95], device=device),
    }

    check_device(mock_calculator, device)

    mock_calculator.compute.assert_called_once()


def test_check_device_fails(mocker) -> None:
    """Test check raises error when metrics are on the wrong device."""
    mock_calculator = mocker.MagicMock(spec=p.ObjectiveProtocol)
    mock_calculator.compute.return_value = {
        'loss': torch.tensor([0.5], device='meta'),
    }

    with pytest.raises(exceptions.DeviceMismatchError):
        check_device(mock_calculator, torch.device('cpu'))


@pytest.mark.parametrize(
    'compute_return, class_name, expected',
    [
        # Case 1: Mapping of metrics
        (
            {'metric_1': torch.tensor(1), 'metric_2': torch.tensor(2)},
            None,
            {'metric_1': 1, 'metric_2': 2},
        ),
        # Case 2: Single tensor
        (
            torch.tensor(0.5),
            'metric_1',
            {'metric_1': 0.5},
        ),
    ],
)
def test_repr_metrics(mocker, compute_return, class_name, expected) -> None:
    """Test metric representation with various compute return values."""
    mock_calculator = mocker.MagicMock(spec=p.ObjectiveProtocol)
    mock_calculator.compute.return_value = compute_return

    if class_name:
        mock_calculator.__class__.__name__ = class_name

    # Call the function and assert the result
    result = compute_metrics(mock_calculator)

    assert result == expected


def test_repr_metrics_fail(mocker) -> None:
    """Test metric representation fails with invalid return."""
    mock_calculator = mocker.MagicMock(spec=p.ObjectiveProtocol)

    with pytest.raises(exceptions.ComputedMetricsTypeError):
        _ = compute_metrics(mock_calculator)
