"""Module containing gradient operations."""

import abc
import copy
import math

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TypeAlias

import torch

from typing_extensions import override

from drytorch.core import protocols as p
from drytorch.core.protocols import GradientOpProtocol


ClipFunction: TypeAlias = Callable[[float, float], float]


def _validate_threshold(threshold: float) -> None:
    if threshold <= 0:
        raise ValueError('Gradient threshold must be positive.')


class GradParamNormalizer(p.GradientOpProtocol):
    """Strategy that normalizes each parameter's gradient to unit norm."""

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Normalize gradients to unit norm in-place."""
        for param in params:
            grad = param.grad
            if grad is None:
                continue
            norm = grad.norm(2)
            if norm:
                param.grad = grad / norm


class GradZScoreNormalizer(p.GradientOpProtocol):
    """Gradient normalizing strategy using Z-score normalization."""

    def __init__(self, eps: float = 1e-8) -> None:  # type: ignore
        """Constructor.

        Args:
            eps: Small constant for numerical stability.
        """
        self._eps: float = eps

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Normalize gradients using Z-score in-place."""
        for param in params:
            grad = param.grad
            if grad is None:
                continue
            param.grad = (grad - grad.mean()) / (grad.std() + self._eps)


class ClipOperation(p.GradientOpProtocol, abc.ABC):
    """Abstract base class for gradient operations."""

    @abc.abstractmethod
    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Apply the gradient operation to the given parameters."""


class GradNormClipper(ClipOperation):
    """Gradient norm clipping strategy.

    Attributes:
        threshold: Maximum norm value of the clipped gradients.
    """

    def __init__(self, threshold: float = 1) -> None:
        """Constructor.

        Args:
            threshold: Maximum norm value of the clipped gradients.
        """
        super().__init__()
        _validate_threshold(threshold)
        self.threshold: float = threshold

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Clip gradients by norm in-place."""
        torch.nn.utils.clip_grad_norm_(params, max_norm=self.threshold)


class GradValueClipper(ClipOperation):
    """Gradient value clipping strategy.

    Attributes:
            threshold: Maximum absolute value of the clipped gradients.
    """

    def __init__(self, threshold: float = 1) -> None:
        """Constructor.

        Args:
            threshold: Maximum absolute value of the clipped gradients.
        """
        super().__init__()
        _validate_threshold(threshold)
        self.threshold: float = threshold

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Clip gradients by value in-place."""
        torch.nn.utils.clip_grad_value_(params, clip_value=self.threshold)


def reciprocal_clipping(zt: float, z_thresh: float) -> float:
    """Reciprocal clipping as recommended in https://arxiv.org/pdf/2504.02507.

    Instead of clipping to the threshold value, reciprocal clipping decreases
    the norm of the gradient even further as the spike gets larger.

    Args:
        zt: the Z-statistic or ratio of the current gradient norm.
        z_thresh: the threshold for the z-statistic values.

    Returns:
        Renormalization factor (between 0 and 1).
    """
    return z_thresh**2 / zt


def mean_clipping(zt: float, z_thresh: float) -> float:
    """Clip to the mean value (effectively setting gradient to running mean).

    Args:
        zt: the Z-statistic or ratio of the current gradient norm.
        z_thresh: the threshold for the z-statistic values.

    Returns:
        Renormalization factor of 0 (clips to mean).
    """
    _not_used = zt, z_thresh
    return 0.0


def max_clipping(zt: float, z_thresh: float) -> float:
    """Standard clipping to the threshold value.

    Args:
        zt: the Z-statistic or ratio of the current gradient norm.
        z_thresh: the threshold for the z-statistic values.

    Returns:
        The threshold value as the renormalization factor.
    """
    _not_used = zt
    return z_thresh


class ClippingCriterion(abc.ABC):
    """Criteria that detects when to clip snd determines the clipping value."""

    @abc.abstractmethod
    def should_clip(self, value: float) -> bool:
        """Determine whether to clip gradients based on the current value.

        Args:
            value: current gradient norm or value to evaluate.

        Returns:
            True if gradients should be clipped, False otherwise.
        """

    @abc.abstractmethod
    def get_clip_value(self, value: float) -> float:
        """Calculate the clipping threshold based on current statistics.

        Args:
            value: Current gradient norm or value.

        Returns:
            The value to clip gradients to.
        """

    def update(self, value: float) -> None:
        """Update internal statistics with a new observed value.

        Args:
            value: new gradient norm or value to incorporate.
        """
        _unused = value
        return

    def set_statistics(self, mean: float, variance: float = 0.0) -> None:
        """Initialize statistics from warmup data.

        Args:
            mean: mean value from the warmup period.
            variance: variance from the warmup period (if applicable).
        """
        _unused = mean, variance
        return

    def reset(self) -> None:
        """Reset all internal statistics to initial state."""
        return


class EMACriterion(ClippingCriterion):
    """Clipping criterion based on Exponential Moving Average.

    It uses only the running mean of gradient norms to detect outliers.
    It clips when the current norm exceeds the mean by a factor of r_thresh.

    Attributes:
        alpha: exponential moving average decay factor.
        r_thresh: ratio threshold between current_norm and mean_norm.
        clipping_function: function to determine clipping behavior.
    """

    def __init__(
        self,
        alpha: float = 0.98,
        r_thresh: float = 1.05,
        clipping_function: ClipFunction = max_clipping,
    ):
        """Constructor.

        Args:
            alpha: exponential moving average decay factor.
            r_thresh: ratio threshold between current_norm and mean_norm.
            clipping_function: function to determine clipping behavior.
        """
        self.alpha: float = alpha
        self.r_thresh: float = r_thresh
        self.clipping_function: ClipFunction = clipping_function
        self._mu_t: float = 0.0
        _validate_threshold(r_thresh)
        return

    @override
    def should_clip(self, value: float) -> bool:
        if self._mu_t == 0.0:
            return False

        return value / self._mu_t > self.r_thresh

    @override
    def get_clip_value(self, value: float) -> float:
        if self._mu_t == 0.0:
            return value

        ratio = value / self._mu_t
        clipping_factor = self.clipping_function(ratio, self.r_thresh)
        return self._mu_t * clipping_factor

    @override
    def update(self, value: float) -> None:
        self._mu_t = self.alpha * self._mu_t + (1 - self.alpha) * value
        return

    @override
    def set_statistics(self, mean: float, variance: float = 0.0) -> None:
        self._mu_t = mean
        return

    @override
    def reset(self) -> None:
        self._mu_t = 0.0
        return


class ZStatCriterion(ClippingCriterion):
    """Clipping criterion based on the Z-statistic.

    Tracks both mean and variance using exponential moving averages. The
    clipping threshold is on the Z-score (standardized deviation). See also
    https://arxiv.org/pdf/2504.02507.

    Attributes:
            alpha: exponential moving average decay factor (0 < alpha < 1).
            z_thresh: Z-score threshold between !z_score| and z_thresh.
            clipping_function: function to determine clipping behavior.
    """

    def __init__(
        self,
        alpha: float = 0.97,
        z_thresh: float = 2.5,
        clipping_function: ClipFunction = reciprocal_clipping,
        eps: float = 1e-06,
    ):
        """Constructor.

        Args:
            alpha: exponential moving average decay factor (0 < alpha < 1).
            z_thresh: threshold for the Z-score.
            clipping_function: function to determine clipping behavior.
            eps: small constant for numerical stability.
        """
        self.alpha: float = alpha
        self.z_thresh: float = z_thresh
        self.clipping_function: ClipFunction = clipping_function
        self._eps: float = eps
        self._mu_t: float = 0.0
        self._v_t: float = 1.0
        _validate_threshold(z_thresh)
        return

    @override
    def should_clip(self, value: float) -> bool:
        """Check if the Z-score exceeds the threshold."""
        if self._mu_t == 0.0:
            return False

        z_score = (value - self._mu_t) / (math.sqrt(self._v_t) + self._eps)
        return abs(z_score) > self.z_thresh

    @override
    def get_clip_value(self, value: float) -> float:
        if self._mu_t == 0.0:
            return value

        z_score = (value - self._mu_t) / (math.sqrt(self._v_t) + self._eps)
        if abs(z_score) <= self.z_thresh:
            return value

        new_z_score = self.clipping_function(abs(z_score), self.z_thresh)
        return self._mu_t + new_z_score * math.sqrt(self._v_t)

    @override
    def update(self, value: float) -> None:
        variance = (value - self._mu_t) ** 2
        self._v_t = self.alpha * self._v_t + (1 - self.alpha) * variance
        self._mu_t = self.alpha * self._mu_t + (1 - self.alpha) * value
        return

    @override
    def set_statistics(self, mean: float, variance: float = 0.0) -> None:
        self._mu_t = mean
        if variance > 0:
            self._v_t = variance
        return

    @override
    def reset(self) -> None:
        self._mu_t = 0.0
        self._v_t = 1.0
        return


class StatsCollector:
    """Constructor.

    Attributes:
            max_samples: the number of collected samples for completion.
            active: whether the collector is currently in use.
    """

    def __init__(self, max_samples: int):
        """Initialize warmup handler.

        Args:
            max_samples: the number of collected samples for completion.
        """
        self.max_samples: int = max_samples
        self._data: list[float] = []
        self.active = True
        return

    def __len__(self) -> int:
        """Return the number of collected warmup samples."""
        return len(self._data)

    @property
    def mean(self) -> float:
        """Calculate mean of collected warmup samples."""
        if not self._data:
            return 0.0
        return sum(self._data) / len(self._data)

    @property
    def variance(self) -> float:
        """Calculate variance of collected samples."""
        if len(self._data) <= 1:
            return 1.0

        mean_val = self.mean
        variance_sum = sum((x - mean_val) ** 2 for x in self._data)
        return variance_sum / (len(self._data) - 1)

    def is_complete(self) -> bool:
        """Check if the collection is complete."""
        completed = len(self) >= self.max_samples
        if completed:
            self.active = False
        return completed

    def append(self, value: float) -> None:
        """Add a new datum to the collection."""
        if len(self) < self.max_samples:
            self._data.append(value)
        return

    def reset(self) -> None:
        """Reset the collection."""
        self._data.clear()
        self.active = True
        return


class HistClipper(ClipOperation):
    """Global gradient clipping strategy that uses previous gradient statistics.

    The gradients' norm is renormalized according to a clipping criterion.

    Attributes:
        criterion: the clipping criterion to determine when and how to clip.
        warmup_clip_strategy: the clipping strategy used during warmup.
        n_warmup_steps: the number of warmup steps to collect initial stats.
    """

    _default_criterion: ZStatCriterion = ZStatCriterion()
    _default_grad_op: GradNormClipper = GradNormClipper()

    def __init__(
        self,
        criterion: ClippingCriterion = _default_criterion,
        warmup_clip_strategy: p.GradientOpProtocol = _default_grad_op,
        n_warmup_steps: int = 20,
    ) -> None:
        """Constructor.

        Args:
            criterion: the clipping criterion to determine when and how to clip.
            warmup_clip_strategy: the clipping strategy used during warmup.
            n_warmup_steps: the number of warmup steps to collect initial stats.
        """
        super().__init__()
        self.criterion: ClippingCriterion = criterion
        self.warmup_clip_strategy: GradientOpProtocol = warmup_clip_strategy
        self.n_warmup_steps: int = n_warmup_steps
        self._warmup_handler: StatsCollector = StatsCollector(n_warmup_steps)
        return

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Apply global gradient clipping.

        Args:
            params: model parameters to clip.

        Side Effects:
            Modifies gradients in-place if clipping is applied.
        """
        # needed to allow multiple iterations
        params_list = list(params)
        squared_norms = [
            (param.grad**2).sum().item()
            for param in params_list
            if param.grad is not None
        ]
        global_norm = math.sqrt(sum(squared_norms))
        if self._warmup_handler.active:
            if not self._warmup_handler.is_complete():
                self._warmup_handler.append(global_norm)
                self.warmup_clip_strategy(params_list)
                return
            else:
                self.criterion.set_statistics(
                    self._warmup_handler.mean, self._warmup_handler.variance
                )

        if self.criterion.should_clip(global_norm):
            clip_value = self.criterion.get_clip_value(global_norm)
            torch.nn.utils.clip_grad_norm_(params_list, clip_value)

        self.criterion.update(global_norm)
        return

    def reset(self):
        """Reset the state."""
        self._warmup_handler.reset()
        self.criterion.reset()
        return


class ParamHistClipper(ClipOperation):
    """Gradient clipping strategy that keeps per-parameter statistics.

    The gradients' norm is renormalized according to a clipping criterion.

    Attributes:
        criterion: the clipping criterion to determine when and how to clip.
        warmup_clip_strategy: the clipping strategy used during warmup.
        n_warmup_steps: the number of warmup steps to collect initial stats.
    """

    _default_criterion: ZStatCriterion = ZStatCriterion()
    _default_grad_op: GradNormClipper = GradNormClipper()

    def __init__(
        self,
        criterion: ClippingCriterion = _default_criterion,
        warmup_clip_strategy: p.GradientOpProtocol = _default_grad_op,
        n_warmup_steps: int = 20,
    ) -> None:
        """Constructor.

        Args:
            criterion: the clipping criterion to determine when and how to clip.
            warmup_clip_strategy: the clipping strategy used during warmup.
            n_warmup_steps: the number of warmup steps to collect initial stats.
        """
        super().__init__()
        self.criterion: ClippingCriterion = criterion
        self.n_warmup_steps: int = n_warmup_steps
        self.warmup_clip_strategy: GradientOpProtocol = warmup_clip_strategy
        self._dict_criterion: defaultdict[int, ClippingCriterion] = defaultdict(
            lambda: copy.copy(criterion)
        )
        self._dict_warmup_handler = defaultdict[int, StatsCollector](
            lambda: StatsCollector(n_warmup_steps)
        )
        return

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Apply global gradient clipping.

        Args:
            params: Model parameters to clip.

        Side Effects:
            Modifies gradients in-place if clipping is applied.
        """
        for param in params:
            grad = param.grad
            if grad is None:
                continue
            grad_norm: float = grad.norm(2, dtype=float).item()
            param_id = id(param)
            warmup_handler = self._dict_warmup_handler[param_id]
            criterion = self._dict_criterion[param_id]
            if warmup_handler.active:
                if not warmup_handler.is_complete():
                    warmup_handler.append(grad_norm)
                    self.warmup_clip_strategy([param])
                    continue
                else:
                    criterion.set_statistics(
                        warmup_handler.mean, warmup_handler.variance
                    )
            if criterion.should_clip(grad_norm):
                clip_value = self.criterion.get_clip_value(grad_norm)
                torch.nn.utils.clip_grad_norm_(param, clip_value)

            criterion.update(grad_norm)
        return

    def reset(self):
        """Reset the state."""
        self._dict_criterion.clear()
        self._dict_warmup_handler.clear()
        return
