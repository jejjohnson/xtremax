"""Equinox-based temporal point process operators.

These classes bundle an intensity specification with numerical-config
defaults (grid size for quadrature, for instance) and expose a rich,
self-contained API — log-likelihood, sampling, survival/hazard,
predictions, diagnostics — without any NumPyro dependency. They are
the neural_stpp-style primary surface for users doing MLE / variational
inference / neural intensity modelling.

A :class:`HomogeneousPoissonProcess` is fully parameterised by ``rate``
and ``observation_window`` — both stored as arrays, so the whole
module is a valid PyTree that ``optax`` can optimise through.

An :class:`InhomogeneousPoissonProcess` additionally holds a
``log_intensity_fn``. If that function is itself an ``eqx.Module``
(e.g. an MLP), its parameters become part of the PyTree automatically.
If it is a plain Python callable, mark it as a static field via
:meth:`InhomogeneousPoissonProcess.from_fn`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes.primitives import (
    compensator_curve,
    hpp_cumulative_intensity,
    hpp_exceedance_log_prob,
    hpp_hazard,
    hpp_intensity,
    hpp_inter_event_log_prob,
    hpp_log_prob,
    hpp_mean_residual_life,
    hpp_predict_count,
    hpp_return_period,
    hpp_sample,
    hpp_survival,
    ipp_cumulative_hazard,
    ipp_cumulative_intensity,
    ipp_hazard,
    ipp_intensity,
    ipp_inter_event_log_prob,
    ipp_log_prob,
    ipp_predict_count,
    ipp_sample_inversion,
    ipp_sample_thinning,
    ipp_survival,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
)


class PiecewiseConstantLogIntensity(eqx.Module):
    """Piecewise-constant log-intensity as an ``eqx.Module``.

    Storing ``bin_edges`` and ``log_rates`` as PyTree leaves (rather
    than captured in a closure) lets them be inspected, vmapped over,
    and optimised via ``eqx.filter_grad`` when used inside an
    :class:`InhomogeneousPoissonProcess`. Parameterising the *log*
    rates means the rate is always positive without a further
    constraint — useful for unconstrained gradient-based fitting.
    """

    bin_edges: Float[Array, ...]
    log_rates: Float[Array, ...]

    def __init__(
        self,
        bin_edges: ArrayLike,
        rates: ArrayLike | None = None,
        log_rates: ArrayLike | None = None,
    ) -> None:
        if (rates is None) == (log_rates is None):
            raise ValueError("Provide exactly one of `rates` or `log_rates`.")
        self.bin_edges = jnp.asarray(bin_edges)
        self.log_rates = (
            jnp.log(jnp.asarray(rates)) if log_rates is None else jnp.asarray(log_rates)
        )

    def __call__(self, t: Array) -> Array:
        # ``searchsorted`` on interior edges maps each ``t`` to a bin
        # index in ``[0, E-1]``; clamp so values on or beyond the right
        # edge still fall into the last bin rather than off the end.
        idx = jnp.searchsorted(self.bin_edges[1:], t, side="right")
        idx = jnp.clip(idx, 0, self.log_rates.shape[-1] - 1)
        return self.log_rates[idx]


class GoodnessOfFit(NamedTuple):
    """Bundle of diagnostics returned by :meth:`goodness_of_fit`.

    Attributes:
        residuals: Time-rescaled inter-event residuals :math:`\\tau_i`.
        mask: Real-event mask aligned with ``residuals``.
        ks_statistic: Kolmogorov–Smirnov statistic versus ``Exp(1)``.
        theoretical_quantiles: QQ-plot theoretical quantiles.
        empirical_quantiles: QQ-plot empirical quantiles.
    """

    residuals: Float[Array, ...]
    mask: Bool[Array, ...]
    ks_statistic: Float[Array, ...]
    theoretical_quantiles: Float[Array, ...]
    empirical_quantiles: Float[Array, ...]


class HomogeneousPoissonProcess(eqx.Module):
    """Homogeneous Poisson process on ``[0, T]``.

    Args:
        rate: Intensity ``λ > 0``. Stored as a JAX array.
        observation_window: Window length ``T > 0``.
    """

    rate: Float[Array, ...]
    observation_window: Float[Array, ...]

    def __init__(
        self,
        rate: ArrayLike,
        observation_window: ArrayLike,
    ) -> None:
        self.rate = jnp.asarray(rate)
        self.observation_window = jnp.asarray(observation_window)

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(self, n_events: Int[Array, ...]) -> Float[Array, ...]:
        """Log-likelihood :math:`n \\log \\lambda - \\lambda T`."""
        return hpp_log_prob(n_events, self.rate, self.observation_window)

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Draw a padded event sequence.

        Returns the same ``(times, mask, n_events)`` triple as
        :func:`hpp_sample`.
        """
        return hpp_sample(
            key, self.rate, self.observation_window, max_events, sample_shape
        )

    # ------------------------------------------------------------
    # Intensity, hazard, survival
    # ------------------------------------------------------------

    def intensity(self, t: Float[Array, ...]) -> Float[Array, ...]:
        """Pointwise intensity (constant, equal to ``rate``)."""
        return hpp_intensity(t, self.rate)

    def cumulative_intensity(self, t: Float[Array, ...]) -> Float[Array, ...]:
        """Compensator :math:`\\Lambda(t) = \\lambda t`."""
        return hpp_cumulative_intensity(t, self.rate)

    def survival(
        self,
        t: Float[Array, ...],
        given_time: Float[Array, ...] | float = 0.0,
    ) -> Float[Array, ...]:
        """:math:`S(t|s) = e^{-\\lambda(t - s)}`."""
        return hpp_survival(t, given_time, self.rate)

    def hazard(self, t: Float[Array, ...]) -> Float[Array, ...]:
        """Constant hazard equal to ``rate``."""
        return hpp_hazard(t, self.rate)

    def inter_event_log_prob(
        self,
        tau: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Log density of inter-event delays ``τ ~ Exp(λ)``."""
        return hpp_inter_event_log_prob(tau, self.rate)

    def mean_residual_life(
        self,
        given_time: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        """Memorylessness: :math:`1/\\lambda` regardless of ``given_time``."""
        return hpp_mean_residual_life(self.rate, given_time)

    # ------------------------------------------------------------
    # Predictions and extreme-value style summaries
    # ------------------------------------------------------------

    def predict_count(
        self,
        start_time: Float[Array, ...],
        end_time: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Expected count on ``[start, end]``."""
        return hpp_predict_count(start_time, end_time, self.rate)

    def return_period(
        self,
        probability: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Time :math:`\\tau` with :math:`P(\\text{event within } \\tau) = p`."""
        return hpp_return_period(probability, self.rate)

    def exceedance_log_prob(
        self,
        k: Int[Array, ...],
        start_time: Float[Array, ...],
        end_time: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Log-probability of more than ``k`` events on ``[start, end]``."""
        return hpp_exceedance_log_prob(k, start_time, end_time, self.rate)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def residuals(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Time-rescaling residuals using ``Λ(t) = λ t``."""
        return time_rescaling_residuals(event_times, mask, self.cumulative_intensity)

    def goodness_of_fit(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> GoodnessOfFit:
        """Bundle residuals + KS + QQ for plotting / hypothesis testing."""
        residuals, res_mask = self.residuals(event_times, mask)
        ks = ks_statistic_exp1(residuals, res_mask)
        theoretical, empirical = qq_exp1_quantiles(residuals, res_mask)
        return GoodnessOfFit(residuals, res_mask, ks, theoretical, empirical)

    def compensator_curve(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Pairs ``(t_i, Λ(t_i))`` for a compensator plot."""
        return compensator_curve(event_times, mask, self.cumulative_intensity)


class InhomogeneousPoissonProcess(eqx.Module):
    """Inhomogeneous Poisson process on ``[0, T]``.

    Args:
        log_intensity_fn: Callable mapping an array of times to an array
            of log-intensities (same shape). If this is itself an
            ``eqx.Module`` its parameters are part of the PyTree.
        observation_window: Window length ``T > 0``.
        integrated_intensity: :math:`\\Lambda(T)`. If ``None``, it is
            computed on the fly from ``log_intensity_fn`` via
            trapezoidal quadrature using ``n_integration_points``
            nodes.
        lambda_max: Upper bound on :math:`\\lambda(t)` over ``[0, T]``.
            Required for :meth:`sample` via thinning. If ``None``,
            thinning will raise.
        n_integration_points: Static grid size used by every method
            that needs quadrature. Override once here rather than
            plumbing kwargs through every call.

    Note:
        ``log_intensity_fn`` is stored as a regular PyTree leaf — **not**
        a static field. This way, when the function is itself an
        ``eqx.Module`` (e.g. an MLP), its array leaves participate in the
        PyTree and are differentiable through :meth:`log_prob`. Plain
        Python callables carry no array leaves and are therefore
        effectively static with no extra ceremony.
    """

    log_intensity_fn: Callable[[Array], Array]
    observation_window: Float[Array, ...]
    integrated_intensity: Float[Array, ...] | None
    lambda_max: Float[Array, ...] | None
    n_integration_points: int = eqx.field(static=True, default=100)

    def __init__(
        self,
        log_intensity_fn: Callable[[Array], Array],
        observation_window: ArrayLike,
        integrated_intensity: ArrayLike | None = None,
        lambda_max: ArrayLike | None = None,
        n_integration_points: int = 100,
    ) -> None:
        self.log_intensity_fn = log_intensity_fn
        self.observation_window = jnp.asarray(observation_window)
        self.integrated_intensity = (
            None if integrated_intensity is None else jnp.asarray(integrated_intensity)
        )
        self.lambda_max = None if lambda_max is None else jnp.asarray(lambda_max)
        self.n_integration_points = n_integration_points

    # ------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------

    @classmethod
    def from_fn(
        cls,
        log_intensity_fn: Callable[[Array], Array],
        observation_window: Float[Array, ...],
        **kwargs,
    ) -> InhomogeneousPoissonProcess:
        """Constructor alias for clarity at call sites."""
        return cls(log_intensity_fn, observation_window, **kwargs)

    @classmethod
    def from_piecewise_constant(
        cls,
        bin_edges: ArrayLike,
        rates: ArrayLike,
        **kwargs,
    ) -> InhomogeneousPoissonProcess:
        """Build an IPP with piecewise-constant intensity.

        Args:
            bin_edges: Sorted array of length ``E`` with
                ``bin_edges[0] = 0`` and ``bin_edges[-1] = T``.
            rates: Array of length ``E - 1`` giving the rate on each
                bin. Must be strictly positive.
            **kwargs: Forwarded to :meth:`__init__` (e.g.
                ``n_integration_points``).

        Returns:
            Operator whose ``log_intensity_fn`` is a
            :class:`PiecewiseConstantLogIntensity` module — its
            ``bin_edges`` and ``log_rates`` are real PyTree leaves,
            so you can differentiate through them with
            ``eqx.filter_grad``. ``integrated_intensity`` is the exact
            sum of ``rates × bin_widths``, and ``lambda_max`` is
            ``max(rates)``.
        """
        bin_edges_arr = jnp.asarray(bin_edges)
        rates_arr = jnp.asarray(rates)
        widths = jnp.diff(bin_edges_arr)
        integrated = jnp.sum(widths * rates_arr)
        lam_max = jnp.max(rates_arr)

        log_intensity_fn = PiecewiseConstantLogIntensity(
            bin_edges=bin_edges_arr, rates=rates_arr
        )

        return cls(
            log_intensity_fn,
            observation_window=bin_edges_arr[-1],
            integrated_intensity=integrated,
            lambda_max=lam_max,
            **kwargs,
        )

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood :math:`\\sum_i \\log \\lambda(t_i) - \\Lambda(T)`.

        If ``integrated_intensity`` was not supplied at construction,
        it is computed on the fly by quadrature with
        ``n_integration_points`` nodes.
        """
        if self.integrated_intensity is None:
            Lambda_T = ipp_predict_count(
                0.0,
                self.observation_window,
                self.log_intensity_fn,
                n_points=self.n_integration_points,
            )
        else:
            Lambda_T = self.integrated_intensity
        return ipp_log_prob(event_times, mask, self.log_intensity_fn, Lambda_T)

    def sample(
        self,
        key: PRNGKeyArray,
        max_candidates: int,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Thinning-based sampler. ``lambda_max`` must be set."""
        if self.lambda_max is None:
            raise ValueError(
                "Cannot sample via thinning without lambda_max; "
                "pass it at construction time or use sample_inversion."
            )
        return ipp_sample_thinning(
            key,
            self.log_intensity_fn,
            self.observation_window,
            self.lambda_max,
            max_candidates,
        )

    def sample_inversion(
        self,
        key: PRNGKeyArray,
        inverse_cumulative_intensity_fn: Callable[[Array], Array],
        max_events: int,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Exact sampler when the inverse compensator is available."""
        if self.integrated_intensity is None:
            Lambda_T = ipp_predict_count(
                0.0,
                self.observation_window,
                self.log_intensity_fn,
                n_points=self.n_integration_points,
            )
        else:
            Lambda_T = self.integrated_intensity
        return ipp_sample_inversion(
            key, inverse_cumulative_intensity_fn, Lambda_T, max_events
        )

    # ------------------------------------------------------------
    # Intensity, hazard, survival
    # ------------------------------------------------------------

    def intensity(self, t: Float[Array, ...]) -> Float[Array, ...]:
        """``λ(t) = exp(log_intensity_fn(t))``."""
        return ipp_intensity(t, self.log_intensity_fn)

    def cumulative_intensity(self, t: Float[Array, ...]) -> Float[Array, ...]:
        """Compensator :math:`\\Lambda(t)` via quadrature."""
        return ipp_cumulative_intensity(
            t, self.log_intensity_fn, n_points=self.n_integration_points
        )

    def survival(
        self,
        t: Float[Array, ...],
        given_time: Float[Array, ...] | float = 0.0,
    ) -> Float[Array, ...]:
        """:math:`S(t|s) = \\exp(-[\\Lambda(t) - \\Lambda(s)])`."""
        return ipp_survival(
            t, given_time, self.log_intensity_fn, n_points=self.n_integration_points
        )

    def hazard(self, t: Float[Array, ...]) -> Float[Array, ...]:
        """Hazard equals intensity for an IPP."""
        return ipp_hazard(t, self.log_intensity_fn)

    def cumulative_hazard(
        self,
        t: Float[Array, ...],
        given_time: Float[Array, ...] | float = 0.0,
    ) -> Float[Array, ...]:
        """:math:`\\Lambda(t) - \\Lambda(s)`."""
        return ipp_cumulative_hazard(
            t, given_time, self.log_intensity_fn, n_points=self.n_integration_points
        )

    def inter_event_log_prob(
        self,
        tau: Float[Array, ...],
        current_time: Float[Array, ...] | float = 0.0,
    ) -> Float[Array, ...]:
        """Log density of next-event delay given last event at ``current_time``."""
        return ipp_inter_event_log_prob(
            tau,
            current_time,
            self.log_intensity_fn,
            n_points=self.n_integration_points,
        )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    def predict_count(
        self,
        start_time: Float[Array, ...],
        end_time: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Expected count :math:`\\Lambda(\\text{end}) - \\Lambda(\\text{start})`."""
        return ipp_predict_count(
            start_time,
            end_time,
            self.log_intensity_fn,
            n_points=self.n_integration_points,
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def residuals(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Time-rescaling residuals under :math:`\\Lambda(t)`."""
        return time_rescaling_residuals(event_times, mask, self.cumulative_intensity)

    def goodness_of_fit(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> GoodnessOfFit:
        """Bundle residuals + KS + QQ for plotting / hypothesis testing."""
        residuals, res_mask = self.residuals(event_times, mask)
        ks = ks_statistic_exp1(residuals, res_mask)
        theoretical, empirical = qq_exp1_quantiles(residuals, res_mask)
        return GoodnessOfFit(residuals, res_mask, ks, theoretical, empirical)

    def compensator_curve(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Pairs ``(t_i, Λ(t_i))`` for a compensator plot."""
        return compensator_curve(event_times, mask, self.cumulative_intensity)
