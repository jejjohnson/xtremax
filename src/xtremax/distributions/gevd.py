"""
Comprehensive Generalized Extreme Value Distribution (GEVD) for NumPyro

This module provides a robust implementation of the GEVD with extensive statistical
methods and proper NumPyro integration.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.scipy.special import gammaln
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_mean,
    gev_return_level,
)


# Minimal scalar-or-array alias for broadcastable GEVD parameters.
ArrayLike = float | jnp.ndarray
from xtremax._rng import check_prng_key


class GeneralizedExtremeValueDistribution(dist.Distribution):
    """
    Generalized Extreme Value Distribution (GEVD) for NumPyro.

    The GEVD is a family of continuous probability distributions developed within
    extreme value theory. It encompasses three types of extreme value distributions:

    - Type I (Gumbel): ξ = 0, exponential tails
    - Type II (Fréchet): ξ > 0, heavy polynomial tails
    - Type III (Weibull): ξ < 0, bounded upper tail

    **Probability Density Function:**

    For ξ ≠ 0:
        f(x) = (1/σ) * t(x)^(-(1/ξ + 1)) * exp(-t(x)^(-1/ξ))
        where t(x) = 1 + ξ(x - μ)/σ

    For ξ = 0 (Gumbel limit):
        f(x) = (1/σ) * exp(-(x - μ)/σ) * exp(-exp(-(x - μ)/σ))

    **Cumulative Distribution Function:**

    For ξ ≠ 0:
        F(x) = exp(-t(x)^(-1/ξ))
        where t(x) = 1 + ξ(x - μ)/σ

    For ξ = 0:
        F(x) = exp(-exp(-(x - μ)/σ))

    **Support:**

    - ξ > 0: x ≥ μ - σ/ξ (lower bounded)
    - ξ = 0: x ∈ ℝ (unbounded)
    - ξ < 0: x ≤ μ - σ/ξ (upper bounded)

    Parameters:
        loc (float): Location parameter μ ∈ ℝ
        scale (float): Scale parameter σ > 0
        shape (float): Shape parameter ξ ∈ ℝ

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create GEVD instances for each type
        >>> gumbel = GeneralizedExtremeValueDistribution(loc=0, scale=1, shape=0)
        >>> frechet = GeneralizedExtremeValueDistribution(loc=0, scale=1, shape=0.2)
        >>> weibull = GeneralizedExtremeValueDistribution(loc=0, scale=1, shape=-0.2)
        >>>
        >>> # Sample and evaluate
        >>> key = jax.random.PRNGKey(42)
        >>> samples = frechet.sample(key, sample_shape=(1000,))
        >>> log_probs = frechet.log_prob(samples)
    """

    # NumPyro distribution interface requirements
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    reparametrized_params = ["loc", "scale", "concentration"]

    def __init__(
        self,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        concentration: ArrayLike | None = None,
        shape: ArrayLike | None = None,
        validate_args: bool | None = None,
    ):
        """
        Initialize the Generalized Extreme Value Distribution.

        Args:
            loc: Location parameter μ (real number)
            scale: Scale parameter σ (positive real number)
            concentration: Shape parameter ξ (real number)
            shape: Deprecated backward-compatible alias for ``concentration``
            validate_args: Whether to validate input arguments

        Raises:
            ValueError: If scale <= 0
        """
        if shape is not None:
            if concentration is not None:
                raise ValueError(
                    "Pass only one of 'concentration' or the deprecated 'shape' alias."
                )
            warnings.warn(
                "'shape' is deprecated; use 'concentration' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            concentration = shape

        if concentration is None:
            concentration = 0.0

        self.loc, self.scale, self.concentration = promote_shapes(
            loc, scale, concentration
        )

        # Determine batch shape from broadcasted parameters
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.loc), jnp.shape(self.scale), jnp.shape(self.concentration)
        )

        # Numerical threshold for Gumbel approximation
        self._gumbel_threshold = 1e-7

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        """
        Generate samples from the GEVD using inverse transform sampling.

        The sampling uses the quantile function (inverse CDF):

        For ξ ≠ 0:
            Q(p) = μ + (σ/ξ) * ((-ln(p))^(-ξ) - 1)

        For ξ = 0:
            Q(p) = μ - σ * ln(-ln(p))

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate

        Returns:
            Array of samples from the GEVD
        """
        check_prng_key(key)
        shape = sample_shape + self.batch_shape

        # Generate uniform random variables U ~ Uniform(0,1)
        uniform_samples = dist.Uniform(0.0, 1.0).sample(key, shape)

        # Apply inverse CDF transformation
        return self.icdf(uniform_samples)

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Log PDF. Thin wrapper for :func:`~xtremax.primitives.gev.gev_log_prob`."""
        return gev_log_prob(value, self.loc, self.scale, self.concentration)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.gev.gev_cdf`."""
        return gev_cdf(value, self.loc, self.scale, self.concentration)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``gev_icdf``."""
        return gev_icdf(q, self.loc, self.scale, self.concentration)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support of the distribution.

        Returns:
            Constraint object representing the support:
            - ξ > 0: [μ - σ/ξ, +∞)
            - ξ = 0: (-∞, +∞)
            - ξ < 0: (-∞, μ - σ/ξ]
        """
        return constraints.real

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        Returns:
            Upper bound: μ - σ/ξ for ξ < 0, +∞ otherwise
        """
        return jnp.where(
            self.concentration < 0, self.loc - self.scale / self.concentration, jnp.inf
        )

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        Returns:
            Lower bound: μ - σ/ξ for ξ > 0, -∞ otherwise
        """
        return jnp.where(
            self.concentration > 0, self.loc - self.scale / self.concentration, -jnp.inf
        )

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.gev.gev_mean`."""
        return gev_mean(self.loc, self.scale, self.concentration)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the distribution.

        For ξ ≠ 0:
            mode = μ + (σ/ξ) * ((1+ξ)^ξ - 1)

        For ξ = 0:
            mode = μ

        Returns:
            Mode of the distribution
        """
        loc, scale, shape = self.loc, self.scale, self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold

        def gumbel_mode():
            return loc

        def gevd_mode():
            return loc + (scale / shape) * (jnp.power(1.0 + shape, shape) - 1.0)

        mode_gumbel = gumbel_mode()
        mode_gevd = gevd_mode()

        return jnp.where(is_gumbel, mode_gumbel, mode_gevd)

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the distribution.

        The variance exists when ξ < 1/2:

        For ξ ≠ 0, ξ < 1/2:
            Var[X] = (σ²/ξ²) * (Γ(1-2ξ) - Γ²(1-ξ))

        For ξ = 0:
            Var[X] = σ² * π²/6

        Returns:
            Variance or NaN/∞ when it doesn't exist
        """
        _loc, scale, shape = self.loc, self.scale, self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        var_exists = shape < 0.5

        def gumbel_variance():
            return (scale**2) * (jnp.pi**2) / 6.0

        def gevd_variance():
            gamma1 = jnp.exp(gammaln(1.0 - 2.0 * shape))
            gamma2 = jnp.exp(2.0 * gammaln(1.0 - shape))
            return (scale**2 / shape**2) * (gamma1 - gamma2)

        var_gumbel = gumbel_variance()
        var_gevd = gevd_variance()

        result = jnp.where(is_gumbel, var_gumbel, var_gevd)
        return jnp.where(var_exists, result, jnp.inf)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the distribution.

        Excess kurtosis exists when ξ < 1/4. The formula involves fourth-order
        moments and gamma functions:

        κ = μ₄/σ⁴ - 3

        where μ₄ is the fourth central moment.

        Returns:
            Excess kurtosis or NaN/∞ when it doesn't exist
        """
        shape = self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        kurt_exists = shape < 0.25

        def gumbel_kurtosis():
            # For Gumbel: excess kurtosis = 12/5
            return 12.0 / 5.0

        def gevd_kurtosis():
            # Complex formula involving gamma functions
            g1 = jnp.exp(gammaln(1.0 - shape))
            g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))
            g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))
            g4 = jnp.exp(gammaln(1.0 - 4.0 * shape))

            # Central moments
            mu2 = g2 - g1**2
            mu4 = g4 - 4.0 * g1 * g3 + 6.0 * g1**2 * g2 - 3.0 * g1**4

            return (mu4 / mu2**2) - 3.0

        kurt_gumbel = gumbel_kurtosis()
        kurt_gevd = gevd_kurtosis()

        result = jnp.where(is_gumbel, kurt_gumbel, kurt_gevd)
        return jnp.where(kurt_exists, result, jnp.inf)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the distribution.

        Skewness exists when ξ < 1/3:

        For Gumbel (ξ = 0): skew ≈ 1.1396
        For general GEVD: involves third-order moments and gamma functions

        Returns:
            Skewness or NaN/∞ when it doesn't exist
        """
        shape = self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        skew_exists = shape < 1.0 / 3.0

        def gumbel_skewness():
            # Analytical value for Gumbel distribution
            return 1.1395470994046486

        def gevd_skewness():
            g1 = jnp.exp(gammaln(1.0 - shape))
            g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))
            g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))

            mu2 = g2 - g1**2
            mu3 = g3 - 3.0 * g1 * g2 + 2.0 * g1**3

            return mu3 / jnp.power(mu2, 1.5)

        skew_gumbel = gumbel_skewness()
        skew_gevd = gevd_skewness()

        result = jnp.where(is_gumbel, skew_gumbel, skew_gevd)
        return jnp.where(skew_exists, result, jnp.inf)

    def entropy(self) -> jnp.ndarray:
        r"""Differential entropy of the GEV distribution (in nats).

        For any :math:`\xi` (including the Gumbel limit),

        .. math:: H = \log\sigma + 1 + \gamma(1 + \xi)

        where :math:`\gamma` is the Euler–Mascheroni constant. At
        :math:`\xi = 0` this correctly reduces to :math:`\log\sigma + 1 + \gamma`
        (the Gumbel entropy), so no separate branch is needed.
        """
        scale, shape = self.scale, self.concentration
        euler_gamma = 0.5772156649015329
        return jnp.log(scale) + 1.0 + euler_gamma * (1.0 + shape)

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        return 1.0 - self.cdf(value)

    def log_survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log survival function log S(x) = log( 1 - F(x) ).

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        return jnp.log(self.survival_function(value))

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the hazard rate h(x) = f(x) / S(x).

        The hazard rate represents the instantaneous rate of failure
        given survival up to time x.

        Args:
            value: Points at which to evaluate the hazard rate

        Returns:
            Hazard rate values
        """
        return jnp.exp(self.log_prob(value) - self.log_survival_function(value))

    def cumulative_hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the cumulative hazard rate Λ(x) = -log(S(x)).

        This represents the accumulated hazard up to time x.

        Args:
            value: Points at which to evaluate the cumulative hazard rate

        Returns:
            Cumulative hazard rate values
        """
        return -self.log_survival_function(value)

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``gev_return_level``."""
        return gev_return_level(return_period, self.loc, self.scale, self.concentration)

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index α = 1/ξ for ξ > 0.

        For Fréchet distributions (ξ > 0), the tail follows a power law:
        P(X > x) ~ x^(-1/ξ) as x → ∞

        Returns:
            Tail index (1/ξ for ξ > 0, ∞ for ξ ≤ 0)
        """
        return jnp.where(self.concentration > 0, 1.0 / self.concentration, jnp.inf)

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        This is equivalent to the survival function at the threshold.

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the mean excess function: E[X - u | X > u].

        For GEVD with ξ < 1, this is:
        E[X - u | X > u] = (σ + ξ(u - μ)) / (1 - ξ)

        This is fundamental in Peaks-Over-Threshold modeling.

        Args:
            threshold: Threshold value u

        Returns:
            Conditional excess mean values
        """
        loc, scale, shape = self.loc, self.scale, self.concentration

        # Only valid for ξ < 1 and threshold in support
        valid_shape = shape < 1.0
        in_support = ((shape > 0) & (threshold >= self.lower_bound())) | (
            (shape <= 0) & (threshold <= self.upper_bound())
        )
        valid = valid_shape & in_support

        # Mean excess formula
        excess_mean = (scale + shape * (threshold - loc)) / (1.0 - shape)

        return jnp.where(valid, excess_mean, jnp.nan)

    def reliability_function(self, time: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the reliability function R(t) = P(X > t).

        This is equivalent to the survival function but emphasizes
        the reliability engineering interpretation for Weibull Type III.

        Args:
            time: Time points for reliability evaluation

        Returns:
            Reliability values (probability of surviving past time t)
        """
        return self.survival_function(time)

    def mean_residual_life(self, time: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the mean residual life function: E[X - t | X > t].

        For reliability applications, this represents the expected
        additional lifetime given survival to time t.

        Args:
            time: Current time/age

        Returns:
            Mean residual life values
        """
        return self.conditional_excess_mean(time)

    def percentile_residual_life(
        self, time: jnp.ndarray, percentile: float
    ) -> jnp.ndarray:
        """
        Compute percentile residual life: (Q(p|X>t) - t).

        This gives the additional time until the p-th percentile failure,
        conditional on survival to time t.

        Args:
            time: Current time/age
            percentile: Percentile level (0 < percentile < 1)

        Returns:
            Percentile residual life values
        """
        # Conditional CDF: F(x | X > t) = p ⟺ F(x) = 1 - (1 - p) S(t).
        # (Previously used `1 - p S(t)`, which mapped p=0 to F^{-1}(1),
        # i.e. the upper endpoint, instead of returning zero residual.)
        survival_prob = self.survival_function(time)
        total_prob = 1.0 - (1.0 - percentile) * survival_prob
        return self.icdf(total_prob) - time

    def expand(
        self, batch_shape: tuple, _instance: dist.Distribution | None = None
    ) -> dist.Distribution:
        """
        Expand the distribution to a new batch shape.

        Args:
            batch_shape: New batch shape
            _instance: Optional instance for type checking

        Returns:
            Expanded distribution instance
        """
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = lax.broadcast_shapes(self.batch_shape, batch_shape)
        new.loc = jnp.broadcast_to(self.loc, batch_shape)
        new.scale = jnp.broadcast_to(self.scale, batch_shape)
        new.concentration = jnp.broadcast_to(self.concentration, batch_shape)
        if hasattr(self, "_gumbel_threshold"):
            new._gumbel_threshold = self._gumbel_threshold
        return new._validate_args(self._validate_args)


# Convenient alias for backward compatibility and shorter imports
GEVD = GeneralizedExtremeValueDistribution
