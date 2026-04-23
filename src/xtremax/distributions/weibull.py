"""
Weibull Type III Generalized Extreme Value Distribution for NumPyro

This module provides a robust implementation of the GEVD Type III (reversed Weibull)
with extensive statistical methods and proper NumPyro integration for bounded upper
tail extreme value modeling.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.scipy.special import gammaln
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax._rng import check_prng_key
from xtremax.primitives.weibull import (
    weibull_cdf,
    weibull_icdf,
    weibull_log_prob,
    weibull_mean,
    weibull_return_level,
)


class WeibullType3GEVD(dist.Distribution):
    """
    Weibull Type III Generalized Extreme Value Distribution for NumPyro.

    The Weibull Type III (reversed Weibull) is the GEVD with shape parameter ξ < 0,
    characterized by a bounded upper tail. This distribution is fundamental in extreme
    value theory for modeling bounded maxima and reliability applications with upper
    failure limits.

    **Key Characteristics:**
    - Bounded upper tail: x ≤ μ - σ/ξ (since ξ < 0)
    - Light tail behavior with exponential decay
    - Often used for modeling material strength limits, maximum system capacity
    - Opposite tail behavior to Fréchet (Type II)

    **Probability Density Function:**

    f(x) = (1/σ) * t(x)^(-(1/ξ + 1)) * exp(-t(x)^(-1/ξ))
    where t(x) = 1 + ξ(x - μ)/σ and ξ < 0

    **Cumulative Distribution Function:**

    F(x) = exp(-t(x)^(-1/ξ)) for t(x) > 0
    F(x) = 1 for t(x) ≤ 0 (above upper bound)
    where t(x) = 1 + ξ(x - μ)/σ

    **Support:**

    x ∈ (-∞, μ - σ/ξ] where μ - σ/ξ is the upper bound

    **Parameters:**

    - loc (μ): Location parameter ∈ ℝ
    - scale (σ): Scale parameter > 0
    - shape (ξ): Shape parameter < 0 (enforced for Type III)

    **Statistical Properties:**

    - Mean exists for ξ > -1: E[X] = μ + (σ/ξ)(Γ(1-ξ) - 1)
    - Variance exists for ξ > -1/2: Var[X] = (σ²/ξ²)(Γ(1-2ξ) - Γ²(1-ξ))
    - Upper bounded support makes it suitable for reliability analysis

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create Weibull Type III GEVD
        >>> weibull_gevd = WeibullType3GEVD(loc=10.0, scale=2.0, shape=-0.3)
        >>>
        >>> # Key properties
        >>> print(f"Upper bound: {weibull_gevd.upper_bound()}")
        >>> print(f"Mean: {weibull_gevd.mean}")
        >>> print(f"Mode: {weibull_gevd.mode}")
        >>>
        >>> # Sample and evaluate
        >>> key = jax.random.PRNGKey(42)
        >>> samples = weibull_gevd.sample(key, sample_shape=(1000,))
        >>> log_probs = weibull_gevd.log_prob(samples)
    """

    # NumPyro distribution interface requirements
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "shape": constraints.less_than(0.0),  # Enforce ξ < 0 for Type III
    }
    reparametrized_params = ["loc", "scale", "shape"]

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        shape: float = -0.1,
        validate_args: bool | None = None,
    ):
        """
        Initialize the Weibull Type III GEVD.

        Args:
            loc: Location parameter μ (real number)
            scale: Scale parameter σ (positive real number)
            shape: Shape parameter ξ (negative real number, ξ < 0)
            validate_args: Whether to validate input arguments

        Raises:
            ValueError: If scale <= 0 or shape >= 0
        """
        # Enforce Type III constraint: ξ < 0
        if jnp.any(shape >= 0):
            raise ValueError(
                "Shape parameter must be negative for Weibull Type III GEVD (ξ < 0)"
            )

        self.loc, self.scale, self.shape = promote_shapes(loc, scale, shape)

        # Determine batch shape from broadcasted parameters
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.loc), jnp.shape(self.scale), jnp.shape(self.shape)
        )

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        """
        Generate samples from the Weibull Type III GEVD via inverse transform sampling.

        The sampling uses the quantile function:
        Q(p) = μ + (σ/ξ) * ((-ln(p))^(-ξ) - 1)

        Since ξ < 0, this becomes:
        Q(p) = μ + (σ/|ξ|) * (1 - (-ln(p))^(|ξ|))

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate

        Returns:
            Array of samples from the Weibull Type III GEVD, all bounded above
        """
        check_prng_key(key)
        shape = sample_shape + self.batch_shape

        # Generate uniform random variables U ~ Uniform(0,1)
        uniform_samples = dist.Uniform(0.0, 1.0).sample(key, shape)

        # Apply inverse CDF transformation
        return self.icdf(uniform_samples)

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Log PDF. Thin wrapper for ``weibull_log_prob``."""
        return weibull_log_prob(value, self.loc, self.scale, self.shape)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.weibull.weibull_cdf`."""
        return weibull_cdf(value, self.loc, self.scale, self.shape)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``weibull_icdf``."""
        return weibull_icdf(q, self.loc, self.scale, self.shape)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support constraint for Weibull Type III GEVD.

        Returns:
            Constraint representing x ≤ μ - σ/ξ (upper bounded)
        """
        upper_bound = self.upper_bound()
        return constraints.less_than(upper_bound)

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        For Weibull Type III (ξ < 0): upper bound = μ - σ/ξ

        Returns:
            Upper bound of the distribution support
        """
        # Since ξ < 0, -σ/ξ = σ/|ξ| > 0
        return self.loc - self.scale / self.shape

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        For Weibull Type III: lower bound = -∞

        Returns:
            Lower bound (-∞ for Type III)
        """
        return jnp.full_like(self.loc, -jnp.inf)

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.weibull.weibull_mean`."""
        return weibull_mean(self.loc, self.scale, self.shape)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the Weibull Type III GEVD.

        The mode is:
        mode = μ + (σ/ξ) * ((1+ξ)^(-ξ) - 1)

        Since ξ < 0, this represents the most likely value below the upper bound.

        Returns:
            Mode of the distribution
        """
        loc, scale, shape = self.loc, self.scale, self.shape

        # Mode formula: mode = μ + (σ/ξ) * ((1+ξ)^(-ξ) - 1)
        return loc + (scale / shape) * (jnp.power(1.0 + shape, -shape) - 1.0)

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the Weibull Type III GEVD.

        The variance exists when ξ > -1/2:
        Var[X] = (σ²/ξ²) * (Γ(1-2ξ) - Γ²(1-ξ))

        Returns:
            Variance or NaN when it doesn't exist (ξ ≤ -1/2)
        """
        _loc, scale, shape = self.loc, self.scale, self.shape

        # Variance exists for ξ > -1/2
        var_exists = shape > -0.5

        # Compute variance using gamma functions
        gamma1 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        gamma2 = jnp.exp(2.0 * gammaln(1.0 - shape))  # Γ²(1-ξ)

        var_val = (scale**2 / shape**2) * (gamma1 - gamma2)

        return jnp.where(var_exists, var_val, jnp.nan)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the Weibull Type III GEVD.

        Excess kurtosis exists when ξ > -1/4 and involves fourth-order moments:
        κ = μ₄/σ⁴ - 3

        Returns:
            Excess kurtosis or NaN when it doesn't exist (ξ ≤ -1/4)
        """
        shape = self.shape

        # Kurtosis exists for ξ > -1/4
        kurt_exists = shape > -0.25

        # Complex formula involving gamma functions for fourth moment
        g1 = jnp.exp(gammaln(1.0 - shape))  # Γ(1-ξ)
        g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))  # Γ(1-3ξ)
        g4 = jnp.exp(gammaln(1.0 - 4.0 * shape))  # Γ(1-4ξ)

        # Central moments
        mu2 = g2 - g1**2
        mu4 = g4 - 4.0 * g1 * g3 + 6.0 * g1**2 * g2 - 3.0 * g1**4

        excess_kurt = (mu4 / mu2**2) - 3.0

        return jnp.where(kurt_exists, excess_kurt, jnp.nan)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the Weibull Type III GEVD.

        Skewness exists when ξ > -1/3 and involves third-order moments.

        For Weibull Type III, skewness is typically negative due to the
        upper bound creating left-skewed distributions.

        Returns:
            Skewness or NaN when it doesn't exist (ξ ≤ -1/3)
        """
        shape = self.shape

        # Skewness exists for ξ > -1/3
        skew_exists = shape > -1.0 / 3.0

        # Compute using gamma functions
        g1 = jnp.exp(gammaln(1.0 - shape))  # Γ(1-ξ)
        g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))  # Γ(1-3ξ)

        # Central moments
        mu2 = g2 - g1**2
        mu3 = g3 - 3.0 * g1 * g2 + 2.0 * g1**3

        skewness = mu3 / jnp.power(mu2, 1.5)

        return jnp.where(skew_exists, skewness, jnp.nan)

    def entropy(self) -> jnp.ndarray:
        """
        Compute the differential entropy of the Weibull Type III GEVD.

        The entropy is:
        H = log(σ) + 1 + γ * (1 + ξ)

        where γ is the Euler-Mascheroni constant ≈ 0.5772. This is the
        standard GEV entropy formula evaluated at the Weibull branch
        (ξ < 0); at ξ = 0 it reduces to the Gumbel entropy
        ``log σ + 1 + γ``.

        Returns:
            Differential entropy in nats
        """
        scale, shape = self.scale, self.shape
        euler_gamma = 0.5772156649015329

        return jnp.log(scale) + 1.0 + euler_gamma * (1.0 + shape)

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        For Weibull Type III, this represents the probability of not
        exceeding the value x, which is relevant for reliability analysis.

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        return 1.0 - self.cdf(value)

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the hazard rate h(x) = f(x) / S(x).

        The hazard rate represents the instantaneous failure rate
        given survival up to time x. For Weibull Type III, the
        hazard rate approaches infinity as x approaches the upper bound.

        Args:
            value: Points at which to evaluate the hazard rate

        Returns:
            Hazard rate values
        """
        log_hazard = self.log_prob(value) - jnp.log(self.survival_function(value))
        return jnp.exp(log_hazard)

    def cumulative_hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the cumulative hazard rate Λ(x) = -log(S(x)).

        This represents the accumulated hazard up to time x.
        For Weibull Type III, this is finite at the upper bound.

        Args:
            value: Points at which to evaluate the cumulative hazard rate

        Returns:
            Cumulative hazard rate values
        """
        return -jnp.log(self.survival_function(value))

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``weibull_return_level``."""
        return weibull_return_level(return_period, self.loc, self.scale, self.shape)

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index for Weibull Type III GEVD.

        For Weibull Type III (ξ < 0), there is no power law tail behavior.
        Instead, the tail is exponentially bounded. The concept of tail
        index (1/ξ) is not meaningful for bounded distributions.

        Returns:
            NaN (tail index not defined for bounded distributions)
        """
        return jnp.full_like(self.shape, jnp.nan)

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        For Weibull Type III, this is 0 when threshold ≥ upper bound.

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities (0 above upper bound)
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the mean excess function: E[X - u | X > u].

        For Weibull Type III GEVD with ξ < 0 and ξ > -1:
        E[X - u | X > u] = (σ + ξ(u - μ)) / (1 - ξ)

        This is particularly important for understanding the expected
        exceedance size given that an exceedance occurs.

        Args:
            threshold: Threshold value u

        Returns:
            Conditional excess mean values
        """
        loc, scale, shape = self.loc, self.scale, self.shape

        # Only valid for ξ > -1 and threshold below upper bound
        mean_exists = shape > -1.0
        below_upper_bound = threshold < self.upper_bound()
        valid = mean_exists & below_upper_bound

        # Mean excess formula for GEVD
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

    def expand(self, batch_shape: tuple[int, ...]) -> dist.Distribution:
        """Expand to ``batch_shape`` by reconstructing via ``__init__``."""
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        return type(self)(
            loc=jnp.broadcast_to(self.loc, batch_shape),
            scale=jnp.broadcast_to(self.scale, batch_shape),
            shape=jnp.broadcast_to(self.shape, batch_shape),
            validate_args=self._validate_args,
        )


# Convenient aliases
WeibullGEVD = WeibullType3GEVD
BoundedGEVD = WeibullType3GEVD
