"""Smoke tests for the five NumPyro-compatible extreme value distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax.distributions import (
    FrechetType2GEVD,
    GeneralizedExtremeValueDistribution,
    GeneralizedParetoDistribution,
    GumbelType1GEVD,
    WeibullType3GEVD,
)


@pytest.fixture
def key():
    return jax.random.key(0)


class TestGEVD:
    def test_log_prob_and_sample_shape(self, key):
        dist = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, shape=0.1)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert lp.shape == (32,)
        assert jnp.all(jnp.isfinite(lp))

    def test_cdf_icdf_round_trip(self):
        dist = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, shape=0.2)
        q = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
        x = dist.icdf(q)
        q_round = dist.cdf(x)
        assert jnp.allclose(q, q_round, atol=1e-4)

    def test_small_negative_shape(self, key):
        dist = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, shape=-0.2)
        samples = dist.sample(key, sample_shape=(16,))
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))


class TestGPD:
    def test_log_prob_and_sample_shape(self, key):
        dist = GeneralizedParetoDistribution(scale=1.0, shape=0.2)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))
        assert jnp.all(samples >= 0)  # GPD support is x >= 0 when loc = 0

    def test_survival_uses_scale_not_shape(self):
        """Regression: survival_function previously aliased `scale = self.shape`.

        With scale=2 and shape=0.3 the correct S(1) is a specific number;
        the buggy version used `shape` as both the scale and shape, giving
        a wildly different value. Verify against 1 - cdf().
        """
        d = GeneralizedParetoDistribution(scale=2.0, shape=0.3)
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        assert jnp.allclose(d.survival_function(x), 1.0 - d.cdf(x), atol=1e-6)


class TestGumbel:
    def test_log_prob_and_sample_shape(self, key):
        dist = GumbelType1GEVD(loc=0.0, scale=1.0)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))

    def test_hazard_rate_matches_f_over_S(self):
        """h(x) must equal f(x) / S(x), not exp(z)/σ."""
        d = GumbelType1GEVD(loc=0.0, scale=2.0)
        x = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        pdf = jnp.exp(d.log_prob(x))
        surv = 1.0 - d.cdf(x)
        expected = pdf / surv
        assert jnp.allclose(d.hazard_rate(x), expected, rtol=1e-5)

    def test_cumulative_hazard_matches_neg_log_survival(self):
        """Λ(x) = -log S(x)."""
        d = GumbelType1GEVD(loc=0.0, scale=2.0)
        x = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        expected = -jnp.log(1.0 - d.cdf(x))
        assert jnp.allclose(d.cumulative_hazard_rate(x), expected, rtol=1e-5)

    def test_hazard_asymptotes_to_inverse_scale(self):
        """At the far upper tail, h(x) → 1/σ (exponential-tail limit)."""
        scale = 1.5
        d = GumbelType1GEVD(loc=0.0, scale=scale)
        x_far = jnp.array([20.0])  # deep in upper tail
        h = d.hazard_rate(x_far)
        assert jnp.allclose(h, 1.0 / scale, atol=1e-5)

    def test_conditional_excess_mean_uses_survival_not_cdf(self):
        """Regression: survival_u was previously `exp(-exp(-z_u))` (the CDF).

        With loc=0, scale=1, threshold=-2 we have F(-2) ≈ 5.7e-4 but
        S(-2) ≈ 0.9994. The bug returned NaN at low thresholds where the
        CDF is tiny; the fix should return a finite mean excess.
        """
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        m = d.conditional_excess_mean(jnp.array(-2.0))
        assert jnp.isfinite(m)


class TestFrechet:
    def test_log_prob_and_sample_shape(self, key):
        dist = FrechetType2GEVD(loc=0.0, scale=1.0, shape=0.2)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))


class TestWeibull:
    def test_log_prob_and_sample_shape(self, key):
        dist = WeibullType3GEVD(loc=0.0, scale=1.0, shape=-0.2)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))


class TestPRNGKeyValidation:
    """sample() must raise TypeError (not AssertionError) on bad keys."""

    def test_rejects_non_key_with_typeerror(self):
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        with pytest.raises(TypeError, match="JAX PRNG key"):
            d.sample(42, sample_shape=(4,))  # plain int, not a key

    def test_accepts_legacy_prngkey(self):
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        legacy = jax.random.PRNGKey(0)
        samples = d.sample(legacy, sample_shape=(4,))
        assert samples.shape == (4,)
