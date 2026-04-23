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


class TestGumbel:
    def test_log_prob_and_sample_shape(self, key):
        dist = GumbelType1GEVD(loc=0.0, scale=1.0)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))


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
