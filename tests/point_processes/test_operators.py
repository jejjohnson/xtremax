"""Tests for the ``equinox``-based temporal point process operators."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes.operators import (
    GoodnessOfFit,
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
)


class TestHppOperator:
    def test_log_prob_matches_closed_form(self):
        op = HomogeneousPoissonProcess(rate=2.0, observation_window=5.0)
        n = jnp.asarray(4)
        expected = 4 * jnp.log(2.0) - 2.0 * 5.0
        assert jnp.allclose(op.log_prob(n), expected)

    def test_sample_returns_padded_triple(self):
        op = HomogeneousPoissonProcess(rate=3.0, observation_window=4.0)
        times, mask, n = op.sample(random.PRNGKey(0), max_events=64)
        assert times.shape == (64,)
        assert mask.shape == (64,)
        assert mask.sum() <= n

    def test_operator_is_pytree(self):
        op = HomogeneousPoissonProcess(rate=1.0, observation_window=2.0)
        leaves = jax.tree_util.tree_leaves(op)
        assert len(leaves) == 2

    def test_is_optimisable_via_grad(self):
        # Fit rate to maximise log_prob of a known n_events over window T.
        op = HomogeneousPoissonProcess(
            rate=jnp.asarray(1.0), observation_window=jnp.asarray(5.0)
        )
        n_observed = jnp.asarray(20)

        def neg_log_lik(op):
            return -op.log_prob(n_observed)

        # Gradient w.r.t. the PyTree leaves (rate, observation_window).
        grad = eqx.filter_grad(neg_log_lik)(op)
        # ∂(-log L)/∂λ = -n/λ + T = -20 + 5 = -15. We take grad wrt op —
        # the rate leaf should hold that value.
        assert jnp.allclose(grad.rate, -15.0)

    def test_survival_at_given_time_is_one(self):
        op = HomogeneousPoissonProcess(rate=2.0, observation_window=10.0)
        assert jnp.allclose(op.survival(3.0, given_time=3.0), 1.0)

    def test_return_period_round_trip(self):
        op = HomogeneousPoissonProcess(rate=1.5, observation_window=10.0)
        tau = op.return_period(0.8)
        assert jnp.allclose(1.0 - jnp.exp(-1.5 * tau), 0.8)

    def test_goodness_of_fit_on_own_samples(self):
        op = HomogeneousPoissonProcess(rate=2.0, observation_window=20.0)
        times, mask, _ = op.sample(random.PRNGKey(0), max_events=128)
        gof = op.goodness_of_fit(times, mask)
        assert isinstance(gof, GoodnessOfFit)
        assert gof.residuals.shape == times.shape
        assert jnp.isfinite(gof.ks_statistic)


class TestIppOperator:
    def test_piecewise_constant_factory(self):
        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 2.0, 5.0]),
            rates=jnp.array([1.0, 4.0]),
        )
        # Λ(T) = 1*2 + 4*3 = 14.
        assert jnp.allclose(op.integrated_intensity, 14.0)
        # λ_max = 4.
        assert jnp.allclose(op.lambda_max, 4.0)
        # Intensity at t=1 is 1, at t=3 is 4.
        assert jnp.allclose(op.intensity(jnp.asarray(1.0)), 1.0)
        assert jnp.allclose(op.intensity(jnp.asarray(3.0)), 4.0)

    def test_ipp_with_constant_matches_hpp(self):
        rate = 2.5
        T = 5.0

        def fn(t):
            return jnp.full_like(t, jnp.log(rate))

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=T,
            integrated_intensity=rate * T,
            lambda_max=rate,
            n_integration_points=200,
        )
        hpp = HomogeneousPoissonProcess(rate=rate, observation_window=T)

        times = jnp.array([0.5, 1.0, 2.0, 3.5, 4.2])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        assert jnp.allclose(
            ipp.log_prob(times, mask),
            hpp.log_prob(jnp.asarray(5)),
        )

    def test_sample_requires_lambda_max(self):
        def fn(t):
            return jnp.full_like(t, 0.0)

        ipp = InhomogeneousPoissonProcess(log_intensity_fn=fn, observation_window=1.0)
        with pytest.raises(ValueError, match="lambda_max"):
            ipp.sample(random.PRNGKey(0), max_candidates=32)

    def test_integrated_intensity_computed_when_none(self):
        # If integrated_intensity is omitted, log_prob should still be finite.
        def fn(t):
            return jnp.log(1.0 + 0.5 * t)

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=4.0,
            n_integration_points=400,
        )
        times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        log_L = ipp.log_prob(times, mask)
        assert jnp.isfinite(log_L)

    def test_piecewise_sampling_is_jittable(self):
        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 2.0, 5.0]),
            rates=jnp.array([1.0, 4.0]),
        )
        jitted = jax.jit(lambda k: op.sample(k, max_candidates=256))
        times, _mask, _ = jitted(random.PRNGKey(0))
        assert times.shape == (256,)

    def test_inversion_sampling(self):
        rate = 2.0
        T = 5.0

        def fn(t):
            return jnp.full_like(t, jnp.log(rate))

        def inv_cum(y):
            return y / rate

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=T,
            integrated_intensity=rate * T,
        )
        times, mask, _n = ipp.sample_inversion(
            random.PRNGKey(0), inv_cum, max_events=128
        )
        valid = times[mask]
        assert jnp.all(jnp.diff(valid) >= -1e-7)

    def test_goodness_of_fit_on_piecewise_sample(self):
        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 5.0, 10.0]),
            rates=jnp.array([2.0, 5.0]),
            n_integration_points=400,
        )

        # Use inversion sampling for exactness with piecewise constant.
        # Λ(t) piecewise linear. Inverse:
        # y in [0, 10] → t = y/2; y in [10, 35] → t = 5 + (y-10)/5.
        def inv_cum(y):
            t_bin1 = y / 2.0
            t_bin2 = 5.0 + (y - 10.0) / 5.0
            return jnp.where(y <= 10.0, t_bin1, t_bin2)

        times, mask, _ = op.sample_inversion(random.PRNGKey(2), inv_cum, max_events=512)
        gof = op.goodness_of_fit(times, mask)
        assert jnp.isfinite(gof.ks_statistic)
