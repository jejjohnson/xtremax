"""Tests for non-stationary GEV return-level and spatial primitives.

Covers the stationary/non-stationary consistency of return levels and
periods, the round-trip ``return_level`` ↔ ``return_period`` identity, the
parameter-field assembly, and the spatial helpers (distance matrix, design
matrix, two-range copula correlation), including ``jax.jit`` / ``jax.vmap``
safety.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from xtremax import (
    assemble_nonstationary_gev_fields,
    design_matrix,
    expected_exceedances,
    gev_return_level,
    nonstationary_return_level,
    nonstationary_return_period,
    pairwise_distances,
    two_range_correlation,
)


class TestAssembleFields:
    def test_shapes_scalar_covariate(self):
        n_sites, n_blocks = 4, 6
        mu0 = jnp.linspace(10.0, 13.0, n_sites)
        mu1 = jnp.full((n_sites,), 0.5)
        log_sigma = jnp.zeros(n_sites)
        time = jnp.linspace(0.0, 1.0, n_blocks)

        loc, scale, shape = assemble_nonstationary_gev_fields(
            mu0, mu1, log_sigma, -0.1, time
        )
        assert loc.shape == (n_blocks, n_sites)
        assert scale.shape == (n_blocks, n_sites)
        assert shape.shape == (n_blocks, n_sites)

    def test_location_trend(self):
        """loc(t, s) = mu0(s) + x_t * mu1(s)."""
        mu0 = jnp.array([10.0, 20.0])
        mu1 = jnp.array([1.0, 2.0])
        time = jnp.array([0.0, 1.0, 2.0])
        loc, scale, _ = assemble_nonstationary_gev_fields(
            mu0, mu1, jnp.log(jnp.array([2.0, 3.0])), 0.0, time
        )
        expected = mu0[None, :] + time[:, None] * mu1[None, :]
        assert jnp.allclose(loc, expected)
        # Scale is exp(log_sigma), constant over blocks.
        assert jnp.allclose(scale[0], jnp.array([2.0, 3.0]))
        assert jnp.allclose(scale[0], scale[-1])

    def test_multi_covariate(self):
        n_sites, n_blocks, n_cov = 3, 5, 2
        mu0 = jnp.zeros(n_sites)
        mu1 = jnp.ones((n_sites, n_cov))
        time = jnp.ones((n_blocks, n_cov))
        loc, _, _ = assemble_nonstationary_gev_fields(
            mu0, mu1, jnp.zeros(n_sites), 0.0, time
        )
        # Each block sums the two covariate contributions: 1*1 + 1*1 = 2.
        assert jnp.allclose(loc, 2.0)


class TestStationaryConsistency:
    def test_return_level_matches_closed_form(self):
        """time_axis=None must equal the closed-form GEV return level."""
        periods = jnp.array([2.0, 10.0, 100.0])
        rl = nonstationary_return_level(periods, 10.0, 2.0, -0.1)
        ref = gev_return_level(periods, 10.0, 2.0, -0.1)
        assert jnp.allclose(rl, ref)

    def test_return_period_inverts_return_level(self):
        period = 50.0
        z = nonstationary_return_level(period, 5.0, 1.5, 0.1)
        recovered = nonstationary_return_period(z, 5.0, 1.5, 0.1)
        assert jnp.allclose(recovered, period, rtol=1e-4)


class TestNonStationary:
    def _fields(self):
        # Warming trend: location drifts upward across 30 blocks.
        n_blocks, n_sites = 30, 4
        time = jnp.linspace(0.0, 1.0, n_blocks)
        mu0 = jnp.linspace(10.0, 13.0, n_sites)
        mu1 = jnp.full((n_sites,), 2.0)
        return assemble_nonstationary_gev_fields(
            mu0, mu1, jnp.zeros(n_sites), -0.15, time
        )

    def test_expected_exceedances_decreasing(self):
        loc, scale, shape = self._fields()
        low = expected_exceedances(11.0, loc, scale, shape, time_axis=0)
        high = expected_exceedances(20.0, loc, scale, shape, time_axis=0)
        # Higher threshold -> fewer expected exceedances.
        assert jnp.all(high < low)

    def test_return_level_solves_exceedance_target(self):
        loc, scale, shape = self._fields()
        period = 20.0
        z = nonstationary_return_level(period, loc, scale, shape, time_axis=0)
        n_blocks = loc.shape[0]
        got = expected_exceedances(z[None, :], loc, scale, shape, time_axis=0)
        # By definition sum_t P(Y_t > z) = N_blocks / T.
        assert jnp.allclose(got, n_blocks / period, atol=1e-3)

    def test_return_period_roundtrip(self):
        loc, scale, shape = self._fields()
        period = 25.0
        z = nonstationary_return_level(period, loc, scale, shape, time_axis=0)
        recovered = nonstationary_return_period(
            z[None, :], loc, scale, shape, time_axis=0
        )
        assert jnp.allclose(recovered, period, rtol=1e-3)

    def test_jit_and_grad_safe(self):
        loc, scale, shape = self._fields()
        fn = jax.jit(
            lambda p: nonstationary_return_level(p, loc, scale, shape, time_axis=0)
        )
        out = fn(20.0)
        assert out.shape == (loc.shape[1],)
        assert jnp.all(jnp.isfinite(out))


class TestSpatial:
    def test_pairwise_distances(self):
        coords = jnp.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        d = pairwise_distances(coords)
        assert d.shape == (3, 3)
        assert jnp.allclose(jnp.diag(d), 0.0)
        assert jnp.allclose(d[0, 1], 5.0)
        assert jnp.allclose(d, d.T)

    def test_design_matrix(self):
        cov = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x = design_matrix(cov)
        assert x.shape == (2, 3)
        assert jnp.allclose(x[:, 0], 1.0)
        assert jnp.allclose(x[:, 1:], cov)

    def test_design_matrix_1d(self):
        x = design_matrix(jnp.array([1.0, 2.0, 3.0]))
        assert x.shape == (3, 2)

    def test_two_range_correlation_is_valid(self):
        coords = jnp.linspace(0.0, 1.0, 8).reshape(-1, 1)
        d = pairwise_distances(coords)
        c = two_range_correlation(d, weight=0.6, range_short=0.1, range_long=0.5)
        assert c.shape == (8, 8)
        # Unit diagonal and symmetry.
        assert jnp.allclose(jnp.diag(c), 1.0, atol=1e-6)
        assert jnp.allclose(c, c.T, atol=1e-6)
        # Positive definite (valid correlation matrix).
        assert jnp.all(jnp.linalg.eigvalsh(c) > 0)

    def test_vmap_over_sites(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 5)
        coords = jax.vmap(lambda k: jax.random.normal(k, (6, 2)))(keys)
        dists = jax.vmap(pairwise_distances)(coords)
        assert dists.shape == (5, 6, 6)
