"""Tests for :mod:`xtremax.point_processes.primitives.diagnostics`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from xtremax.point_processes.primitives import (
    compensator_curve,
    hpp_cumulative_intensity,
    hpp_sample,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
)


RATE = 2.0
T = 10.0


def hpp_cum(t):
    return hpp_cumulative_intensity(t, RATE)


class TestTimeRescalingResiduals:
    def test_hpp_residuals_equal_scaled_inter_event_times(self):
        # For HPP with Λ(t) = λ t, residuals are λ * Δt.
        times = jnp.array([0.5, 1.0, 2.5, 3.0])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        residuals, out_mask = time_rescaling_residuals(times, mask, hpp_cum)
        expected = RATE * jnp.array([0.5, 0.5, 1.5, 0.5])
        assert jnp.allclose(residuals, expected)
        assert jnp.all(out_mask == mask)

    def test_padding_positions_are_zero(self):
        times = jnp.array([1.0, 2.0, 99.0, 99.0])
        mask = jnp.array([True, True, False, False])
        residuals, _ = time_rescaling_residuals(times, mask, hpp_cum)
        assert residuals[2] == 0.0
        assert residuals[3] == 0.0


class TestKsStatisticExp1:
    def test_hpp_samples_pass_ks(self):
        # For an HPP under its own compensator, rescaled residuals are Exp(1).
        keys = random.split(random.PRNGKey(0), 50)

        def once(k):
            times, mask, _ = hpp_sample(k, RATE, T, max_events=128)
            residuals, _ = time_rescaling_residuals(times, mask, hpp_cum)
            return ks_statistic_exp1(residuals, mask), mask.sum()

        ks_values, _ = jax.vmap(once)(keys)
        # Median should be well below the asymptotic 95% critical value (~0.21 at n=20).
        assert float(jnp.median(ks_values)) < 0.3

    def test_wrong_compensator_fails_ks(self):
        # Pass a compensator that is far too weak: residuals will have
        # mean << 1 ⇒ KS large.
        def wrong(t):
            return 0.1 * t

        key = random.PRNGKey(1)
        times, mask, _ = hpp_sample(key, RATE, T, max_events=128)
        residuals, _ = time_rescaling_residuals(times, mask, wrong)
        ks = ks_statistic_exp1(residuals, mask)
        assert float(ks) > 0.5


class TestQqExp1:
    def test_quantile_shape_and_nan_padding(self):
        key = random.PRNGKey(2)
        times, mask, _ = hpp_sample(key, RATE, T, max_events=128)
        residuals, _ = time_rescaling_residuals(times, mask, hpp_cum)
        theoretical, empirical = qq_exp1_quantiles(residuals, mask)
        assert theoretical.shape == residuals.shape
        # At ranks > n_real, both vectors should be NaN.
        n_real = int(mask.sum())
        assert jnp.all(jnp.isnan(theoretical[n_real:]))
        assert jnp.all(jnp.isnan(empirical[n_real:]))
        # Real entries are finite and monotone.
        assert jnp.all(jnp.isfinite(theoretical[:n_real]))
        assert jnp.all(jnp.isfinite(empirical[:n_real]))
        assert jnp.all(jnp.diff(theoretical[:n_real]) >= -1e-7)


class TestCompensatorCurve:
    def test_curve_padding_is_nan(self):
        key = random.PRNGKey(3)
        times, mask, _ = hpp_sample(key, RATE, T, max_events=128)
        ts, cum = compensator_curve(times, mask, hpp_cum)
        n_real = int(mask.sum())
        assert jnp.all(jnp.isfinite(ts[:n_real]))
        assert jnp.all(jnp.isfinite(cum[:n_real]))
        assert jnp.all(jnp.isnan(ts[n_real:]))
        assert jnp.all(jnp.isnan(cum[n_real:]))
