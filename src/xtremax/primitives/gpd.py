"""Pure-JAX primitives for the Generalized Pareto Distribution.

GPD models threshold exceedances in the peaks-over-threshold (POT)
framework. The location is pinned to zero (``x`` is the excess over
the threshold); the scale ``σ > 0`` and shape ``ξ`` determine the tail.

The :math:`\\xi = 0` limit is the exponential distribution. As with the GEV
primitives, the limit is handled *smoothly* via
:func:`xtremax.primitives._common.log1p_over_x` / ``expm1_over_x`` rather than a
hard ``jnp.where`` branch, so values stay accurate and gradients finite through
:math:`\\xi = 0` — and there is no threshold constant to drift against the
Distribution layer.

All functions are stateless, ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from xtremax.primitives._common import expm1_over_x, log1p_over_x


def _reduced_exponent(
    x: Float[Array, ...], scale: Float[Array, ...], v_safe: Float[Array, ...]
) -> Array:
    r"""Return :math:`w = \log(1 + \xi x/\sigma)/\xi`, so ``t^{-1/ξ} = exp(-w)``.

    ``v_safe = ξ x/σ`` must already be masked to ``> -1``. As ``ξ → 0`` this
    tends to ``x/σ`` (the exponential rate term), smoothly.
    """
    return (x / scale) * log1p_over_x(v_safe)


def gpd_log_prob(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Log PDF of the Generalized Pareto Distribution (loc = 0)."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    v = shape * x / scale
    valid = v > -1.0
    v_safe = jnp.where(valid, v, 0.0)

    w = _reduced_exponent(x, scale, v_safe)
    log_t = jnp.log1p(v_safe)
    # -log σ - (1/ξ + 1) log(1 + ξx/σ) = -log σ - (w + log_t); → -log σ - x/σ.
    log_pdf = -jnp.log(scale) - (w + log_t)

    in_support = (x >= 0.0) & valid
    return jnp.where(in_support, log_pdf, -jnp.inf)


def gpd_cdf(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """CDF of the Generalized Pareto Distribution."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    v = shape * x / scale
    valid = v > -1.0
    v_safe = jnp.where(valid, v, 0.0)

    w = _reduced_exponent(x, scale, v_safe)
    # 1 - t^{-1/ξ} = 1 - exp(-w) = -expm1(-w); → 1 - exp(-x/σ) in the limit.
    cdf_inside = -jnp.expm1(-w)
    # Beyond the support, ξ > 0 tails map out-of-support CDF to 0 (lower
    # bound); ξ < 0 tails map it to 1 (finite upper bound, including the
    # endpoint x = -σ/ξ where 1 + ξx/σ = 0 exactly).
    boundary = jnp.where(shape < 0, 1.0, 0.0)
    cdf = jnp.where(valid, cdf_inside, boundary)
    # Clamp the universal lower bound (x < 0 is always out of support).
    return jnp.where(x >= 0.0, cdf, 0.0)


def gpd_survival(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Survival function :math:`S(x) = (1 + \xi x/\sigma)^{-1/\xi}` of the GPD.

    Computed as ``exp(-w)`` directly (rather than ``1 - cdf``) so the deep tail
    stays accurate. Equals ``exp(-x/σ)`` in the exponential limit.
    """
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    v = shape * x / scale
    valid = v > -1.0
    v_safe = jnp.where(valid, v, 0.0)

    w = _reduced_exponent(x, scale, v_safe)
    s_inside = jnp.exp(-w)
    # Above the finite Weibull-type upper bound (ξ < 0) S = 0; ξ > 0 has no
    # upper bound so `valid` never fails there for x ≥ 0.
    s = jnp.where(valid, s_inside, 0.0)
    # Below x = 0 the exceedance is certain, S = 1.
    return jnp.where(x >= 0.0, s, 1.0)


def gpd_log_survival(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Log survival :math:`\log S(x) = -\tfrac{1}{\xi}\log(1 + \xi x/\sigma)`.

    Equals ``-w`` inside the support, so the deep tail is exact instead of
    passing through ``log(1 - cdf)``.
    """
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    v = shape * x / scale
    valid = v > -1.0
    v_safe = jnp.where(valid, v, 0.0)

    w = _reduced_exponent(x, scale, v_safe)
    ls = jnp.where(valid, -w, -jnp.inf)
    return jnp.where(x >= 0.0, ls, 0.0)


def _gpd_icdf_from_log_exceedance(
    log_p: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""GPD quantile from :math:`\log p`, the log exceedance probability ``1 - q``.

    .. math:: Q = \frac{\sigma}{\xi}\big(p^{-\xi} - 1\big)
        = -\sigma \log p \cdot \frac{\exp(-\xi \log p) - 1}{-\xi \log p}

    which tends to :math:`-\sigma \log p` (the exponential quantile) as
    :math:`\xi \to 0`. Taking ``log p`` directly avoids the ``1 - q``
    cancellation that wrecks large return periods.
    """
    a = -shape * log_p
    return -scale * log_p * expm1_over_x(a)


def gpd_icdf(
    q: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Quantile function (inverse CDF) of the GPD."""
    # log(1 - q) via log1p keeps the upper tail (q → 1) accurate.
    return _gpd_icdf_from_log_exceedance(jnp.log1p(-q), scale, shape)


def gpd_mean(
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""GPD mean: :math:`\sigma / (1 - \xi)` for :math:`\xi < 1`, else ``+inf``."""
    shape = jnp.asarray(shape)
    # Substitute a safe divisor where shape >= 1 so the masked-out branch
    # traces without a concrete Python ZeroDivisionError.
    denom = jnp.where(shape < 1.0, 1.0 - shape, 1.0)
    mean_val = scale / denom
    return jnp.where(shape < 1.0, mean_val, jnp.inf)


def gpd_return_level(
    period: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""T-period return level: :math:`Q(1 - 1/T)`.

    Parameterized by the exceedance probability ``1/T`` directly (``log p =
    -log T``) so it stays accurate for large ``T``.
    """
    return _gpd_icdf_from_log_exceedance(-jnp.log(period), scale, shape)
