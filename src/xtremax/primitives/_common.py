"""Shared numerical constants and stable series helpers for EVT primitives.

The GEV / GPD kernels all reduce to two removable-singularity quantities as
the shape parameter approaches its Gumbel / exponential limit:

- ``log1p(u) / u`` — appears as :math:`\\log(1 + \\xi z)/\\xi` (via ``u = ξz``),
- ``expm1(a) / a`` — appears in the quantile and mean expansions.

Both tend to ``1`` as their argument tends to ``0``, but the naive quotient is
``0 / 0`` there — numerically ``nan`` in float32 and, worse, ``nan`` in the
reverse-mode gradient. Evaluating them through these helpers keeps the value
accurate and the gradient finite across the whole shape range, which lets the
kernels drop their hard ``jnp.where`` branch on the Gumbel limit entirely: the
formulas are smooth through :math:`\\xi = 0`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


EULER_GAMMA = 0.5772156649015329

# Below this magnitude the exact quotient loses precision (and its gradient is
# 0/0), so we switch to a Taylor series. 1e-3 keeps the 4th-order series within
# float64 eps at the crossover while staying comfortably accurate in float32.
_SERIES_THRESHOLD = 1e-3


def clamped_standardize(
    x: Float[Array, ...], loc: Float[Array, ...], scale: Float[Array, ...]
) -> Array:
    r"""Return ``(x - loc) / scale`` clamped only against literal overflow.

    Two guards keep ``z`` finite without ever distorting a physically
    meaningful standardized value:

    - A non-finite ``x`` is replaced by ``loc`` (so ``z → 0``) *before* the
      division. Callers override the value for ``±inf`` via their own
      extended-real limit; sanitizing the input here means a ``±inf`` never
      forms an ``inf / scale²`` term in the reverse-mode gradient.
    - The ratio is clipped to ``±sqrt(max)`` afterwards. A tiny (even
      subnormal) ``scale`` sends the raw ratio to ``±inf``, which would poison
      ``ξz`` and ``log1p(ξz)/ξ`` with a ``nan``; ``sqrt(max)`` is loose enough
      that both ``z`` and ``z²`` (the latter appears in the shape gradient) stay
      representable, and sits far beyond any real standardized value, so genuine
      heavy (Fréchet) upper tails are returned untouched. The deep-tail
      ``exp(-w)`` overflow is handled separately by :func:`safe_exp_neg`.
    """
    x = jnp.asarray(x)
    x_safe = jnp.where(jnp.isfinite(x), x, loc)
    z = (x_safe - loc) / scale
    z_max = jnp.sqrt(jnp.finfo(z.dtype).max)
    return jnp.clip(z, -z_max, z_max)


def safe_exp_neg(w: Float[Array, ...]) -> Array:
    r"""Return ``exp(-w)`` with the argument floored so it can never overflow.

    In the Gumbel deep lower tail ``w → -∞`` and the raw ``exp(-w)`` overflows to
    ``inf``; worse, the reverse-mode shape gradient of the log density scales as
    ``exp(-w) · z²``, so an unbounded ``exp(-w)`` yields ``0 · inf = nan`` in the
    discarded ``jnp.where`` branch. Flooring ``w`` at ``-(log(max) - 2·log log
    max)`` caps ``exp(-w)`` a couple of e-folds below the dtype ceiling — chosen
    so ``exp(-w) · z²`` also stays finite (``z ≈ -w`` there) — and zeroes the
    gradient of the saturated branch. The value cap is immaterial: the density is
    already numerically zero that far into the tail.
    """
    w = jnp.asarray(w)
    log_max = jnp.log(jnp.finfo(w.dtype).max)
    exp_arg_max = log_max - 2.0 * jnp.log(log_max)
    return jnp.exp(-jnp.maximum(w, -exp_arg_max))


def log1p_over_x(u: Float[Array, ...]) -> Float[Array, ...]:
    r"""Stable :math:`\log(1 + u) / u`, with the removable singularity at 0.

    Value and reverse-mode gradient are both finite for every ``u > -1``. The
    caller must sanitize out-of-support values (``u <= -1``) before calling —
    ``log1p`` is undefined there.
    """
    u = jnp.asarray(u)
    small = jnp.abs(u) < _SERIES_THRESHOLD
    # Double-``where`` so neither branch feeds a singular value to autodiff.
    u_exact = jnp.where(small, 1.0, u)
    exact = jnp.log1p(u_exact) / u_exact
    u_series = jnp.where(small, u, 0.0)
    # log1p(u)/u = 1 - u/2 + u²/3 - u³/4 + u⁴/5 - …
    series = 1.0 + u_series * (
        -0.5 + u_series * (1.0 / 3.0 + u_series * (-0.25 + u_series * 0.2))
    )
    return jnp.where(small, series, exact)


def expm1_over_x(a: Float[Array, ...]) -> Float[Array, ...]:
    r"""Stable :math:`(\exp(a) - 1) / a`, with the removable singularity at 0.

    Value and reverse-mode gradient are both finite everywhere.
    """
    a = jnp.asarray(a)
    small = jnp.abs(a) < _SERIES_THRESHOLD
    a_exact = jnp.where(small, 1.0, a)
    exact = jnp.expm1(a_exact) / a_exact
    a_series = jnp.where(small, a, 0.0)
    # expm1(a)/a = 1 + a/2 + a²/6 + a³/24 + a⁴/120 + …
    series = 1.0 + a_series * (
        0.5
        + a_series * (1.0 / 6.0 + a_series * (1.0 / 24.0 + a_series * (1.0 / 120.0)))
    )
    return jnp.where(small, series, exact)
