"""Primitives for marked spatial point processes.

Under the separable factorisation
:math:`\\lambda(s, m) = \\lambda(s) \\, f(m \\mid s)` the joint
log-likelihood decomposes into a ground (location) component and a
mark component:

.. math::
    \\log L = \\sum_i [\\log \\lambda(s_i) + \\log f(m_i \\mid s_i)]
              - \\Lambda(D).

These primitives focus on the mark contribution conditional on
locations and delegate the ground term to
:func:`~xtremax.point_processes.primitives.ipp_spatial.ipp_spatial_log_prob`.
The mark distribution is supplied as a callable
``mark_distribution_fn(s) -> numpyro.Distribution`` so users can
condition on location (e.g. magnitude that scales with risk gradient).
For location-independent marks pass a function that ignores its input.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jaxtyping import Array, Bool, Float, PRNGKeyArray


def spatial_marks_log_prob(
    locations: Float[Array, ...],
    marks: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array], dist.Distribution],
) -> Float[Array, ...]:
    """Sum of ``log f(mᵢ | sᵢ)`` over real events.

    Args:
        locations: Padded locations ``(..., max_events, d)``.
        marks: Padded marks. Shape ``(..., max_events)`` for scalar
            marks or ``(..., max_events, mark_dim)`` for vector marks.
        mask: Event mask shape ``(..., max_events)``.
        mark_distribution_fn: Callable ``s -> Distribution``. The
            distribution must have ``event_shape == ()`` for scalar
            marks or ``event_shape == (mark_dim,)`` for vector marks.

    Returns:
        Scalar (or batched) log-likelihood contribution from marks.

    Notes:
        Padding positions still call ``mark_distribution_fn`` for shape
        stability — choose padding locations inside the domain so the
        callable does not raise. Their log-probs are masked out.
    """
    locations = jnp.asarray(locations)
    marks = jnp.asarray(marks)

    # ``log_prob`` is still called at padding positions for shape
    # stability, but padding marks are forced to a value inside the
    # distribution's support before the call so the masked-out result
    # carries finite gradients. Without this, distributions with
    # bounded support (Gamma on ``(0, ∞)``, Beta on ``(0, 1)``) produce
    # ``-inf`` log-probs and NaN gradients when padding marks are
    # zeroed out — even though the ``where`` masks the value out, the
    # gradient still flows through both branches.
    if marks.ndim == mask.ndim:
        safe_marks = jnp.where(mask, marks, 1.0)
    else:
        safe_marks = jnp.where(mask[..., None], marks, 1.0)

    def per_event(s_i: Array, m_i: Array) -> Array:
        d = mark_distribution_fn(s_i)
        return d.log_prob(m_i)

    # Vectorise across the event axis. Locations are 2-D
    # ``(max_events, d)`` and marks are 1-D or 2-D — vmap with
    # ``in_axes=0`` over the leading axis works in both cases.
    per_event_log_prob = jax.vmap(per_event)(locations, safe_marks)
    while per_event_log_prob.ndim > mask.ndim:
        per_event_log_prob = jnp.sum(per_event_log_prob, axis=-1)
    return jnp.sum(jnp.where(mask, per_event_log_prob, 0.0), axis=-1)


def sample_spatial_marks_at_locations(
    key: PRNGKeyArray,
    locations: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array], dist.Distribution],
) -> Float[Array, ...]:
    """Draw marks ``mᵢ ~ f(· | sᵢ)`` at every location.

    Independent across events, so a single ``vmap`` over PRNG splits is
    sufficient — no scan is needed (unlike the temporal marked PP,
    which threads history through each step).

    Args:
        key: JAX PRNG key.
        locations: Padded locations.
        mask: Event mask.
        mark_distribution_fn: Mark distribution callable.

    Returns:
        Marks aligned with ``locations``. Padding positions are zeroed.
        Scalar marks have shape ``locations.shape[:-1]``; vector marks
        have shape ``locations.shape[:-1] + (mark_dim,)``.
    """
    n_max = locations.shape[-2]
    keys = random.split(key, n_max)

    def per_event(s_i: Array, k: PRNGKeyArray) -> Array:
        d = mark_distribution_fn(s_i)
        return d.sample(k)

    marks = jax.vmap(per_event)(locations, keys)
    if marks.ndim == mask.ndim:
        return jnp.where(mask, marks, 0.0)
    return jnp.where(mask[..., None], marks, 0.0)
