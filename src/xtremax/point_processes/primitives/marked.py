"""Primitives for marked temporal point processes.

Separable factorisation
:math:`\\lambda(t, m) = \\lambda_g(t) \\, f(m \\mid t, \\mathcal{H})`
with a user-supplied mark distribution ``mark_distribution_fn`` that
returns a NumPyro ``Distribution`` conditioned on time and (optionally)
history. Under separability the joint log-likelihood decomposes into
the ground-process log-likelihood plus a sum of mark log-densities,
so these primitives focus on the mark contribution and leave the
ground-process term to whichever family (IPP, Hawkes, renewal) the
caller is using.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._history import EventHistory


def marks_log_prob(
    event_times: Float[Array, ...],
    marks: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution],
    history_at_each_event: bool = True,
) -> Float[Array, ...]:
    r"""Sum of ``log f(mᵢ | tᵢ, Hᵢ)`` over real events.

    Args:
        event_times: Padded event times ``(max_events,)``.
        marks: Padded marks aligned with ``event_times``. Shape
            ``(max_events,)`` for scalar marks, ``(max_events, d)`` for
            ``d``-dimensional marks.
        mask: Event mask.
        mark_distribution_fn: Callable ``(t, history) -> Distribution``.
        history_at_each_event: When ``True`` the history passed to
            ``mark_distribution_fn(tᵢ, Hᵢ)`` contains events
            ``{t_j : j < i}``; when ``False`` (the default used by
            callers that do not need prior-event conditioning) it is
            the full observed history — this saves the per-event
            vmap but is only correct for history-independent mark
            distributions.

    Returns:
        Scalar log-likelihood contribution from the marks.

    Notes:
        At padding positions ``mark_distribution_fn`` is still called
        (to keep shapes static) but the log-prob is masked out.
    """
    event_times = jnp.asarray(event_times)
    marks = jnp.asarray(marks)

    if not history_at_each_event:
        full_history = EventHistory(
            times=event_times, mask=mask, marks=marks if marks.ndim == 2 else None
        )

        def per_event(i: Int[Array, ...]) -> Float[Array, ...]:
            t_i = event_times[i]
            d = mark_distribution_fn(t_i, full_history)
            mark_i = marks[i]
            return d.log_prob(mark_i)

    else:
        n_max = event_times.shape[-1]
        idx_grid = jnp.arange(n_max)
        # A per-event history uses mask[..., j] & (j < i).
        marks_2d = marks if marks.ndim == 2 else marks[..., None]

        def per_event(i: Int[Array, ...]) -> Float[Array, ...]:
            before_i = (idx_grid < i) & mask
            h_i = EventHistory(
                times=jnp.where(before_i, event_times, 0.0),
                mask=before_i,
                marks=jnp.where(before_i[..., None], marks_2d, 0.0),
            )
            d = mark_distribution_fn(event_times[i], h_i)
            mark_i = marks[i]
            return d.log_prob(mark_i)

    idx = jnp.arange(event_times.shape[-1])
    per_event_log_prob = jax.vmap(per_event)(idx)
    # Mark distributions on multi-dim marks return per-draw scalars, on
    # batched marks they return per-component values — sum any trailing dims.
    while per_event_log_prob.ndim > mask.ndim:
        per_event_log_prob = jnp.sum(per_event_log_prob, axis=-1)
    return jnp.sum(jnp.where(mask, per_event_log_prob, 0.0), axis=-1)


def sample_marks_at_times(
    key: PRNGKeyArray,
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution],
    mark_shape: tuple[int, ...] = (),
    history_at_each_event: bool = True,
) -> Float[Array, ...]:
    """Draw marks ``mᵢ ~ f(· | tᵢ, Hᵢ)`` at each observed time.

    Args:
        key: JAX PRNG key.
        event_times: Padded event times.
        mask: Event mask.
        mark_distribution_fn: Mark distribution callable.
        mark_shape: Event shape of each mark (``()`` for scalar,
            ``(d,)`` for vector).
        history_at_each_event: Same semantics as in
            :func:`marks_log_prob`.

    Returns:
        Array of marks shaped ``event_times.shape + mark_shape``,
        padded with zeros at non-event positions.
    """
    n_max = event_times.shape[-1]
    keys = random.split(key, n_max)

    if history_at_each_event:
        idx_grid = jnp.arange(n_max)

        def per_event(i: Int[Array, ...], k: PRNGKeyArray) -> Array:
            before_i = (idx_grid < i) & mask
            h_i = EventHistory(
                times=jnp.where(before_i, event_times, 0.0),
                mask=before_i,
                marks=None,
            )
            d = mark_distribution_fn(event_times[i], h_i)
            return d.sample(k)

    else:
        full_history = EventHistory(times=event_times, mask=mask, marks=None)

        def per_event(i: Int[Array, ...], k: PRNGKeyArray) -> Array:
            d = mark_distribution_fn(event_times[i], full_history)
            return d.sample(k)

    idx = jnp.arange(n_max)
    marks = jax.vmap(per_event)(idx, keys)
    # Zero-out padding positions for cleanliness.
    if marks.ndim == mask.ndim:
        return jnp.where(mask, marks, 0.0)
    return jnp.where(mask[..., None], marks, 0.0)
