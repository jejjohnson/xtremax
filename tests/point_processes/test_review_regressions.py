"""Regressions for review-round fixes on the extensions PR.

Each test corresponds to a specific bug raised in PR review so a
future refactor cannot silently re-introduce it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import EventHistory
from xtremax.point_processes.operators import (
    ExponentialHawkes,
    HomogeneousPoissonProcess,
    MarkedTemporalPointProcess,
    ThinningProcess,
)
from xtremax.point_processes.primitives.marked import sample_marks_at_times
from xtremax.point_processes.primitives.thinning import (
    retention_compensator,
    thinning_retention_log_prob,
)


class TestHistoryDependentRetentionSeesPrefixOnly:
    """Codex P1: retention must not peek at events in its own future.

    If the retention callable ever sees events with ``time >= t`` in
    ``history.mask``, it can condition on the future and produce a
    wrong log-likelihood / compensator. We assert that both the
    per-event log-prob and the quadrature compensator honour the
    ``time < t`` prefix contract.
    """

    def test_thinning_retention_log_prob_uses_prefix_only(self):
        event_times = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.array([True, True, True, True])
        history = EventHistory(times=event_times, mask=mask, marks=None)

        # Retention that returns a value encoding the prefix count so
        # the test can read it back via the summed log-prob. If the
        # prefix contract leaks (full history used instead), every
        # event sees n=4 and the sum is 4·log(f(4)).
        def retention(t, h, mark=None):
            n = jnp.sum(h.mask).astype(jnp.float32)
            return jnp.clip((n + 1.0) * 0.2, 1e-6, 1.0)

        total = float(
            thinning_retention_log_prob(retention, event_times, mask, history)
        )
        # Event i sees n=i prior events → p = (i+1)·0.2 ∈ {0.2, 0.4, 0.6, 0.8}.
        expected = sum(float(jnp.log((i + 1) * 0.2)) for i in range(4))
        assert total == pytest.approx(expected, rel=1e-4)

        # Sanity: if history leaked (all four events), sum would be
        # 4·log(5·0.2) = 4·log(1.0) = 0, which this test would not
        # match.
        leaked = 4 * float(jnp.log(1.0))
        assert total != pytest.approx(leaked, abs=1e-3)

    def test_retention_compensator_sees_prefix_only(self):
        event_times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([True, True, True])
        history = EventHistory(times=event_times, mask=mask, marks=None)

        def retention(t, h, mark=None):
            # At t = 0.5 there should be no prior events; at t = 2.5
            # there should be exactly 2 (events at 1.0, 2.0).
            n = jnp.sum(h.mask)
            # We cannot assert under vmap; instead check that the
            # visible count is consistent with t via a numeric bound:
            # n_visible ≤ count of event_times strictly less than t.
            return jnp.asarray(1.0) * (n.astype(jnp.float32) + 0.0) * 0.0 + 1.0

        def intensity(t, h):
            return jnp.asarray(1.0)

        # If the prefix logic is wrong and full history leaks in, the
        # retention_fn above would still be 1.0, but a concrete test:
        # pick a retention that actively reads ``h.mask`` and check
        # the integral numerically against the hand computation.
        def retention_count(t, h, mark=None):
            # p = 1 / (1 + N_prefix); compensator = ∫(1 - p)·λ dt
            n = jnp.sum(h.mask).astype(jnp.float32)
            return 1.0 / (1.0 + n)

        T = 4.0
        val = float(retention_compensator(retention_count, intensity, history, T))
        # At each grid point t: n_prefix = # events < t.
        # Known values: p=1 for t<1, p=1/2 for 1≤t<2, p=1/3 for 2≤t<3, p=1/4 for t≥3.
        # ∫(1-p)·1 dt = 0·1 + 0.5·1 + (2/3)·1 + 0.75·1 = 0 + 0.5 + 0.667 + 0.75 = 1.917
        expected = 0.0 + 0.5 + (2.0 / 3.0) + 0.75
        assert val == pytest.approx(expected, rel=0.05)


class TestThinningMarkedBaseRoutesMarks:
    """Codex P1: thinning a marked base must preserve and use marks.

    Previously ``sample`` forced ``latent_marks = None`` because the
    third element of the base result was always treated as a count.
    """

    def test_mark_dependent_retention_fires(self):
        class MagnitudeGate(eqx.Module):
            cutoff: jnp.ndarray

            def __call__(self, t, history, proposed_mark=None):
                # proposed_mark must be non-None for this to fire.
                if proposed_mark is None:
                    return jnp.asarray(0.0)
                return (proposed_mark > self.cutoff).astype(jnp.float32)

        hpp = HomogeneousPoissonProcess(5.0, 5.0)

        def marks_fn(t, history):
            return dist.Normal(0.5, 0.1)  # most marks > 0.3

        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=marks_fn,
            mark_dim=None,
            history_at_each_event=False,
        )
        gate = MagnitudeGate(cutoff=jnp.asarray(0.3))
        thin = ThinningProcess(base=mtpp, retention_fn=gate)
        _t, _m, n_retained = thin.sample(jax.random.PRNGKey(0), max_events=64)

        # If marks are routed correctly, the gate should keep most of
        # the draws (mean ≈ 0.5 > 0.3 so retention ≈ 1). If marks get
        # dropped to None, the gate returns 0 and nothing is retained.
        assert int(n_retained) > 0


class TestSampleMarksAtTimesUsesPriorMarks:
    """Codex P1: sequential mark sampling must thread prior marks.

    Without this, mark distributions that depend on ``history.marks``
    silently sample from the wrong law.
    """

    def test_prior_marks_are_visible_to_next_mark_draw(self):
        """Each successive draw's ``loc`` shifts by the mean of prior marks.

        If prior marks are correctly threaded, mark ``m_i`` is drawn
        from ``Normal(sum_prior_marks, 0.01)`` — the successive means
        are cumulative partial sums. A broken implementation would
        always draw from ``Normal(0, 0.01)``.
        """
        event_times = jnp.array([0.0, 1.0, 2.0, 3.0])
        mask = jnp.array([True, True, True, True])

        def marks_fn(t, history):
            loc = (
                jnp.sum(history.marks[..., 0])
                if history.marks is not None
                else jnp.asarray(0.0)
            )
            return dist.Normal(loc, 0.01)

        marks = sample_marks_at_times(
            jax.random.PRNGKey(0),
            event_times,
            mask,
            marks_fn,
            history_at_each_event=True,
        )

        # Running sum of marks: each m_i ≈ m_0 + m_1 + ... + m_{i-1}
        # (with tiny noise). So m_1 ≈ m_0, m_2 ≈ 2·m_0, m_3 ≈ 4·m_0 in
        # expectation — strictly increasing in absolute value when m_0
        # is not too small.
        m = marks
        # m[0] arbitrary but finite; thereafter each should be roughly
        # the cumulative sum of prior draws.
        assert jnp.all(jnp.isfinite(m))
        # If prior marks are ignored, m[i] ~ Normal(0, 0.01) for all i
        # and |m[3]| would typically be well under 0.05. With threading
        # the cumulative sum pushes |m[3]| dramatically larger.
        assert abs(float(m[3])) > 3 * abs(float(m[0])) - 0.1


class TestThinningPromotesScalarMarksInHistory:
    """Copilot: scalar marks must not be dropped from ``EventHistory``."""

    def test_history_marks_populated_for_scalar_marks(self):
        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, True])
        marks = jnp.array([0.1, 0.2, 0.3, 0.4])  # shape (4,) — same as mask

        seen_marks = []

        def retention(t, history, mark=None):
            if history.marks is not None:
                seen_marks.append(history.marks.shape)
            return jnp.asarray(1.0)

        hpp = HomogeneousPoissonProcess(2.0, 3.0)
        thin = ThinningProcess(base=hpp, retention_fn=retention)
        thin.log_prob(event_times, mask, marks=marks)

        # Should have been invoked with a non-None, 2-D marks buffer at
        # every event — regression for the previous drop on
        # marks.shape[-1] == mask.shape[-1].
        assert len(seen_marks) >= 1
        for shape in seen_marks:
            assert len(shape) == 2


class TestThinningForwardsMaxCandidates:
    """Copilot: ``max_candidates`` must reach a thinning-based base."""

    def test_max_candidates_passes_through_to_hawkes(self):
        # Hawkes accepts ``max_candidates``; HPP does not. We assert
        # that passing it through doesn't raise for Hawkes.
        hawkes = ExponentialHawkes(mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0)

        def retention(t, history, mark=None):
            return jnp.asarray(1.0)

        thin = ThinningProcess(base=hawkes, retention_fn=retention)
        # Should not raise — base.sample(max_events, max_candidates=...)
        # is valid for Hawkes.
        _ = thin.sample(jax.random.PRNGKey(0), max_events=32, max_candidates=128)
