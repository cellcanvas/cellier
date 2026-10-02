"""The property test with cache policies (``plans/mesh_refactor_v3.md`` 5.2).

The environment of ``test_property.py`` with each cache given, at random, a
resource (``"io"`` or ``"compute"``), a cap on target reads (``None``, 1, 2
or 3) and a failure rule, and the core a random compute budget.  Every
invariant of ``test_property.py`` is kept; these change or are new:

2'   one budget per lane (shared, backstop, compute), and together they
     account for every read outstanding;
9'   no target read starts while a ``VISIBLE`` backstop is queued in a cache
     of the **same resource**;
13   target reads of a cache in flight never exceed its cap, wanted or not;
13b  the cap never withholds a backstop: after an issue round, a cache with
     a backstop queued has no room left in its lane;
14   work conserving: after an issue round, no cache has a head its lane
     has room for, so a blocked resource never delays the other;
15   an I/O read never holds the compute lane, and a compute read never
     holds the shared window or the backstop lane;
16   a capped cache starts no target read while a ``VISIBLE`` backstop
     arrival of its own waits for its commit round;
17   a cache with ``retry_on_pass=False`` never reads a given-up key again
     without an invalidation;
5    liveness, unchanged, now with caps and holds.

Each mutant is a deliberate bug in the core; the invariants must catch it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cellier.render.scheduling import (
    COMPUTE_LANE,
    SHARED_LANE,
    CachePolicy,
    ChunkClass,
    ChunkState,
    SchedulerCore,
    Tier,
)
from tests.render.scheduling import test_property as tp
from tests.render.scheduling._fakes import FakeResidency

_QUEUED = int(ChunkState.QUEUED)
_FETCHING = int(ChunkState.FETCHING)
_FAILED = int(ChunkState.FAILED)
_VISIBLE = int(Tier.VISIBLE)
_BACKSTOP = int(ChunkClass.BACKSTOP)

STEPS = 600


class GeneralEnvironment(tp.Environment):
    """``Environment`` with a random policy on every cache."""

    def _register(self, cache_id: int) -> None:
        rng = self.rng
        if not hasattr(self, "_budget_set"):
            self._budget_set = True
            self.config.compute_budget = int(rng.integers(1, 4))
            # (cache_id, incarnation, key) -> reads issued since it was last
            # invalidated or left the registry, for invariant 17.
            self.issued: Counter[tuple[int, int, int]] = Counter()
        n_slots = int(rng.integers(8, 40))
        residency = FakeResidency(n_slots)
        cap = int(rng.integers(0, 4))
        attempts = int(rng.integers(0, 3))
        residency.policy = CachePolicy(
            max_target_fetching=cap or None,
            resource=str(rng.choice(("io", "compute"))),
            retry_max_attempts=attempts or None,
            retry_on_pass=bool(rng.random() < 0.5),
        )
        self.residency[cache_id] = residency
        self.core.register(cache_id, residency, scene=cache_id % 2)
        self.latest[cache_id] = {}
        self.t_of.pop(cache_id, None)
        self.retired.discard(cache_id)
        self.incarnation[cache_id] += 1
        policy = residency.policy
        self.stats[f"register_{policy.resource}"] += 1
        self.stats["register_capped" if cap else "register_uncapped"] += 1
        self.stats[
            "register_retry_on_pass" if policy.retry_on_pass else "register_no_retry"
        ] += 1

    def _policy(self, cache_id: int) -> CachePolicy:
        return self.core.policy_of(cache_id)

    def _max_attempts(self, cache_id: int) -> int:
        own = self._policy(cache_id).retry_max_attempts
        return self.config.retry_max_attempts if own is None else own

    def _target_reads_in_flight(self, cache_id: int) -> int:
        reg = self.core.registry(cache_id)
        return int(((reg.state == _FETCHING) & (reg.cls != _BACKSTOP)).sum())

    def _backstop_arrival_waiting(self, cache_id: int) -> bool:
        reg = self.core.registry(cache_id)
        for key in self.core.arrived_keys(cache_id):
            row = reg.find(key)
            if row >= 0 and reg.cls[row] == _BACKSTOP and reg.tier[row] == _VISIBLE:
                return True
        return False

    # -- issue rounds --------------------------------------------------------

    def issue(self) -> None:
        self._round: list[int] = []  # cache ids, in issue order
        super().issue()
        core = self.core
        blocked: set[str] = set()
        for cache in core._caches.values():
            cache_id = cache.cache_id
            policy = cache.policy
            head = core._head(cache)
            if head is not None:
                # 14: work conserving.
                assert core._lane_for(cache, head[0]) is None, (
                    f"cache {cache_id} ({policy.resource}) has an issuable head "
                    f"{head} left after an issue round"
                )
                blocked.add(policy.resource)
            # 13b: the cap never withholds a backstop.
            reg = cache.registry
            queued_backstop = (
                (reg.state == _QUEUED) & (reg.tier == _VISIBLE) & (reg.cls == _BACKSTOP)
            )
            if queued_backstop.any():
                assert core._lane_for(cache, _BACKSTOP) is None, (
                    f"cache {cache_id}: a backstop is queued with room in its lane"
                )
                if policy.max_target_fetching is not None:
                    self.stats["capped_backstop_waits_for_its_lane"] += 1
        issued = {self._policy(c).resource for c in self._round}
        for resource in blocked:
            self.stats[f"round_blocked_{resource}"] += 1
            other = "compute" if resource == "io" else "io"
            if other in issued:
                self.stats[f"{other}_issued_past_blocked_{resource}"] += 1

    def _on_trace(self, kind: str, payload: tuple) -> None:
        if kind == "invalidate":
            cache_id, keys = payload
            for key in keys.tolist():
                self.issued.pop((cache_id, self.incarnation[cache_id], int(key)), None)
        if kind != "issue":
            super()._on_trace(kind, payload)
            return
        self.stats["trace_issue"] += 1
        cache_id, key, cls, lane = payload
        core = self.core
        policy = self._policy(cache_id)
        resource = policy.resource
        self._round.append(cache_id)
        self.stats[f"issue_{resource}"] += 1
        # 12: a retired cache starts no read.
        assert cache_id not in self.retired, f"retired cache {cache_id} issued"
        # 15: lanes follow the resource.
        if resource == "compute":
            assert lane == COMPUTE_LANE, (cache_id, lane)
            assert core.in_flight[COMPUTE_LANE] <= self.config.compute_budget
        else:
            assert lane != COMPUTE_LANE, (cache_id, lane)
        # 17: a given-up key is not read again without an invalidation.
        mark = (cache_id, self.incarnation[cache_id], int(key))
        self.issued[mark] += 1
        if not policy.retry_on_pass:
            assert self.issued[mark] <= self._max_attempts(cache_id), (
                f"cache {cache_id}: key {key} read {self.issued[mark]} times "
                f"with retry_on_pass=False"
            )
        cap = policy.max_target_fetching
        if cls == _BACKSTOP:
            if cap is not None and self._target_reads_in_flight(cache_id) >= cap:
                self.stats["backstop_issued_past_full_cap"] += 1
            return
        if resource == "io":
            assert lane == SHARED_LANE
        if cap is not None:
            # 13: the cap.  The record is FETCHING already, so it is counted.
            n_target = self._target_reads_in_flight(cache_id)
            assert n_target <= cap, f"cache {cache_id}: {n_target} > cap {cap}"
            # 16: not while a backstop arrival waits for its commit.
            assert not self._backstop_arrival_waiting(cache_id), (
                f"cache {cache_id}: target {key} issued ahead of a backstop commit"
            )
        # 9': no backstop of this resource is waiting.
        for other in core.cache_ids:
            if self._policy(other).resource != resource:
                continue
            reg = core.registry(other)
            queued_backstop = (
                (reg.state == _QUEUED) & (reg.tier == _VISIBLE) & (reg.cls == _BACKSTOP)
            )
            assert not queued_backstop.any(), (
                f"target {key} issued ahead of a backstop in cache {other}"
            )

    # -- state checks --------------------------------------------------------

    def check(self) -> None:
        super().check()
        core = self.core
        by_lane = Counter(ticket.lane for _, _, ticket, _ in self.reads)
        # 2': each lane's count is the reads that hold it.
        assert [by_lane[lane] for lane in range(3)] == core.in_flight
        for _, _, ticket, _ in self.reads:
            if not core.is_registered(ticket.cache_id):
                continue
            if ticket.token is not core._caches[ticket.cache_id].token:
                continue
            # 15, for reads in flight.
            compute = self._policy(ticket.cache_id).resource == "compute"
            assert (ticket.lane == COMPUTE_LANE) == compute
        for cache_id in core.cache_ids:
            cap = self._policy(cache_id).max_target_fetching
            if cap is None:
                continue
            # 13: the cap, in every state.
            n_target = self._target_reads_in_flight(cache_id)
            assert n_target <= cap, (cache_id, n_target, cap)
            reg = core.registry(cache_id)
            waiting = (
                (reg.state == _QUEUED) & (reg.tier == _VISIBLE) & (reg.cls != _BACKSTOP)
            )
            if n_target >= cap and waiting.any():
                self.stats["cap_holds_a_target"] += 1
            if self._backstop_arrival_waiting(cache_id) and waiting.any():
                self.stats["commit_holds_a_target"] += 1

    def _check_cache(self, cache_id: int) -> None:
        super()._check_cache(cache_id)
        # A key that left the registry starts over (invariant 17's count).
        reg = self.core.registry(cache_id)
        incarnation = self.incarnation[cache_id]
        live = set(reg.key[reg.state != tp.DEAD].tolist())
        for mark in [m for m in self.issued if m[0] == cache_id]:
            if mark[1] != incarnation or mark[2] not in live:
                del self.issued[mark]
        if not self._policy(cache_id).retry_on_pass:
            given_up = (reg.state == _FAILED) & (
                reg.attempts >= self._max_attempts(cache_id)
            )
            if given_up.any():
                self.stats["given_up_and_kept"] += 1


def _run(seed: int, steps: int, core_class=SchedulerCore) -> Counter[str]:
    class Env(GeneralEnvironment):
        pass

    Env.core_class = core_class
    env = Env(seed)
    if seed % 2 == 1:
        env.prefill()
    actions: Counter[str] = Counter()
    for step in range(steps):
        actions[env.act()] += 1
        env.check()
        if step % 300 == 299:
            env.quiesce()
    env.quiesce()
    return env.stats + actions


@pytest.mark.parametrize("seed", range(12))
def test_invariants_hold_with_random_policies(seed: int) -> None:
    stats = _run(seed, STEPS)
    assert stats["trace_commit"] > 0
    assert stats["trace_issue"] > 0


#: Events the invariants guard; each must happen somewhere in the traces.
REACH = (
    "issue_io",
    "issue_compute",
    "cap_holds_a_target",
    "commit_holds_a_target",
    "backstop_issued_past_full_cap",
    "capped_backstop_waits_for_its_lane",
    "round_blocked_io",
    "round_blocked_compute",
    "io_issued_past_blocked_compute",
    "compute_issued_past_blocked_io",
    "given_up_and_kept",
    "trace_evict",
    "trace_discard",
    "invalidate",
    "retire",
    "remove",
)


def test_the_traces_reach_every_rule() -> None:
    total: Counter[str] = Counter()
    for seed in range(12, 36):
        total += _run(seed, 300)
    for name in REACH:
        assert total[name] > 0, f"no {name} in the traces: {dict(total)}"


# ---------------------------------------------------------------------------
# Mutants
# ---------------------------------------------------------------------------


class _IgnoresTheCap(SchedulerCore):
    """Starts a target read whatever is in flight."""

    def _target_held(self, cache) -> bool:
        return False


class _CapsEveryRead(SchedulerCore):
    """The rule D22 replaced: a backstop is held by the cap too."""

    def _head(self, cache):
        cap = cache.policy.max_target_fetching
        if cap is not None and cache.n_fetching >= cap:
            return None
        return super()._head(cache)


class _IgnoresTheComputeBudget(SchedulerCore):
    def _lane_for(self, cache, cls):
        if cache.policy.resource == "compute":
            return COMPUTE_LANE
        return super()._lane_for(cache, cls)


class _EndsTheRoundOnABlockedHead(SchedulerCore):
    """The loop's rule before the compute lane: right for one budget only."""

    def next_reads(self):
        self._round_over = False
        return super().next_reads()

    def _head(self, cache):
        if getattr(self, "_round_over", False):
            return None
        return super()._head(cache)

    def _lane_for(self, cache, cls):
        lane = super()._lane_for(cache, cls)
        if lane is None:
            self._round_over = True
        return lane


class _PutsComputeReadsInTheSharedWindow(SchedulerCore):
    def _lane_for(self, cache, cls):
        lane = super()._lane_for(cache, cls)
        return SHARED_LANE if lane == COMPUTE_LANE else lane


class _DoesNotWaitForTheBackstopCommit(SchedulerCore):
    def _target_held(self, cache) -> bool:
        cap = cache.policy.max_target_fetching
        if cap is None:
            return False
        reg = cache.registry
        fetching = (reg.state == _FETCHING) & (reg.cls != _BACKSTOP)
        return int(fetching.sum()) >= cap


class _RetriesAGivenUpKeyOnEveryPass(SchedulerCore):
    def _apply(self, cache, desired) -> None:
        if desired is not None and not cache.policy.retry_on_pass:
            reg = cache.registry
            reg.compact()
            rows, found = reg.find_many(desired.keys)
            hit = rows[found]
            again = hit[reg.state[hit] == _FAILED]
            reg.state[again] = _QUEUED
            reg.attempts[again] = 0
        super()._apply(cache, desired)


MUTANTS = (
    _IgnoresTheCap,
    _CapsEveryRead,
    _IgnoresTheComputeBudget,
    _EndsTheRoundOnABlockedHead,
    _PutsComputeReadsInTheSharedWindow,
    _DoesNotWaitForTheBackstopCommit,
    _RetriesAGivenUpKeyOnEveryPass,
)


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda cls: cls.__name__.strip("_"))
def test_a_mutant_core_is_caught(mutant) -> None:
    caught = 0
    seeds = range(20)
    for seed in seeds:
        try:
            _run(seed, 300, core_class=mutant)
        except AssertionError:
            caught += 1
    assert caught > 0, f"{mutant.__name__} passed every invariant on {len(seeds)} seeds"


def test_random_policies_draw_from_the_same_stream_each_run() -> None:
    """The environment is deterministic per seed, so a failure reproduces."""
    assert _run(3, 120) == _run(3, 120)
    assert np.random.default_rng(3).integers(0, 10) == np.random.default_rng(
        3
    ).integers(0, 10)
