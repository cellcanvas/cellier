"""Cache policies, rule by rule (``plans/mesh_refactor_v3.md`` 5.2, K1-K6).

A cache declares a :class:`CachePolicy` once, at registration: a cap on
target reads in flight, the resource a read occupies, and what a failure
costs.  Driven synchronously, as ``test_core.py`` is.
"""

from __future__ import annotations

import asyncio

import pytest

from cellier.render import SchedulerConfig
from cellier.render.scheduling import (
    BACKSTOP_LANE,
    COMPUTE_LANE,
    SHARED_LANE,
    CachePolicy,
    ChunkClass,
    ChunkScheduler,
    ChunkState,
    ReadOutcome,
    SchedulerCore,
)
from tests.render.scheduling._fakes import (
    AsyncStore,
    FakeResidency,
    FakeStore,
    coarse_keys,
    desired,
    fine_keys,
    pack,
)

_TARGET = int(ChunkClass.TARGET)
_BACKSTOP = int(ChunkClass.BACKSTOP)

#: A mesh level cache: one fine read at a time, compute work, one attempt.
LEVELS = CachePolicy(
    max_target_fetching=1,
    resource="compute",
    retry_max_attempts=1,
    retry_on_pass=False,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _core(**config) -> tuple[SchedulerCore, Clock]:
    clock = Clock()
    return SchedulerCore(SchedulerConfig(**config), now=clock), clock


def _residency(n_slots: int = 16, policy: CachePolicy | None = None) -> FakeResidency:
    residency = FakeResidency(n_slots)
    if policy is not None:
        residency.policy = policy
    return residency


def _issue(core: SchedulerCore) -> list:
    core.process()
    return core.next_reads()


def _state(core: SchedulerCore, cache_id: int, key: int) -> int | None:
    reg = core.registry(cache_id)
    row = reg.find(key)
    return None if row < 0 else int(reg.state[row])


# A two-level visual: one coarse key and one fine key per slider position.
def _coarse(t: int) -> int:
    return pack(4, t, 0, 0)


def _fine(t: int) -> int:
    return pack(1, t, 0, 0)


# ---------------------------------------------------------------------------
# The policy itself
# ---------------------------------------------------------------------------


def test_the_default_policy_is_no_cap_io_and_retry_on_pass() -> None:
    policy = CachePolicy()
    assert policy.max_target_fetching is None
    assert policy.resource == "io"
    assert policy.retry_max_attempts is None
    assert policy.retry_on_pass is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_target_fetching": 0},
        {"resource": "gpu"},
        {"retry_max_attempts": 0},
    ],
)
def test_a_policy_rejects_values_the_scheduler_cannot_honour(kwargs) -> None:
    with pytest.raises(ValueError):
        CachePolicy(**kwargs)


def test_an_adapter_without_a_policy_gets_the_default() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(8))
    assert core.policy_of(1) == CachePolicy()


def test_the_policy_is_read_at_registration() -> None:
    core, _ = _core()
    residency = _residency(policy=LEVELS)
    core.register(1, residency)
    assert core.policy_of(1) is LEVELS
    # Changing the adapter afterwards changes nothing, like ``n_slots``.
    residency.policy = CachePolicy()
    assert core.policy_of(1) is LEVELS


def test_a_policy_of_the_wrong_type_is_refused() -> None:
    core, _ = _core()
    residency = FakeResidency(8)
    residency.policy = {"max_target_fetching": 1}
    with pytest.raises(TypeError, match="CachePolicy"):
        core.register(1, residency)


# ---------------------------------------------------------------------------
# K1: the cap counts target reads only
# ---------------------------------------------------------------------------


def test_one_target_read_in_flight_per_capped_cache() -> None:
    core, _ = _core()
    core.register(1, _residency(policy=CachePolicy(max_target_fetching=1)))
    core.set_desired(desired(1, target=fine_keys(1, n=3)))
    first = _issue(core)
    assert [t.key for t in first] == fine_keys(1, n=1)
    # Nothing more until it lands.
    assert core.next_reads() == []
    core.complete_read(first[0], data=0)
    assert [t.key for t in core.next_reads()] == fine_keys(1, n=2)[1:]


def test_an_uncapped_cache_issues_every_target() -> None:
    core, _ = _core()
    core.register(1, _residency())
    core.set_desired(desired(1, target=fine_keys(1, n=3)))
    assert len(_issue(core)) == 3


def test_a_coarse_read_starts_while_a_fine_read_is_in_flight() -> None:
    """The gap D22 closes: a slider moved again while the fine read of the
    position it left is still running must get its coarse read at once."""
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], target=[_fine(0)], store=store))
    # Coarse first, then fine: the planner's order.
    assert [t.key for t in _issue(core)] == [_coarse(0), _fine(0)]

    # The slider moves on with the fine read still running.
    core.set_desired(desired(1, backstop=[_coarse(1)], store=store))
    assert [t.key for t in _issue(core)] == [_coarse(1)]
    assert _state(core, 1, _fine(0)) == int(ChunkState.FETCHING)

    # And again: still not held, and the next fine read still is.
    core.set_desired(desired(1, backstop=[_coarse(2)], target=[_fine(2)], store=store))
    assert [t.key for t in _issue(core)] == [_coarse(2)]
    assert _state(core, 1, _fine(2)) == int(ChunkState.QUEUED)


def test_a_target_read_no_longer_wanted_still_holds_the_cap() -> None:
    """A read cannot be cancelled.  If an unwanted one did not count, fine
    reads would pile up to the compute budget and block the coarse ones."""
    core, _ = _core()
    core.register(1, _residency(policy=CachePolicy(max_target_fetching=1)))
    store = FakeStore()
    core.set_desired(desired(1, target=[_fine(0)], store=store))
    fine0 = _issue(core)
    core.set_desired(desired(1, target=[_fine(1)], store=store))
    assert _issue(core) == []
    core.complete_read(fine0[0], data=0)
    assert [t.key for t in core.next_reads()] == [_fine(1)]


def test_two_coarse_reads_of_one_cache_can_overlap() -> None:
    """A consequence to build for: arrivals can land out of tick order."""
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    first = _issue(core)
    core.set_desired(desired(1, backstop=[_coarse(1)], store=store))
    second = _issue(core)
    assert [t.key for t in first + second] == [_coarse(0), _coarse(1)]
    assert core.in_flight[COMPUTE_LANE] == 2
    # The later one lands first; both are kept for the commit round.
    assert core.complete_read(second[0], data=1) == ReadOutcome.ARRIVED
    assert core.complete_read(first[0], data=0) == ReadOutcome.ARRIVED


def test_the_cap_is_per_cache() -> None:
    core, _ = _core()
    for cache_id in (1, 2):
        core.register(cache_id, _residency(policy=CachePolicy(max_target_fetching=1)))
        core.set_desired(desired(cache_id, target=fine_keys(1, n=2)))
    tickets = _issue(core)
    assert sorted(t.cache_id for t in tickets) == [1, 2]


# ---------------------------------------------------------------------------
# K2: the compute lane
# ---------------------------------------------------------------------------


def test_compute_reads_take_the_compute_lane_and_nothing_else() -> None:
    core, _ = _core(compute_budget=2, max_in_flight=8, backstop_reserved=2)
    core.register(1, _residency(policy=CachePolicy(resource="compute")))
    core.set_desired(desired(1, backstop=coarse_keys(), target=fine_keys(1, n=3)))
    tickets = _issue(core)
    assert len(tickets) == 2
    assert {t.lane for t in tickets} == {COMPUTE_LANE}
    assert core.in_flight == [0, 0, 2]
    core.complete_read(tickets[0], data=0)
    assert core.in_flight == [0, 0, 1]


def test_io_reads_never_take_the_compute_lane() -> None:
    core, _ = _core(compute_budget=2, max_in_flight=2, backstop_reserved=1)
    core.register(1, _residency())
    core.set_desired(desired(1, backstop=coarse_keys(), target=fine_keys(1, n=3)))
    tickets = _issue(core)
    assert sorted(t.lane for t in tickets) == [SHARED_LANE, SHARED_LANE, BACKSTOP_LANE]
    assert core.in_flight[COMPUTE_LANE] == 0


def test_the_compute_budget_is_shared_by_every_compute_cache() -> None:
    core, _ = _core(compute_budget=3)
    for cache_id in (1, 2):
        core.register(cache_id, _residency(policy=CachePolicy(resource="compute")))
        core.set_desired(desired(cache_id, target=fine_keys(1, n=4)))
    tickets = _issue(core)
    assert len(tickets) == 3
    assert core.in_flight == [0, 0, 3]
    # Round robin between the two.
    assert sorted(t.cache_id for t in tickets) in ([1, 1, 2], [1, 2, 2])


def test_the_default_compute_budget_is_four() -> None:
    assert SchedulerConfig().compute_budget == 4
    with pytest.raises(ValueError, match="compute_budget"):
        SchedulerConfig(compute_budget=0)


def test_an_idle_core_has_no_compute_read_outstanding() -> None:
    core, _ = _core()
    core.register(1, _residency(policy=CachePolicy(resource="compute")))
    core.set_desired(desired(1, target=fine_keys(1, n=1)))
    tickets = _issue(core)
    assert not core.idle()
    core.complete_read(tickets[0], data=0)
    core.commit_round()
    assert core.idle()


# ---------------------------------------------------------------------------
# K3: a blocked resource does not end the round
# ---------------------------------------------------------------------------


def test_a_full_compute_lane_does_not_hold_io_reads() -> None:
    core, _ = _core(compute_budget=1, max_in_flight=4, backstop_reserved=0)
    # The compute cache's backstop head outranks the io cache's target head,
    # so the round meets the blocked lane first.
    core.register(1, _residency(policy=CachePolicy(resource="compute")))
    core.register(2, _residency())
    core.set_desired(desired(1, backstop=coarse_keys()))
    core.set_desired(desired(2, target=fine_keys(1, n=3)))
    tickets = _issue(core)
    assert [t.cache_id for t in tickets].count(1) == 1
    assert [t.cache_id for t in tickets].count(2) == 3


def test_a_full_shared_window_does_not_hold_compute_reads() -> None:
    core, _ = _core(compute_budget=2, max_in_flight=1, backstop_reserved=0)
    core.register(1, _residency())
    core.register(2, _residency(policy=CachePolicy(resource="compute")))
    core.set_desired(desired(1, backstop=coarse_keys()))
    core.set_desired(desired(2, target=fine_keys(1, n=3)))
    tickets = _issue(core)
    assert [t.cache_id for t in tickets].count(1) == 1
    assert [t.cache_id for t in tickets].count(2) == 2


# ---------------------------------------------------------------------------
# K4: a target waits for the backstop's commit
# ---------------------------------------------------------------------------


def test_a_capped_cache_holds_its_target_until_the_backstop_is_committed() -> None:
    """A scrub's end: the last tick's coarse read has landed and waits for
    its commit round when the end plans the fine level."""
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    coarse = _issue(core)
    core.complete_read(coarse[0], data=0)
    core.set_desired(desired(1, backstop=[_coarse(0)], target=[_fine(0)], store=store))
    # Landed, not committed: the fine read would compete with the commit.
    assert _issue(core) == []
    core.commit_round()
    assert [t.key for t in core.next_reads()] == [_fine(0)]


def test_an_uncapped_cache_does_not_wait_for_the_commit() -> None:
    core, _ = _core(max_in_flight=1, backstop_reserved=0)
    core.register(1, _residency())
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    coarse = _issue(core)
    core.complete_read(coarse[0], data=0)
    core.set_desired(desired(1, backstop=[_coarse(0)], target=[_fine(0)], store=store))
    assert [t.key for t in _issue(core)] == [_fine(0)]


def test_an_arrival_that_is_no_longer_wanted_does_not_hold_the_target() -> None:
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    coarse0 = _issue(core)
    # The slider moved on, and its new coarse level is resident already.
    core.set_desired(desired(1, backstop=[_coarse(1)], store=store))
    coarse1 = _issue(core)
    core.complete_read(coarse1[0], data=1)
    core.commit_round()
    core.complete_read(coarse0[0], data=0)
    core.set_desired(desired(1, backstop=[_coarse(1)], target=[_fine(1)], store=store))
    assert [t.key for t in _issue(core)] == [_fine(1)]


async def test_a_commit_round_pumps() -> None:
    """The scheduler issues the held target when the round has committed."""
    store = AsyncStore(latency=0.001)
    residency = _residency(policy=LEVELS)
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=30.0))
    scheduler.register(1, residency, scene="scene")
    scheduler.pass_([desired(1, backstop=[_coarse(0)], store=store)])
    for _ in range(200):
        await asyncio.sleep(0.001)
        if scheduler.core.arrived_keys(1):
            break
    assert scheduler.core.arrived_keys(1) == [_coarse(0)]
    scheduler.pass_([desired(1, backstop=[_coarse(0)], target=[_fine(0)], store=store)])
    await asyncio.sleep(0.01)
    # Held: the coarse arrival has not been committed.
    assert store.calls == [_coarse(0)]

    scheduler.commit_round("scene")
    await asyncio.sleep(0.01)
    assert store.calls == [_coarse(0), _fine(0)]
    scheduler.close()


async def test_a_round_that_only_discards_pumps_too() -> None:
    """A round can free the hold without committing: the arrival was for a
    key invalidated or dropped.  The pump is unconditional."""
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=30.0))
    scheduler.register(1, _residency(policy=LEVELS), scene="scene")
    pumps: list[int] = []
    original = scheduler._pump
    scheduler._pump = lambda: (pumps.append(1), original())[1]
    assert scheduler.commit_round("scene") == []
    assert pumps == [1]
    scheduler.close()


# ---------------------------------------------------------------------------
# K5: the draw is rebuilt before completion fires
# ---------------------------------------------------------------------------


def test_completion_fires_after_the_draw_is_rebuilt() -> None:
    core, _ = _core()
    residency = _residency()
    core.register(1, residency)
    drawn_at_completion: list[dict] = []
    drawn_at_backstop: list[dict] = []
    core.on_complete = lambda c, g: drawn_at_completion.append(dict(residency.lut))
    core.on_backstop_complete = lambda c, g: drawn_at_backstop.append(
        dict(residency.lut)
    )
    key = _coarse(0)
    core.set_desired(desired(1, backstop=[key]))
    tickets = _issue(core)
    core.complete_read(tickets[0], data=0)
    core.commit_round()
    assert len(drawn_at_completion) == len(drawn_at_backstop) == 1
    # The callback saw the committed key already painted.
    assert key in set(drawn_at_completion[0].values())
    assert key in set(drawn_at_backstop[0].values())


def test_an_all_kept_pass_still_completes_with_its_draw_built() -> None:
    """A static visual on an unrelated slider: nothing is read, and the
    generation completes in the pass itself."""
    core, _ = _core()
    residency = _residency(policy=LEVELS)
    core.register(1, residency)
    store = FakeStore()
    done: list[tuple[int, int, int]] = []
    core.on_complete = lambda c, g: done.append((c, g, residency.n_rebuilds))
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    tickets = _issue(core)
    core.complete_read(tickets[0], data=0)
    core.commit_round()
    rebuilds = residency.n_rebuilds
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    assert _issue(core) == []
    assert done[-1] == (1, 2, rebuilds + 1)


# ---------------------------------------------------------------------------
# K6: failures per cache
# ---------------------------------------------------------------------------


def test_a_cache_overrides_the_number_of_attempts() -> None:
    core, _ = _core(retry_max_attempts=3, retry_backoff_s=0.0)
    core.register(1, _residency(policy=CachePolicy(retry_max_attempts=1)))
    core.register(2, _residency())
    for cache_id in (1, 2):
        core.set_desired(desired(cache_id, target=[_fine(0)]))
    tickets = {t.cache_id: t for t in _issue(core)}
    error = ValueError("kernel")
    assert core.complete_read(tickets[1], error=error) == ReadOutcome.GAVE_UP
    assert core.complete_read(tickets[2], error=error) == ReadOutcome.RETRY
    assert core.progress(1).failed == 1
    assert core.progress(2).failed == 0


def test_a_given_up_key_is_not_read_again_by_a_pass_that_wants_it() -> None:
    """Every reslice of a scene passes every visual in it: a deterministic
    kernel error would otherwise rerun on every slider tick."""
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    done: list[int] = []
    core.on_complete = lambda c, g: done.append(g)
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    tickets = _issue(core)
    assert core.complete_read(tickets[0], error=ValueError("kernel")) == (
        ReadOutcome.GAVE_UP
    )
    assert done == [1]
    for generation in (2, 3, 4):
        core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
        assert _issue(core) == []
        # Given up still counts as done: completion fires for each pass.
        assert done[-1] == generation
        assert core.is_complete(1)
        assert core.progress(1).failed == 1
    assert core.idle()


def test_the_default_policy_grants_one_more_attempt_per_pass() -> None:
    core, _ = _core(retry_max_attempts=1)
    core.register(1, _residency())
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    tickets = _issue(core)
    core.complete_read(tickets[0], error=OSError("down"))
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    assert [t.key for t in _issue(core)] == [_coarse(0)]


def test_a_given_up_key_is_read_again_after_an_invalidation() -> None:
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    tickets = _issue(core)
    core.complete_read(tickets[0], error=ValueError("kernel"))
    assert core.next_reads() == []
    assert core.invalidate(store.id) == [1]
    again = core.next_reads()
    assert [t.key for t in again] == [_coarse(0)]
    assert core.complete_read(again[0], data=0) == ReadOutcome.ARRIVED


def test_a_given_up_key_is_read_again_when_the_key_changes() -> None:
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    store = FakeStore()
    core.set_desired(desired(1, backstop=[_coarse(0)], store=store))
    tickets = _issue(core)
    core.complete_read(tickets[0], error=ValueError("kernel"))
    core.set_desired(desired(1, backstop=[_coarse(1)], store=store))
    assert [t.key for t in _issue(core)] == [_coarse(1)]
    # The failed record went with the pass that stopped wanting it.
    assert _state(core, 1, _coarse(0)) is None


def test_a_given_up_level_does_not_stop_the_other_level_loading() -> None:
    core, _ = _core()
    core.register(1, _residency(policy=LEVELS))
    core.set_desired(desired(1, backstop=[_coarse(0)], target=[_fine(0)]))
    coarse, fine = _issue(core)
    assert (coarse.key, fine.key) == (_coarse(0), _fine(0))
    core.complete_read(coarse, error=ValueError("kernel"))
    core.complete_read(fine, data=0)
    core.commit_round()
    assert core.is_complete(1)
    assert core.progress(1).failed == 1
    assert core.progress(1).resident_target == 1
