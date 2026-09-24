"""The scheduler core, rule by rule (design v3, sections 5.2-5.7, 5.13-5.14).

Driven synchronously: the test issues reads with ``next_reads``, answers
them with ``complete_read`` and runs ``commit_round`` itself, on a manual
clock.  ``test_property.py`` checks the invariants over random traces.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render import SchedulerConfig
from cellier.render.scheduling import (
    BACKSTOP_LANE,
    SHARED_LANE,
    ChunkClass,
    ChunkState,
    ReadOutcome,
    SchedulerCore,
    Tier,
)
from tests.render.scheduling._fakes import (
    FakeResidency,
    FakeStore,
    coarse_keys,
    desired,
    fine_keys,
    pack,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _core(**config) -> tuple[SchedulerCore, Clock]:
    clock = Clock()
    core = SchedulerCore(SchedulerConfig(**config), now=clock)
    return core, clock


def _state(core: SchedulerCore, cache_id: int, key: int) -> tuple[int, int] | None:
    reg = core.registry(cache_id)
    row = reg.find(key)
    return None if row < 0 else (int(reg.state[row]), int(reg.tier[row]))


def _land(core: SchedulerCore, tickets, data=None) -> list[str]:
    return [
        core.complete_read(t, data=t.key if data is None else data) for t in tickets
    ]


# ---------------------------------------------------------------------------
# Registration and validation
# ---------------------------------------------------------------------------


def test_register_twice_is_refused() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(8))
    with pytest.raises(ValueError, match="already registered"):
        core.register(1, FakeResidency(8))


def test_a_cache_needs_two_slots() -> None:
    core, _ = _core()
    with pytest.raises(ValueError, match="at least 2 slots"):
        core.register(1, FakeResidency(1))


def test_desired_set_for_an_unknown_cache() -> None:
    core, _ = _core()
    with pytest.raises(KeyError):
        core.set_desired(desired(9, target=[pack(1, 0, 0, 0)]))


def test_planner_must_truncate_to_n_slots_minus_one() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(4))
    core.set_desired(desired(1, target=fine_keys(1, n=3)))  # fits
    with pytest.raises(ValueError, match="n_slots - 1"):
        core.set_desired(desired(1, target=fine_keys(1, n=4)))


def test_repeated_keys_are_refused() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(8))
    k = pack(1, 0, 0, 0)
    with pytest.raises(ValueError, match="repeat"):
        core.set_desired(desired(1, target=[k, k]))


def test_desired_set_arrays_must_line_up() -> None:
    from cellier.render.scheduling import DesiredSet

    with pytest.raises(ValueError, match="same length"):
        DesiredSet(
            cache_id=1,
            keys=np.array([1, 2]),
            cls=np.array([0]),
            slice_ids=np.array([0, 0]),
            build_request=list,
            store=FakeStore(),
        )


# ---------------------------------------------------------------------------
# The pass (5.4)
# ---------------------------------------------------------------------------


def test_pass_queues_in_planner_order_backstop_first() -> None:
    core, _ = _core(max_in_flight=100, backstop_reserved=0)
    core.register(1, FakeResidency(64))
    target = fine_keys(2, n=5)
    backstop = coarse_keys()
    # The planner's target order is kept, not the key order.
    core.set_desired(desired(1, backstop=backstop, target=target[::-1]))
    core.process()
    issued = [t.key for t in core.next_reads()]
    assert issued == backstop + target[::-1]


def test_passes_coalesce_to_the_latest() -> None:
    core, _ = _core()
    residency = FakeResidency(16)
    core.register(1, residency)
    core.set_desired(desired(1, target=fine_keys(1, t=0, n=3)))
    core.set_desired(desired(1, target=fine_keys(1, t=1, n=3)))
    assert core.process() == [1]
    assert core.generation(1) == 1
    assert sorted(core.registry(1).key.tolist()) == sorted(fine_keys(1, t=1, n=3))


def test_dropping_deletes_queued_and_keeps_fetching() -> None:
    core, _ = _core(max_in_flight=2, backstop_reserved=0)
    core.register(1, FakeResidency(16))
    old = fine_keys(1, t=0, n=4)
    core.set_desired(desired(1, target=old))
    core.process()
    fetching = core.next_reads()
    assert [t.key for t in fetching] == old[:2]

    core.set_desired(desired(1, target=fine_keys(1, t=1, n=2)))
    core.process()
    # In flight: kept, now RECENT.  Queued and never read: deleted.
    for key in old[:2]:
        assert _state(core, 1, key) == (ChunkState.FETCHING, Tier.RECENT)
    for key in old[2:]:
        assert _state(core, 1, key) is None


def test_a_superseded_read_is_kept_and_committed_as_recent() -> None:
    core, _ = _core(max_in_flight=1, backstop_reserved=0)
    residency = FakeResidency(16)
    core.register(1, residency)
    old = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[old]))
    core.process()
    (ticket,) = core.next_reads()
    core.set_desired(desired(1, target=[pack(1, 1, 0, 0)]))
    core.process()

    assert core.complete_read(ticket, data="bytes") == ReadOutcome.ARRIVED
    assert _state(core, 1, old) == (ChunkState.ARRIVED, Tier.RECENT)
    core.commit_round()
    assert _state(core, 1, old) == (ChunkState.RESIDENT, Tier.RECENT)
    assert (0, old) in residency.writes


def test_wanted_again_resident_costs_nothing() -> None:
    core, _ = _core()
    residency = FakeResidency(16)
    core.register(1, residency)
    key = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[key]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    core.set_desired(desired(1, target=[pack(1, 1, 0, 0)]))
    core.process()
    core.next_reads()
    assert _state(core, 1, key) == (ChunkState.RESIDENT, Tier.RECENT)

    core.set_desired(desired(1, target=[key]))
    core.process()
    assert _state(core, 1, key) == (ChunkState.RESIDENT, Tier.VISIBLE)
    assert key not in [t.key for t in core.next_reads()]
    # It is foreground in the draw straight away.
    assert residency.lut[(0, 0)] == key


def test_a_pass_redraws_its_cache() -> None:
    core, _ = _core()
    residency = FakeResidency(16)
    core.register(1, residency)
    core.set_desired(desired(1, target=[pack(1, 0, 0, 0)]))
    core.process()
    assert residency.n_rebuilds == 1


# ---------------------------------------------------------------------------
# Fetching: budget, lane, class first, round robin (5.5, 5.6)
# ---------------------------------------------------------------------------


def test_budget_and_backstop_lane() -> None:
    core, _ = _core(max_in_flight=3, backstop_reserved=2)
    core.register(1, FakeResidency(64))
    backstop = coarse_keys()  # 4 keys
    core.set_desired(desired(1, backstop=backstop, target=fine_keys(1, n=10)))
    core.process()
    tickets = core.next_reads()
    # Backstops take the lane first, then shared capacity; targets get the rest.
    lanes = [(t.key in backstop, t.lane) for t in tickets]
    assert lanes == [
        (True, BACKSTOP_LANE),
        (True, BACKSTOP_LANE),
        (True, SHARED_LANE),
        (True, SHARED_LANE),
        (False, SHARED_LANE),
    ]
    assert core.in_flight == [3, 2]
    assert core.next_reads() == []

    # A finished backstop read frees a lane slot, which a target cannot use.
    core.complete_read(tickets[0], data=1)
    assert core.next_reads() == []
    core.complete_read(tickets[4], data=1)
    (more,) = core.next_reads()
    assert more.lane == SHARED_LANE


def test_every_backstop_goes_before_any_target() -> None:
    core, _ = _core(max_in_flight=4, backstop_reserved=0)
    for cache_id in (1, 2):
        core.register(cache_id, FakeResidency(64))
    core.set_desired(desired(1, target=fine_keys(1, n=6)))
    core.set_desired(desired(2, backstop=coarse_keys(), target=fine_keys(1, n=6)))
    core.process()
    tickets = core.next_reads()
    assert [(t.cache_id, t.key) for t in tickets] == [(2, k) for k in coarse_keys()]


def test_round_robin_between_caches() -> None:
    core, _ = _core(max_in_flight=6, backstop_reserved=0)
    for cache_id in (1, 2, 3):
        core.register(cache_id, FakeResidency(64))
        core.set_desired(desired(cache_id, target=fine_keys(1, n=10)))
    core.process()
    order: list[int] = []
    core.trace = lambda kind, payload: (
        order.append(payload[0]) if kind == "issue" else None
    )
    core.next_reads()
    assert sorted(order[:3]) == [1, 2, 3]  # each cache once...
    assert order == order[:3] * 2  # ...then again in the same rotation


def test_a_small_cache_is_not_paced_by_a_big_one() -> None:
    """D8: round robin finishes a small layer early."""
    core, _ = _core(max_in_flight=2, backstop_reserved=0)
    core.register(1, FakeResidency(256))
    core.register(2, FakeResidency(8))
    core.set_desired(desired(1, target=fine_keys(1, n=200)))
    core.set_desired(desired(2, target=fine_keys(2, n=4)))
    core.process()
    served = []
    for _ in range(4):
        tickets = core.next_reads()
        served += [t.cache_id for t in tickets]
        _land(core, tickets)
    assert served.count(2) == 4


def test_batches_call_build_request_once_per_cache() -> None:
    core, _ = _core(max_in_flight=8, backstop_reserved=0)
    core.register(1, FakeResidency(64))
    calls = []
    ds = desired(1, target=fine_keys(1, n=6))
    builder = ds.build_request

    def counting(keys):
        calls.append(len(keys))
        return builder(keys)

    object.__setattr__(ds, "build_request", counting)
    core.set_desired(ds)
    core.process()
    assert len(core.next_reads()) == 6
    assert calls == [6]


def test_a_failing_request_builder_fails_the_reads() -> None:
    core, _ = _core(retry_max_attempts=1)
    core.register(1, FakeResidency(8))
    ds = desired(1, target=fine_keys(1, n=2))

    def broken(keys):
        raise RuntimeError("no")

    object.__setattr__(ds, "build_request", broken)
    core.set_desired(ds)
    core.process()
    assert core.next_reads() == []
    assert core.in_flight == [0, 0]
    assert core.progress(1).failed == 2
    assert core.is_complete(1)


# ---------------------------------------------------------------------------
# Commit rounds and eviction (5.7)
# ---------------------------------------------------------------------------


def test_commit_round_writes_and_redraws_once_per_cache() -> None:
    core, _ = _core()
    residency = FakeResidency(16)
    core.register(1, residency)
    keys = fine_keys(1, n=5)
    core.set_desired(desired(1, target=keys))
    core.process()
    _land(core, core.next_reads())
    rebuilds = residency.n_rebuilds
    assert core.commit_round() == [1]
    assert residency.n_rebuilds == rebuilds + 1
    assert sorted(k for _, k in residency.writes) == sorted(keys)
    assert sorted(s for s, _ in residency.writes) == list(range(5))
    assert core.commit_round() == []


def _fill_with_recent(core: SchedulerCore, cache_id: int, n: int, t: int) -> list[int]:
    """Make *n* residents at slice *t*, then drop them (RECENT)."""
    keys = fine_keys(1, t=t, n=n)
    core.set_desired(desired(cache_id, target=keys))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    return keys


def test_visible_arrival_evicts_the_least_important_recent() -> None:
    core, _ = _core(max_in_flight=100)
    core.register(1, FakeResidency(4))
    # Gen 1 and gen 2 residents, both dropped by gen 3.
    older = _fill_with_recent(core, 1, 2, t=0)
    newer = fine_keys(1, t=1, n=2)
    core.set_desired(desired(1, target=newer))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    assert core.free_slots(1) == []

    want = fine_keys(1, t=2, n=1)
    core.set_desired(desired(1, target=want))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    # The victim is the least important RECENT: the oldest wanted_gen, and
    # within it the highest rank.
    assert _state(core, 1, older[1]) is None
    assert _state(core, 1, older[0]) == (ChunkState.RESIDENT, Tier.RECENT)
    assert _state(core, 1, want[0]) == (ChunkState.RESIDENT, Tier.VISIBLE)


def test_stale_backstops_outlive_stale_targets() -> None:
    """Class first: every stale target goes before any stale backstop."""
    core, _ = _core(max_in_flight=100)
    core.register(1, FakeResidency(4))
    backstop = pack(4, 0, 0, 0)
    core.set_desired(desired(1, backstop=[backstop]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    # Newer, but target-class, residents.
    targets = fine_keys(1, t=1, n=3)
    core.set_desired(desired(1, target=targets))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()

    core.set_desired(desired(1, target=fine_keys(1, t=2, n=1)))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    assert _state(core, 1, backstop) == (ChunkState.RESIDENT, Tier.RECENT)
    assert sum(_state(core, 1, k) is None for k in targets) == 1


def test_kept_arrival_evicts_only_a_lower_ranked_recent() -> None:
    core, _ = _core(max_in_flight=100)
    core.register(1, FakeResidency(3))
    # Two backstop residents at gen 1.
    core.set_desired(desired(1, backstop=coarse_keys()[:2]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    # A target read at gen 2, superseded in flight at gen 3.
    stray = pack(1, 1, 0, 0)
    core.set_desired(desired(1, target=[stray]))
    core.process()
    (ticket,) = core.next_reads()
    core.set_desired(desired(1, target=[pack(1, 2, 0, 0)]))
    core.process()
    blocker = core.next_reads()
    _land(core, blocker)
    core.commit_round()  # the gen-3 target takes the free slot
    assert core.free_slots(1) == []

    # The kept target ranks below every stale backstop: discarded.
    assert core.complete_read(ticket, data=1) == ReadOutcome.ARRIVED
    core.commit_round()
    assert _state(core, 1, stray) is None
    for key in coarse_keys()[:2]:
        assert _state(core, 1, key) == (ChunkState.RESIDENT, Tier.RECENT)


def test_kept_backstop_replaces_a_stale_target() -> None:
    core, _ = _core(max_in_flight=100)
    core.register(1, FakeResidency(3))
    stale = _fill_with_recent(core, 1, 2, t=0)
    backstop = pack(4, 1, 0, 0)
    core.set_desired(desired(1, backstop=[backstop]))
    core.process()
    (ticket,) = core.next_reads()
    core.set_desired(desired(1, target=[pack(1, 2, 0, 0)]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    assert core.free_slots(1) == []

    core.complete_read(ticket, data=1)
    core.commit_round()
    assert _state(core, 1, backstop) == (ChunkState.RESIDENT, Tier.RECENT)
    assert sum(_state(core, 1, k) is None for k in stale) == 1


def test_scene_scoped_rounds() -> None:
    core, _ = _core()
    a, b = FakeResidency(8), FakeResidency(8)
    core.register(1, a, scene="A")
    core.register(2, b, scene="B")
    for cache_id in (1, 2):
        core.set_desired(desired(cache_id, target=fine_keys(1, n=2)))
    core.process()
    _land(core, core.next_reads())
    assert core.commit_round("A") == [1]
    assert a.writes and not b.writes
    assert core.oldest_arrival() is not None
    assert core.commit_round() == [2]
    assert core.oldest_arrival() is None


def test_a_failing_write_gives_the_record_up() -> None:
    core, _ = _core(retry_max_attempts=3)

    class Broken(FakeResidency):
        def write(self, slot, key, data):
            raise RuntimeError("upload failed")

    core.register(1, Broken(8))
    key = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[key]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    assert _state(core, 1, key) == (ChunkState.FAILED, Tier.VISIBLE)
    assert len(core.free_slots(1)) == 8
    assert core.next_retry_at() is None  # no retry storm
    assert core.is_complete(1)
    assert core.progress(1).failed == 1


# ---------------------------------------------------------------------------
# Completion and progress (5.13)
# ---------------------------------------------------------------------------


def test_completion_fires_once_per_generation() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(16))
    done, backstop_done = [], []
    core.on_complete = lambda c, g: done.append((c, g))
    core.on_backstop_complete = lambda c, g: backstop_done.append((c, g))
    core.set_desired(desired(1, backstop=coarse_keys(), target=fine_keys(2, n=3)))
    core.process()
    tickets = core.next_reads()
    _land(core, tickets[:4])  # the backstop
    core.commit_round()
    assert backstop_done == [(1, 1)]
    assert done == []
    _land(core, tickets[4:])
    core.commit_round()
    core.commit_round()
    assert done == [(1, 1)]


def test_superseded_generation_never_completes() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(16))
    done = []
    core.on_complete = lambda c, g: done.append(g)
    core.set_desired(desired(1, target=fine_keys(1, t=0, n=2)))
    core.process()
    first = core.next_reads()
    core.set_desired(desired(1, target=fine_keys(1, t=1, n=2)))
    core.process()
    second = core.next_reads()
    _land(core, first)
    core.commit_round()
    assert done == []
    _land(core, second)
    core.commit_round()
    assert done == [2]


def test_background_is_drawn_until_complete() -> None:
    core, _ = _core(max_in_flight=100)
    residency = FakeResidency(16)
    core.register(1, residency)
    old = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[old]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()

    new = pack(1, 1, 0, 0)
    core.set_desired(desired(1, target=[new]))
    core.process()
    # The new slice is loading: the old one shows underneath.
    assert residency.lut[(0, 0)] == old
    _land(core, core.next_reads())
    core.commit_round()
    assert residency.lut[(0, 0)] == new
    assert residency.last_complete is True


def test_newest_background_slice_is_on_top() -> None:
    """D7: a newer slice's backstop covers an older slice's fine tiles."""
    core, _ = _core(max_in_flight=100)
    residency = FakeResidency(32)
    core.register(1, residency)
    fine = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[fine]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    backstop = pack(4, 1, 0, 0)
    core.set_desired(desired(1, backstop=[backstop]))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    # Move on to t=2, which has not loaded: both older slices are background.
    core.set_desired(desired(1, target=[pack(1, 2, 0, 0)]))
    core.process()
    assert residency.lut[(0, 0)] == backstop


def test_progress_counts() -> None:
    core, _ = _core(max_in_flight=2, backstop_reserved=1, retry_max_attempts=1)
    core.register(1, FakeResidency(16))
    core.set_desired(
        desired(
            1,
            backstop=coarse_keys()[:2],
            target=fine_keys(1, n=3),
            n_truncated_target=7,
            n_truncated_backstop=1,
        )
    )
    core.process()
    tickets = core.next_reads()  # 2 backstop + 1 target
    core.complete_read(tickets[0], data=1)
    core.complete_read(tickets[2], error=OSError("x"))
    core.commit_round()
    p = core.progress(1)
    assert (p.needed_backstop, p.resident_backstop) == (2, 1)
    assert (p.needed_target, p.resident_target) == (3, 0)
    assert p.in_flight == 1
    assert p.failed == 1
    assert (p.truncated_target, p.truncated_backstop) == (7, 1)
    assert (p + p).needed_target == 6


# ---------------------------------------------------------------------------
# Retry (5.2)
# ---------------------------------------------------------------------------


def test_failed_reads_retry_with_backoff_then_give_up() -> None:
    core, clock = _core(retry_max_attempts=3, retry_backoff_s=0.25)
    core.register(1, FakeResidency(8))
    done = []
    core.on_complete = lambda c, g: done.append(g)
    key = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[key]))
    core.process()

    delays = []
    for attempt in (1, 2, 3):
        (ticket,) = core.next_reads()
        outcome = core.complete_read(ticket, error=OSError("x"))
        if attempt < 3:
            assert outcome == ReadOutcome.RETRY
            due = core.next_retry_at()
            delays.append(due - clock.t)
            assert core.requeue_due() == 0  # not yet
            clock.t = due
            assert core.requeue_due() == 1
        else:
            assert outcome == ReadOutcome.GAVE_UP
    assert delays == [0.25, 0.5]
    assert core.next_retry_at() is None
    assert done == [1]  # given up counts as done
    assert core.idle()

    # A later pass that still wants it grants one more attempt.
    core.set_desired(desired(1, target=[key]))
    core.process()
    (ticket,) = core.next_reads()
    assert core.complete_read(ticket, error=OSError("x")) == ReadOutcome.GAVE_UP


def test_a_failed_read_no_longer_wanted_is_deleted() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(8))
    key = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[key]))
    core.process()
    (ticket,) = core.next_reads()
    core.set_desired(desired(1, target=[pack(1, 1, 0, 0)]))
    core.process()
    assert core.complete_read(ticket, error=OSError("x")) == ReadOutcome.DROPPED
    assert _state(core, 1, key) is None


# ---------------------------------------------------------------------------
# Invalidation (5.14)
# ---------------------------------------------------------------------------


def test_invalidating_a_fetch_drops_its_arrival_and_requeues() -> None:
    core, _ = _core()
    store = FakeStore()
    core.register(1, FakeResidency(8))
    key = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[key], store=store))
    core.process()
    (ticket,) = core.next_reads()
    assert core.invalidate(store.id) == [1]
    assert core.complete_read(ticket, data="old") == ReadOutcome.DROPPED
    assert _state(core, 1, key) == (ChunkState.QUEUED, Tier.VISIBLE)
    (again,) = core.next_reads()
    assert again.key == key


def test_invalidating_residents_requeues_visible_and_deletes_recent() -> None:
    core, _ = _core(max_in_flight=100)
    store = FakeStore()
    residency = FakeResidency(8)
    core.register(1, residency)
    old = pack(1, 0, 0, 0)
    core.set_desired(desired(1, target=[old], store=store))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    new = pack(1, 1, 0, 0)
    core.set_desired(desired(1, target=[new], store=store))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    assert len(core.free_slots(1)) == 6

    core.invalidate(store.id)
    assert _state(core, 1, old) is None
    assert _state(core, 1, new) == (ChunkState.QUEUED, Tier.VISIBLE)
    assert len(core.free_slots(1)) == 8
    assert residency.lut == {}


def test_invalidation_is_limited_to_the_region_and_the_store() -> None:
    core, _ = _core(max_in_flight=100)
    store, other = FakeStore(), FakeStore()
    core.register(1, FakeResidency(8))
    core.register(2, FakeResidency(8))
    keys = fine_keys(1, n=2)
    core.set_desired(desired(1, target=keys, store=store))
    core.set_desired(desired(2, target=keys, store=other))
    core.process()
    _land(core, core.next_reads())
    core.commit_round()
    touched = core.invalidate(store.id, region=lambda k: k == keys[0])
    assert touched == [1]
    assert _state(core, 1, keys[0]) == (ChunkState.QUEUED, Tier.VISIBLE)
    assert _state(core, 1, keys[1]) == (ChunkState.RESIDENT, Tier.VISIBLE)
    assert _state(core, 2, keys[0]) == (ChunkState.RESIDENT, Tier.VISIBLE)


# ---------------------------------------------------------------------------
# Retire and remove (5.4)
# ---------------------------------------------------------------------------


def test_retire_stops_fetching_and_keeps_residents() -> None:
    core, _ = _core(max_in_flight=1, backstop_reserved=0)
    core.register(1, FakeResidency(8))
    keys = fine_keys(1, n=3)
    core.set_desired(desired(1, target=keys))
    core.process()
    (first,) = core.next_reads()
    _land(core, [first])
    core.commit_round()
    (second,) = core.next_reads()

    core.retire(1)
    core.process()
    reg = core.registry(1)
    assert not (reg.tier == Tier.VISIBLE).any()
    assert not (reg.state == ChunkState.QUEUED).any()
    assert _state(core, 1, keys[0]) == (ChunkState.RESIDENT, Tier.RECENT)
    # The read in flight lands and is kept.
    core.complete_read(second, data=1)
    core.commit_round()
    assert _state(core, 1, keys[1]) == (ChunkState.RESIDENT, Tier.RECENT)
    assert core.next_reads() == []
    assert core.is_complete(1)


def test_remove_drops_reads_that_land_later() -> None:
    core, _ = _core()
    residency = FakeResidency(8)
    core.register(1, residency)
    core.set_desired(desired(1, target=fine_keys(1, n=2)))
    core.process()
    tickets = core.next_reads()
    core.remove(1)
    # Re-registering the id does not adopt the old reads.
    replacement = FakeResidency(8)
    core.register(1, replacement)
    assert _land(core, tickets) == [ReadOutcome.DROPPED] * 2
    assert core.in_flight == [0, 0]
    core.commit_round()
    assert residency.writes == []
    assert replacement.writes == []
    assert len(core.registry(1)) == 0


def test_remove_forgets_a_pending_pass() -> None:
    core, _ = _core()
    core.register(1, FakeResidency(8))
    core.set_desired(desired(1, target=fine_keys(1, n=2)))
    core.remove(1)
    assert core.process() == []
    assert core.idle()


def test_class_values() -> None:
    """The packed class values the planners emit."""
    assert int(ChunkClass.BACKSTOP) > int(ChunkClass.TARGET)
    assert int(Tier.VISIBLE) > int(Tier.PREFETCH) > int(Tier.RECENT)
