"""``CachePolicy.commit_stale``: stale results drawn as kept in the commit round.

``plans/scheduler_only_loading_design_v2.md`` 4.1 (D13, D14).  A cache whose
policy has ``commit_stale`` keeps a result an invalidation makes stale -- a
read invalidated in flight, or an arrival invalidated before its round -- and
hands it to the adapter's ``write_kept`` in the next commit round, never
before; the record is requeued either way.  A cache without it behaves as it
always did (the other core tests and the trace guard).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from cellier.render import SchedulerConfig
from cellier.render.scheduling import (
    CachePolicy,
    ChunkScheduler,
    ChunkState,
    ReadOutcome,
    SchedulerCore,
)
from tests.render.scheduling._fakes import AsyncStore, FakeStore, desired

STALE = CachePolicy(
    max_target_fetching=1,
    resource="compute",
    retry_max_attempts=1,
    retry_on_pass=False,
    commit_stale=True,
)
PLAIN = CachePolicy(
    max_target_fetching=1,
    resource="compute",
    retry_max_attempts=1,
    retry_on_pass=False,
)
KEY = 7


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class KeptResidency:
    """Two slots, a record of writes, kept writes and rebuilds."""

    n_slots = 2

    def __init__(self, policy: CachePolicy = STALE) -> None:
        self.policy = policy
        self.writes: list[tuple[int, Any]] = []
        self.kept: list[tuple[int, Any]] = []
        self.rebuilds = 0

    def write(self, slot: int, key: int, data: Any) -> None:
        self.writes.append((key, data))

    def write_kept(self, key: int, data: Any) -> None:
        self.kept.append((key, data))

    def rebuild_draw(self, view) -> None:
        self.rebuilds += 1

    def keys_in_region(self, keys: np.ndarray, region: Any) -> np.ndarray:
        return np.ones(len(keys), dtype=bool)


def _rig(policy: CachePolicy = STALE):
    clock = Clock()
    core = SchedulerCore(SchedulerConfig(), now=clock)
    residency = KeptResidency(policy)
    core.register(1, residency, scene="scene")
    store = FakeStore()
    core.set_desired(desired(1, target=[KEY], store=store))
    core.process()
    return core, residency, store, clock


def _state(core: SchedulerCore, key: int = KEY) -> int | None:
    reg = core.registry(1)
    row = reg.find(key)
    return None if row < 0 else int(reg.state[row])


def test_a_read_invalidated_in_flight_is_kept_for_the_next_round() -> None:
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    outcome = core.complete_read(ticket, data="old")
    assert outcome == ReadOutcome.STALE_KEPT
    # Not before the round.
    assert residency.kept == []
    # The record is requeued: the re-read still lands.
    assert _state(core) == int(ChunkState.QUEUED)
    assert not core.idle()
    assert core.commit_round("scene") == [1]
    assert residency.kept == [(KEY, "old")]
    assert residency.writes == []
    (again,) = core.next_reads()
    assert again.key == KEY
    core.complete_read(again, data="new")
    core.commit_round("scene")
    assert residency.writes == [(KEY, "new")]


def test_an_invalidated_arrival_is_kept_for_the_next_round() -> None:
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    assert core.complete_read(ticket, data="old") == ReadOutcome.ARRIVED
    core.invalidate(store.id)
    assert core.arrived_keys(1) == []
    assert _state(core) == int(ChunkState.QUEUED)
    assert residency.kept == []
    core.commit_round("scene")
    assert residency.kept == [(KEY, "old")]
    assert residency.writes == []


def test_a_newer_stale_result_replaces_an_older_one() -> None:
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    core.complete_read(ticket, data="first")
    core.invalidate(store.id)
    (again,) = core.next_reads()
    core.invalidate(store.id)
    assert core.complete_read(again, data="second") == ReadOutcome.STALE_KEPT
    core.commit_round("scene")
    assert residency.kept == [(KEY, "second")]


def test_a_stale_arrival_no_longer_wanted_is_dropped() -> None:
    """A ``RECENT`` record's stale read is never drawn."""
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    core.retire(1)
    core.process()
    core.invalidate(store.id)
    assert core.complete_read(ticket, data="old") == ReadOutcome.DROPPED
    core.commit_round("scene")
    assert residency.kept == []
    assert core.idle()


def test_a_failed_read_invalidated_in_flight_keeps_nothing() -> None:
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    outcome = core.complete_read(ticket, error=OSError("torn"))
    assert outcome == ReadOutcome.DROPPED
    core.commit_round("scene")
    assert residency.kept == []
    assert _state(core) == int(ChunkState.QUEUED)


def test_without_commit_stale_the_stale_read_is_dropped() -> None:
    core, residency, store, _clock = _rig(PLAIN)
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    assert core.complete_read(ticket, data="old") == ReadOutcome.DROPPED
    core.commit_round("scene")
    assert residency.kept == []
    (ticket,) = core.next_reads()
    core.complete_read(ticket, data="mid")
    core.invalidate(store.id)
    core.commit_round("scene")
    assert residency.kept == []
    assert residency.writes == []


def test_a_waiting_stale_result_is_the_oldest_arrival() -> None:
    """``oldest_arrival`` and ``idle`` count it, so the fallback commits it."""
    core, _residency, store, clock = _rig()
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    clock.t = 3.0
    core.complete_read(ticket, data="old")
    # The re-read is queued, but take it off so only the stale entry waits.
    (again,) = core.next_reads()
    core.complete_read(again, error=OSError("gone"))
    assert core.oldest_arrival() == 3.0
    assert not core.idle()
    core.commit_round("scene")
    assert core.oldest_arrival() is None
    assert core.idle()


def test_a_round_of_another_scene_leaves_the_stale_result() -> None:
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    core.complete_read(ticket, data="old")
    assert core.commit_round("other") == []
    assert residency.kept == []
    core.commit_round("scene")
    assert residency.kept == [(KEY, "old")]


def test_retire_drops_the_stale_results() -> None:
    core, residency, store, _clock = _rig()
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    core.complete_read(ticket, data="old")
    core.retire(1)
    core.process()
    core.commit_round("scene")
    assert residency.kept == []


def test_the_round_emits_a_commit_stale_trace() -> None:
    core, _residency, store, _clock = _rig()
    trace: list = []
    core.trace = lambda kind, payload: trace.append((kind, payload))
    (ticket,) = core.next_reads()
    core.invalidate(store.id)
    core.complete_read(ticket, data="old")
    core.commit_round("scene")
    assert ("commit_stale", (1, KEY)) in trace


async def test_a_stale_kept_read_draws_and_arms_the_fallback() -> None:
    """With no frame drawn, the fallback round commits the stale result."""
    gate = asyncio.Event()
    store = AsyncStore(latency=0.0, gate=gate)
    draws: list = []
    scheduler = ChunkScheduler(
        SchedulerConfig(commit_fallback_s=0.01), request_draw=draws.append
    )
    residency = KeptResidency()
    scheduler.register(1, residency, scene="scene")
    scheduler.pass_([desired(1, target=[KEY], store=store)])
    for _ in range(50):
        await asyncio.sleep(0)
        if store.calls:
            break
    scheduler.invalidate(store.id)
    draws.clear()
    gate.set()
    for _ in range(200):
        await asyncio.sleep(0.002)
        if residency.kept:
            break
    assert residency.kept == [(KEY, KEY * 10)]
    assert "scene" in draws
    await scheduler.drain(timeout_s=2.0)
    assert residency.writes == [(KEY, KEY * 10)]
    scheduler.close()


async def test_drain_with_commit_commits_a_stale_result() -> None:
    gate = asyncio.Event()
    store = AsyncStore(latency=0.0, gate=gate)
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=30.0))
    residency = KeptResidency()
    scheduler.register(1, residency, scene="scene")
    scheduler.pass_([desired(1, target=[KEY], store=store)])
    for _ in range(50):
        await asyncio.sleep(0)
        if store.calls:
            break
    scheduler.invalidate(store.id)
    gate.set()
    await scheduler.drain(timeout_s=2.0, commit=True)
    assert residency.kept == [(KEY, KEY * 10)]
    assert residency.writes == [(KEY, KEY * 10)]
    assert len(store.calls) == 2
    scheduler.close()
