"""The scheduler on asyncio: coalescing, reads, timers, draws and teardown."""

from __future__ import annotations

import asyncio

import pytest

from cellier.render import RenderManagerConfig, SchedulerConfig
from cellier.render.scheduling import ChunkScheduler, ChunkState, Tier
from tests.render.scheduling._fakes import (
    AsyncStore,
    FakeResidency,
    coarse_keys,
    desired,
    fine_keys,
    pack,
)


def _written(residency: FakeResidency) -> dict[int, object]:
    return {key: residency.slots[slot][1] for slot, key in residency.writes}


def test_render_manager_config_carries_the_scheduler_config() -> None:
    config = RenderManagerConfig(scheduler=SchedulerConfig(max_in_flight=4))
    restored = RenderManagerConfig.model_validate_json(config.model_dump_json())
    assert restored.scheduler.max_in_flight == 4
    assert RenderManagerConfig().scheduler == SchedulerConfig()


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_in_flight", 0),
        ("backstop_reserved", -1),
        ("commit_fallback_s", 0),
        ("dims_settle_s", 0),
        ("retry_max_attempts", 0),
        ("retry_backoff_s", -1),
    ],
)
def test_scheduler_config_rejects_bad_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        SchedulerConfig(**{field: value})


async def test_cold_load_commits_through_the_fallback() -> None:
    """No canvas draws: the fallback timer commits, and completion fires."""
    store = AsyncStore()
    residency = FakeResidency(32)
    done: list[tuple[int, int]] = []
    draws: list[object] = []
    scheduler = ChunkScheduler(
        SchedulerConfig(commit_fallback_s=0.01),
        request_draw=draws.append,
        on_complete=lambda c, g: done.append((c, g)),
    )
    scheduler.register(1, residency, scene="scene")
    keys = coarse_keys() + fine_keys(2, n=10)
    scheduler.pass_([desired(1, backstop=coarse_keys(), target=keys[4:], store=store)])
    await scheduler.drain()

    assert _written(residency) == {k: k * 10 for k in keys}
    assert done == [(1, 1)]
    assert draws and set(draws) == {"scene"}
    progress = scheduler.progress(1)
    assert progress.resident_backstop == 4
    assert progress.resident_target == 10
    scheduler.close()


async def test_frame_rounds_commit_before_the_fallback() -> None:
    """A canvas that draws on request commits in its own ``before_draw``."""
    store = AsyncStore()
    residency = FakeResidency(32)
    scheduler: ChunkScheduler

    def request_draw(scene: object) -> None:
        # Stand-in for a canvas: draw on the next loop iteration, calling
        # the before_draw hook for its scene.
        asyncio.get_running_loop().call_soon(scheduler.commit_round, scene)

    scheduler = ChunkScheduler(
        SchedulerConfig(commit_fallback_s=60.0), request_draw=request_draw
    )
    scheduler.register(1, residency, scene="scene")
    keys = fine_keys(1, n=12)
    scheduler.pass_([desired(1, target=keys, store=store)])
    await scheduler.drain(timeout_s=5.0)  # far shorter than the fallback
    assert sorted(_written(residency)) == sorted(keys)
    scheduler.close()


async def test_passes_in_one_iteration_coalesce() -> None:
    store = AsyncStore()
    residency = FakeResidency(32)
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=0.01))
    scheduler.register(1, residency)
    for t in range(5):
        scheduler.pass_([desired(1, target=fine_keys(1, t=t, n=3), store=store)])
    await scheduler.drain()
    assert scheduler.core.generation(1) == 1
    # Only the last pass's keys were ever read.
    assert sorted(store.calls) == sorted(fine_keys(1, t=4, n=3))
    scheduler.close()


async def test_the_in_flight_budget_holds() -> None:
    store = AsyncStore(latency=0.005)
    scheduler = ChunkScheduler(
        SchedulerConfig(max_in_flight=3, backstop_reserved=2, commit_fallback_s=0.01)
    )
    for cache_id in (1, 2):
        scheduler.register(cache_id, FakeResidency(64))
        scheduler.pass_(
            [
                desired(
                    cache_id,
                    backstop=coarse_keys(t=cache_id),
                    target=fine_keys(1, t=cache_id, n=20),
                    store=store,
                )
            ]
        )
    await scheduler.drain()
    assert store.max_concurrent == 5
    assert len(store.calls) == 2 * 24
    scheduler.close()


async def test_a_failed_read_is_retried_on_a_timer() -> None:
    bad = pack(1, 0, 0, 0)
    store = AsyncStore(fail=lambda key, attempt: key == bad and attempt < 3)
    residency = FakeResidency(8)
    scheduler = ChunkScheduler(
        SchedulerConfig(
            retry_max_attempts=3, retry_backoff_s=0.005, commit_fallback_s=0.01
        )
    )
    scheduler.register(1, residency)
    scheduler.pass_([desired(1, target=[bad], store=store)])
    await scheduler.drain()
    assert store.calls.count(bad) == 3
    assert _written(residency) == {bad: bad * 10}
    scheduler.close()


async def test_a_read_that_keeps_failing_is_given_up() -> None:
    bad, good = pack(1, 0, 0, 0), pack(1, 0, 0, 1)
    store = AsyncStore(fail=lambda key, attempt: key == bad)
    residency = FakeResidency(8)
    done: list[int] = []
    scheduler = ChunkScheduler(
        SchedulerConfig(
            retry_max_attempts=2, retry_backoff_s=0.002, commit_fallback_s=0.01
        ),
        on_complete=lambda c, g: done.append(g),
    )
    scheduler.register(1, residency)
    scheduler.pass_([desired(1, target=[bad, good], store=store)])
    await scheduler.drain()
    assert store.calls.count(bad) == 2
    assert _written(residency) == {good: good * 10}
    assert scheduler.progress(1).failed == 1
    assert done == [1]
    scheduler.close()


async def test_draws_follow_every_rebuild_outside_a_frame() -> None:
    """A pass and a give-up rebuild the draw, so their scene must redraw."""
    bad = pack(1, 0, 0, 0)
    store = AsyncStore(fail=lambda key, attempt: True)
    draws: list[object] = []
    scheduler = ChunkScheduler(
        SchedulerConfig(retry_max_attempts=1), request_draw=draws.append
    )
    scheduler.register(1, FakeResidency(8), scene="s")
    scheduler.pass_([desired(1, target=[bad], store=store)])
    await asyncio.sleep(0)
    assert draws == ["s"]  # the pass
    await scheduler.drain()
    assert draws == ["s", "s"]  # the give-up
    scheduler.close()


async def test_superseded_reads_land_and_are_kept() -> None:
    store = AsyncStore(latency=0.02)
    residency = FakeResidency(16)
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=0.01))
    scheduler.register(1, residency)
    old = fine_keys(1, t=0, n=2)
    scheduler.pass_([desired(1, target=old, store=store)])
    await asyncio.sleep(0.005)  # issued, not landed
    new = fine_keys(1, t=1, n=2)
    scheduler.pass_([desired(1, target=new, store=store)])
    await scheduler.drain()
    reg = scheduler.core.registry(1)
    for key in old:
        row = reg.find(key)
        assert (reg.state[row], reg.tier[row]) == (ChunkState.RESIDENT, Tier.RECENT)
    assert sorted(_written(residency)) == sorted(old + new)
    scheduler.close()


async def test_invalidate_refetches_what_is_wanted() -> None:
    store = AsyncStore()
    residency = FakeResidency(8)
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=0.01))
    scheduler.register(1, residency)
    keys = fine_keys(1, n=2)
    scheduler.pass_([desired(1, target=keys, store=store)])
    await scheduler.drain()
    assert scheduler.invalidate(store.id, region=lambda k: k == keys[0]) == [1]
    await scheduler.drain()
    assert store.calls.count(keys[0]) == 2
    assert store.calls.count(keys[1]) == 1
    scheduler.close()


async def test_remove_mid_flight_drops_the_reads() -> None:
    store = AsyncStore(latency=0.01)
    residency = FakeResidency(8)
    scheduler = ChunkScheduler(SchedulerConfig(commit_fallback_s=0.005))
    scheduler.register(1, residency)
    scheduler.pass_([desired(1, target=fine_keys(1, n=3), store=store)])
    await asyncio.sleep(0.002)
    scheduler.remove(1)
    await scheduler.drain()
    assert residency.writes == []
    assert scheduler.core.in_flight == [0, 0]
    scheduler.close()


async def test_retire_stops_new_reads() -> None:
    # The reads are held open until the retire has been applied.  A retire
    # is a pass, applied on the next loop iteration; a read landing before
    # then refills from the old queue, and a timed sleep could not rule that
    # out (on Windows a 2 ms sleep can oversleep a 10 ms read).
    gate = asyncio.Event()
    store = AsyncStore(gate=gate)
    residency = FakeResidency(64)
    scheduler = ChunkScheduler(
        SchedulerConfig(max_in_flight=2, backstop_reserved=0, commit_fallback_s=0.005)
    )
    scheduler.register(1, residency)
    scheduler.pass_([desired(1, target=fine_keys(1, n=20), store=store)])
    async with asyncio.timeout(5.0):
        while len(store.calls) < 2:
            await asyncio.sleep(0)
    scheduler.retire(1)
    await asyncio.sleep(0)  # the retire's pass runs
    gate.set()
    await scheduler.drain()
    assert len(store.calls) == 2  # only what was in flight
    assert len(residency.writes) == 2  # kept, as RECENT
    scheduler.close()


async def test_close_cancels_reads_and_is_idempotent() -> None:
    store = AsyncStore(latency=10.0)
    residency = FakeResidency(8)
    scheduler = ChunkScheduler()
    scheduler.register(1, residency)
    scheduler.pass_([desired(1, target=fine_keys(1, n=3), store=store)])
    await asyncio.sleep(0.001)
    tasks = list(scheduler._tasks)
    assert len(tasks) == 3
    scheduler.close()
    scheduler.close()
    await asyncio.sleep(0)
    assert all(task.cancelled() for task in tasks)
    assert residency.writes == []
    # Passes after close are ignored rather than raising in a callback.
    scheduler.retire(1)


async def test_drain_times_out_when_busy() -> None:
    store = AsyncStore(latency=10.0)
    scheduler = ChunkScheduler()
    scheduler.register(1, FakeResidency(8))
    scheduler.pass_([desired(1, target=fine_keys(1, n=1), store=store)])
    with pytest.raises(TimeoutError):
        await scheduler.drain(timeout_s=0.01)
    scheduler.close()


async def test_a_failing_draw_request_does_not_break_loading(caplog) -> None:
    store = AsyncStore()
    residency = FakeResidency(8)

    def broken(scene: object) -> None:
        raise RuntimeError("canvas gone")

    scheduler = ChunkScheduler(
        SchedulerConfig(commit_fallback_s=0.005), request_draw=broken
    )
    scheduler.register(1, residency)
    scheduler.pass_([desired(1, target=fine_keys(1, n=2), store=store)])
    await scheduler.drain()
    assert len(residency.writes) == 2
    assert "request_draw failed" in caplog.text
    scheduler.close()
