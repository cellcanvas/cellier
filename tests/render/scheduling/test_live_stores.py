"""Live stores: an announced change reaches the screen, rate-capped (Phase 7).

Design v3 5.14.  A store another process writes to announces the change
(``notify_changed``).  The store revalidates its chunk cache once, the
scheduler drops the GPU bricks read from it, and the readers are resliced at
most ``SchedulerConfig.store_change_max_hz`` times a second.  A multiscale
reader needs no new plan for a contents change: invalidation requeues what
it wants.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
import tensorstore as ts

from tests.render.conftest import drain_loading
from tests.render.scheduling.test_backstop_integration import _add


def _write_externally(root, value: float) -> None:
    """Overwrite every level through a separate handle and context."""
    for name in ("s0", "s1"):
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": f"{root}/{name}"},
        }
        handle = ts.open(spec).result()
        handle[...] = np.full(handle.shape, value, dtype=np.float32)


def _record_reads(monkeypatch, store) -> list[np.ndarray]:
    """Every array the store returns, in order."""
    reads: list[np.ndarray] = []
    original = type(store).get_data

    async def recording(self_store, request):
        data = await original(self_store, request)
        reads.append(np.asarray(data))
        return data

    monkeypatch.setattr(type(store), "get_data", recording)
    return reads


def _count_plans(monkeypatch) -> list[int]:
    from cellier.render.scene_manager import SceneManager

    calls: list[int] = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        calls.append(len(visual_configs))
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return calls


def _count_reslices(monkeypatch, controller) -> list[float]:
    """Loop times of every ``reslice_visual``."""
    times: list[float] = []
    original = controller.reslice_visual

    def spy(visual_id):
        times.append(asyncio.get_running_loop().time())
        return original(visual_id)

    monkeypatch.setattr(controller, "reslice_visual", spy)
    return times


# -- the whole path, on a multiscale store --------------------------------------------


@pytest.mark.parametrize("dim", ["3d", "2d"])
async def test_an_external_write_reaches_the_gpu_without_a_new_plan(
    controller, multiscale_image_store, tmp_path, monkeypatch, dim
):
    scene, visual, _gfx = _add(controller, multiscale_image_store, dim=dim)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    loaded = controller.loading_progress(visual.id)
    assert loaded.complete

    reads = _record_reads(monkeypatch, multiscale_image_store)
    plans = _count_plans(monkeypatch)
    _write_externally(tmp_path, 5.0)
    multiscale_image_store.notify_changed("contents")
    await drain_loading(controller)

    # Every wanted chunk was read again, and saw the new values: the chunk
    # cache was revalidated, not served stale.
    wanted = loaded.needed_backstop + loaded.needed_target
    assert len(reads) == wanted
    # The old data was 0 with a 1.0 interior; a brick's halo past the edge
    # of the volume reads as 0.
    assert all(set(np.unique(r).tolist()) <= {0.0, 5.0} for r in reads)
    assert all((r == 5.0).any() for r in reads)
    assert controller.loading_progress(visual.id).complete
    # No replan: invalidation requeued the chunks the plan already wants.
    assert plans == []


async def test_without_an_announcement_the_write_stays_hidden(
    controller, multiscale_image_store, tmp_path, monkeypatch
):
    """Rechecks stay off: the promise the default makes (design 5.12)."""
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    reads = _record_reads(monkeypatch, multiscale_image_store)
    _write_externally(tmp_path, 5.0)
    controller.reslice_all()
    await drain_loading(controller)
    assert reads == []  # everything is resident; nothing is re-read


async def test_an_extent_change_replans_a_multiscale_reader(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    plans = _count_plans(monkeypatch)
    multiscale_image_store.notify_changed("extent")
    await drain_loading(controller)
    assert plans


# -- the rate cap ---------------------------------------------------------------------


def _min_gap(interval: float) -> float:
    """The shortest spacing the cap can show between two reslices.

    asyncio fires a timer up to one clock tick early (``_run_once`` runs
    every handle due before ``time() + clock_resolution``), and the loop
    clock only moves in ticks.  On Windows before Python 3.13 that tick is
    15.6 ms, so a 33 ms interval can read as 32 ms; elsewhere it is well
    under a microsecond.  0.99 absorbs float noise.
    """
    return 0.99 * interval - time.get_clock_info("monotonic").resolution


async def test_a_burst_of_changes_reslices_once_now_and_once_after(
    controller, image_volume, monkeypatch
):
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(image_volume, scene.id)
    controller.add_canvas(scene_id=scene.id)
    await drain_loading(controller)
    times = _count_reslices(monkeypatch, controller)

    for _ in range(20):
        image_volume.notify_changed("contents")
    assert len(times) == 1  # the first change, at once
    assert controller._deferred_reslice_tasks()  # the rest, folded

    await drain_loading(controller)
    assert len(times) == 2
    interval = 1.0 / controller._render_manager.config.scheduler.store_change_max_hz
    assert times[1] - times[0] >= _min_gap(interval)


async def test_a_steady_stream_is_capped_and_the_last_change_is_resliced(
    controller, image_volume, monkeypatch
):
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(image_volume, scene.id)
    controller.add_canvas(scene_id=scene.id)
    await drain_loading(controller)
    times = _count_reslices(monkeypatch, controller)
    loop = asyncio.get_running_loop()
    max_hz = controller._render_manager.config.scheduler.store_change_max_hz

    start = loop.time()
    last_change = start
    while loop.time() - start < 0.3:  # about 150 Hz of changes
        image_volume.notify_changed("contents")
        last_change = loop.time()
        await asyncio.sleep(0.005)
    await drain_loading(controller)

    elapsed = times[-1] - times[0]
    min_gap = _min_gap(1.0 / max_hz)
    assert len(times) - 1 <= elapsed / min_gap + 1
    assert all(b - a >= min_gap for a, b in zip(times, times[1:], strict=False))
    # Nothing is dropped: a reslice runs after the final change.
    assert times[-1] >= last_change


async def test_each_store_has_its_own_cap(
    controller, image_volume, labels_volume, monkeypatch
):
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(image_volume, scene.id)
    controller.add_labels(labels_volume, scene.id)
    controller.add_canvas(scene_id=scene.id)
    await drain_loading(controller)
    times = _count_reslices(monkeypatch, controller)

    image_volume.notify_changed("contents")
    labels_volume.notify_changed("contents")
    assert len(times) == 2  # neither waits for the other
    await drain_loading(controller)


async def test_close_cancels_a_pending_reslice(controller, image_volume):
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(image_volume, scene.id)
    controller.add_canvas(scene_id=scene.id)
    image_volume.notify_changed("contents")
    image_volume.notify_changed("contents")
    [task] = controller._deferred_reslice_tasks()

    controller.close()
    await asyncio.sleep(0)
    assert task.cancelled()
    assert not controller._deferred_reslice_tasks()
