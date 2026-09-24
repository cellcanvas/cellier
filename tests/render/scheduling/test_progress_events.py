"""Progress events for multiscale visuals (design v3 5.13, Phase 6).

The fixture is a 16^3 volume with two levels and ``force_level=1``: the
target is level 1 (store scale 0) and the backstop level 2 (scale 1).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cellier.events import (
    BackstopCompleteEvent,
    LoadingProgress,
    ResliceCompletedEvent,
    ResliceProgressEvent,
)
from cellier.visuals import ProgressiveLoadingConfig
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading
from tests.render.scheduling.test_backstop_integration import _add, _Gate, _until


class _Recorder:
    """Every progress, backstop and completion event of one visual, in order."""

    def __init__(self, controller, visual_id) -> None:
        self.events: list = []
        owner = uuid4()
        controller.on_reslice_progress(visual_id, self.events.append, owner_id=owner)
        controller.on_backstop_complete(visual_id, self.events.append, owner_id=owner)
        controller.on_reslice_completed(visual_id, self.events.append, owner_id=owner)

    def of(self, event_type) -> list:
        return [e for e in self.events if isinstance(e, event_type)]


async def _load(controller, scene) -> None:
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    await asyncio.sleep(0)  # the last progress flush


@pytest.mark.parametrize("dim", ["3d", "2d"])
async def test_progress_climbs_to_complete(controller, multiscale_image_store, dim):
    scene, visual, _gfx = _add(controller, multiscale_image_store, dim=dim)
    rec = _Recorder(controller, visual.id)
    await _load(controller, scene)

    progress = [e.progress for e in rec.of(ResliceProgressEvent)]
    assert progress
    assert all(isinstance(p, LoadingProgress) for p in progress)
    # Counts stay consistent (invariant 8), and never go backwards.
    for p in progress:
        assert p.resident_target <= p.needed_target
        assert p.resident_backstop <= p.needed_backstop
    resident = [p.resident_target + p.resident_backstop for p in progress]
    assert resident == sorted(resident)
    last = progress[-1]
    assert last.complete and last.backstop_complete
    assert last.resident_target == last.needed_target > 0
    assert last.needed_backstop > 0
    assert last.fraction == 1.0
    assert all(e.scene_id == scene.id for e in rec.of(ResliceProgressEvent))
    # The pull API reads the same counts.
    assert controller.loading_progress(visual.id) == last


async def test_backstop_complete_fires_while_the_target_is_held(
    controller, multiscale_image_store, monkeypatch
):
    gate = _Gate(monkeypatch, multiscale_image_store)
    scene, visual, _gfx = _add(controller, multiscale_image_store)
    rec = _Recorder(controller, visual.id)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    scheduler = controller._render_manager.scheduler

    def backstop_in() -> bool:
        scheduler.commit_round()  # as a drawn frame would
        return bool(rec.of(BackstopCompleteEvent))

    await _until(backstop_in)
    [event] = rec.of(BackstopCompleteEvent)
    assert event.scene_id == scene.id
    progress = controller.loading_progress(visual.id)
    assert progress.backstop_complete and not progress.complete
    assert progress.resident_target == 0
    assert progress.fraction == 0.0
    assert not rec.of(ResliceCompletedEvent)

    gate.open.set()
    await drain_loading(controller)
    await asyncio.sleep(0)
    # Completion comes after the backstop, and the backstop fires once.
    kinds = [type(e) for e in rec.events if not isinstance(e, ResliceProgressEvent)]
    assert kinds == [BackstopCompleteEvent, ResliceCompletedEvent]
    assert rec.of(ResliceProgressEvent)[-1].progress.complete


async def test_no_backstop_event_without_a_backstop(controller, multiscale_image_store):
    scene, visual, _gfx = _add(
        controller,
        multiscale_image_store,
        loading=ProgressiveLoadingConfig(backstop=False),
    )
    rec = _Recorder(controller, visual.id)
    await _load(controller, scene)
    assert not rec.of(BackstopCompleteEvent)
    last = rec.of(ResliceProgressEvent)[-1].progress
    assert last.needed_backstop == 0
    assert last.complete and last.backstop_complete


async def test_one_event_per_loop_iteration_over_every_channel(
    controller, tmp_path, monkeypatch
):
    """A composite visual has one atlas per channel; its event sums them."""
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from cellier.scene import spatial_axes
    from cellier.visuals import (
        MultiscaleImageChannelAppearance,
        MultiscaleImageSingleAppearance,
    )
    from cellier.visuals._image import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
    )
    from tests.render.conftest import _write_multiscale_zarr

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (2, 16, 16, 16)), ("s1", (2, 8, 8, 8))],
        fill=lambda a: a.__setitem__(slice(None), 1.0),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0,) * 4, (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0,) * 4, (0.0, 0.5, 0.5, 0.5)],
    )
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    visual = controller.add_image_multiscale(
        store,
        scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        channel_axis=0,
        composite=True,
        channels={
            0: MultiscaleImageChannelAppearance(color_map="red"),
            1: MultiscaleImageChannelAppearance(color_map="green"),
        },
        single=MultiscaleImageSingleAppearance(color_map="grays"),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    rec = _Recorder(controller, visual.id)
    scheduler = controller._render_manager.scheduler
    # No fallback round: arrivals wait for the round this test runs.
    monkeypatch.setattr(scheduler, "_arm_fallback", lambda: None)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await _until(
        lambda: (
            scheduler._process_handle is None
            and not scheduler._tasks
            and scheduler.core.oldest_arrival() is not None
        )
    )

    before = len(rec.of(ResliceProgressEvent))
    assert scheduler.commit_round()  # both channels commit in one round
    await asyncio.sleep(0)
    events = rec.of(ResliceProgressEvent)[before:]
    assert len(events) == 1
    total = events[0].progress
    per_channel = [
        scheduler.progress(gfx.slots[i].residency_3d().cache_id)
        for i in gfx._drawn.values()
    ]
    assert len(per_channel) == 2
    assert total.needed_target == sum(p.needed_target for p in per_channel)
    assert total.resident_target == total.needed_target
    assert total.complete


async def test_no_progress_for_a_visual_that_is_not_loaded(controller):
    assert controller.loading_progress(uuid4()) is None


async def test_a_dims_drag_marks_the_target_deferred_until_the_settle(
    controller, tmp_path
):
    from tests.render.scheduling.test_dims_drag import DRAG, _settle_s, _tzyx_store
    from tests.render.scheduling.test_dims_drag import _add as _add_drag

    scene, visual, _gfx = _add_drag(controller, _tzyx_store(tmp_path), loading=DRAG)
    await _load(controller, scene)
    assert not controller.loading_progress(visual.id).target_deferred

    rec = _Recorder(controller, visual.id)
    controller.update_slice_indices(scene.id, {0: 6.0})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    progress = controller.loading_progress(visual.id)
    assert progress.target_deferred
    assert progress.needed_target == 0

    await asyncio.sleep(_settle_s(controller) * 2)
    await drain_loading(controller)
    await asyncio.sleep(0)
    last = rec.of(ResliceProgressEvent)[-1].progress
    assert not last.target_deferred
    assert last.complete and last.needed_target > 0
    assert any(e.progress.target_deferred for e in rec.of(ResliceProgressEvent))
