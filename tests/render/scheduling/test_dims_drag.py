"""``coarsest_while_moving`` and the dims scrub (interaction tracker design 4.3, 4.4).

A scrub is a run of interactive ticks: ``update_slice_indices`` with
``interactive=True`` or inside ``dims_interaction``.  During one, a visual with
``coarsest_while_moving`` on for the view plans backstop-only; the scrub's end (release
or stillness) plans it in full.  A plain ``update_slice_indices`` is a jump.

The fixture is a 16^3 volume with two levels and ``force_level=1``: the
target is level 1 and the backstop level 2.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.events import DimsChangedEvent
from cellier.render.scheduling import ChunkClass
from cellier.visuals import MultiscaleImageSingleAppearance, ProgressiveLoadingConfig
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)
from cellier.visuals._labels import MultiscaleLabelsAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

#: Appearance fields: plan the backstop only while moving, in 2D and 3D.
DRAG = {"coarsest_while_moving_2d": True, "coarsest_while_moving_3d": True}
#: The defaults of this phase: plan in full on every tick.
EAGER: dict = {}


def _tzyx_store(tmp_path):
    """Two timepoints of the 16^3, two-level volume: a collapsed t in 3D."""
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from tests.render.conftest import _write_multiscale_zarr

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (8, 16, 16, 16)), ("s1", (8, 8, 8, 8))],
        fill=lambda a: a.__setitem__(slice(None), 1.0),
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0,) * 4, (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0,) * 4, (0.0, 0.5, 0.5, 0.5)],
    )


def _add(controller, store, dim="2d", moving=DRAG, name="scene"):
    if len(store.level_shapes[0]) == 4:
        from cellier.scene import spatial_axes

        scene = controller.add_scene(
            coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
            dim=dim,
            name=name,
        )
    else:
        scene = controller.add_scene(dim=dim, name=name)
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1, **moving),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx


def _cache_id(gfx, dim="2d") -> int:
    slot = gfx.slots[0]
    return (slot.residency_2d() if dim == "2d" else slot.residency_3d()).cache_id


def _wanted_classes(controller, cache_id) -> set[int]:
    """Classes of the records the latest pass wanted (``VISIBLE``)."""
    from cellier.render.scheduling import Tier

    reg = controller._render_manager.scheduler.core.registry(cache_id)
    return {int(c) for c in reg.cls[reg.tier == Tier.VISIBLE]}


async def _loaded(controller, scene) -> None:
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)


def _settle_s(controller) -> float:
    return controller._render_manager.config.scheduler.dims_settle_s


def _record_plans(monkeypatch) -> list[tuple]:
    """Record ``(visual_id, plan mode name)`` for every visual a pass plans."""
    from cellier.render.scene_manager import SceneManager

    plans: list[tuple] = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        for visual_id, cfg in visual_configs.items():
            targets = request.target_visual_ids
            if targets is None or visual_id in targets:
                plans.append((visual_id, cfg.plan_mode.name))
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


def _modes(plans: list[tuple], visual_id=None) -> list[str]:
    return [mode for vid, mode in plans if visual_id is None or vid == visual_id]


def _record_events(controller, scene) -> list[tuple]:
    """Record ``(phase, reason)`` of every ``DimsInteractionEvent``."""
    events: list[tuple] = []
    controller.on_dims_interaction(
        scene.id,
        lambda event: events.append((event.phase, event.reason)),
        owner_id=controller._id,
    )
    return events


def _timers(controller) -> list:
    return controller._dims_driver.tasks()


# -- the config -------------------------------------------------------------------


@pytest.mark.parametrize(
    "appearance_cls", [MultiscaleImageAppearance, MultiscaleLabelsAppearance]
)
def test_eager_is_the_default(appearance_cls) -> None:
    appearance = appearance_cls()
    assert appearance.coarsest_while_moving_2d is False
    assert appearance.coarsest_while_moving_3d is False


@pytest.mark.parametrize("field", ["dims_drag", "backstop"])
def test_the_removed_loading_settings_are_refused(field) -> None:
    """``dims_drag`` and the ``backstop`` switch are gone, not ignored."""
    with pytest.raises(ValidationError, match=field):
        ProgressiveLoadingConfig(**{field: "eager"})


def test_the_opt_in_is_a_model_property(multiscale_image_store, controller) -> None:
    _scene, drag, _gfx = _add(controller, multiscale_image_store)
    _other, eager, _gfx2 = _add(
        controller,
        multiscale_image_store,
        moving=EAGER,
        name="eager",
    )
    assert drag.plans_coarse_while_moving(2) is True
    assert drag.plans_coarse_while_moving(3) is True
    assert eager.plans_coarse_while_moving(2) is False
    assert eager.plans_coarse_while_moving(3) is False


def test_each_view_dimension_has_its_own_setting() -> None:
    appearance = MultiscaleImageAppearance(coarsest_while_moving_3d=True)
    visual = MultiscaleImageVisual.model_construct(appearance=appearance)
    assert visual.plans_coarse_while_moving(3) is True
    assert visual.plans_coarse_while_moving(2) is False


# -- a scrub ends on stillness ----------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_a_tick_plans_the_backstop_and_the_settle_the_target(
    controller, tmp_path, dim
):
    scene, _visual, gfx = _add(controller, _tzyx_store(tmp_path), dim=dim)
    await _loaded(controller, scene)
    cache_id = _cache_id(gfx, dim)
    assert _wanted_classes(controller, cache_id) == {
        int(ChunkClass.BACKSTOP),
        int(ChunkClass.TARGET),
    }

    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    assert controller.dims_interaction_state(scene.id) == "active"
    await asyncio.sleep(0)  # the pass
    await asyncio.sleep(0)
    assert _wanted_classes(controller, cache_id) == {int(ChunkClass.BACKSTOP)}
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_target == 0

    await asyncio.sleep(_settle_s(controller) * 2)
    assert controller.dims_interaction_state(scene.id) == "idle"
    await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_target > 0
    assert progress.resident_target == progress.needed_target
    assert int(ChunkClass.TARGET) in _wanted_classes(controller, cache_id)


async def test_ticks_move_the_deadline_and_the_end_plans_once(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)

    settle = _settle_s(controller)
    for z in (2.0, 4.0, 6.0, 8.0):
        controller.update_slice_indices(scene.id, {0: z}, interactive=True)
        await asyncio.sleep(settle / 3)  # faster than the settle
    assert _modes(plans) == ["BACKSTOP_ONLY"] * 4
    assert len(_timers(controller)) == 1  # one timer for the whole scrub
    assert events == [("start", None)]

    await asyncio.sleep(settle * 2)
    await drain_loading(controller)
    assert _modes(plans).count("FULL") == 1  # one end, one canvas
    assert events == [("start", None), ("end", "settle")]
    assert not _timers(controller)


# -- a scrub ends on release ------------------------------------------------------


async def test_a_release_plans_the_target_at_once(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)

    with controller.dims_interaction(scene.id):
        assert events == []  # a scope starts nothing
        for z in (2.0, 4.0, 6.0):
            controller.update_slice_indices(scene.id, {0: z})  # in scope
        assert _modes(plans) == ["BACKSTOP_ONLY"] * 3
        assert controller.dims_interaction_state(scene.id) == "active"

    # No wait: the release planned in full before the block returned.
    assert _modes(plans) == ["BACKSTOP_ONLY"] * 3 + ["FULL"]
    assert events == [("start", None), ("end", "release")]
    assert controller.dims_interaction_state(scene.id) == "idle"
    # No timer is left sleeping to hold a drain up (design 4.2).
    assert not _timers(controller)
    assert not controller._deferred_reslice_tasks()
    await drain_loading(controller)


async def test_a_scope_with_no_tick_emits_nothing(controller, multiscale_image_store):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    events = _record_events(controller, scene)
    source = uuid4()
    controller.begin_dims_interaction(scene.id, source_id=source)
    controller.end_dims_interaction(scene.id, source_id=source)
    controller.end_dims_interaction(scene.id, source_id=source)  # not open: no-op
    assert events == []


async def test_a_scope_on_an_unknown_scene_is_refused(controller):
    with pytest.raises(KeyError):
        controller.begin_dims_interaction(uuid4(), source_id=uuid4())


async def test_holding_still_settles_and_the_next_tick_starts_again(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)
    source = uuid4()

    controller.begin_dims_interaction(scene.id, source_id=source)
    controller.update_slice_indices(scene.id, {0: 4.0})
    await asyncio.sleep(_settle_s(controller) * 2)  # held still, still pressed
    assert events == [("start", None), ("end", "settle")]
    assert _modes(plans) == ["BACKSTOP_ONLY", "FULL"]

    controller.update_slice_indices(scene.id, {0: 6.0})  # the scope is still open
    assert events[-1] == ("start", None)
    assert _modes(plans)[-1] == "BACKSTOP_ONLY"
    controller.end_dims_interaction(scene.id, source_id=source)
    assert events[-1] == ("end", "release")
    assert _modes(plans) == ["BACKSTOP_ONLY", "FULL", "BACKSTOP_ONLY", "FULL"]
    await drain_loading(controller)


# -- jumps ------------------------------------------------------------------------


async def test_a_programmatic_move_is_a_jump_and_plans_in_full(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)

    controller.update_slice_indices(scene.id, {0: 6.0})
    assert _modes(plans) == ["FULL"]
    assert events == []
    assert controller.dims_interaction_state(scene.id) == "idle"
    assert not _timers(controller)
    await drain_loading(controller)


async def test_a_jump_ends_a_scrub(controller, multiscale_image_store, monkeypatch):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)

    controller.update_slice_indices(scene.id, {0: 4.0}, interactive=True)
    controller.update_slice_indices(scene.id, {0: 6.0})  # a script moves the dims
    assert events == [("start", None), ("end", "jump")]
    # The jump's own reslice is the full plan: one, not two.
    assert _modes(plans) == ["BACKSTOP_ONLY", "FULL"]
    assert not _timers(controller)
    assert not controller._deferred_reslice_tasks()
    assert not controller._dims_scrub_pending
    await drain_loading(controller)


async def test_eager_ticks_plan_in_full(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store, moving=EAGER)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)
    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    # A scene with no opted-in visual still tracks and emits its state.
    assert events == [("start", None)]
    await asyncio.sleep(_settle_s(controller) * 2)
    await drain_loading(controller)
    assert events == [("start", None), ("end", "settle")]
    assert _modes(plans) == ["FULL"]  # the tick; the end plans nothing


async def test_a_displayed_axes_change_cancels_and_plans_in_full(
    controller, tmp_path, monkeypatch
):
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from tests.render.conftest import _write_multiscale_zarr

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=lambda a: a.__setitem__(slice(None), 1.0),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    )
    scene, _visual, _gfx = _add(controller, store)
    await _loaded(controller, scene)
    events = _record_events(controller, scene)
    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    assert _timers(controller)

    plans = _record_plans(monkeypatch)
    controller.set_displayed_axes(scene.id, (0, 2))
    assert events == [("start", None), ("end", "cancel")]
    assert not _timers(controller)
    assert not controller._deferred_reslice_tasks()
    await drain_loading(controller)
    assert plans and set(_modes(plans)) == {"FULL"}


async def test_other_visuals_in_the_scene_still_plan_in_full(
    controller, multiscale_image_store, monkeypatch
):
    scene, drag_visual, _gfx = _add(controller, multiscale_image_store)
    eager_visual = controller.add_image_multiscale(
        data=multiscale_image_store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        single=MultiscaleImageSingleAppearance(color_map="viridis"),
    )
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    assert _modes(plans, drag_visual.id) == ["BACKSTOP_ONLY"]
    assert _modes(plans, eager_visual.id) == ["FULL"]
    await asyncio.sleep(_settle_s(controller) * 2)
    await drain_loading(controller)
    # The end plans only what planned coarse.
    assert _modes(plans, drag_visual.id) == ["BACKSTOP_ONLY", "FULL"]
    assert _modes(plans, eager_visual.id) == ["FULL"]


# -- what is and is not a tick ----------------------------------------------------


async def test_a_call_that_changes_nothing_is_not_a_tick(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    events = _record_events(controller, scene)
    plans = _record_plans(monkeypatch)
    dims_events: list = []
    controller.on_dims_changed(scene.id, dims_events.append, owner_id=controller._id)

    controller.update_slice_indices(scene.id, {0: 4.0}, interactive=True)
    assert events == [("start", None)]
    # None of these moves a sliced axis: the same position, an unchanged
    # thickness (the ortho mirror re-writes it on every mirrored move), and
    # the stored position of a displayed axis.
    controller.update_slice_indices(scene.id, {0: 4.0})
    controller.update_thickness(scene.id, dict(scene.dims.selection.thickness))
    controller.update_slice_indices(scene.id, {1: 3.0})
    assert events == [("start", None)]  # not ended by a "jump"
    assert controller.dims_interaction_state(scene.id) == "active"
    assert _modes(plans) == ["BACKSTOP_ONLY"]  # and nothing was resliced
    assert [event.region_changed for event in dims_events] == [True, False]
    await asyncio.sleep(_settle_s(controller) * 2)
    await drain_loading(controller)


async def test_a_thickness_change_is_a_jump(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    events = _record_events(controller, scene)
    plans = _record_plans(monkeypatch)
    controller.update_slice_indices(scene.id, {0: 4.0}, interactive=True)
    controller.update_thickness(scene.id, {0: 2.0})
    assert events == [("start", None), ("end", "jump")]
    assert _modes(plans) == ["BACKSTOP_ONLY", "FULL"]
    assert not _timers(controller)
    await drain_loading(controller)


# -- events -----------------------------------------------------------------------


async def test_the_start_is_emitted_before_the_dims_model_changes(
    controller, multiscale_image_store
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    before = scene.dims.selection.slice_indices[0]
    seen: list = []
    source = uuid4()

    def on_interaction(event) -> None:
        seen.append(
            (
                event.phase,
                event.reason,
                event.source_id,
                set(event.axes),
                scene.dims.selection.slice_indices[0],
            )
        )

    controller.on_dims_interaction(scene.id, on_interaction, owner_id=controller._id)
    controller.update_slice_indices(
        scene.id, {0: 6.0}, source_id=source, interactive=True
    )
    assert seen == [("start", None, source, {0}, before)]
    releaser = uuid4()
    controller.begin_dims_interaction(scene.id, source_id=releaser)
    controller.end_dims_interaction(scene.id, source_id=releaser)
    assert seen[-1] == ("end", "release", releaser, {0}, 6.0)
    await drain_loading(controller)


async def test_dims_changed_events_say_whether_they_are_interactive(
    controller, multiscale_image_store
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    flags: list[bool] = []
    controller._outgoing_events.subscribe(
        DimsChangedEvent,
        lambda event: flags.append(event.interactive),
        entity_id=scene.id,
        owner_id=controller._id,
    )
    controller.update_slice_indices(scene.id, {0: 2.0})  # a jump
    controller.update_slice_indices(scene.id, {0: 4.0}, interactive=True)
    with controller.dims_interaction(scene.id):
        controller.update_slice_indices(scene.id, {0: 6.0})  # in a scope
    controller.update_slice_indices(scene.id, {0: 8.0})
    assert flags == [False, True, True, False]
    await drain_loading(controller)


async def test_a_gui_update_event_carries_the_interactive_flag(
    controller, multiscale_image_store, monkeypatch
):
    from cellier.events import DimsInteractionUpdateEvent, DimsUpdateEvent

    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    events = _record_events(controller, scene)
    widget = uuid4()
    bus = controller._incoming_events
    bus.emit(DimsInteractionUpdateEvent(widget, scene.id, "begin"))
    bus.emit(DimsUpdateEvent(widget, scene.id, {0: 6.0}, None, interactive=True))
    assert _modes(plans) == ["BACKSTOP_ONLY"]
    bus.emit(DimsInteractionUpdateEvent(widget, scene.id, "end"))
    assert events == [("start", None), ("end", "release")]
    assert _modes(plans) == ["BACKSTOP_ONLY", "FULL"]
    # Without the flag it is a jump.
    bus.emit(DimsUpdateEvent(widget, scene.id, {0: 2.0}, None))
    assert _modes(plans)[-1] == "FULL"
    assert events == [("start", None), ("end", "release")]
    await drain_loading(controller)


# -- every reslice during a scrub plans coarse (design 4.4) -----------------------


def _reslice_visual(controller, scene, visual, store) -> None:
    controller.reslice_visual(visual.id)


def _reslice_scene(controller, scene, visual, store) -> None:
    controller.reslice_scene(scene.id)


def _loading_config(controller, scene, visual, store) -> None:
    controller.set_loading_config(visual.id, backstop_max_slot_fraction=0.3)


def _shown(controller, scene, visual, store) -> None:
    controller.set_visual_visible(visual.id, True)


def _transform(controller, scene, visual, store) -> None:
    controller.set_visual_transform(
        visual.id,
        controller.data_to_world(scene.id, store, translation=(0.0, 1.0, 0.0)),
    )


def _store_extent(controller, scene, visual, store) -> None:
    store.notify_changed("extent")


@pytest.mark.parametrize(
    "trigger",
    [
        _reslice_visual,
        _reslice_scene,
        _loading_config,
        _shown,
        _transform,
        _store_extent,
    ],
)
async def test_a_reslice_during_a_scrub_plans_coarse_and_the_end_in_full(
    controller, multiscale_image_store, monkeypatch, trigger
):
    scene, visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    if trigger is _shown:
        controller.set_visual_visible(visual.id, False)
        await drain_loading(controller)
    plans = _record_plans(monkeypatch)

    with controller.dims_interaction(scene.id):
        controller.update_slice_indices(scene.id, {0: 6.0})
        ticked = len(plans)
        trigger(controller, scene, visual, multiscale_image_store)
        await asyncio.sleep(0)
        assert len(plans) > ticked, "the trigger did not reslice"
        assert set(_modes(plans, visual.id)) == {"BACKSTOP_ONLY"}
        assert visual.id in controller._dims_scrub_pending[scene.id]
        during = len(plans)

    assert _modes(plans, visual.id)[during:] == ["FULL"]  # once, at the end
    assert not controller._dims_scrub_pending
    await drain_loading(controller)


async def test_a_visual_added_during_a_scrub_plans_coarse_then_in_full(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)

    with controller.dims_interaction(scene.id):
        controller.update_slice_indices(scene.id, {0: 4.0})
        added = controller.add_image_multiscale(
            data=multiscale_image_store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(force_level=1, **DRAG),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
            single=MultiscaleImageSingleAppearance(color_map="viridis"),
        )
        controller.update_slice_indices(scene.id, {0: 6.0})
        assert set(_modes(plans, added.id)) == {"BACKSTOP_ONLY"}
        during = len(_modes(plans, added.id))

    assert _modes(plans, added.id)[during:] == ["FULL"]
    await drain_loading(controller)


async def test_a_config_change_mid_scrub_still_plans_in_full_at_the_end(
    controller, multiscale_image_store, monkeypatch
):
    """Pending is never trimmed by config (design 4.4)."""
    scene, visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    with controller.dims_interaction(scene.id):
        controller.update_slice_indices(scene.id, {0: 6.0})
        visual.appearance.coarsest_while_moving_2d = False
        # The setting is read by the next plan; changing it plans nothing.
        assert _modes(plans, visual.id) == ["BACKSTOP_ONLY"]
    assert _modes(plans, visual.id) == ["BACKSTOP_ONLY", "FULL"]
    await drain_loading(controller)


# -- the plan mode reaches any chunked visual that opts in ------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_a_visual_that_opts_in_through_its_model_receives_the_plan_mode(
    controller, tmp_path, monkeypatch, dim
):
    """The opt-in is the model's property, not the visual's type or config.

    This is the path the multiscale mesh uses: an ``eager`` image here stands
    in for a visual whose model answers ``plans_coarse_while_moving`` itself.
    """
    scene, visual, gfx = _add(controller, _tzyx_store(tmp_path), dim=dim, moving=EAGER)
    await _loaded(controller, scene)
    assert visual.plans_coarse_while_moving(int(dim[0])) is False
    monkeypatch.setattr(
        MultiscaleImageVisual, "plans_coarse_while_moving", lambda self, n: True
    )
    received: list[str] = []
    original = gfx.plan

    def plan(request, config, plan_mode):
        received.append(plan_mode.name)
        return original(request, config, plan_mode)

    monkeypatch.setattr(gfx, "plan", plan)

    controller.update_slice_indices(scene.id, {0: 2.0}, interactive=True)
    controller.update_slice_indices(scene.id, {0: 4.0}, interactive=True)
    assert received == ["BACKSTOP_ONLY", "BACKSTOP_ONLY"]
    await asyncio.sleep(_settle_s(controller) * 2)
    assert received == ["BACKSTOP_ONLY", "BACKSTOP_ONLY", "FULL"]  # the end
    controller.update_slice_indices(scene.id, {0: 6.0})  # a jump
    assert received[-1] == "FULL"
    assert len(received) == 4
    await drain_loading(controller)


# -- teardown ---------------------------------------------------------------------


async def test_removing_the_scene_or_closing_drops_the_scrub(
    controller, multiscale_image_store
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    other, _v2, _g2 = _add(controller, multiscale_image_store, name="other")
    await _loaded(controller, scene)
    events = _record_events(controller, scene)
    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    controller.update_slice_indices(other.id, {0: 6.0}, interactive=True)
    task, other_task = _timers(controller)

    controller.remove_scene(scene.id)
    await asyncio.sleep(0)
    assert task.cancelled()
    assert events == [("start", None)]  # dropped: no end event
    assert controller.dims_interaction_state(scene.id) == "idle"
    assert scene.id not in controller._dims_scrub_pending
    assert _timers(controller) == [other_task]

    controller.close()
    await asyncio.sleep(0)
    assert other_task.cancelled()
    assert not _timers(controller)
    assert not controller._dims_scrub_pending


async def test_the_settle_time_is_read_at_each_tick(controller, multiscale_image_store):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    events = _record_events(controller, scene)
    controller._render_manager.config.scheduler.dims_settle_s = 0.02
    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    await asyncio.sleep(0.08)  # under the default 0.15 s
    assert events == [("start", None), ("end", "settle")]
    await drain_loading(controller)


# -- no event loop ----------------------------------------------------------------


def test_with_no_event_loop_a_scrub_settles_at_once(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    events = _record_events(controller, scene)
    order: list = []
    monkeypatch.setattr(
        type(controller._render_manager),
        "reslice_scene",
        lambda self, *args, **kwargs: order.append(
            (
                kwargs.get("target_visual_ids") is not None,
                {
                    cfg.plan_mode.name
                    for cfg in (kwargs.get("visual_configs") or args[2]).values()
                },
            )
        ),
    )
    controller.update_slice_indices(scene.id, {0: 6.0}, interactive=True)
    assert events == [("start", None), ("end", "settle")]
    # The tick planned coarse, then the end planned its pending set in full.
    assert order == [(False, {"BACKSTOP_ONLY"}), (True, {"FULL"})]
    assert controller.dims_interaction_state(scene.id) == "idle"
    assert np.isclose(scene.dims.selection.slice_indices[0], 6.0)


# -- end to end: a slider release -------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_a_slider_release_reaches_the_target_without_the_settle_wait(
    controller, tmp_path, dim
):
    """What a dims slider sends: press, interactive ticks, release.

    The stillness time is set far beyond the test, so only the release can
    have planned the target.
    """
    from cellier.events import DimsInteractionUpdateEvent, DimsUpdateEvent

    scene, _visual, gfx = _add(controller, _tzyx_store(tmp_path), dim=dim)
    await _loaded(controller, scene)
    controller._render_manager.config.scheduler.dims_settle_s = 60.0
    cache_id = _cache_id(gfx, dim)
    widget = uuid4()
    bus = controller._incoming_events

    bus.emit(DimsInteractionUpdateEvent(widget, scene.id, "begin"))
    for t in (2.0, 4.0, 6.0):
        bus.emit(DimsUpdateEvent(widget, scene.id, {0: t}, None, interactive=True))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert _wanted_classes(controller, cache_id) == {int(ChunkClass.BACKSTOP)}

    bus.emit(DimsInteractionUpdateEvent(widget, scene.id, "end"))
    assert not controller._deferred_reslice_tasks()  # nothing left to wait for
    await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_target > 0
    assert progress.resident_target == progress.needed_target
    assert int(ChunkClass.TARGET) in _wanted_classes(controller, cache_id)
