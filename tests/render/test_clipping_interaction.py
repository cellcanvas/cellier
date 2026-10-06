"""The clipping interaction tracker: scopes, ticks and the start / end event.

A drag is announced and nothing else: no plan mode depends on it (clipping
plane gizmo design, section 6).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.data import PointsMemoryStore
from cellier.events import ClippingPlanesChangedEvent, PlaneInteractionEvent
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane
from tests.render import _clipping as h
from tests.render.conftest import drain_loading


def _store() -> PointsMemoryStore:
    system = DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space") for n in "zyx"),
    )
    rng = np.random.default_rng(0)
    return PointsMemoryStore(
        positions=(rng.random((200, 3)) * 32.0).astype(np.float32),
        data_coordinate_systems=[system],
    )


def _plane(system, x: float) -> ClippingPlane:
    return ClippingPlane.from_point_normal(
        system, (16, 16, x), (0, 0, 1), axes=("z", "y", "x")
    )


def _scene(controller):
    """A points visual with one plane, and the log of what the bus said."""
    store = _store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    item = _plane(system, 16.0)
    visual = controller.add_points(
        data=store, scene_id=scene.id, clipping_planes=(item,)
    )
    controller.add_canvas(scene_id=scene.id)
    log: list = []
    owner = uuid4()
    controller.on_plane_interaction(visual.id, log.append, owner_id=owner)
    controller.on_clipping_planes_changed(visual.id, log.append, owner_id=owner)
    return visual, item, system, log


def _move(controller, visual, item, system, x: float, **kwargs) -> None:
    controller.set_clipping_plane(visual.id, item.id, _plane(system, x).plane, **kwargs)


def _interactions(log) -> list[tuple]:
    return [
        (event.phase, event.reason)
        for event in log
        if isinstance(event, PlaneInteractionEvent)
    ]


def test_a_change_outside_a_scope_is_a_jump(controller):
    visual, item, system, log = _scene(controller)
    _move(controller, visual, item, system, 20.0)
    assert _interactions(log) == []
    assert controller.plane_interaction_state(visual.id) == "idle"


def test_opening_a_scope_starts_nothing(controller):
    visual, _item, _system, log = _scene(controller)
    source = uuid4()
    controller.begin_plane_interaction(visual.id, source_id=source)
    controller.end_plane_interaction(visual.id, source_id=source)
    controller.end_plane_interaction(visual.id, source_id=source)  # not open
    assert log == []


def test_an_unknown_visual_is_refused(controller):
    with pytest.raises(KeyError):
        controller.begin_plane_interaction(uuid4(), source_id=uuid4())


async def test_a_drag_is_one_start_and_one_release(controller):
    visual, item, system, log = _scene(controller)
    source = uuid4()
    controller.begin_plane_interaction(visual.id, source_id=source)
    for x in (18.0, 20.0, 22.0):
        _move(controller, visual, item, system, x, source_id=source)
    assert controller.plane_interaction_state(visual.id) == "active"
    controller.end_plane_interaction(visual.id, source_id=source)

    assert _interactions(log) == [("start", None), ("end", "release")]
    assert controller.plane_interaction_state(visual.id) == "idle"
    # The start is announced ahead of the drag's first change.
    assert isinstance(log[0], PlaneInteractionEvent)
    assert isinstance(log[1], ClippingPlanesChangedEvent)
    assert [e.source_id for e in log] == [source] * len(log)


async def test_stillness_ends_a_drag_and_the_next_change_starts_another(controller):
    controller._render_manager.config.scheduler.dims_settle_s = 0.02
    visual, item, system, log = _scene(controller)
    source = uuid4()
    controller.begin_plane_interaction(visual.id, source_id=source)
    _move(controller, visual, item, system, 18.0)
    await asyncio.sleep(0.1)
    assert _interactions(log) == [("start", None), ("end", "settle")]

    _move(controller, visual, item, system, 20.0)
    controller.end_plane_interaction(visual.id, source_id=source)
    assert _interactions(log)[2:] == [("start", None), ("end", "release")]


def test_with_no_event_loop_each_change_settles(controller):
    visual, item, system, log = _scene(controller)
    with controller.plane_interaction(visual.id):
        _move(controller, visual, item, system, 18.0)
        _move(controller, visual, item, system, 20.0)
    assert _interactions(log) == [("start", None), ("end", "settle")] * 2


async def test_a_drag_changes_no_planning(controller, reslice, tmp_path, monkeypatch):
    scene = controller.add_scene(dim="3d", name="ms")
    visual, store = h.add_visual(
        "image_multiscale_mip", controller, scene.id, h.image_data(), tmp_path, "ms"
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    modes: list = []
    original = controller._render_config_for

    def _recording(*args, **kwargs):
        config = original(*args, **kwargs)
        modes.append(config.plan_mode)
        return config

    monkeypatch.setattr(controller, "_render_config_for", _recording)
    visual.clipping_planes = h.clipping_planes(store, [((0, 0, 16.5), (0, 0, 1))])
    outside = list(modes)
    assert outside  # the change planned
    modes.clear()
    item = visual.clipping_planes[0]
    moved = h.clipping_planes(store, [((0, 0, 20.5), (0, 0, 1))])[0].plane
    with controller.plane_interaction(visual.id):
        controller.set_clipping_plane(visual.id, item.id, moved)
    assert modes == outside
    await drain_loading(controller)


async def test_removing_the_visual_forgets_its_drag(controller):
    visual, item, system, log = _scene(controller)
    controller.begin_plane_interaction(visual.id, source_id=uuid4())
    _move(controller, visual, item, system, 18.0)
    assert controller.plane_interaction_state(visual.id) == "active"
    controller.remove_visual(visual.id)
    assert controller.plane_interaction_state(visual.id) == "idle"
    assert _interactions(log) == [("start", None)]
    assert controller._plane_interaction_driver.tasks() == []
