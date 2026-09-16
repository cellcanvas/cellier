"""Tests for scene-level readiness: reslice_scene(on_ready=...) / on_scene_ready.

These exercise the armed-quiescence tracker that fires a single callback once
every visual loaded by a reslice has committed to the GPU, across mixed
geometry types.  They also assert the ``canvas_id`` now carried by the reslice
events.

Async tests take ``qtbot`` so a ``QApplication`` exists for the offscreen
canvas, and pytest-asyncio (mode=auto) installs a running loop for the slicer.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events._events import ResliceCompletedEvent, ResliceStartedEvent
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


def _build_mixed_scene(controller: CellierController):
    """A 3D scene with an in-memory image and an in-memory points visual."""
    scene = controller.add_scene(dim="3d", name="scene")

    img_data = np.random.rand(16, 16, 16).astype(np.float32)
    img_store = ImageMemoryStore(data=img_data, name="img")
    img_visual = controller.add_image(
        data=img_store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    positions = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32)
    pts_store = PointsMemoryStore(positions=positions)
    pts_visual = controller.add_points(data=pts_store, scene_id=scene.id)

    controller.add_canvas(scene_id=scene.id)
    return scene, img_visual, pts_visual


async def test_on_ready_fires_once_after_mixed_geometry_commits(qtbot):
    """reslice_scene(on_ready=...) fires exactly once after image + points load."""
    controller = CellierController()
    scene, _img, _pts = _build_mixed_scene(controller)

    calls: list[int] = []
    controller.reslice_scene(scene.id, on_ready=lambda: calls.append(1))

    # Not ready yet: the async brick/tile reads have not been driven.
    assert calls == []

    tasks = list(controller._render_manager._slicer._tasks.values())
    if tasks:
        await asyncio.gather(*tasks)

    assert calls == [1], "on_ready must fire exactly once after all data commits"


async def test_on_scene_ready_is_equivalent(qtbot):
    """on_scene_ready triggers the load and fires the callback once."""
    controller = CellierController()
    scene, _img, _pts = _build_mixed_scene(controller)

    calls: list[int] = []
    controller.on_scene_ready(scene.id, lambda: calls.append(1))

    tasks = list(controller._render_manager._slicer._tasks.values())
    if tasks:
        await asyncio.gather(*tasks)

    assert calls == [1]


def test_on_ready_fires_immediately_when_nothing_to_load(qtbot):
    """A scene with no visuals has nothing to load, so on_ready fires at once."""
    controller = CellierController()
    scene = controller.add_scene(dim="3d", name="empty")
    controller.add_canvas(scene_id=scene.id)

    calls: list[int] = []
    controller.reslice_scene(scene.id, on_ready=lambda: calls.append(1))

    # No visuals -> no ResliceStartedEvent -> pending stays 0 -> fires
    # synchronously when the arming window closes (no event loop needed).
    assert calls == [1]


async def test_reslice_events_carry_canvas_id(qtbot):
    """ResliceStartedEvent and ResliceCompletedEvent both report the canvas id."""
    controller = CellierController()
    scene, _img, _pts = _build_mixed_scene(controller)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    owner_id = uuid4()
    started: list[ResliceStartedEvent] = []
    completed: list[ResliceCompletedEvent] = []
    controller._outgoing_events.subscribe(
        ResliceStartedEvent, started.append, owner_id=owner_id
    )
    controller._outgoing_events.subscribe(
        ResliceCompletedEvent, completed.append, owner_id=owner_id
    )

    controller.reslice_scene(scene.id)

    # Drive the async reads so the completion path runs too.
    tasks = list(controller._render_manager._slicer._tasks.values())
    if tasks:
        await asyncio.gather(*tasks)

    assert started, "expected a ResliceStartedEvent"
    assert all(ev.canvas_id == canvas_id for ev in started)
    assert completed, "expected a ResliceCompletedEvent"
    assert all(ev.canvas_id == canvas_id for ev in completed)
