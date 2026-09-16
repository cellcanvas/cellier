"""Integration tests for ResliceCompletedEvent emission.

These exercise the full reslice path through the controller, render manager,
slice coordinator, and async slicer so that ``on_reslice_completed`` fires for
every visual once its bricks/tiles have been committed.

The tests are ``async`` so pytest-asyncio (mode=auto) installs a running event
loop — the async slicer schedules its work via ``asyncio.ensure_future`` — and
take ``qtbot`` so a ``QApplication`` exists for the offscreen canvas.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


async def test_on_reslice_completed_fires_for_geometry_and_image(qtbot):
    """on_reslice_completed must fire for both a points and an image visual."""
    controller = CellierController()
    scene = controller.add_scene(dim="3d", name="scene")

    # In-memory volume image visual.
    img_data = np.random.rand(16, 16, 16).astype(np.float32)
    img_store = ImageMemoryStore(data=img_data, name="img")
    img_visual = controller.add_image(
        data=img_store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    # In-memory points (geometry) visual.
    positions = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32)
    pts_store = PointsMemoryStore(positions=positions)
    pts_visual = controller.add_points(data=pts_store, scene_id=scene.id)

    # A canvas is required for reslicing (one request is submitted per canvas).
    controller.add_canvas(scene_id=scene.id)

    owner_id = uuid4()
    events: list = []
    controller.on_reslice_completed(img_visual.id, events.append, owner_id=owner_id)
    controller.on_reslice_completed(pts_visual.id, events.append, owner_id=owner_id)

    controller.reslice_all()

    # reslice_all schedules one async task per visual synchronously; drive them
    # to completion so the on_complete -> ResliceCompletedEvent path runs.
    tasks = list(controller._render_manager._slicer._tasks.values())
    if tasks:
        await asyncio.gather(*tasks)

    fired = {event.visual_id for event in events}
    assert img_visual.id in fired, "image visual reslice never completed"
    assert pts_visual.id in fired, "points visual reslice never completed"

    # Every emitted event reports a non-negative brick count.
    assert all(event.brick_count >= 0 for event in events)
