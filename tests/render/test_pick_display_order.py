"""Image and labels picks decode to the right data axes in any display order.

The render layer decodes a pick in the visual's texture order, which is the
drawn data axes ascending whatever ``displayed_axes`` says: only the node
matrix carries the display permutation.  A transposed display used to come
back with its displayed data axes swapped (found by the B8 hand check,
``scripts/b8_mouse_world_coordinate_check.py --transposed``).

The landmark block sits at ``P0`` = data ``(z, y, x)`` ``(7, 20, 28)``, so a
swap of ``y`` and ``x`` misses it by 8 voxels.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience.capture import _wait_until_loaded
from cellier.data import ImageMemoryStore
from cellier.events import MaskedImagePickEvent, ViewRay
from cellier.events._events import _CanvasRawPointerEvent
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes
from cellier.visuals import MaskedImageSingleAppearance
from tests.render._affine import (
    P0,
    RASTER,
    add_landmark,
    block_volume,
    drawn,
    load,
    render,
    world_matrix,
)
from tests.render._masked import masked_config

ORDERS = [
    ("y", "x"),
    ("x", "y"),
    ("z", "y", "x"),
    ("x", "z", "y"),
    ("y", "x", "z"),
]


def _viewer(displayed: tuple[str, ...]) -> Viewer:
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=displayed, gui="offscreen"
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    return viewer


@pytest.mark.parametrize("family", RASTER)
@pytest.mark.parametrize("displayed", ORDERS, ids="".join)
def test_picks_land_on_the_landmark_in_any_display_order(tmp_path, family, displayed):
    viewer = _viewer(displayed)
    try:
        visual = add_landmark(viewer, family, world_matrix(np.eye(3)), tmp_path)
        load(viewer, slice_z=float(P0[0]) if len(displayed) == 2 else None)
        frame, renderer, _ = render(viewer, 480)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        dims = viewer.scene.dims
        displayed_axes = tuple(dims.selection.displayed_axes)
        slice_indices = {
            axis: position
            for axis, position in dims.selection.slice_indices.items()
            if axis not in displayed_axes
        }
        rows, cols = np.nonzero(drawn(frame))
        assert rows.size > 0
        rng = np.random.default_rng(0)
        picks = rng.choice(rows.size, size=min(20, rows.size), replace=False)
        for index in picks:
            info = renderer.get_pick_info((cols[index] + 0.5, rows[index] + 0.5))
            coord = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
            assert coord is not None
            full = viewer.controller._promote_pick_coordinate(
                coord,
                displayed_axes=displayed_axes,
                slice_indices=slice_indices,
                world_ndim=3,
                hit_visual_id=visual.id,
            )
            voxel = np.floor(np.asarray(full))
            assert np.all(np.abs(voxel - P0) <= 2.0), (voxel, P0)
    finally:
        viewer.controller.close()


def _masked_pick_events(viewer: Viewer, frame, renderer, n_picks: int = 20) -> list:
    """Pick drawn pixels and send each through the controller's pick path."""
    controller = viewer.controller
    scene_id = viewer.scene.id
    canvas_id = controller.get_canvas_ids(scene_id)[0]
    manager = controller._render_manager
    camera = "2d" if len(viewer.scene.dims.selection.displayed_axes) == 2 else "3d"
    events: list[MaskedImagePickEvent] = []
    controller.on_pick(canvas_id, MaskedImagePickEvent, events.append, owner_id=uuid4())

    rows, cols = np.nonzero(drawn(frame))
    assert rows.size > 0
    rng = np.random.default_rng(0)
    raw_events = []
    for index in rng.choice(rows.size, size=min(n_picks, rows.size), replace=False):
        info = renderer.get_pick_info((cols[index] + 0.5, rows[index] + 0.5))
        hit = info.get("world_object")
        assert hit is not None
        details = manager._extract_pick_details(scene_id, hit, info)
        assert details is not None
        raw_events.append(
            _CanvasRawPointerEvent(
                canvas_id=canvas_id,
                scene_id=scene_id,
                action="press",
                camera_type=camera,
                position_2d=np.array([1.0, 2.0]) if camera == "2d" else None,
                ray=(
                    ViewRay(origin=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]))
                    if camera == "3d"
                    else None
                ),
                hit_visual_id=manager._scenes[scene_id].get_visual_id_for_node(hit),
                button=1,
                modifiers=(),
                buttons=(1,),
                gesture_id=uuid4(),
                pick_details=details,
            )
        )

    async def _run() -> None:
        for raw in raw_events:
            controller._on_raw_pointer_event(raw)
        await _wait_until_loaded(controller)

    asyncio.run(_run())
    assert len(events) == len(raw_events)
    return events


@pytest.mark.parametrize(
    "displayed", [("x", "y"), ("x", "z", "y"), ("y", "x", "z")], ids="".join
)
def test_masked_image_picks_land_on_the_landmark_in_a_permuted_order(displayed):
    """A MaskedImage pick goes through the same promotion as an image pick.

    In 3D that is the paired branch the transposed display broke.  A 2D
    MaskedImage always draws through the slice node, whose decode is laid out
    by its own branch; ``xy`` checks that one in a permuted order.
    """
    viewer = _viewer(displayed)
    try:
        landmark = block_volume(np.float32)
        viewer.controller.add_masked_image(
            ImageMemoryStore(data=landmark, name="mask"),
            ImageMemoryStore(data=landmark, name="value"),
            viewer.scene.id,
            single=MaskedImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), iso_threshold=0.5
            ),
            render_config=masked_config(),
        )
        load(viewer, slice_z=float(P0[0]) if len(displayed) == 2 else None)
        frame, renderer, _ = render(viewer, 480)
        for event in _masked_pick_events(viewer, frame, renderer):
            voxel = np.floor(np.asarray(event.pick_info.data_coordinate))
            assert np.all(np.abs(voxel - P0) <= 2.0), (voxel, P0)
    finally:
        viewer.controller.close()
