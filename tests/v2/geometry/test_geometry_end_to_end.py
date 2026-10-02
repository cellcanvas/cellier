"""The geometry slicing fix, driven through a live viewer.

This is the phase that changes what a user sees, so the check that matters is
the one that goes through ``CellierController``: the region is built from the
scene's dims and the canvas's rendered system, and the store filters against
it.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance
from tests._v2 import bound, data_region

# Design 3.12's six points and its 2 um z spacing.
_POSITIONS = np.array(
    [
        [1.6, 10, 20],
        [2.0, 12, 22],
        [2.4, 14, 24],
        [2.6, 16, 26],
        [13.0, 18, 28],
        [14.0, 30, 30],
    ],
    dtype=np.float32,
)


def _viewer(scale=(2.0, 0.5, 0.5), translation=(10.0, 0.0, 0.0)):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"), dim="2d"
    )
    controller.add_canvas(scene.id)
    store = PointsMemoryStore(positions=_POSITIONS.copy(), name="pts")
    visual = controller.add_points(
        data=store, scene_id=scene.id, appearance=PointsMarkerAppearance()
    )
    if scale is not None:
        visual.transform = bound(controller, scene.id, store, scale, translation)
    return controller, scene, visual, store


async def _selected(controller, scene, visual, store, *, with_region=True):
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    selection = (
        controller._selections_for_scene(scene.id)[canvas_id] if with_region else None
    )
    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=scene.dims.to_state(),
        selection=selection,
    )
    data = await store.get_data(requests[0])
    if data.original_indices is None:
        return []
    return data.original_indices.tolist()


async def test_the_visual_now_slices_against_its_transform():
    """Design 3.12's headline: the code this replaced drew the one point
    24 um off the slice plane and hid the three that are on it.

    The old path was still reachable, and asserted against here, until
    Phase 8 deleted it (R8.3); what is left is the number it was wrong about.
    The points sit at data z ``0, 2, 4, 6, 8, 14`` under a 4x z scale, so
    world Z 14 +/- 1 is data z ``[3.25, 3.75]``... only on the reading that
    ``contains`` does, against the transform.
    """
    controller, scene, visual, store = _viewer()
    controller.update_slice_indices(scene.id, {0: 14.0})
    scene.dims.selection.thickness = {0: 1.0}
    assert await _selected(controller, scene, visual, store) == [0, 1, 2]


async def test_planning_without_a_region_is_now_an_error():
    """R8.3.  A request carrying no region used to compare a world value
    against data coordinates; there is no second path now."""
    controller, scene, visual, store = _viewer()
    controller.update_slice_indices(scene.id, {0: 14.0})
    with pytest.raises(RuntimeError, match="no region to plan from"):
        await _selected(controller, scene, visual, store, with_region=False)


async def test_no_thickness_is_a_plane():
    """The scene's thickness is the only one: with none stated, only what
    lies on the plane is drawn.  There is no per-family floor."""
    controller, scene, visual, store = _viewer()
    controller.update_slice_indices(scene.id, {0: 14.0})
    assert scene.dims.selection.thickness == {}
    # world Z = 14 is data z = 2, where p1 sits.
    assert await _selected(controller, scene, visual, store) == [1]
    # world Z = 14.2 is data z = 2.1: inside the old +/-0.5 floor, on no point.
    controller.update_slice_indices(scene.id, {0: 14.2})
    assert await _selected(controller, scene, visual, store) == []


async def test_a_three_dimensional_view_keeps_every_point():
    """No axis is collapsed, so the region bounds nothing and ``contains`` is
    all-True -- the same outcome as the old "slice_indices is empty so the
    mask loop does not run", by one rule instead of two."""
    controller, scene, visual, store = _viewer()
    controller.update_displayed_axes(scene.id, (0, 1, 2))
    controller.update_slice_indices(scene.id, {})
    assert await _selected(controller, scene, visual, store) == [0, 1, 2, 3, 4, 5]


async def test_the_uploaded_positions_are_in_ascending_data_axis_order():
    """Design 3.14: ``positions[:, displayed]`` baked the display permutation
    into the vertex buffer, which is the one place it must never live."""
    controller, scene, visual, store = _viewer(scale=None)
    controller.update_slice_indices(scene.id, {0: 2.0})
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    selection = controller._selections_for_scene(scene.id)[canvas_id]

    scene.dims.selection.displayed_axes = (2, 1)
    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=scene.dims.to_state(),
        selection=selection,
    )
    reversed_order = await store.get_data(requests[0])

    scene.dims.selection.displayed_axes = (1, 2)
    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=scene.dims.to_state(),
        selection=selection,
    )
    ascending = await store.get_data(requests[0])

    np.testing.assert_array_equal(reversed_order.positions, ascending.positions)


async def test_lines_require_both_endpoints_as_they_always_did():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"), dim="2d"
    )
    controller.add_canvas(scene.id)
    # Two segments: the first lies on the slice, the second straddles it.
    positions = np.array(
        [[2.0, 0, 0], [2.0, 1, 1], [2.0, 2, 2], [9.0, 3, 3]], dtype=np.float32
    )
    store = LinesMemoryStore(positions=positions, name="lines")
    visual = controller.add_lines(
        data=store, scene_id=scene.id, appearance=LinesMemoryAppearance()
    )
    visual.transform = bound(controller, scene.id, store, (1.0, 1.0, 1.0))
    controller.update_slice_indices(scene.id, {0: 2.0})

    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
    )
    data = await store.get_data(requests[0])
    assert data.original_edge_indices.tolist() == [0]


async def test_a_store_reached_without_a_controller_still_slices():
    """A store driven directly takes the region like any other caller.

    Until Phase 8 a request could carry ``slice_indices`` plus ``thickness``
    instead, which is what kept a headlessly constructed visual drivable --
    at the cost of comparing a **world** value against **data** coordinates.
    The region is built in the store's own space here, so the same slab
    selects the same point.
    """
    store = PointsMemoryStore(positions=_POSITIONS.copy(), name="pts")
    from cellier.data.points._points_requests import PointsSliceRequest

    request = PointsSliceRequest(
        slice_request_id=UUID(int=1),
        chunk_request_id=UUID(int=1),
        scale_index=0,
        displayed_axes=(1, 2),
        retained_axes=(1, 2),
        region=data_region(3, {0: (14, 0.5)}),
    )
    data = await store.get_data(request)
    assert data.original_indices.tolist() == [5]


@pytest.mark.parametrize("thickness", [0.5, 1.0, 2.0])
async def test_thickness_is_a_world_quantity_end_to_end(thickness):
    controller, scene, visual, store = _viewer()
    controller.update_slice_indices(scene.id, {0: 14.0})
    scene.dims.selection.thickness = {0: thickness}
    expected = {0.5: [1], 1.0: [0, 1, 2], 2.0: [0, 1, 2, 3]}[thickness]
    assert await _selected(controller, scene, visual, store) == expected
