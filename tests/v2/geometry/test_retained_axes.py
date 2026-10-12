"""Which data columns a geometry visual uploads.

``displayed_axes`` indexes the **world**.  A store reaches the world through
its transform, so the data axes it keeps are whatever that transform maps the
displayed world axes from -- not the same numbers, and not even the same
count, once the store is of lower rank than the world.

Reading them positionally raised ``IndexError`` on a broadcast store and
silently uploaded the wrong columns on a permuted one, which is a wrong
picture with no error attached.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.data.points._points_requests import PointsSliceRequest
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform, Axis
from cellier.visuals._points_memory import PointsMarkerAppearance
from tests._displayed import trailing_axes
from tests._v2 import data_region, data_system

_POSITIONS = np.array(
    [[1.0, 10.0, 20.0], [2.0, 12.0, 22.0], [3.0, 14.0, 24.0]], dtype=np.float32
)


def _channel_world():
    """A ``(c, z, y, x)`` world: a channel axis in front of three spatial ones."""
    return (Axis(name="c", axis_type="channel"), *spatial_axes("z", "y", "x"))


def _viewer(world_axes, *, dim="3d"):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=world_axes, displayed_axes=trailing_axes(world_axes, dim)
    )
    controller.add_canvas(scene.id)
    store = PointsMemoryStore(
        positions=_POSITIONS.copy(),
        name="pts",
        data_coordinate_systems=[data_system(("z", "y", "x"))],
    )
    return controller, scene, store


def test_a_broadcast_store_keeps_its_own_axes_not_the_worlds():
    """A ``zyx`` store in a ``czyx`` world retains (0, 1, 2) while the world
    displays (1, 2, 3).  Asking it for data axis 3 is an IndexError."""
    controller, scene, store = _viewer(_channel_world())
    data = store.data_coordinate_systems[0]
    world = scene.dims.world_coordinate_system
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axis_by_name(name).id: world.axis_by_name(name).id
            for name in ("z", "y", "x")
        },
        broadcast_output_axes=[world.axis_by_name("c").id],
    )
    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(),
        transform=transform,
    )

    assert scene.dims.selection.displayed_axes == (1, 2, 3)
    assert controller.render_spaces(visual.id).retained_axes == (0, 1, 2)


def test_a_permuted_transform_keeps_the_axes_it_maps_from():
    """data z -> world y, data y -> world x, data x -> world z.

    Displaying world (y, x) means uploading data (z, y) -- columns 0 and 1 --
    and a positional reading uploads columns 1 and 2 instead.  Both are
    in range, so nothing raises; the markers simply land somewhere else.
    """
    controller, scene, store = _viewer(spatial_axes("z", "y", "x"), dim="2d")
    data = store.data_coordinate_systems[0]
    world = scene.dims.world_coordinate_system
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axis_by_name("z").id: world.axis_by_name("y").id,
            data.axis_by_name("y").id: world.axis_by_name("x").id,
            data.axis_by_name("x").id: world.axis_by_name("z").id,
        },
    )
    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(),
        transform=transform,
    )

    assert scene.dims.selection.displayed_axes == (1, 2)
    assert controller.render_spaces(visual.id).retained_axes == (0, 1)


def test_an_order_preserving_transform_is_unchanged():
    """The fix is invisible where the two happened to agree, which is every
    case the suite had before it."""
    controller, scene, store = _viewer(spatial_axes("z", "y", "x"), dim="2d")
    visual = controller.add_points(
        data=store, scene_id=scene.id, appearance=PointsMarkerAppearance()
    )
    assert scene.dims.selection.displayed_axes == (1, 2)
    assert controller.render_spaces(visual.id).retained_axes == (1, 2)


async def test_the_store_uploads_the_columns_the_request_names():
    """``retained_axes`` is what indexes the position array."""
    store = PointsMemoryStore(
        positions=_POSITIONS.copy(),
        name="pts",
        data_coordinate_systems=[data_system(("z", "y", "x"))],
    )
    request = PointsSliceRequest(
        slice_request_id=(request_id := uuid4()),
        chunk_request_id=request_id,
        scale_index=0,
        displayed_axes=(1, 2, 3),
        retained_axes=(0, 1, 2),
        region=data_region(3),
    )
    data = await store.get_data(request)
    np.testing.assert_array_equal(data.positions, _POSITIONS)


async def test_retained_axes_is_required_and_world_axes_cannot_stand_in():
    """The unplaced fallback is gone (R8.3).

    It read ``displayed_axes`` -- **world** axes -- as data columns, correct
    exactly when the store was of the world's rank and its transform
    preserved axis order, and silently wrong otherwise.  A request now has to
    say which data axes it means, and there is no way to leave it out."""
    with pytest.raises(TypeError, match="retained_axes"):
        PointsSliceRequest(
            slice_request_id=(request_id := uuid4()),
            chunk_request_id=request_id,
            scale_index=0,
            displayed_axes=(1, 2),
            region=data_region(3),
        )
