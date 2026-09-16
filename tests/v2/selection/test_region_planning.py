"""What planning from the region selects, on every in-memory family.

Design 3.7's claim: because this phase bounds only the collapsed axes at zero
thickness, ``bounding_box()`` returns ``lo == hi`` on a collapsed axis -- the
slice position pulled into voxel space, rounded by
``round_world_to_voxel`` -- and ``(-inf, +inf)`` on a displayed one, clamped
to the full extent.

These tests were originally written as an equality against planning from
``dims_state``, which was the path the region replaced.  Phase 8 deleted that
path, so the agreement they recorded is now pinned as the expected selection
itself -- the same numbers, asserted against the rule rather than against a
second implementation of it.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.render.visuals._slicing import round_world_to_voxel
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from tests._v2 import bound

_WORLD = [("t", "time"), *spatial_axes("z", "y", "x")]
_SHAPE = (10, 20, 30, 40)


def _viewer(kind, transform_spec=None):
    """Add one visual of *kind* to a ``TZYX`` scene with one canvas."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_WORLD, dim="3d")
    controller.add_canvas(scene.id)
    if kind == "image":
        store = ImageMemoryStore(data=np.zeros(_SHAPE, dtype=np.float32))
        visual = controller.add_image(
            data=store,
            scene_id=scene.id,
            appearance=InMemoryImageAppearance(),
            single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
        )
    elif kind == "labels":
        store = LabelMemoryStore(data=np.zeros(_SHAPE, dtype=np.int32))
        visual = controller.add_labels(
            data=store, scene_id=scene.id, appearance=InMemoryLabelsAppearance()
        )
    else:
        store = ImageMemoryStore(data=np.zeros((3, 20, 30, 40), dtype=np.float32))
        visual = controller.add_image(
            data=store,
            scene_id=scene.id,
            channel_axis=0,
            composite=True,
            channels={
                0: InMemoryImageChannelAppearance(color_map="red"),
                1: InMemoryImageChannelAppearance(color_map="green"),
            },
        )
    if transform_spec is not None:
        visual.transform = bound(controller, scene.id, store, *transform_spec)
    return controller, scene, visual


def _plan(controller, scene, visual, *, with_region: bool = True):
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    selection = (
        controller._selections_for_scene(scene.id)[canvas_id] if with_region else None
    )
    requests = gfx.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=100.0,
        dims_state=scene.dims.to_state(),
        selection=selection,
    )
    return [request.axis_selections for request in requests]


@pytest.mark.parametrize("kind", ["image", "labels"])
@pytest.mark.parametrize("slice_position", [0.0, 1.0, 2.5, 9.0])
async def test_the_region_collapses_to_the_rounded_plane(kind, slice_position):
    """Identity transform, so the world position *is* the voxel coordinate and
    the only thing left to check is the rounding rule."""
    controller, scene, visual = _viewer(kind)
    controller.update_slice_indices(scene.id, {0: slice_position})
    expected = round_world_to_voxel(slice_position, _SHAPE[0])
    assert _plan(controller, scene, visual) == [(expected, (0, 20), (0, 30), (0, 40))]


@pytest.mark.parametrize("slice_position", [9.5, 99.0, -0.6])
async def test_labels_clamp_a_slice_outside_the_data(slice_position):
    """Labels keep the clamping assembler (design 3.2)."""
    controller, scene, visual = _viewer("labels")
    controller.update_slice_indices(scene.id, {0: slice_position})
    expected = round_world_to_voxel(slice_position, _SHAPE[0])
    assert _plan(controller, scene, visual) == [(expected, (0, 20), (0, 30), (0, 40))]


@pytest.mark.parametrize("slice_position", [9.5, 99.0, -0.6])
async def test_an_image_draws_nothing_for_a_slice_outside_the_data(slice_position):
    """``[-0.5, size - 0.5)`` is the data's extent; outside it an image plans no
    request and hides its data node (design 3.2)."""
    controller, scene, visual = _viewer("image")
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    controller.update_slice_indices(scene.id, {0: slice_position})
    assert _plan(controller, scene, visual) == []
    assert gfx._inner_node_3d.visible is False

    controller.update_slice_indices(scene.id, {0: 3.0})
    assert _plan(controller, scene, visual) == [(3, (0, 20), (0, 30), (0, 40))]
    assert gfx._inner_node_3d.visible is True


@pytest.mark.parametrize("slice_position", [0.0, 1.0, 2.5, 9.0, 99.0])
async def test_a_multichannel_visual_plans_one_request_per_channel(slice_position):
    """The channel axis is this family's own concern: the region names it like
    any other collapsed axis and each request then overwrites it.  The image
    slicing rule skips it, so a channel slider outside the data still draws."""
    controller, scene, visual = _viewer("multichannel")
    controller.update_slice_indices(scene.id, {0: slice_position})
    assert _plan(controller, scene, visual) == [
        (0, (0, 20), (0, 30), (0, 40)),
        (1, (0, 20), (0, 30), (0, 40)),
    ]


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_an_awkward_transform_still_rounds_half_up(kind):
    """Design 3.7's numbers: ``T = 0.5 t + 0.25`` and ``Z = 2 z + 10``, chosen
    so the rounding step has something to do."""
    controller, scene, visual = _viewer(
        kind, ((0.5, 2.0, 0.5, 0.5), (0.25, 10.0, 0.0, 0.0))
    )
    controller.update_slice_indices(scene.id, {0: 1.0})
    # world T = 1 -> voxel t = 1.5 -> half-up -> 2, and the design says so.
    assert _plan(controller, scene, visual) == [(2, (0, 20), (0, 30), (0, 40))]


@pytest.mark.parametrize("kind", ["image", "labels", "multichannel"])
async def test_planning_without_a_region_is_now_an_error(kind):
    """R8.3.  Until v1 was retired, a request carrying no region fell back to
    reading ``dims_state.slice_indices`` as world positions.  There is no
    second path now, and a caller who reaches this has a bug upstream."""
    controller, scene, visual = _viewer(kind)
    with pytest.raises(RuntimeError, match="no region to plan from"):
        _plan(controller, scene, visual, with_region=False)


async def test_a_thickness_on_the_collapsed_axis_fetches_a_slab_for_labels():
    """The extension design 3.7 promises: everything downstream of the region
    is untouched, and only the editor's thickness moves."""
    controller, scene, visual = _viewer("labels")
    controller.update_slice_indices(scene.id, {0: 4.0})
    scene.dims.selection.thickness = {0: 1.5}
    assert _plan(controller, scene, visual, with_region=True) == [
        ((3, 7), (0, 20), (0, 30), (0, 40))
    ]


@pytest.mark.parametrize(
    "slice_position, half_thickness, expected",
    [(4.0, 1.5, 4), (4.4, 1.5, 4), (11.0, 2.0, 9), (-2.0, 1.5, 0)],
)
async def test_a_thickness_on_an_image_draws_the_nearest_plane_in_it(
    slice_position, half_thickness, expected
):
    """An image draws one plane per sliced axis: the sample nearest the slice
    position whose extent overlaps the band (design 3.2)."""
    controller, scene, visual = _viewer("image")
    controller.update_slice_indices(scene.id, {0: slice_position})
    scene.dims.selection.thickness = {0: half_thickness}
    assert _plan(controller, scene, visual) == [(expected, (0, 20), (0, 30), (0, 40))]


async def test_a_thickness_too_thin_to_reach_the_data_draws_nothing_for_an_image():
    controller, scene, visual = _viewer("image")
    controller.update_slice_indices(scene.id, {0: 11.0})
    scene.dims.selection.thickness = {0: 1.0}
    assert _plan(controller, scene, visual) == []


async def test_a_sheared_transform_is_rejected_when_it_is_assigned():
    """D7, caught one step earlier than the assembler.

    Today's in-memory path builds a zero-filled world point, writes the slice
    positions into it and inverts -- which answers "the ``t`` at world
    ``Z = Y = X = 0``", an arbitrary choice nothing states, and no error.  A
    sheared transform now fails when it is *assigned*, because the render
    spaces read the axis correspondence off the matrix and there is none.
    The assembler's own half-bounded guard is the second line of defence, and
    is covered in the assembler's tests.
    """
    from cellier.transform import AffineTransform

    controller, scene, visual = _viewer("image")
    store = controller.get_data_store(UUID(visual.data_store_id))
    matrix = np.eye(5)
    matrix[0, 1] = 0.1  # world T picks up a little of data z
    sheared = AffineTransform.from_matrix(
        matrix, store.data_coordinate_system, scene.dims.world_coordinate_system
    )
    with pytest.raises(Exception, match="one output axis per input axis"):
        visual.transform = sheared
