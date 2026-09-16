"""``RegionSelection`` is the artifact the slicer consumes (D43).

``DimsManager`` becomes an *editor* -- it holds an index and a thickness, the
thing a GUI binds to -- and emits one of these.  An oblique editor would hold
a plane and emit the same type, and the slicer would not know the difference
(R6).
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.scene.dims import (
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.transform import RegionSelection, RenderedCoordinateSystem


@pytest.fixture
def viewer():
    """A ``TZYX`` scene showing ``ZYX`` on one canvas, sliced at ``T = 1``."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")], dim="3d"
    )
    controller.add_canvas(scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    return controller, scene, canvas_id


def _selection(viewer) -> RegionSelection:
    controller, scene, canvas_id = viewer
    rendered, embedding = controller._rendered[canvas_id]
    return scene.dims.to_selection(rendered, embedding)


# ---------------------------------------------------------------------------
# What the editor emits
# ---------------------------------------------------------------------------


async def test_the_collapsed_axis_is_bounded_and_the_displayed_ones_are_not(viewer):
    """Design 3.7 step 2: in this phase the region bounds only the collapsed
    axes, which is what makes the output identical to what came before."""
    controller, scene, _ = viewer
    controller.update_slice_indices(scene.id, {0: 1.0})
    box = _selection(viewer).region.bounding_box()
    np.testing.assert_array_equal(box.min_coordinate, [1.0, -np.inf, -np.inf, -np.inf])
    np.testing.assert_array_equal(box.max_coordinate, [1.0, np.inf, np.inf, np.inf])


def test_the_region_is_in_world_coordinates(viewer):
    """Not rendered: rendered space has only the displayed axes and so cannot
    express a thickness on a collapsed one."""
    _controller, scene, _ = viewer
    selection = _selection(viewer)
    assert selection.region.coordinate_system == (scene.dims.world_coordinate_system.id)
    assert selection.region.ndim == 4


async def test_the_slice_position_lives_in_the_transforms_constant_column(viewer):
    """D35: the numbers stay in one place.  The transform's constant column
    holds the slice *position* and the region holds its *extent*; both are
    downstream of the editor's index and neither is stored twice."""
    controller, scene, _ = viewer
    controller.update_slice_indices(scene.id, {0: 2.5})
    selection = _selection(viewer)
    assert selection.slice_position[0] == pytest.approx(2.5)


async def test_an_axis_with_no_thickness_is_a_plane(viewer):
    """``half_thickness`` defaults to 0.5 because that is the number the
    geometry request builders hardcoded, but the *region* says only what the
    user asked for.  Bounding a collapsed axis by default would make an image
    visual -- which draws a single plane -- fetch a slab it cannot show."""
    controller, scene, _ = viewer
    controller.update_slice_indices(scene.id, {0: 1.0})
    box = _selection(viewer).region.bounding_box()
    assert box.min_coordinate[0] == box.max_coordinate[0] == 1.0


async def test_an_explicit_thickness_makes_a_slab(viewer):
    """The one change that turns a plane into a slab: everything downstream --
    the pull-back, the bounding box, the assembler -- is untouched."""
    controller, scene, _ = viewer
    controller.update_slice_indices(scene.id, {0: 4.0})
    scene.dims.selection.thickness = {0: 1.5}
    box = _selection(viewer).region.bounding_box()
    assert box.min_coordinate[0] == pytest.approx(2.5)
    assert box.max_coordinate[0] == pytest.approx(5.5)


def test_a_displayed_axis_position_is_left_unbounded(viewer):
    """D36: a displayed axis keeps a stored position, which never bounds the
    region -- it is drawn, not sliced."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")], dim="3d"
    )
    controller.update_slice_indices(scene.id, {1: 4.0, 2: 5.0, 3: 6.0})
    controller.add_canvas(scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    rendered, embedding = controller._rendered[canvas_id]
    box = scene.dims.to_selection(rendered, embedding).region.bounding_box()
    for axis in (1, 2, 3):
        assert box.min_coordinate[axis] == -np.inf
        assert box.max_coordinate[axis] == np.inf


def test_the_embedding_never_declares_a_broadcast_axis(viewer):
    """F1.1: a world axis the canvas does not display sits at a definite
    position, not free.  ``RegionSelection`` rejects an embedding that says
    otherwise, and this is what keeps that invariant honest."""
    selection = _selection(viewer)
    assert not selection.transform.broadcast_axes


def test_a_selection_whose_slice_misses_its_region_is_rejected():
    """D43's invariant: pulling the region back through the transform must
    leave a non-empty set.  A selection whose own slice position selects
    nothing is incoherent."""
    from cellier.transform import AffineTransform, ConvexRegion

    world = world_coordinate_system([("t", "time"), *spatial_axes("z", "y", "x")])
    canvas_id = __import__("uuid").uuid4()
    rendered = RenderedCoordinateSystem.from_world(
        world, [world.axes[a].id for a in (1, 2, 3)], canvas_id
    )
    embedding = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={
            rendered.axes[i].id: world.axes[a].id for i, a in enumerate((1, 2, 3))
        },
        constant_output_axes={world.axes[0].id: 0.0},
    )
    # The canvas is showing T = 0 but the region wants T = 9.
    region = ConvexRegion.from_axis_slabs(world, {world.axes[0].id: (9.0, 0.0)})
    with pytest.raises(ValueError, match="would select nothing"):
        RegionSelection(transform=embedding, region=region)


def test_a_plane_selection_cannot_emit_one_yet():
    from cellier.scene.dims import PlaneSelection

    # model_construct rather than the constructor: the axis-coverage
    # validator has nothing to check a PlaneSelection against.
    dims_plane = DimsManager.model_construct(
        world_coordinate_system=world_coordinate_system(spatial_axes("z", "y", "x")),
        selection=PlaneSelection(),
    )
    with pytest.raises(NotImplementedError, match="RegionSelection"):
        dims_plane.to_selection(None, None)


# ---------------------------------------------------------------------------
# The transport
# ---------------------------------------------------------------------------


def test_the_selection_rides_on_the_reslicing_request(viewer):
    """It travels beside the camera fields the request already carried."""
    controller, scene, canvas_id = viewer
    captured = []
    controller._render_manager._slice_coordinator.submit = (
        lambda request, configs=None: captured.append(request)
    )
    controller.reslice_scene(scene.id)
    assert len(captured) == 1
    assert isinstance(captured[0].selection, RegionSelection)
    assert captured[0].canvas_id == canvas_id


def test_each_canvas_gets_its_own_selection():
    """One per canvas, because the embedding is built from that canvas's
    rendered coordinate system."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")], dim="3d"
    )
    controller.add_canvas(scene.id)
    controller.add_canvas(scene.id)
    selections = controller._selections_for_scene(scene.id)
    assert set(selections) == set(controller.get_canvas_ids(scene.id))
    first, second = selections.values()
    assert first.transform.input_coordinate_system != (
        second.transform.input_coordinate_system
    )
