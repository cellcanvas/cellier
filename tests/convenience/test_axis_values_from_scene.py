"""World-space axis ranges from store extents (implementation plan, Phase 1).

``_axis_values_from_scene`` used to read ``level_shapes``, which only image
and label stores have.  Now every store answers ``axis_extents``, so the
geometry-only scene below -- which raised before this phase -- works.
"""

import numpy as np
import pytest

from cellier.convenience import ContinuousAxisValues, Viewer, axis_values_from_viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


def _viewer() -> Viewer:
    return Viewer(spatial_axes("z", "y", "x"))


def _add_image(viewer: Viewer, data: np.ndarray, name: str = "img"):
    store = ImageMemoryStore(data=data, name=name)
    viewer.controller.add_data_store(store)
    viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(),
        name=name,
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    return store


def _add_points(viewer: Viewer, positions: np.ndarray, name: str = "pts"):
    store = PointsMemoryStore(positions=positions, name=name)
    viewer.controller.add_data_store(store)
    viewer.add_points(store, name=name)
    return store


def test_points_only_scene_has_ranges():
    """The bug this phase fixes: a geometry-only scene used to raise."""
    viewer = _viewer()
    _add_points(viewer, np.array([[0.0, 1.0, 2.0], [10.0, 5.0, 3.0]]))

    assert axis_values_from_viewer(viewer) == {
        0: ContinuousAxisValues(min=0.0, max=10.0),
        1: ContinuousAxisValues(min=1.0, max=5.0),
        2: ContinuousAxisValues(min=2.0, max=3.0),
    }


def test_image_ranges_use_the_edge_convention():
    """Regression value for the half-voxel widening this phase introduces."""
    viewer = _viewer()
    _add_image(viewer, np.zeros((8, 16, 24), dtype=np.float32))

    assert axis_values_from_viewer(viewer) == {
        0: ContinuousAxisValues(min=-0.5, max=7.5),
        1: ContinuousAxisValues(min=-0.5, max=15.5),
        2: ContinuousAxisValues(min=-0.5, max=23.5),
    }


def test_mixed_scene_takes_the_union():
    """The union is what makes a mixed-extent scene scrubbable across both."""
    viewer = _viewer()
    _add_image(viewer, np.zeros((8, 16, 24), dtype=np.float32))
    _add_points(viewer, np.array([[-4.0, 1.0, 2.0], [20.0, 5.0, 3.0]]))

    ranges = axis_values_from_viewer(viewer)
    # Axis 0: points reach further at both ends than the 8-voxel image.
    assert ranges[0] == ContinuousAxisValues(min=-4.0, max=20.0)
    # Axes 1 and 2: the image is wider, so its edges win.
    assert ranges[1] == ContinuousAxisValues(min=-0.5, max=15.5)
    assert ranges[2] == ContinuousAxisValues(min=-0.5, max=23.5)


def test_an_empty_store_does_not_drag_the_union_to_zero():
    """An empty store occupies nothing, so it must be skipped, not counted."""
    viewer = _viewer()
    _add_image(viewer, np.ones((8, 16, 24), dtype=np.float32))
    _add_points(viewer, np.zeros((0, 3), dtype=np.float32), name="empty")

    assert axis_values_from_viewer(viewer) == {
        0: ContinuousAxisValues(min=-0.5, max=7.5),
        1: ContinuousAxisValues(min=-0.5, max=15.5),
        2: ContinuousAxisValues(min=-0.5, max=23.5),
    }


def test_a_scene_with_nothing_to_measure_raises():
    viewer = _viewer()
    _add_points(viewer, np.zeros((0, 3), dtype=np.float32), name="empty")

    with pytest.raises(ValueError, match="No visuals with extents"):
        axis_values_from_viewer(viewer)


def test_an_axis_only_broadcast_visuals_reach_falls_back_to_the_origin():
    """A broadcast axis has no extent, so a slider range needs a fallback.

    Ranges come from ``cellier.scene._bounds``, which skips broadcast axes
    rather than reading the zero row's translation off them.  With nothing
    else reaching ``c`` the range falls back to the origin -- where the zero
    row put it before -- and the other axes are unaffected.
    """
    from uuid import UUID

    from cellier.transform import AffineTransform

    viewer = Viewer([("c", "channel"), *spatial_axes("z", "y", "x")])
    _add_image(viewer, np.zeros((4, 6, 8), dtype=np.float32))
    (visual,) = viewer.scene.visuals
    store = viewer.controller.get_data_store(UUID(visual.data_store_id))
    data = store.data_coordinate_system
    world = viewer.scene.dims.world_coordinate_system
    visual.transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={data.axes[i].id: world.axes[i + 1].id for i in range(3)},
        broadcast_output_axes=(world.axes[0].id,),
    )

    ranges = axis_values_from_viewer(viewer)

    assert ranges[0] == ContinuousAxisValues(min=0.0, max=0.0)
    assert ranges[1] == ContinuousAxisValues(min=-0.5, max=3.5)
