"""The scene's thickness is the only thickness (mesh refactor v3, T1-T3).

No geometry family adds a floor: an axis the user gave no thickness is a
plane.  A continuous axis keeps pure containment.  A discrete axis is
anchored at the sample the slider selects, so thickness 0 draws exactly the
current sample.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.graph import GraphMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.render._spaces import geometry_data_region
from cellier.scene.dims import spatial_axes
from cellier.transform import (
    AffineTransform,
    Axis,
    ConvexRegion,
    DataCoordinateSystem,
    RegionSelection,
    RenderedCoordinateSystem,
    WorldCoordinateSystem,
)
from cellier.visuals._graph_memory import GraphAppearance, TrailConfig
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._mesh_memory import MeshFlatAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance

# z of each element; y and x are irrelevant to the slice.
_Z = (1.0, 2.0, 2.4, 3.0, 5.0)


def _points_positions() -> np.ndarray:
    return np.array([[z, 10.0 + i, 20.0 + i] for i, z in enumerate(_Z)], np.float32)


def _viewer():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"), dim="2d"
    )
    controller.add_canvas(scene.id)
    return controller, scene


def _request_2d(controller, scene, visual):
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    return gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
    )[0]


async def _points_selected(controller, scene) -> list[int]:
    store = PointsMemoryStore(positions=_points_positions(), name="pts")
    visual = controller.add_points(
        data=store, scene_id=scene.id, appearance=PointsMarkerAppearance()
    )
    data = await store.get_data(_request_2d(controller, scene, visual))
    return [] if data.original_indices is None else data.original_indices.tolist()


async def _lines_selected(controller, scene) -> list[int]:
    # One flat segment per z, so a segment is in the slab or out of it.
    positions = np.repeat(_points_positions(), 2, axis=0)
    positions[1::2, 1:] += 1.0
    store = LinesMemoryStore(positions=positions, name="lines")
    visual = controller.add_lines(
        data=store, scene_id=scene.id, appearance=LinesMemoryAppearance()
    )
    data = await store.get_data(_request_2d(controller, scene, visual))
    if data.is_empty or data.original_edge_indices is None:
        return []
    # One two-vertex segment per element.
    return np.unique(np.asarray(data.original_edge_indices)).tolist()


async def _mesh_selected(controller, scene) -> list[int]:
    # One flat triangle per z.
    base = _points_positions()
    positions = np.repeat(base, 3, axis=0)
    positions[1::3, 1] += 1.0
    positions[2::3, 2] += 1.0
    indices = np.arange(len(positions), dtype=np.int32).reshape(-1, 3)
    store = MeshMemoryStore(positions=positions, indices=indices, name="mesh")
    visual = controller.add_mesh(
        data=store, scene_id=scene.id, appearance=MeshFlatAppearance()
    )
    data = await store.get_data(_request_2d(controller, scene, visual))
    if data.is_empty or data.original_face_indices is None:
        return []
    return np.asarray(data.original_face_indices).tolist()


async def _graph_selected(controller, scene) -> list[int]:
    store = GraphMemoryStore.from_arrays(
        _points_positions(), np.zeros((0, 2), dtype=np.int32)
    )
    visual = controller.add_graph(
        data=store, scene_id=scene.id, appearance=GraphAppearance()
    )
    data = await store.get_data(_request_2d(controller, scene, visual))
    if data.original_node_rows is None:
        return []
    return np.asarray(data.original_node_rows).tolist()


_FAMILIES = {
    "points": _points_selected,
    "lines": _lines_selected,
    "mesh": _mesh_selected,
    "graph": _graph_selected,
}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
async def test_no_thickness_draws_only_what_lies_on_the_plane(family):
    controller, scene = _viewer()
    controller.update_slice_indices(scene.id, {0: 2.0})
    assert scene.dims.selection.thickness == {}
    # Only the element at z = 2.0; the one at 2.4 was inside the old +/-0.5.
    assert await _FAMILIES[family](controller, scene) == [1]


@pytest.mark.parametrize("family", sorted(_FAMILIES))
async def test_a_plane_between_elements_draws_nothing(family):
    controller, scene = _viewer()
    controller.update_slice_indices(scene.id, {0: 2.2})
    assert await _FAMILIES[family](controller, scene) == []


@pytest.mark.parametrize("family", sorted(_FAMILIES))
async def test_a_thickness_draws_the_band(family):
    controller, scene = _viewer()
    controller.update_slice_indices(scene.id, {0: 2.0})
    controller.update_thickness(scene.id, {0: 1.0})
    # z in [1, 3]: both edges are inside.
    assert await _FAMILIES[family](controller, scene) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Discrete axes (T2)
# ---------------------------------------------------------------------------


def _time_systems(scale: float, sampling: str = "discrete"):
    data = DataCoordinateSystem(
        name="d",
        axes=(
            Axis(name="t", axis_type="time", sampling=sampling),
            Axis(name="y", axis_type="space"),
            Axis(name="x", axis_type="space"),
        ),
        datastore_id=uuid4(),
    )
    world = WorldCoordinateSystem(
        name="w",
        axes=(
            Axis(name="t", axis_type="time"),
            Axis(name="y", axis_type="space"),
            Axis(name="x", axis_type="space"),
        ),
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={d.id: w.id for d, w in zip(data.axes, world.axes, strict=True)},
        scale={data.axes[0].id: scale},
    )
    return data, world, transform


def _selection(world, position: float, half_thickness: float) -> RegionSelection:
    canvas_id = uuid4()
    rendered = RenderedCoordinateSystem.from_world(
        world, [world.axes[1].id, world.axes[2].id], canvas_id
    )
    embedding = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={rendered.axes[i].id: world.axes[i + 1].id for i in range(2)},
        constant_output_axes={world.axes[0].id: float(position)},
    )
    return RegionSelection(
        transform=embedding,
        region=ConvexRegion.from_axis_slabs(
            world, {world.axes[0].id: (float(position), float(half_thickness))}
        ),
    )


_N_SAMPLES = 200


@pytest.mark.parametrize("scale", [0.1, 0.3, 1.0 / 3.0, 0.5, 7.0])
def test_a_discrete_axis_at_thickness_zero_draws_exactly_the_current_sample(scale):
    """The slider's value is a rounded world position; the sample it selects
    is found by snapping in data units, where the comparison is exact."""
    data, world, transform = _time_systems(scale)
    samples = np.zeros((_N_SAMPLES, 3))
    samples[:, 0] = np.arange(_N_SAMPLES)
    for index in range(_N_SAMPLES):
        # What a slider with two decimals would hold.
        position = round(index * scale, 2)
        region = geometry_data_region(
            _selection(world, position, 0.0), transform, world, data
        )
        expected = int(np.floor(position / scale + 0.5))
        assert np.flatnonzero(region.contains(samples)).tolist() == [expected]


def test_without_the_anchor_a_rounded_position_misses_its_sample():
    """What the rule is for: the same plane, pulled back with pure
    containment, selects nothing whenever the position was rounded."""
    data, world, transform = _time_systems(1.0 / 3.0, sampling="continuous")
    samples = np.zeros((_N_SAMPLES, 3))
    samples[:, 0] = np.arange(_N_SAMPLES)
    missed = 0
    for index in range(_N_SAMPLES):
        position = round(index / 3.0, 2)
        region = geometry_data_region(
            _selection(world, position, 0.0), transform, world, data
        )
        missed += not region.contains(samples).any()
    assert missed > _N_SAMPLES // 2


def test_a_discrete_slab_keeps_the_samples_on_its_edges():
    """A 0.3 window on a 0.1-spaced axis is +/-3 samples: 7, not 5."""
    data, world, transform = _time_systems(0.1)
    samples = np.zeros((_N_SAMPLES, 3))
    samples[:, 0] = np.arange(_N_SAMPLES)
    region = geometry_data_region(_selection(world, 5.0, 0.3), transform, world, data)
    assert np.flatnonzero(region.contains(samples)).tolist() == list(range(47, 54))


def test_a_discrete_position_between_samples_selects_the_nearer_one():
    """Round half up, the rule an image uses, so the two change frame at the
    same instant."""
    data, world, transform = _time_systems(1.0)
    samples = np.zeros((10, 3))
    samples[:, 0] = np.arange(10)

    def selected(position):
        region = geometry_data_region(
            _selection(world, position, 0.0), transform, world, data
        )
        return np.flatnonzero(region.contains(samples)).tolist()

    assert selected(4.4) == [4]
    assert selected(4.5) == [5]
    assert selected(4.6) == [5]


def test_a_continuous_axis_is_not_snapped():
    data, world, transform = _time_systems(1.0, sampling="continuous")
    samples = np.zeros((10, 3))
    samples[:, 0] = np.arange(10)
    region = geometry_data_region(_selection(world, 4.4, 0.0), transform, world, data)
    assert not region.contains(samples).any()


# ---------------------------------------------------------------------------
# The graph's window (T3)
# ---------------------------------------------------------------------------


def _graph_extents(controller, scene, visual) -> dict[int, tuple[float, float]]:
    return dict(_request_2d(controller, scene, visual).extents)


def _graph(controller, scene, trail=None):
    store = GraphMemoryStore.from_arrays(
        _points_positions(), np.zeros((0, 2), dtype=np.int32)
    )
    return controller.add_graph(
        data=store,
        scene_id=scene.id,
        appearance=GraphAppearance(),
        **({} if trail is None else {"trail": trail}),
    )


async def test_a_graph_with_no_trail_takes_the_scene_slab():
    controller, scene = _viewer()
    visual = _graph(controller, scene)
    controller.update_slice_indices(scene.id, {0: 2.0})
    assert _graph_extents(controller, scene, visual) == {0: (0.0, 0.0)}
    controller.update_thickness(scene.id, {0: 1.5})
    assert _graph_extents(controller, scene, visual) == {0: (1.5, 1.5)}


async def test_a_trail_widens_the_scene_slab_and_never_narrows_it():
    controller, scene = _viewer()
    visual = _graph(controller, scene, {0: TrailConfig(before=2.0, after=0.5)})
    controller.update_slice_indices(scene.id, {0: 3.0})
    # No thickness: the trail alone.
    assert _graph_extents(controller, scene, visual) == {0: (2.0, 0.5)}
    # A slab thinner than the trail's reach on one side, thicker on the other.
    controller.update_thickness(scene.id, {0: 1.0})
    assert _graph_extents(controller, scene, visual) == {0: (2.0, 1.0)}
    # A slab thicker than the trail on both sides.
    controller.update_thickness(scene.id, {0: 4.0})
    assert _graph_extents(controller, scene, visual) == {0: (4.0, 4.0)}
