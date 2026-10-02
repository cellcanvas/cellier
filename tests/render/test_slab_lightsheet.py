"""A slab on a light-sheet style scene (mesh refactor v3, T3, T4 and T7).

World ``(t, c, z, y, x)`` with ``t`` in seconds through one shared
non-uniform transform; multiscale labels ``tzyx`` broadcast over ``c``; a
graph on a discrete ``t`` index, also broadcast over ``c``.

Before labels took the one-plane rule, a z half-thickness made the multiscale
labels plan a range of planes, and every 2D tile write failed: a tile holds
one plane.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.data.graph import GraphMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.transform import (
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    CoordinateSystem,
    DataCoordinateSystem,
    NonUniformAxisTransform,
)
from cellier.visuals._graph_memory import GraphAppearance, TrailConfig
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

# Unevenly spaced frames, in seconds.
_TIMES = (0.0, 1.0, 2.5, 3.0, 6.0)
_Z_SCALE = 2.0  # world micrometers per label voxel
_Z_HALF = 1.0  # the scene's z half-thickness, in world micrometers


def _space(name: str) -> Axis:
    return Axis(name=name, axis_type="space", unit="micrometer")


def _world_axes() -> tuple[Axis, ...]:
    return (
        Axis(name="t", axis_type="time", unit="second"),
        Axis(name="c", axis_type="channel"),
        _space("z"),
        _space("y"),
        _space("x"),
    )


def _time_block(store_cs, world) -> NonUniformAxisTransform:
    return NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=_TIMES),
        input_coordinate_system=CoordinateSystem(name="t", axes=(store_cs.axes[0],)).id,
        output_coordinate_system=CoordinateSystem(
            name="t", axes=(world.axis_by_name("t"),)
        ).id,
    )


def _to_world(store_cs, world, time_name, z_scale):
    return ByDimensionTransform.from_axis_map(
        store_cs,
        world,
        axis_map={time_name: "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": z_scale, "y": 1.0, "x": 1.0},
        broadcast_output_axes=["c"],
        axis_transforms={time_name: _time_block(store_cs, world)},
        name="to_world",
    )


def _labels_store(tmp_path) -> MultiscaleZarrDataStore:
    def _fill(arr: np.ndarray) -> None:
        _t, _d, h, w = arr.shape
        arr[:, :, h // 4 : h // 2, w // 4 : w // 2] = 3
        arr[:, :, h // 2 : 3 * h // 4, w // 2 : 3 * w // 4] = 7

    n_t = len(_TIMES)
    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (n_t, 16, 16, 16)), ("s1", (n_t, 8, 8, 8))],
        fill=_fill,
        dtype="int32",
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        data_coordinate_system=DataCoordinateSystem(
            name="labels",
            axes=(
                Axis(name="t", axis_type="time", unit="frame"),
                _space("z"),
                _space("y"),
                _space("x"),
            ),
            datastore_id=uuid4(),
        ),
        name="labels",
    )


def _graph_store() -> GraphMemoryStore:
    # One node per frame, at world z = 16 (the scene's slice) +/- a little.
    positions = np.array(
        [[t, 16.0 + 0.4 * t, 8.0, 8.0] for t in range(len(_TIMES))],
        dtype=np.float32,
    )
    edges = np.array([[i, i + 1] for i in range(len(_TIMES) - 1)], dtype=np.int32)
    return GraphMemoryStore.from_arrays(
        positions,
        edges,
        data_coordinate_system=DataCoordinateSystem(
            name="tracks",
            axes=(
                Axis(name="t_idx", axis_type="time", sampling="discrete"),
                _space("z"),
                _space("y"),
                _space("x"),
            ),
            datastore_id=uuid4(),
        ),
    )


def _scene(controller, tmp_path):
    scene = controller.add_scene(
        dim="2d", name="scene", coordinate_system=_world_axes()
    )
    world = scene.dims.world_coordinate_system
    labels_store = _labels_store(tmp_path)
    labels = controller.add_labels_multiscale(
        data=labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        transform=_to_world(
            labels_store.data_coordinate_systems[0], world, "t", _Z_SCALE
        ),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    graph_store = _graph_store()
    graph = controller.add_graph(
        data=graph_store,
        scene_id=scene.id,
        appearance=GraphAppearance(),
        transform=_to_world(
            graph_store.data_coordinate_systems[0], world, "t_idx", 1.0
        ),
        trail={0: TrailConfig(before=10.0, after=0.0)},
    )
    controller.add_canvas(scene_id=scene.id)
    return scene, labels, graph


def _gfx(controller, scene, visual):
    return controller._render_manager._scenes[scene.id].get_visual(visual.id)


def _graph_request(controller, scene, graph):
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    return _gfx(controller, scene, graph).build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=100.0,
        world_width=10.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
    )[0]


async def test_labels_write_tiles_at_a_z_thickness(
    controller, render_scene, reslice, tmp_path
):
    scene, labels, _graph = _scene(controller, tmp_path)
    # World z = 16 um is label plane 8; t = 2.5 s is frame 2.
    controller.update_slice_indices(scene.id, {0: 2.5, 2: 16.0})
    controller.update_thickness(scene.id, {2: _Z_HALF})

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)

    gfx = _gfx(controller, scene, labels)
    # One plane on z, not the range the slab covers.
    selections = gfx._level0_axis_selections()
    assert selections[0] == 2
    assert selections[1] == 8
    # Tiles were written, and nothing failed on the way.
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0
    progress = controller._render_manager._slice_coordinator.visual_progress(labels.id)
    assert progress is None or progress.failed == 0
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_labels_past_the_data_draw_nothing(controller, reslice, tmp_path):
    scene, labels, _graph = _scene(controller, tmp_path)
    # z = 60 um is past the 32 um the labels cover; the slab does not reach.
    controller.update_slice_indices(scene.id, {0: 2.5, 2: 60.0})
    controller.update_thickness(scene.id, {2: _Z_HALF})
    await reslice(controller, scene.id)
    gfx = _gfx(controller, scene, labels)
    assert gfx._slice_empty is True
    assert gfx._inner_node_2d.visible is False

    controller.update_slice_indices(scene.id, {2: 16.0})
    await reslice(controller, scene.id)
    assert gfx._slice_empty is False
    assert gfx._inner_node_2d.visible is True


async def test_the_graphs_z_window_is_the_scene_thickness(controller, tmp_path):
    scene, _labels, graph = _scene(controller, tmp_path)
    controller.update_slice_indices(scene.id, {0: 2.5, 2: 16.0})

    # No thickness: a plane on z.  The trail is on t and is untouched.
    request = _graph_request(controller, scene, graph)
    assert request.extents[1] == (0.0, 0.0)

    controller.update_thickness(scene.id, {2: _Z_HALF})
    request = _graph_request(controller, scene, graph)
    # The graph's z is already in world micrometers.
    assert request.extents[1] == (_Z_HALF, _Z_HALF)
    # t is anchored at the frame the slider selects (2.5 s is frame 2), and
    # its window is the trail: 10 s back reaches frame 0, nothing forward.
    assert request.slice_positions[0] == 2.0
    before, after = request.extents[0]
    assert before >= 2.0
    assert after < 1.0


async def test_time_stays_in_step_across_families(controller, reslice, tmp_path):
    """Labels pick the frame by the image rule, the graph by snapping; both
    round half up on the same transform."""
    scene, labels, graph = _scene(controller, tmp_path)
    for seconds, frame in [(0.4, 0), (0.5, 1), (2.7, 2), (2.8, 3), (6.0, 4)]:
        controller.update_slice_indices(scene.id, {0: seconds, 2: 16.0})
        await reslice(controller, scene.id)
        assert _gfx(controller, scene, labels)._level0_axis_selections()[0] == frame
        assert _graph_request(controller, scene, graph).slice_positions[0] == frame
