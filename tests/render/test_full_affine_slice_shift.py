"""Full affine design v2, Phase 5: ``SLICE_SHIFT`` views, 2D and 3D.

A view whose sliced data axes reach a displayed world axis -- a drift
``x += v t`` scrubbed in ``t``, a shear ``x += s z`` sliced in ``z`` -- moves
the drawn slab across the view as the slice moves.  Image, labels and mesh
nodes follow the slice; points, lines and graph place each element at its
own collapsed position by the model-frame fold (D11).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience._geometry import axis_values_from_viewer
from cellier.convenience.capture import _ensure_canvases, _load_data
from cellier.data import ImageMemoryStore
from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.gui._axis_values import DiscreteAxisValues
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import LinesMemoryAppearance, PointsMarkerAppearance
from tests.render._affine import (
    FAMILIES,
    GEOMETRY,
    LANDMARK_PX,
    LINEARS,
    N_FRAMES,
    P0,
    TIME_FAMILIES,
    add_drift_landmark,
    add_landmark,
    bind,
    canvas_view,
    centroid,
    drawn,
    drift_matrix,
    expected_centroid,
    frame_matrix,
    landmark_error,
    load,
    new_tzyx_viewer,
    new_viewer,
    nudge_2d_camera,
    project,
    render,
    to_world,
    world_matrix,
)

# ---------------------------------------------------------------------------
# shear_x_by_z in a (y, x) view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_shear_x_by_z_in_a_yx_view_every_family(tmp_path, family):
    assert landmark_error(family, "shear_x_by_z", "2d", tmp_path) < LANDMARK_PX


@pytest.mark.parametrize("family", ["points", "lines", "graph"])
def test_elements_in_a_thick_slab_land_at_their_own_depths(family):
    """Elements at three depths, one ``(y, x)``: three ``x`` offsets.

    Before the fold every element in the slab was drawn at the slab's one
    collapsed coordinate, all on top of each other.
    """
    matrix = world_matrix(LINEARS["shear_x_by_z"])
    depths = (P0[0] - 1.0, P0[0], P0[0] + 1.0)
    centres = np.array([[z, P0[1], P0[2]] for z in depths], dtype=np.float32)
    # Lines and graph edges: a short segment along y through each centre.
    ends = np.repeat(centres, 2, axis=0)
    ends[0::2, 1] -= 1.5
    ends[1::2, 1] += 1.5
    viewer = new_viewer("2d")
    try:
        if family == "points":
            store = PointsMemoryStore(positions=centres, name="p")
            viewer.add_points(
                store,
                appearance=PointsMarkerAppearance(size=5.0, color=(1, 1, 1, 1)),
                transform=bind(viewer, store, matrix),
            )
        elif family == "lines":
            store = LinesMemoryStore(positions=ends, name="l")
            viewer.add_lines(
                store,
                appearance=LinesMemoryAppearance(color=(1, 1, 1, 1), thickness=3.0),
                transform=bind(viewer, store, matrix),
            )
        else:
            edges = np.arange(len(ends), dtype=np.int64).reshape(-1, 2)
            store = GraphMemoryStore.from_arrays(ends, edges, name="g")
            viewer.add_graph(store, transform=bind(viewer, store, matrix))
        load(viewer, slice_z=to_world(matrix, P0)[0], half_thickness=1.0)
        nudge_2d_camera(canvas_view(viewer).camera)
        frame, _, camera = render(viewer)
    finally:
        viewer.controller.close()
    mask = drawn(frame)
    for point in centres:
        row, col = np.round(project(camera, to_world(matrix, point))).astype(int)
        assert mask[row, col], f"nothing drawn where {point} lands"
    expected_cols = sorted(project(camera, to_world(matrix, p))[1] for p in centres)
    assert np.min(np.diff(expected_cols)) > 10  # three separate places
    assert _components(mask) == 3


def _components(mask: np.ndarray) -> int:
    from scipy.ndimage import label

    return int(label(mask)[1])


# ---------------------------------------------------------------------------
# Drift x += v t: scrub t, the landmark tracks v t
# ---------------------------------------------------------------------------


def _scrub(viewer, t: float, z: float | None, thickness: float | None = None) -> None:
    async def _run():
        if thickness is not None:
            viewer.controller.update_thickness(viewer.scene.id, {1: thickness})
        positions = {0: float(t)}
        if z is not None:
            positions[1] = float(z)
        viewer.set_slice_positions(positions)
        await _load_data(viewer)

    asyncio.run(_run())


@pytest.mark.parametrize("dim", ["2d", "3d"])
@pytest.mark.parametrize("family", TIME_FAMILIES)
def test_a_drift_tracks_v_t_while_scrubbing(tmp_path, dim, family):
    viewer = new_tzyx_viewer(dim)
    try:
        visual = add_drift_landmark(viewer, family, drift_matrix(), tmp_path)
        _ensure_canvases(viewer, (960, 960))
        status = viewer.controller.draw_status(visual.id)
        assert status.alignment == "slice_shift" and status.drawable
        thickness = 0.6 if dim == "2d" and family in GEOMETRY else None
        errors = []
        for t in range(N_FRAMES):
            _scrub(viewer, t, P0[0] if dim == "2d" else None, thickness)
            camera = canvas_view(viewer).camera
            if dim == "2d":
                nudge_2d_camera(camera)
            frame, _, camera = render(viewer)
            got = centroid(frame)
            assert got is not None, f"nothing drawn at t={t}"
            expected = expected_centroid(family, camera, frame_matrix(t), dim=dim)
            errors.append(float(np.linalg.norm(got - expected)))
    finally:
        viewer.controller.close()
    assert max(errors) < LANDMARK_PX, errors


def test_the_time_slider_stays_discrete_under_drift():
    """K15: the ``t`` row reads data ``t`` alone, so its samples survive.

    Before the port the sampler asked for a bijection, got a ``ValueError``
    from the drift, and marked every non-space axis continuous.
    """
    axes = tuple(
        Axis(name=name, axis_type=kind, sampling="discrete")
        for name, kind in (
            ("t", "time"),
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
        )
    )
    store = ImageMemoryStore(
        data=np.zeros((N_FRAMES, 4, 6, 8), dtype=np.float32),
        data_coordinate_systems=[
            DataCoordinateSystem(name="series", axes=axes, datastore_id=uuid4())
        ],
    )
    viewer = new_tzyx_viewer("2d")
    try:
        visual = viewer.add_image(store, transform=bind(viewer, store, drift_matrix()))
        assert viewer.controller.draw_status(visual.id).alignment == "slice_shift"
        values = axis_values_from_viewer(viewer)
        assert values[0] == DiscreteAxisValues(values=(0.0, 1.0, 2.0))
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The refresh runs only where the slice moves the data (spy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, follows",
    [("scale", False), ("rot_yx", False), ("shear_x_by_z", True)],
)
@pytest.mark.parametrize("family", ["image", "labels", "points", "mesh"])
def test_a_slice_move_refreshes_the_node_only_when_it_must(
    tmp_path, family, name, follows
):
    viewer = new_viewer("2d")
    try:
        visual = add_landmark(viewer, family, world_matrix(LINEARS[name]), tmp_path)
        load(viewer, slice_z=10.0)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        calls = []
        original = gfx_visual._update_node_matrix

        def spy(displayed_axes):
            calls.append(displayed_axes)
            return original(displayed_axes)

        gfx_visual._update_node_matrix = spy
        _scrub_z(viewer, 12.0)
    finally:
        viewer.controller.close()
    assert bool(calls) is follows


def _scrub_z(viewer, z: float) -> None:
    async def _run():
        viewer.set_slice_positions({0: float(z)})
        await _load_data(viewer)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# D11: "model" sizes are the same on the fold path
# ---------------------------------------------------------------------------


def test_model_sizes_are_equal_across_the_two_paths(tmp_path):
    """The fold keeps the node matrix, so a pygfx ``"model"`` size is unchanged.

    ``L = diag(1, 2, 1)`` draws a point through today's path (axis aligned);
    the same with ``x += 0.5 z`` draws it through the fold.  The displayed
    block is the same, so the spot must be the same size.
    """
    areas = []
    for shear in (0.0, 0.5):
        linear = np.diag([1.0, 2.0, 1.0])
        linear[2, 0] = shear
        matrix = world_matrix(linear)
        viewer = new_viewer("2d")
        try:
            store = PointsMemoryStore(positions=P0[None].astype(np.float32), name="p")
            visual = viewer.add_points(
                store,
                appearance=PointsMarkerAppearance(size=3.0, color=(1, 1, 1, 1)),
                transform=bind(viewer, store, matrix),
            )
            # pygfx's "model" space, the one that reads the node matrix (the
            # public "world" does not).
            gfx_visual = viewer.controller._render_manager._scenes[
                viewer.scene.id
            ].get_visual(visual.id)
            gfx_visual._material.size_space = "model"
            expected = "slice_shift" if shear else "axis_aligned"
            assert viewer.controller.draw_status(visual.id).alignment == expected
            load(viewer, slice_z=to_world(matrix, P0)[0], half_thickness=0.6)
            frame, _, _ = render(viewer)
            areas.append(int(drawn(frame).sum()))
        finally:
            viewer.controller.close()
    assert areas[0] > 50
    assert areas[0] == areas[1]
