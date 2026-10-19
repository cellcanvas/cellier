"""Full affine design v2, Phase 6: ``OBLIQUE`` views for the geometry families.

A 2D view whose slice is tilted through the data: points, lines, graph and
mesh sections are selected exactly by the tilted slab (K12) and placed in
the world frame (D4's fallback, since no node matrix maps the data to the
screen there).  Raster families still draw nothing there (D1).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from cellier.convenience.capture import _ensure_canvases, _load_data
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.visuals import PointsMarkerAppearance
from tests.render._affine import (
    GEOMETRY,
    LANDMARK_PX,
    LINEARS,
    P0,
    SHAPE,
    TIME_FAMILIES,
    add_drift_landmark,
    add_landmark,
    bind,
    canvas_view,
    centroid,
    cube_corners,
    expected_centroid,
    landmark_error,
    load,
    new_tzyx_viewer,
    new_viewer,
    nudge_2d_camera,
    render,
    to_world,
    world_matrix,
)

# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", GEOMETRY)
@pytest.mark.parametrize("name", ["rot_zx", "general", "shear_z_by_x"])
def test_oblique_landmark_every_geometry_family(tmp_path, name, family):
    viewer_status = _status(tmp_path, family, name)
    assert viewer_status == ("oblique", True)
    assert landmark_error(family, name, "2d", tmp_path) < LANDMARK_PX


def _status(tmp_path, family, name):
    viewer = new_viewer("2d")
    try:
        visual = add_landmark(viewer, family, world_matrix(LINEARS[name]), tmp_path)
        status = viewer.controller.draw_status(visual.id)
        return status.alignment, status.drawable
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", ["image", "image_multiscale", "labels"])
def test_raster_oblique_still_draws_nothing(tmp_path, family):
    """D1: raster ``OBLIQUE`` is out of scope; the status says why."""
    assert _status(tmp_path, family, "rot_zx") == ("oblique", False)


# ---------------------------------------------------------------------------
# A cube's section by a tilted plane, against the analytic intersection
# ---------------------------------------------------------------------------


def _analytic_section(matrix: np.ndarray, level: float) -> np.ndarray:
    """The cube's cross-section by world ``z = level``: pygfx ``(x, y)`` points."""
    corners = np.array([to_world(matrix, c) for c in cube_corners().astype(float)])
    points = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") != 1:
                continue
            za, zb = corners[a, 0], corners[b, 0]
            if (za - level) * (zb - level) < 0:
                t = (level - za) / (zb - za)
                hit = corners[a] + t * (corners[b] - corners[a])
                points.append((hit[2], hit[1]))
    return np.array(points)


@pytest.mark.parametrize("family", ["mesh", "mesh_multiscale"])
def test_a_cube_section_under_the_general_affine_is_the_analytic_polygon(
    tmp_path, family
):
    matrix = world_matrix(LINEARS["general"])
    level = to_world(matrix, P0)[0] + 0.4  # off centre: not a symmetric cut
    viewer = new_viewer("2d")
    try:
        visual = add_landmark(viewer, family, matrix, tmp_path)
        load(viewer, slice_z=level)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        fill = next(
            level_.fill for level_ in gfx_visual._levels.values() if not level_.is_empty
        )
        local = np.asarray(fill.geometry.positions.data, dtype=np.float64)
        world = np.asarray(fill.world.matrix, dtype=np.float64)
        drawn = (np.c_[local, np.ones(len(local))] @ world.T)[:, :2]
    finally:
        viewer.controller.close()
    expected = _analytic_section(matrix, level)
    assert len(expected) >= 3
    hull = ConvexHull(drawn)
    want = ConvexHull(expected)
    assert hull.volume == pytest.approx(want.volume, rel=1e-4)
    for vertex in expected:
        assert np.min(np.linalg.norm(drawn - vertex, axis=1)) < 1e-3


# ---------------------------------------------------------------------------
# Selection and placement under a tilted slab with thickness
# ---------------------------------------------------------------------------


def test_points_in_a_tilted_slab_are_selected_and_placed_exactly():
    rng = np.random.default_rng(3)
    positions = (rng.random((400, 3)) * (np.asarray(SHAPE) - 1)).astype(np.float32)
    matrix = world_matrix(LINEARS["general"])
    level = float(to_world(matrix, P0)[0])
    half = 1.5
    viewer = new_viewer("2d")
    try:
        store = PointsMemoryStore(positions=positions, name="p")
        visual = viewer.add_points(
            store,
            appearance=PointsMarkerAppearance(size=3.0, color=(1, 1, 1, 1)),
            transform=bind(viewer, store, matrix),
        )
        load(viewer, slice_z=level, half_thickness=half)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        selected = np.asarray(gfx_visual._original_indices)
        local = np.asarray(gfx_visual.node.geometry.positions.data, dtype=np.float64)
        node = np.asarray(gfx_visual.node.world.matrix, dtype=np.float64)
    finally:
        viewer.controller.close()
    world = positions.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    inside = np.flatnonzero(np.abs(world[:, 0] - level) <= half)
    assert 10 < len(inside) < 400
    assert np.array_equal(np.sort(selected), inside)
    placed = (np.c_[local, np.ones(len(local))] @ node.T)[:, :2]
    expected = world[selected][:, [2, 1]]  # pygfx (x, y) = world (x, y)
    np.testing.assert_allclose(placed, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Two sliced axes coupled to each other: an aligned view, every family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", TIME_FAMILIES)
def test_a_time_row_reading_depth_is_an_aligned_view_that_draws(tmp_path, family):
    """World ``T = t + 0.1 z``: the ``(y, x)`` view is axis aligned.

    Both sliced rows pin the data plane, so the view draws what the
    identity would; the axis-aligned code paths, which read a bijection,
    must not be the ones taking it (there is none).  Geometry selects by
    plain containment on the tilted ``T`` constraint, so it gets a slab.
    """
    matrix = np.eye(5)
    matrix[0, 1] = 0.1
    t = 1.0
    viewer = new_tzyx_viewer("2d")
    try:
        visual = add_drift_landmark(viewer, family, matrix, tmp_path)
        status = viewer.controller.draw_status(visual.id)
        assert status.alignment == "axis_aligned" and status.drawable
        _ensure_canvases(viewer, (960, 960))

        async def _run():
            if family in GEOMETRY:
                viewer.controller.update_thickness(viewer.scene.id, {0: 0.3, 1: 0.6})
            viewer.set_slice_positions({0: t + 0.1 * P0[0], 1: P0[0]})
            await _load_data(viewer)

        asyncio.run(_run())
        nudge_2d_camera(canvas_view(viewer).camera)
        frame, _, camera = render(viewer)
        got = centroid(frame)
        assert got is not None
        expected = expected_centroid(family, camera, np.eye(4), dim="2d")
        assert np.linalg.norm(got - expected) < LANDMARK_PX
    finally:
        viewer.controller.close()
