"""The polygon a plane makes in a visual (plane outline design, section 5).

The expected polygons are worked out by hand and written here as numbers.
Coordinates are rendered space, pygfx ``(x, y, z)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._clipping import KEEP_EVERYTHING, reduce_clipping_planes
from cellier.render._plane_outline import (
    box_diagonal,
    box_frame,
    clip_polygon,
    clipping_plane_polygon,
    node_box_frame,
    render_plane_polygon,
)
from cellier.render._render_planes import reduce_render_planes
from cellier.visuals import ClippingPlane, RenderPlane
from tests.render.test_clipping_reduction import _points_visual

#: A box from the origin to (4, 6, 8), as rows and corners.
BOX_HIGH = (4.0, 6.0, 8.0)
BOX_ROWS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -4.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, -6.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, -8.0],
    ]
)
BOX_CORNERS = np.array(
    [[x, y, z] for x in (0.0, 4.0) for y in (0.0, 6.0) for z in (0.0, 8.0)]
)
UNIT_ROWS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, -1.0],
    ]
)
UNIT_CORNERS = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
)
X, Y, Z = np.eye(3)
UNBOUNDED = (False, False, False, False)
BOUNDED = (True, True, True, True)
NO_EXTENT = (0.0, 0.0, 0.0, 0.0)


def assert_polygon(found: np.ndarray, expected, tolerance: float = 1e-9) -> None:
    """The same vertices in the same cyclic order, either way round."""
    expected = np.asarray(expected, dtype=np.float64)
    assert found.shape == expected.shape, found
    for candidate in (expected, expected[::-1]):
        for shift in range(len(candidate)):
            if np.abs(found - np.roll(candidate, shift, axis=0)).max() <= tolerance:
                return
    raise AssertionError(f"polygons differ:\n{found}\nexpected\n{expected}")


def _xy_plane(z: float, extent=NO_EXTENT, bounded=UNBOUNDED, clipping=()):
    """The polygon of the plane at height *z*, through (2, 3, z)."""
    return render_plane_polygon(
        (2.0, 3.0, z), X, Y, extent, bounded, BOX_ROWS, BOX_CORNERS, clipping
    )


# -- a render plane ---------------------------------------------------------------


def test_an_axis_aligned_plane_gives_the_box_s_cross_section():
    assert_polygon(_xy_plane(3.0), [(0, 0, 3), (4, 0, 3), (4, 6, 3), (0, 6, 3)])


def test_a_plane_across_the_cube_s_diagonal_gives_the_regular_hexagon():
    polygon = render_plane_polygon(
        (0.5, 0.5, 0.5),
        np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
        np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0),
        NO_EXTENT,
        UNBOUNDED,
        UNIT_ROWS,
        UNIT_CORNERS,
    )
    assert_polygon(
        polygon,
        [
            (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, 1.0, 0.5),
            (0.0, 0.5, 1.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 0.5),
        ],
    )


def test_extents_inside_the_box_give_the_extent_rectangle():
    polygon = _xy_plane(3.0, (-1.0, 1.0, -2.0, 2.0), BOUNDED)
    assert_polygon(polygon, [(1, 1, 3), (3, 1, 3), (3, 5, 3), (1, 5, 3)])


def test_an_unbounded_side_is_ended_by_the_box():
    polygon = _xy_plane(3.0, (-1.0, 0.0, -2.0, 2.0), (True, False, True, True))
    assert_polygon(polygon, [(1, 1, 3), (4, 1, 3), (4, 5, 3), (1, 5, 3)])


def test_extents_past_the_box_are_ended_by_the_box():
    polygon = _xy_plane(3.0, (-50.0, 1.0, -2.0, 50.0), BOUNDED)
    assert_polygon(polygon, [(0, 1, 3), (3, 1, 3), (3, 6, 3), (0, 6, 3)])


def test_a_clipping_plane_across_a_corner_gives_a_pentagon():
    # Kept where x + y >= 3: the corner at (1, 1) is cut off.
    polygon = _xy_plane(
        3.0, (-1.0, 1.0, -2.0, 2.0), BOUNDED, clipping=[(1.0, 1.0, 0.0, 3.0)]
    )
    assert_polygon(polygon, [(2, 1, 3), (3, 1, 3), (3, 5, 3), (1, 5, 3), (1, 2, 3)])


@pytest.mark.parametrize(
    ("z", "clipping"),
    [
        (20.0, ()),  # the plane misses the box
        (-0.01, ()),
        (3.0, [(1.0, 0.0, 0.0, 10.0)]),  # clipped away: kept where x >= 10
        (3.0, [(0.0, 0.0, 1.0, 5.0)]),  # the plane is on the cut side
    ],
)
def test_a_plane_with_nothing_to_draw_gives_no_polygon(z, clipping):
    polygon = _xy_plane(z, clipping=clipping)
    assert polygon.shape == (0, 3)


@pytest.mark.parametrize("z", [0.0, 8.0, 8.0 * (1.0 + 1e-12), -1e-12])
def test_a_plane_lying_in_a_face_gives_the_face(z):
    """The box is closed, and a rounding error does not open it."""
    polygon = _xy_plane(z)
    face = round(z)
    assert_polygon(
        polygon, [(0, 0, face), (4, 0, face), (4, 6, face), (0, 6, face)], 1e-9
    )


def test_a_row_with_no_normal_keeps_everything_or_nothing():
    """A disabled clipping plane, and one parallel to the slice."""
    whole = [(0, 0, 3), (4, 0, 3), (4, 6, 3), (0, 6, 3)]
    assert_polygon(_xy_plane(3.0, clipping=[KEEP_EVERYTHING]), whole)
    assert_polygon(_xy_plane(3.0, clipping=[(0.0, 0.0, 0.0, 0.0)]), whole)
    assert _xy_plane(3.0, clipping=[(0.0, 0.0, 0.0, 1.0)]).shape == (0, 3)


def test_a_normal_of_any_length_cuts_at_the_same_place():
    """A reduced clipping plane's normal is not a unit vector."""
    expected = [(2, 1, 3), (3, 1, 3), (3, 5, 3), (1, 5, 3), (1, 2, 3)]
    for factor in (1e-3, 1.0, 250.0):
        row = factor * np.array([1.0, 1.0, 0.0, 3.0])
        polygon = _xy_plane(3.0, (-1.0, 1.0, -2.0, 2.0), BOUNDED, clipping=[row])
        assert_polygon(polygon, expected)


def test_any_plane_gives_a_polygon_on_it_and_inside_every_cut():
    """Whatever the pose: the vertices are on the plane, inside the box and
    the clipping planes, and at most four, plus one per cut."""
    rng = np.random.default_rng(7)
    scale = box_diagonal(BOX_CORNERS)
    drawn = 0
    for _ in range(200):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        axis_0 = np.cross(normal, rng.normal(size=3))
        axis_0 /= np.linalg.norm(axis_0)
        axis_1 = np.cross(normal, axis_0)
        origin = rng.uniform((-1, -1, -1), (5, 7, 9))
        clipping = np.column_stack([rng.normal(size=(2, 3)), rng.uniform(-3, 3, 2)])
        polygon = render_plane_polygon(
            origin,
            axis_0,
            axis_1,
            rng.uniform(1.0, 6.0, 4) * (-1, 1, -1, 1),
            rng.random(4) < 0.5,
            BOX_ROWS,
            BOX_CORNERS,
            clipping,
        )
        if len(polygon) == 0:
            continue
        drawn += 1
        assert 3 <= len(polygon) <= 4 + 6 + 2
        assert np.abs((polygon - origin) @ normal).max() < 1e-9 * scale
        for row in np.vstack([BOX_ROWS, clipping]):
            inside = (polygon @ row[:3] - row[3]) / np.linalg.norm(row[:3])
            assert inside.min() > -1e-6 * scale
    assert drawn > 60


# -- a clipping plane ---------------------------------------------------------------


def test_a_clipping_plane_s_polygon_is_its_cut_face():
    # Kept where z >= 3.
    polygon = clipping_plane_polygon(0, [(0.0, 0.0, 1.0, 3.0)], BOX_ROWS, BOX_CORNERS)
    assert_polygon(polygon, [(0, 0, 3), (4, 0, 3), (4, 6, 3), (0, 6, 3)])


def test_two_clipping_planes_each_end_at_the_other():
    rows = [(0.0, 0.0, 1.0, 3.0), (1.0, 0.0, 0.0, 1.0)]  # z >= 3 and x >= 1
    assert_polygon(
        clipping_plane_polygon(0, rows, BOX_ROWS, BOX_CORNERS),
        [(1, 0, 3), (4, 0, 3), (4, 6, 3), (1, 6, 3)],
    )
    assert_polygon(
        clipping_plane_polygon(1, rows, BOX_ROWS, BOX_CORNERS),
        [(1, 0, 3), (1, 6, 3), (1, 6, 8), (1, 0, 8)],
    )


def test_a_tilted_clipping_plane_with_a_long_normal():
    # Kept where 2x + 2z >= 8, the plane x + z = 4.
    polygon = clipping_plane_polygon(0, [(2.0, 0.0, 2.0, 8.0)], BOX_ROWS, BOX_CORNERS)
    assert_polygon(polygon, [(0, 0, 4), (4, 0, 0), (4, 6, 0), (0, 6, 4)])


@pytest.mark.parametrize(
    "rows",
    [
        [(0.0, 0.0, 1.0, 20.0)],  # misses the box
        [KEEP_EVERYTHING],  # disabled
        [(0.0, 0.0, 0.0, 1.0)],  # parallel to the slice
        [(0.0, 0.0, 1.0, 3.0), (0.0, 0.0, 1.0, 5.0)],  # behind another plane
    ],
)
def test_a_clipping_plane_with_no_face_gives_no_polygon(rows):
    assert clipping_plane_polygon(0, rows, BOX_ROWS, BOX_CORNERS).shape == (0, 3)


# -- cutting ------------------------------------------------------------------------


def test_clip_polygon_never_returns_a_point_or_a_line():
    square = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=float)
    # Kept where x >= 1: one edge of the square is left.
    assert clip_polygon(square, [(1.0, 0.0, 0.0, 1.0)], 1.0).shape == (0, 3)
    # Kept where x + y >= 2: one corner is left.
    assert clip_polygon(square, [(1.0, 1.0, 0.0, 2.0)], 1.0).shape == (0, 3)
    assert clip_polygon(square, np.zeros((0, 4)), 1.0).shape == (4, 3)


# -- the box of a placed visual -----------------------------------------------------

IDENTITY = {"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0}
NO_SHIFT = {"t": 0.0, "z": 0.0, "y": 0.0, "x": 0.0}
#: A 4 x 6 x 8 voxel volume's edges, data ``(z, y, x)``.
VOXEL_LOW = (-0.5, -0.5, -0.5)
VOXEL_HIGH = (3.5, 5.5, 7.5)


def test_the_box_follows_the_visual_s_transform(controller):
    """Voxels twice as tall in z, shifted in x: data (z, y, x) to world."""
    scale = dict(IDENTITY, z=2.0)
    shift = dict(NO_SHIFT, x=100.0)
    visual, _system, spaces = _points_visual(
        controller, "zyx", "zyx", "zyx", scale, shift
    )
    rows, corners = box_frame(spaces, visual.transform, {}, VOXEL_LOW, VOXEL_HIGH)
    assert rows.shape == (6, 4)
    np.testing.assert_allclose(corners.min(axis=0), (99.5, -0.5, -1.0))
    np.testing.assert_allclose(corners.max(axis=0), (107.5, 5.5, 7.0))
    # Each corner is on three faces and inside the other three.
    inside = corners @ rows[:, :3].T - rows[:, 3]
    assert inside.min() > -1e-9
    assert (np.abs(inside) < 1e-9).sum(axis=1).tolist() == [3] * 8
    assert box_diagonal(corners) == pytest.approx(np.sqrt(8.0**2 + 6.0**2 + 8.0**2))


def test_a_tilted_plane_through_anisotropic_voxels(controller):
    """Voxels of size (2, 1, 1): the box is world z -1..7, y and x as data.

    The plane x + z = 5 leaves it at x = -0.5 (z = 5.5) and z = -1 (x = 6).
    """
    visual, _system, spaces = _points_visual(
        controller, "zyx", "zyx", "zyx", dict(IDENTITY, z=2.0), NO_SHIFT
    )
    rows, corners = box_frame(spaces, visual.transform, {}, VOXEL_LOW, VOXEL_HIGH)
    plane = RenderPlane.from_point_normal(
        spaces.world, (2.0, 2.5, 3.0), (1.0, 0.0, 1.0), axes=("z", "y", "x")
    )
    reduced = reduce_render_planes(spaces, [plane])
    polygon = render_plane_polygon(
        reduced.origin[0],
        reduced.axis_0[0],
        reduced.axis_1[0],
        reduced.extent[0],
        reduced.bounded[0],
        rows,
        corners,
    )
    assert_polygon(
        polygon,
        [(-0.5, -0.5, 5.5), (6.0, -0.5, -1.0), (6.0, 5.5, -1.0), (-0.5, 5.5, 5.5)],
    )


@pytest.mark.parametrize(("t", "z"), [(1.0, 4.0), (3.0, 2.0)])
def test_a_clipping_plane_across_a_sliced_axis_moves_with_the_slice(controller, t, z):
    """Kept where t + z >= 5, in a zyx view of tzyx data: at time *t* the
    cut face is at z = 5 - t."""
    visual, system, spaces = _points_visual(
        controller, "tzyx", "tzyx", "zyx", IDENTITY, NO_SHIFT
    )
    constants = {0: t}
    rows, corners = box_frame(spaces, visual.transform, constants, (0, 0, 0), (8, 6, 4))
    item = ClippingPlane.from_point_normal(system, (5, 0, 0, 0), (1, 1, 0, 0))
    clipping = reduce_clipping_planes(spaces, visual.transform, constants, [item])
    polygon = clipping_plane_polygon(0, clipping, rows, corners)
    assert_polygon(polygon, [(0, 0, z), (4, 0, z), (4, 6, z), (0, 6, z)])


def test_the_box_is_for_a_3d_view(controller):
    visual, _system, spaces = _points_visual(
        controller, "zyx", "zyx", "yx", IDENTITY, NO_SHIFT
    )
    with pytest.raises(ValueError, match="3D view"):
        box_frame(spaces, visual.transform, {0: 0.0}, (0, 0), (1, 1))


# -- a box given in a node's frame ---------------------------------------------------


def test_a_box_in_a_node_s_frame_is_carried_by_the_node_s_matrix():
    """A node that doubles x and shifts y by 10: the unit cube becomes the
    box from (0, 10, 0) to (2, 11, 1)."""
    matrix = np.diag([2.0, 1.0, 1.0, 1.0])
    matrix[1, 3] = 10.0
    rows, corners = node_box_frame(matrix, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    np.testing.assert_allclose(corners.min(axis=0), (0.0, 10.0, 0.0))
    np.testing.assert_allclose(corners.max(axis=0), (2.0, 11.0, 1.0))
    inside = corners @ rows[:, :3].T - rows[:, 3]
    assert inside.min() > -1e-12
    assert (np.abs(inside) < 1e-12).sum(axis=1).tolist() == [3] * 8
    # Kept where x >= 1: the face at x = 1, half way along the box.
    polygon = clipping_plane_polygon(0, [(1.0, 0.0, 0.0, 1.0)], rows, corners)
    assert_polygon(polygon, [(1, 10, 0), (1, 11, 0), (1, 11, 1), (1, 10, 1)])


def test_a_box_in_a_turned_node_s_frame_is_a_turned_box():
    """A quarter turn about z: the box's x side lies along y."""
    matrix = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0, 0, 0, 1],
        ]
    )
    rows, corners = node_box_frame(matrix, (0.0, 0.0, 0.0), (4.0, 1.0, 1.0))
    np.testing.assert_allclose(corners.min(axis=0), (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(corners.max(axis=0), (0.0, 4.0, 1.0), atol=1e-12)
    inside = corners @ rows[:, :3].T - rows[:, 3]
    assert inside.min() > -1e-12


def test_a_node_scaled_to_nothing_has_no_box():
    assert node_box_frame(np.diag([1.0, 0.0, 1.0, 1.0]), (0, 0, 0), (1, 1, 1)) is None
