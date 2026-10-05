"""Reducing data-space clipping planes to a view (clipping planes design 4.1).

The defining property: for a data point drawn in the view, the reduced
plane's value at the point's rendered position equals the data plane's
value at the point.  The rendered position is taken from the real node
matrix, so the reduction is checked against where things are drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data import PointsMemoryStore
from cellier.render._clipping import (
    KEEP_EVERYTHING,
    data_half_space_rows,
    expand_rendered_plane,
    reduce_clipping_planes,
)
from cellier.render._spaces import node_matrix
from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane

RNG = np.random.default_rng(0)


def _points_visual(controller, world_names, data_names, displayed, scale, shift):
    from uuid import uuid4

    from cellier.transform import WorldCoordinateSystem

    world = WorldCoordinateSystem(
        axes=tuple(
            Axis(name=n, axis_type="time" if n == "t" else "space") for n in world_names
        )
    )
    scene = controller.add_scene(
        coordinate_system=world, dim="3d" if len(displayed) == 3 else "2d", name="s"
    )
    controller.set_displayed_axes(
        scene.id, tuple(world_names.index(n) for n in displayed)
    )
    system = DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(
            Axis(name=n, axis_type="time" if n == "t" else "space") for n in data_names
        ),
    )
    store = PointsMemoryStore(
        positions=np.zeros((4, len(data_names)), dtype=np.float32),
        data_coordinate_systems=[system],
    )
    transform = AffineTransform.from_axis_map(
        system,
        world,
        axis_map={n: n for n in data_names},
        scale={n: scale[n] for n in data_names},
        translation={n: shift[n] for n in data_names},
        # A store with fewer axes than the world is drawn at every position
        # of the axes it lacks.
        broadcast_output_axes=[n for n in world_names if n not in data_names],
    )
    visual = controller.add_points(data=store, scene_id=scene.id, transform=transform)
    controller.add_canvas(scene_id=scene.id)
    return visual, system, controller.render_spaces(visual.id)


def _residual(spaces, transform, constants, plane, abcd) -> float:
    """Max ``|(abc . r - d') - (n . p - d)|`` over points in the view."""
    ndim = plane.plane.ndim
    points = RNG.uniform(-40, 120, size=(500, ndim))
    for axis, value in constants.items():
        points[:, axis] = value
    truth = points @ plane.plane.normal - plane.plane.offset
    retained = list(spaces.retained_axes)
    matrix = node_matrix(spaces, transform, constants).astype(np.float64)
    # A points visual uploads its retained data axes, reversed to (x, y, z).
    local = np.zeros((len(points), 4))
    local[:, : len(retained)] = points[:, retained][:, ::-1]
    local[:, 3] = 1.0
    rendered = (local @ matrix.T)[:, :3]
    predicted = rendered @ np.asarray(abcd[:3]) - abcd[3]
    return float(np.abs(predicted - truth).max() / max(1.0, np.abs(truth).max()))


SCALE = {"t": 0.5, "z": 4.0, "y": 0.65, "x": 1.3}
SHIFT = {"t": 3.0, "z": -30.0, "y": 12.0, "x": 100.0}


@pytest.mark.parametrize(
    ("world", "data", "displayed", "constants"),
    [
        ("zyx", "zyx", "zyx", {}),
        ("zyx", "zyx", "yx", {"z": 17.0}),
        ("zyx", "zyx", "zx", {"y": 5.0}),
        ("tzyx", "tzyx", "zyx", {"t": 9.0}),
        ("tzyx", "tzyx", "yx", {"t": 9.0, "z": 31.0}),
        ("tzyx", "zyx", "zyx", {}),
    ],
)
def test_the_reduced_plane_agrees_with_the_node_matrix(
    controller, world, data, displayed, constants
):
    visual, system, spaces = _points_visual(
        controller, world, data, displayed, SCALE, SHIFT
    )
    plane = ClippingPlane.from_point_normal(
        system, (40, 30, 20), (1.0, 0.4, -0.7), axes=("z", "y", "x")
    )
    pinned = {system.resolve(name): value for name, value in constants.items()}
    (abcd,) = reduce_clipping_planes(spaces, visual.transform, pinned, [plane])
    # 32-bit node matrix against a 64-bit reduction.
    assert _residual(spaces, visual.transform, pinned, plane, abcd) < 1e-5
    if len(displayed) == 2:
        assert abcd[2] == 0.0  # a line in the 2D scene


def test_a_zyx_plane_is_the_same_at_every_timepoint(controller):
    visual, system, spaces = _points_visual(
        controller, "tzyx", "tzyx", "zyx", SCALE, SHIFT
    )
    plane = ClippingPlane.from_point_normal(
        system, (40, 30, 20), (1.0, 0.4, -0.7), axes=("z", "y", "x")
    )
    reduced = [
        reduce_clipping_planes(spaces, visual.transform, {0: t}, [plane])[0]
        for t in (0.0, 7.0, 31.0)
    ]
    np.testing.assert_allclose(reduced, [reduced[0]] * 3, rtol=0, atol=1e-12)


def test_a_time_component_moves_the_plane_with_the_slider(controller):
    visual, system, spaces = _points_visual(
        controller, "tzyx", "tzyx", "zyx", SCALE, SHIFT
    )
    # Keeps x >= 10 + 2 t.
    plane = ClippingPlane.from_point_normal(
        system, (0, 0, 0, 10), (-2, 0, 0, 1), axes=("t", "z", "y", "x")
    )
    at = {
        t: reduce_clipping_planes(spaces, visual.transform, {0: t}, [plane])[0]
        for t in (0.0, 5.0)
    }
    # World x = 1.3 * data x + 100; the plane is x_world >= 1.3 * (10 + 2 t) + 100
    # for a normal of (1 / 1.3, 0, 0).
    for t, (a, b, c, d) in at.items():
        assert (b, c) == (0.0, 0.0)
        assert d / a == pytest.approx(1.3 * (10 + 2 * t) + 100)


def test_a_disabled_plane_keeps_everything(controller):
    visual, system, spaces = _points_visual(
        controller, "zyx", "zyx", "zyx", SCALE, SHIFT
    )
    planes = [
        ClippingPlane.from_point_normal(system, (40, 0, 0), (1, 0, 0), enabled=False),
        ClippingPlane.from_point_normal(system, (40, 0, 0), (1, 0, 0)),
    ]
    reduced = reduce_clipping_planes(spaces, visual.transform, {}, planes)
    assert reduced[0] == KEEP_EVERYTHING
    assert len(reduced) == 2
    assert reduce_clipping_planes(spaces, visual.transform, {}, []) == []


def test_a_plane_parallel_to_the_slice_keeps_all_or_nothing(controller):
    visual, system, spaces = _points_visual(
        controller, "zyx", "zyx", "yx", SCALE, SHIFT
    )
    plane = ClippingPlane.from_point_normal(system, (40, 0, 0), (1, 0, 0))  # z >= 40
    for z, kept in ((50.0, True), (30.0, False)):
        a, b, c, d = reduce_clipping_planes(spaces, visual.transform, {0: z}, [plane])[
            0
        ]
        assert (a, b, c) == (0.0, 0.0, 0.0)
        assert (0.0 >= d) is kept


def test_culling_rows_are_in_data_space_in_pygfx_order(controller):
    _visual, system, _spaces = _points_visual(
        controller, "zyx", "zyx", "zyx", SCALE, SHIFT
    )
    planes = [
        ClippingPlane.from_point_normal(system, (1, 2, 3), (4, 5, 6)),
        ClippingPlane.from_point_normal(system, (0, 0, 0), (1, 0, 0), enabled=False),
    ]
    rows = data_half_space_rows(planes, (0, 1, 2), {})
    np.testing.assert_allclose(rows, [[6, 5, 4, -(4 + 10 + 18)]])
    # 2D: z pinned at 7, so it moves into the constant.
    rows = data_half_space_rows(planes, (1, 2), {0: 7.0})
    np.testing.assert_allclose(rows, [[6, 5, 0, -(32 - 4 * 7)]])


# ---------------------------------------------------------------------------
# The inverse: a rendered pose back to a data plane (gizmo design v2, 5.3)
# ---------------------------------------------------------------------------


def _constants(system, constants):
    return {system.resolve(name): value for name, value in constants.items()}


@pytest.mark.parametrize(
    ("world", "data", "constants"),
    [
        ("zyx", "zyx", {}),
        ("tzyx", "tzyx", {"t": 9.0}),
        ("tzyx", "zyx", {}),
    ],
)
def test_a_reduced_plane_expands_back_to_the_data_plane(
    controller, world, data, constants
):
    """Anisotropic and shifted: reduce, read a pose off it, expand."""
    visual, system, spaces = _points_visual(
        controller, world, data, "zyx", SCALE, SHIFT
    )
    constants = _constants(system, constants)
    plane = ClippingPlane.from_point_normal(
        system, (40, 30, 20), (1.0, 0.4, -0.7), axes=("z", "y", "x")
    )
    a, b, c, d = reduce_clipping_planes(spaces, visual.transform, constants, [plane])[0]
    abc = np.array([a, b, c])
    # A gizmo holds a unit normal and a point on the plane.
    unit = abc / np.linalg.norm(abc)
    point = unit * (d / np.linalg.norm(abc)) + np.cross(unit, [0.3, -1.0, 2.0])
    normal, offset = expand_rendered_plane(
        spaces, visual.transform, constants, point, unit
    )
    # The same plane up to the scale the unit normal took out.
    scale = np.linalg.norm(abc)
    np.testing.assert_allclose(normal * scale, plane.plane.normal, atol=1e-9)
    assert offset * scale == pytest.approx(plane.plane.offset, abs=1e-9)
    for axis in constants:
        assert normal[axis] == 0.0


def test_an_expanded_plane_reduces_to_the_pose_it_came_from(controller):
    visual, system, spaces = _points_visual(
        controller, "tzyx", "tzyx", "zyx", SCALE, SHIFT
    )
    constants = _constants(system, {"t": 9.0})
    point = np.array([12.0, -7.5, 30.25])
    unit = np.array([0.36, -0.48, 0.8])
    normal, offset = expand_rendered_plane(
        spaces, visual.transform, constants, point, unit
    )
    from cellier.transform import Plane

    plane = ClippingPlane(
        plane=Plane(coordinate_system=system.id, normal=normal, offset=offset)
    )
    abcd = reduce_clipping_planes(spaces, visual.transform, constants, [plane])[0]
    np.testing.assert_allclose(abcd[:3], unit, atol=1e-12)
    assert abcd[3] == pytest.approx(float(unit @ point), abs=1e-9)
    # 32-bit node matrix against a 64-bit reduction.
    assert _residual(spaces, visual.transform, constants, plane, abcd) < 1e-5


def test_expanding_needs_a_3d_view(controller):
    visual, _system, spaces = _points_visual(
        controller, "zyx", "zyx", "yx", SCALE, SHIFT
    )
    with pytest.raises(ValueError, match="3D view"):
        expand_rendered_plane(spaces, visual.transform, {0: 17.0}, (0, 0, 0), (1, 0, 0))


def test_expanding_refuses_a_zero_normal(controller):
    visual, _system, spaces = _points_visual(
        controller, "zyx", "zyx", "zyx", SCALE, SHIFT
    )
    with pytest.raises(ValueError, match="all zero"):
        expand_rendered_plane(spaces, visual.transform, {}, (0, 0, 0), (0, 0, 0))
