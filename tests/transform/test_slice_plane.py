"""The geometry of one 2D slice through raster data (oblique 2D design v3, Phase 1).

``plans/oblique_2d_raster_design_v3.md`` sections 4.1, 4.2 and 5.2.2, and
D15's footprint.  The transforms are v1 6.1's (``id``, ``rot_yx``,
``shear_x_by_z``, ``rot_zx``, ``general``, ``drift_rot``), tried against
every 2D view of a ``zyx`` and a ``tzyx`` world.
"""

from __future__ import annotations

import itertools
from uuid import uuid4

import numpy as np
import pytest

from cellier.transform import (
    AffineTransform,
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    ConvexRegion,
    CoordinateSystem,
    DataCoordinateSystem,
    DepthAlignment,
    HalfSpace,
    NonAffineTransformError,
    NonUniformAxisTransform,
    ProjectionRefusal,
    ProjectionRefusalReason,
    RegionSelection,
    RenderedCoordinateSystem,
    RouteRefusalReason,
    SliceAlignment,
    SliceRoute,
    WorldCoordinateSystem,
    choose_slice_route,
    classify_projection,
    classify_view,
    projection_axis,
    ray_step,
    slice_footprint,
    slice_plane_frame,
)

A = SliceAlignment
KINDS = {"t": "time", "c": "channel"}


def _system(names: str, name: str) -> CoordinateSystem:
    return CoordinateSystem(
        name=name,
        axes=tuple(Axis(name=n, axis_type=KINDS.get(n, "space")) for n in names),
    )


def _world(names: str) -> WorldCoordinateSystem:
    return WorldCoordinateSystem(
        axes=tuple(Axis(name=n, axis_type=KINDS.get(n, "space")) for n in names)
    )


def _affine(linear, translation, data_names, world):
    linear = np.asarray(linear, dtype=np.float64)
    matrix = np.zeros((linear.shape[0] + 1, linear.shape[1] + 1))
    matrix[:-1, :-1] = linear
    if translation is not None:
        matrix[:-1, -1] = translation
    matrix[-1, -1] = 1.0
    return AffineTransform.from_matrix(matrix, _system(data_names, "data"), world)


def _rot(a: int, b: int, degrees: float, n: int = 3) -> np.ndarray:
    theta = np.deg2rad(degrees)
    m = np.eye(n)
    m[a, a] = m[b, b] = np.cos(theta)
    m[a, b] = -np.sin(theta)
    m[b, a] = np.sin(theta)
    return m


def _shear(row: int, column: int, amount: float = 0.5, n: int = 3) -> np.ndarray:
    m = np.eye(n)
    m[row, column] = amount
    return m


#: v1 6.1's spatial transforms, as ``zyx -> zyx`` linear blocks.
SPATIAL = {
    "id": np.eye(3),
    "rot_yx": _rot(1, 2, 30.0),
    "shear_x_by_z": _shear(2, 0),
    "rot_zx": _rot(0, 2, 30.0),
    "general": _rot(0, 1, 20.0)
    @ _rot(1, 2, 25.0)
    @ _rot(0, 2, 15.0)
    @ np.diag([2.0, 1.0, 1.0]),
}
TRANSLATION_3D = np.array([5.0, -2.0, 1.5])


def _drift_rot() -> np.ndarray:
    """``tzyx -> tzyx``: a ``zx`` rotation, and world x drifts with data t."""
    linear = np.eye(4)
    linear[1:, 1:] = _rot(0, 2, 30.0)
    linear[3, 0] = 0.3
    return linear


#: The same, as ``tzyx -> tzyx`` (``id`` and ``drift_rot`` of v1 6.1 for 4D).
TEMPORAL = {
    "id": np.eye(4),
    "rot_zx": np.block(
        [[np.eye(1), np.zeros((1, 3))], [np.zeros((3, 1)), _rot(0, 2, 30.0)]]
    ),
    "drift_rot": _drift_rot(),
}
TRANSLATION_4D = np.array([1.0, 5.0, -2.0, 1.5])

VIEWS_3D = list(itertools.combinations(range(3), 2))
VIEWS_4D = list(itertools.combinations(range(4), 2))


def _spatial(name: str):
    return _affine(SPATIAL[name], TRANSLATION_3D, "zyx", _world("zyx"))


def _temporal(name: str):
    return _affine(TEMPORAL[name], TRANSLATION_4D, "tzyx", _world("tzyx"))


def _selection(world, displayed, position, thickness=None):
    """A D43 selection displaying *displayed*, sliced at *position*."""
    names = [axis.name for axis in world.axes]
    shown = [names[axis] for axis in displayed]
    rendered = RenderedCoordinateSystem.from_world(world, shown, uuid4())
    sliced = {
        names[axis]: float(position[axis])
        for axis in range(len(names))
        if axis not in displayed
    }
    thickness = thickness or {}
    return RegionSelection(
        transform=AffineTransform.from_axis_map(
            rendered,
            world,
            axis_map={name: name for name in shown},
            constant_output_axes=sliced,
        ),
        region=ConvexRegion.from_axis_slabs(
            world,
            {
                name: (value, thickness.get(names.index(name), 0.0))
                for name, value in sliced.items()
            },
        ),
    )


# -- pullback and V ---------------------------------------------------------


@pytest.mark.parametrize("name", list(SPATIAL))
@pytest.mark.parametrize("view", VIEWS_3D)
def test_every_2d_view_of_a_volume_pulls_back_to_a_plane(name, view):
    classification = classify_view(_spatial(name), view)
    assert classification.pullback_dim == 2
    if classification.alignment is A.OBLIQUE:
        assert len(classification.varying_data_axes) == 3
    else:
        assert classification.varying_data_axes == classification.retained_data_axes


@pytest.mark.parametrize(
    ("name", "view", "varying"),
    [
        ("rot_zx", (1, 2), (0, 1, 2)),
        ("rot_zx", (0, 1), (0, 1, 2)),
        ("rot_zx", (0, 2), (0, 2)),
        ("shear_x_by_z", (1, 2), (1, 2)),
        ("rot_yx", (1, 2), (1, 2)),
        ("rot_yx", (0, 2), (0, 1, 2)),
    ],
)
def test_the_varying_axes_of_a_spatial_view(name, view, varying):
    assert classify_view(_spatial(name), view).varying_data_axes == varying


def test_a_3d_view_varies_over_every_axis():
    classification = classify_view(_spatial("general"), (0, 1, 2))
    assert classification.pullback_dim == 3
    assert classification.varying_data_axes == (0, 1, 2)


@pytest.mark.parametrize("name", list(TEMPORAL))
@pytest.mark.parametrize("view", VIEWS_4D)
def test_every_2d_view_of_a_timelapse_pulls_back_to_a_plane(name, view):
    classification = classify_view(_temporal(name), view)
    assert classification.pullback_dim == 2
    assert len(classification.varying_data_axes) >= 2


@pytest.mark.parametrize(
    ("view", "varying"),
    [
        ((0, 1), (0, 1, 3)),
        ((0, 2), (0, 1, 2, 3)),  # x is sliced and reads t: |V| = 4 (O19)
        ((0, 3), (0, 1, 3)),
        ((1, 2), (1, 2, 3)),
        ((1, 3), (1, 3)),
        ((2, 3), (1, 2, 3)),
    ],
)
def test_the_varying_axes_of_drift_rot(view, varying):
    assert classify_view(_temporal("drift_rot"), view).varying_data_axes == varying


def test_the_drift_kymograph_varies_over_time_and_two_space_axes():
    """K7, K16: the ``(t, x)`` view of ``drift_rot`` is ``V = tzx``."""
    classification = classify_view(_temporal("drift_rot"), (0, 3))
    assert classification.alignment is A.OBLIQUE
    assert classification.varying_data_axes == (0, 1, 3)


def test_a_tilted_2d_store_pulls_back_to_a_line():
    """O18: a ``yx`` image tilted in a ``zyx`` world (world z reads both data
    axes), seen in ``yx``: OBLIQUE, and the slice meets it along a line."""
    linear = np.zeros((3, 2))
    linear[0] = [0.25, 0.5]
    linear[1, 0] = linear[2, 1] = 1.0
    classification = classify_view(_affine(linear, None, "yx", _world("zyx")), (1, 2))
    assert classification.alignment is A.OBLIQUE
    assert classification.pullback_dim == 1
    decision = choose_slice_route(classification)
    assert decision.route is SliceRoute.NONE
    assert decision.reason is RouteRefusalReason.PULLBACK_NOT_2D
    assert decision.dimension == 1
    with pytest.raises(ValueError, match="plane"):
        slice_plane_frame(classification, np.zeros(3))


def _flat_store_linear(z_reads_x: float = 0.0) -> np.ndarray:
    """A ``yx`` store in a ``zyx`` world; world z = *z_reads_x* * data x."""
    linear = np.zeros((3, 2))
    linear[0, 1] = z_reads_x
    linear[1, 0] = linear[2, 1] = 1.0
    return linear


@pytest.mark.parametrize(
    ("linear", "view", "alignment"),
    [
        # Sheared: world z reads only data x; seen in yx.
        (_flat_store_linear(0.5), (1, 2), A.SLICE_SHIFT),
        # Flat, seen edge-on in (z, x): the slice pins data y.
        (_flat_store_linear(), (0, 2), A.AXIS_ALIGNED),
    ],
)
def test_every_slice_along_a_line_is_refused(linear, view, alignment):
    """O18, U13: a 2D slice that meets the data along a line is refused in
    every alignment (today's path raises on it when spaces are pushed)."""
    classification = classify_view(_affine(linear, None, "yx", _world("zyx")), view)
    assert classification.alignment is alignment
    assert classification.pullback_dim == 1
    decision = choose_slice_route(classification)
    assert decision.route is SliceRoute.NONE
    assert decision.reason is RouteRefusalReason.PULLBACK_NOT_2D
    assert decision.dimension == 1


def test_a_flat_2d_store_in_its_own_plane_stays_on_todays_path():
    classification = classify_view(
        _affine(_flat_store_linear(), None, "yx", _world("zyx")), (1, 2)
    )
    assert classification.pullback_dim == 2
    assert choose_slice_route(classification).route is SliceRoute.TODAY


def test_a_flat_2d_store_in_3d_stays_on_todays_path():
    classification = classify_view(
        _affine(_flat_store_linear(0.5), None, "yx", _world("zyx")), (0, 1, 2)
    )
    assert choose_slice_route(classification).route is SliceRoute.TODAY


def test_a_projecting_tilted_2d_store_is_refused():
    """G0-4: the slab meets a tilted 2D store in a strip (10.4), deferred."""
    transform = _affine(_flat_store_linear(0.5), None, "yx", _world("zyx"))
    geometry = classify_projection(transform, (1, 2), 0)
    decision = choose_slice_route(classify_view(transform, (1, 2)), geometry)
    assert decision.route is SliceRoute.NONE
    assert decision.reason is RouteRefusalReason.PULLBACK_NOT_2D


def test_a_volume_broadcast_into_a_channel_world():
    """``zyx`` into ``czyx``: the channel row is empty and drops out of ``V``."""
    linear = np.zeros((4, 3))
    linear[1:, :] = SPATIAL["rot_zx"]
    transform = _affine(linear, np.zeros(4), "zyx", _world("czyx"))
    classification = classify_view(transform, (2, 3))
    assert classification.pullback_dim == 2
    assert classification.varying_data_axes == (0, 1, 2)
    frame = slice_plane_frame(classification, np.array([3.0, 4.0, 0.0, 0.0]))
    world = transform.map_coordinates(frame.data_points(np.array([[1.0, 2.0]])))
    np.testing.assert_allclose(world[0, 1:], [4.0, 1.0, 2.0], atol=1e-12)


def test_a_coupling_d7_keeps_is_varying():
    """A small coupling D7 keeps (``1e-4``) makes its axis vary."""
    linear = np.eye(3)
    linear[0, 1] = 1e-4  # world z reads data y, a little
    classification = classify_view(_affine(linear, None, "zyx", _world("zyx")), (1, 2))
    assert classification.alignment is A.OBLIQUE
    assert classification.varying_data_axes == (0, 1, 2)


def test_noise_d7_snaps_does_not_vary():
    linear = np.eye(3)
    linear[0, 1] = 1e-9
    classification = classify_view(_affine(linear, None, "zyx", _world("zyx")), (1, 2))
    assert classification.varying_data_axes == (1, 2)


# -- the frame ---------------------------------------------------------------


@pytest.mark.parametrize("name", list(SPATIAL))
@pytest.mark.parametrize("view", VIEWS_3D)
def test_frame_points_satisfy_the_slice(name, view):
    transform = _spatial(name)
    classification = classify_view(transform, view)
    position = np.array([7.25, -3.5, 12.0])
    frame = slice_plane_frame(classification, position)
    assert frame.displayed_world_axes == view
    displayed = np.array([[0.0, 0.0], [3.0, -1.5], [-40.0, 250.0]])
    world = transform.map_coordinates(frame.data_points(displayed))
    sliced = [axis for axis in range(3) if axis not in view]
    np.testing.assert_allclose(
        world[:, sliced],
        np.broadcast_to(position[sliced], (len(displayed), len(sliced))),
        atol=1e-12,
    )
    np.testing.assert_allclose(world[:, list(view)], displayed, atol=1e-12)
    assert set(np.flatnonzero(np.abs(frame.axes).sum(axis=0))) <= set(
        classification.varying_data_axes
    )


@pytest.mark.parametrize("name", list(TEMPORAL))
@pytest.mark.parametrize("view", VIEWS_4D)
def test_timelapse_frame_points_satisfy_the_slice(name, view):
    transform = _temporal(name)
    classification = classify_view(transform, view)
    position = np.array([3.0, 7.25, -3.5, 12.0])
    frame = slice_plane_frame(classification, position)
    displayed = np.array([[0.0, 0.0], [2.0, 5.0]])
    world = transform.map_coordinates(frame.data_points(displayed))
    sliced = [axis for axis in range(4) if axis not in view]
    np.testing.assert_allclose(
        world[:, sliced],
        np.broadcast_to(position[sliced], (len(displayed), len(sliced))),
        atol=1e-12,
    )
    np.testing.assert_allclose(world[:, list(view)], displayed, atol=1e-12)


def test_the_frame_reads_the_snapped_transform():
    """D7: a trig-built quarter turn's ``6e-17`` never reaches the frame."""
    transform = _affine(_rot(1, 2, 90.0), None, "zyx", _world("zyx"))
    frame = slice_plane_frame(classify_view(transform, (1, 2)), np.array([4.0, 0, 0]))
    assert np.count_nonzero(frame.axes) == 2
    assert frame.origin[0] == 4.0


def test_the_frame_needs_a_2d_view():
    with pytest.raises(ValueError, match="2D view"):
        slice_plane_frame(classify_view(_spatial("id"), (0, 1, 2)), np.zeros(3))


def test_the_slice_position_has_one_value_per_world_axis():
    with pytest.raises(ValueError, match="axes"):
        slice_plane_frame(classify_view(_spatial("id"), (1, 2)), np.zeros(2))


@pytest.fixture
def nonuniform_time():
    """``tzyx`` into ``tzyx``, a non-uniform ``t`` and a scaled ``z``."""
    data = DataCoordinateSystem(
        name="volume",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="frame"),
            Axis(name="z", axis_type="space"),
            Axis(name="y", axis_type="space"),
            Axis(name="x", axis_type="space"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            Axis(name="z", axis_type="space"),
            Axis(name="y", axis_type="space"),
            Axis(name="x", axis_type="space"),
        )
    )
    leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=(0.0, 1.0, 2.5, 4.0)),
        input_coordinate_system=CoordinateSystem(name="t", axes=(data.axes[0],)).id,
        output_coordinate_system=CoordinateSystem(name="t", axes=(world.axes[0],)).id,
    )
    transform = ByDimensionTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        axis_transforms={"t": leaf},
    )
    return transform, world


def test_a_sliced_non_uniform_axis_is_pinned_to_its_data_value(nonuniform_time):
    transform, _ = nonuniform_time
    classification = classify_view(transform, (2, 3))
    assert classification.varying_data_axes == (2, 3)
    frame = slice_plane_frame(classification, np.array([2.5, 6.0, 0.0, 0.0]))
    np.testing.assert_allclose(frame.origin, [2.0, 3.0, 0.0, 0.0], atol=1e-12)


def test_a_displayed_non_uniform_axis_has_no_frame(nonuniform_time):
    transform, _ = nonuniform_time
    classification = classify_view(transform, (0, 3))
    with pytest.raises(NonAffineTransformError, match="non-affine"):
        slice_plane_frame(classification, np.zeros(4))


# -- projection geometry ---------------------------------------------------


@pytest.mark.parametrize(
    ("name", "view", "axis", "alignment"),
    [
        ("id", (1, 2), 0, DepthAlignment.ALONG_AXIS),
        ("rot_yx", (1, 2), 0, DepthAlignment.ALONG_AXIS),
        ("shear_x_by_z", (1, 2), 0, DepthAlignment.TILTED),
        ("rot_zx", (1, 2), 0, DepthAlignment.TILTED),
        ("rot_zx", (0, 2), 1, DepthAlignment.ALONG_AXIS),
        ("general", (0, 1), 2, DepthAlignment.TILTED),
    ],
)
def test_depth_alignment(name, view, axis, alignment):
    geometry = classify_projection(_spatial(name), view, axis)
    assert geometry.depth_alignment is alignment
    assert geometry.slab_pullback_dim == 3
    assert geometry.slab_varying_axes == (0, 1, 2)


@pytest.mark.parametrize("name", list(SPATIAL))
@pytest.mark.parametrize("view", VIEWS_3D)
def test_the_ray_moves_along_its_world_axis_only(name, view):
    transform = _spatial(name)
    (axis,) = [index for index in range(3) if index not in view]
    direction = classify_projection(transform, view, axis).ray_direction_data
    expected = np.zeros(3)
    expected[axis] = 1.0
    np.testing.assert_allclose(
        transform.to_affine().linear @ direction, expected, atol=1e-12
    )


def test_a_projection_of_a_timelapse_varies_over_space():
    geometry = classify_projection(_temporal("id"), (2, 3), 1)
    assert geometry.depth_alignment is DepthAlignment.ALONG_AXIS
    assert geometry.slab_varying_axes == (1, 2, 3)


def test_a_drift_rot_projection_along_x_holds_time():
    """``drift_rot`` in a ``(z, y)`` view, projected along world x: the ray
    keeps data t fixed (world t is held), so ``V_h`` is ``zyx``."""
    geometry = classify_projection(_temporal("drift_rot"), (1, 2), 3)
    assert geometry.slab_pullback_dim == 3
    assert geometry.slab_varying_axes == (1, 2, 3)
    assert geometry.depth_alignment is DepthAlignment.TILTED


def test_a_projection_along_a_broadcast_axis_is_constant():
    linear = np.zeros((4, 3))
    linear[1:, :] = np.eye(3)
    transform = _affine(linear, None, "zyx", _world("czyx"))
    geometry = classify_projection(transform, (2, 3), 0)
    assert geometry.depth_alignment is DepthAlignment.CONSTANT
    assert ray_step(geometry.ray_direction_data) == np.inf


def test_a_projection_runs_along_a_sliced_axis():
    with pytest.raises(ValueError, match="displayed"):
        classify_projection(_spatial("id"), (1, 2), 1)
    with pytest.raises(ValueError, match="not an axis"):
        classify_projection(_spatial("id"), (1, 2), 5)


def test_a_projection_along_a_non_uniform_axis_is_refused(nonuniform_time):
    with pytest.raises(NonAffineTransformError, match="ray step"):
        classify_projection(nonuniform_time[0], (2, 3), 0)


# -- ray step ------------------------------------------------------------------


def test_one_voxel_per_world_unit_steps_one_unit():
    direction = classify_projection(_spatial("id"), (1, 2), 0).ray_direction_data
    assert ray_step(direction) == 1.0


def test_the_step_follows_the_world_scale_and_the_level():
    transform = _affine(np.diag([2.0, 1.0, 1.0]), None, "zyx", _world("zyx"))
    direction = classify_projection(transform, (1, 2), 0).ray_direction_data
    assert ray_step(direction) == pytest.approx(2.0)
    assert ray_step(direction, level_scale=(2.0, 2.0, 2.0)) == pytest.approx(4.0)
    assert ray_step(direction, steps_per_voxel=2.0) == pytest.approx(1.0)


def test_the_step_is_the_euclidean_voxel_length_k6():
    """``shear_x_by_z`` moves the ray one voxel in z and half a voxel in x per
    world unit: ``|d| = sqrt(1.25)``, where the max-norm would say 1."""
    direction = classify_projection(
        _spatial("shear_x_by_z"), (1, 2), 0
    ).ray_direction_data
    assert ray_step(direction) == pytest.approx(1.0 / np.sqrt(1.25), abs=1e-12)


def test_a_rotation_keeps_the_step():
    direction = classify_projection(_spatial("rot_zx"), (1, 2), 0).ray_direction_data
    assert ray_step(direction) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps_per_voxel": 0.0}, "positive"),
        ({"level_scale": (1.0, 1.0)}, "one positive size"),
        ({"level_scale": (1.0, 0.0, 1.0)}, "one positive size"),
    ],
)
def test_bad_ray_step_arguments_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ray_step(np.array([1.0, 0.0, 0.0]), **kwargs)


# -- which axis projects -----------------------------------------------------


NAMES = ("t", "z", "y", "x")


@pytest.mark.parametrize(
    ("operator", "axes", "thickness", "expected"),
    [
        ("none", ("z",), {1: 4.0}, None),
        ("mip", (), {1: 4.0}, None),
        ("mip", ("z",), {}, None),
        ("mip", ("z",), {1: 0.0}, None),
        ("mip", ("z",), {1: 4.0}, 1),
        ("mean", ("t",), {0: 2.0}, 0),
        (
            "mip",
            ("q",),
            {1: 4.0},
            ProjectionRefusal(ProjectionRefusalReason.UNKNOWN_AXIS, ("q",)),
        ),
        (
            "mip",
            ("z", "t"),
            {0: 2.0, 1: 4.0},
            ProjectionRefusal(ProjectionRefusalReason.MULTIPLE_AXES, ("z", "t")),
        ),
        (
            "minip",
            ("x",),
            {3: 4.0},
            ProjectionRefusal(ProjectionRefusalReason.DISPLAYED_AXIS, ("x",)),
        ),
    ],
)
def test_the_projection_axis_rule(operator, axes, thickness, expected):
    assert projection_axis(operator, axes, NAMES, (2, 3), thickness) == expected


def test_an_unknown_axis_is_reported_before_a_second_axis():
    result = projection_axis("mip", ("z", "q"), NAMES, (2, 3), {1: 1.0})
    assert result == ProjectionRefusal(ProjectionRefusalReason.UNKNOWN_AXIS, ("q",))


# -- the route table ---------------------------------------------------------


@pytest.mark.parametrize("name", list(SPATIAL))
@pytest.mark.parametrize("view", VIEWS_3D)
def test_the_route_of_a_plain_slice(name, view):
    classification = classify_view(_spatial(name), view)
    decision = choose_slice_route(classification)
    if classification.alignment is A.OBLIQUE:
        assert decision.route is SliceRoute.BRICK
    else:
        assert decision.route is SliceRoute.TODAY
    assert decision.reason is None


@pytest.mark.parametrize(
    ("transform", "view", "normal"),
    [
        (lambda: _spatial("rot_zx"), (1, 2), 0),
        (lambda: _spatial("rot_zx"), (0, 1), 2),
        (lambda: _spatial("general"), (0, 2), 1),
        # t is sliced too but pinned: z is the normal.
        (lambda: _temporal("rot_zx"), (2, 3), 1),
        # The drift kymograph: V = tzx, the normal is z (y is pinned).
        (lambda: _temporal("drift_rot"), (0, 3), 1),
    ],
)
def test_a_brick_route_names_its_normal_axis(transform, view, normal):
    decision = choose_slice_route(classify_view(transform(), view))
    assert decision.route is SliceRoute.BRICK
    assert decision.normal_world_axis == normal


def test_a_projection_names_its_axis_as_the_normal():
    transform = _spatial("id")
    geometry = classify_projection(transform, (1, 2), 0)
    decision = choose_slice_route(classify_view(transform, (1, 2)), geometry)
    assert decision.normal_world_axis == 0


def test_a_3d_view_stays_on_todays_path():
    classification = classify_view(_spatial("general"), (0, 1, 2))
    assert choose_slice_route(classification).route is SliceRoute.TODAY


def test_any_projection_takes_the_brick_route():
    """G0-2: aligned or tilted."""
    for name in ("id", "rot_zx"):
        transform = _spatial(name)
        classification = classify_view(transform, (1, 2))
        geometry = classify_projection(transform, (1, 2), 0)
        assert choose_slice_route(classification, geometry).route is SliceRoute.BRICK


def test_a_slice_through_four_data_axes_draws_nothing():
    """O19: world t reads data x, world z is tilted through x: ``|V| = 4``."""
    linear = np.eye(4)
    linear[0, 3] = 0.2
    linear[1:, 1:] = _rot(0, 2, 30.0)
    classification = classify_view(
        _affine(linear, None, "tzyx", _world("tzyx")), (2, 3)
    )
    assert classification.pullback_dim == 2
    decision = choose_slice_route(classification)
    assert decision.route is SliceRoute.NONE
    assert decision.reason is RouteRefusalReason.TOO_MANY_VARYING_AXES
    assert decision.dimension == 4


def test_a_slab_through_four_data_axes_draws_nothing():
    linear = np.eye(4)
    linear[0, 3] = 0.2
    transform = _affine(linear, None, "tzyx", _world("tzyx"))
    geometry = classify_projection(transform, (2, 3), 1)
    decision = choose_slice_route(classify_view(transform, (2, 3)), geometry)
    assert decision.route is SliceRoute.NONE
    assert decision.reason is RouteRefusalReason.TOO_MANY_VARYING_AXES


# -- the footprint -----------------------------------------------------------


SHAPE = (8, 10, 12)


def _inside_box(points: np.ndarray, shape) -> np.ndarray:
    upper = np.asarray(shape, dtype=np.float64) - 0.5
    return np.all((points >= -0.5) & (points <= upper), axis=-1)


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Points strictly inside a counter-clockwise convex polygon."""
    inside = np.ones(points.shape[0], dtype=bool)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        cross = edge[0] * (points[:, 1] - start[1]) - edge[1] * (
            points[:, 0] - start[0]
        )
        inside &= cross > 0.0
    return inside


def _distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    distance = np.full(points.shape[0], np.inf)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        t = np.clip(((points - start) @ edge) / (edge @ edge), 0.0, 1.0)
        nearest = start + t[:, None] * edge
        distance = np.minimum(distance, np.linalg.norm(points - nearest, axis=1))
    return distance


def _assert_convex_ccw(polygon: np.ndarray) -> None:
    assert polygon.shape[0] >= 3
    for a, b, c in zip(
        polygon, np.roll(polygon, -1, axis=0), np.roll(polygon, -2, axis=0), strict=True
    ):
        assert (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0.0


def test_an_aligned_slice_is_the_box_face():
    world = _world("zyx")
    transform = _affine(np.eye(3), None, "zyx", world)
    classification = classify_view(transform, (1, 2))
    polygon = slice_footprint(
        SHAPE, classification, _selection(world, (1, 2), [3.0, 0, 0])
    )
    _assert_convex_ccw(polygon)
    expected = {(-0.5, -0.5), (9.5, -0.5), (9.5, 11.5), (-0.5, 11.5)}
    assert {tuple(np.round(vertex, 12)) for vertex in polygon} == expected


def test_scale_and_translation_move_the_face():
    world = _world("zyx")
    transform = _affine(np.diag([1.0, 2.0, 3.0]), [0.0, 5.0, -1.0], "zyx", world)
    polygon = slice_footprint(
        SHAPE, classify_view(transform, (1, 2)), _selection(world, (1, 2), [2.0, 0, 0])
    )
    np.testing.assert_allclose(polygon.min(axis=0), [4.0, -2.5], atol=1e-12)
    np.testing.assert_allclose(polygon.max(axis=0), [24.0, 33.5], atol=1e-12)


def test_a_slice_outside_the_data_has_no_footprint():
    world = _world("zyx")
    transform = _affine(np.eye(3), None, "zyx", world)
    polygon = slice_footprint(
        SHAPE, classify_view(transform, (1, 2)), _selection(world, (1, 2), [40.0, 0, 0])
    )
    assert polygon.shape == (0, 2)


@pytest.mark.parametrize("name", ["rot_zx", "general", "shear_x_by_z", "rot_yx"])
@pytest.mark.parametrize("view", VIEWS_3D)
@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_the_footprint_is_the_set_of_points_in_the_data(name, view, fraction):
    """A displayed point is inside the polygon exactly when the frame puts it
    inside the data box, checked on a grid (away from the polygon's edge);
    every vertex lies on the slice and on a box face."""
    world = _world("zyx")
    transform = _affine(SPATIAL[name], TRANSLATION_3D, "zyx", world)
    classification = classify_view(transform, view)
    centre = transform.map_coordinates((np.asarray(SHAPE, float)[None, :] - 1) / 2)[0]
    corner = transform.map_coordinates(np.zeros((1, 3)))[0]
    position = corner + fraction * 2.0 * (centre - corner)
    selection = _selection(world, view, position)
    polygon = slice_footprint(SHAPE, classification, selection)
    _assert_convex_ccw(polygon)
    frame = slice_plane_frame(classification, position)

    on_box = frame.data_points(polygon)
    upper = np.asarray(SHAPE, float) - 0.5
    face = np.minimum(np.abs(on_box + 0.5), np.abs(on_box - upper)).min(axis=1)
    np.testing.assert_allclose(face, 0.0, atol=1e-9)
    assert ((on_box >= -0.5 - 1e-9) & (on_box <= upper + 1e-9)).all()

    low, high = polygon.min(axis=0) - 2.0, polygon.max(axis=0) + 2.0
    grid = np.stack(
        np.meshgrid(np.linspace(low[0], high[0], 61), np.linspace(low[1], high[1], 61)),
        axis=-1,
    ).reshape(-1, 2)
    clear = _distance_to_polygon(grid, polygon) > 1e-6
    in_data = _inside_box(frame.data_points(grid), SHAPE)
    np.testing.assert_array_equal(_inside_polygon(grid, polygon)[clear], in_data[clear])


def test_a_slab_footprint_contains_the_slice_and_is_the_slab_cut():
    world = _world("zyx")
    transform = _affine(SPATIAL["rot_zx"], TRANSLATION_3D, "zyx", world)
    classification = classify_view(transform, (1, 2))
    position = np.array([8.0, 0.0, 0.0])
    plane = slice_footprint(SHAPE, classification, _selection(world, (1, 2), position))
    slab_selection = _selection(world, (1, 2), position, thickness={0: 2.0})
    slab = slice_footprint(
        SHAPE, classification, slab_selection, projection_world_axis=0
    )
    _assert_convex_ccw(slab)
    assert (
        _inside_polygon(plane, slab) | (_distance_to_polygon(plane, slab) < 1e-9)
    ).all()
    assert _inside_polygon(plane.mean(axis=0, keepdims=True), slab).all()

    low, high = slab.min(axis=0) - 2.0, slab.max(axis=0) + 2.0
    grid = np.stack(
        np.meshgrid(np.linspace(low[0], high[0], 41), np.linspace(low[1], high[1], 41)),
        axis=-1,
    ).reshape(-1, 2)
    depths = np.linspace(position[0] - 2.0, position[0] + 2.0, 401)
    in_slab = np.zeros(grid.shape[0], dtype=bool)
    for depth in depths:
        frame = slice_plane_frame(classification, np.array([depth, 0.0, 0.0]))
        in_slab |= _inside_box(frame.data_points(grid), SHAPE)
    clear = _distance_to_polygon(grid, slab) > 0.05
    np.testing.assert_array_equal(_inside_polygon(grid, slab)[clear], in_slab[clear])


def test_the_thickness_of_a_plain_slice_does_not_widen_it():
    """Without a projection the plane is drawn, whatever the slab (4.2)."""
    world = _world("zyx")
    transform = _affine(SPATIAL["rot_zx"], TRANSLATION_3D, "zyx", world)
    classification = classify_view(transform, (1, 2))
    position = np.array([8.0, 0.0, 0.0])
    plane = slice_footprint(SHAPE, classification, _selection(world, (1, 2), position))
    thick = slice_footprint(
        SHAPE, classification, _selection(world, (1, 2), position, thickness={0: 2.0})
    )
    np.testing.assert_allclose(thick, plane, atol=1e-12)


def test_a_bound_on_a_displayed_axis_cuts_the_footprint():
    """R3: a region may bound a displayed axis; the polygon honours it."""
    world = _world("zyx")
    transform = _affine(np.eye(3), None, "zyx", world)
    selection = _selection(world, (1, 2), [3.0, 0, 0])
    bounded = RegionSelection(
        transform=selection.transform,
        region=ConvexRegion(
            coordinate_system=selection.region.coordinate_system,
            ndim=3,
            half_spaces=(
                *selection.region.half_spaces,
                HalfSpace(normal=np.array([0.0, 0.0, 1.0]), offset=5.0),
            ),
        ),
    )
    polygon = slice_footprint(SHAPE, classify_view(transform, (1, 2)), bounded)
    assert polygon[:, 1].max() == pytest.approx(5.0)


def test_a_projection_along_a_broadcast_axis_is_the_plane():
    world = _world("czyx")
    linear = np.zeros((4, 3))
    linear[1:, :] = SPATIAL["rot_zx"]
    transform = _affine(linear, None, "zyx", world)
    classification = classify_view(transform, (2, 3))
    position = np.array([0.0, 4.0, 0.0, 0.0])
    plane = slice_footprint(SHAPE, classification, _selection(world, (2, 3), position))
    slab = slice_footprint(
        SHAPE,
        classification,
        _selection(world, (2, 3), position, thickness={0: 3.0}),
        projection_world_axis=0,
    )
    np.testing.assert_allclose(slab, plane, atol=1e-12)


def test_a_timepoint_outside_the_data_has_no_footprint(nonuniform_time):
    transform, world = nonuniform_time
    classification = classify_view(transform, (2, 3))
    inside = _selection(world, (2, 3), [2.5, 2.0, 0.0, 0.0])
    assert slice_footprint((4, 8, 10, 12), classification, inside).shape == (4, 2)
    outside = _selection(world, (2, 3), [9.0, 2.0, 0.0, 0.0])
    assert slice_footprint((4, 8, 10, 12), classification, outside).shape == (0, 2)


def test_the_footprint_refuses_another_world():
    transform = _affine(np.eye(3), None, "zyx", _world("zyx"))
    with pytest.raises(ValueError, match="world coordinate system"):
        slice_footprint(
            SHAPE,
            classify_view(transform, (1, 2)),
            _selection(_world("zyx"), (1, 2), [3.0, 0, 0]),
        )
