"""The per-view classifier (full affine design v2, Phase 1).

``plans/full_affine_transforms_design_v2.md`` sections 3.1 and 4.1, with
decision D7's relative tolerance.  Every case is a ``data -> world``
transform against one view; the expected cases follow section 3.1's rules
and its worked checks.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.transform import (
    COUPLING_TOLERANCE,
    AffineTransform,
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    CoordinateSystem,
    DataCoordinateSystem,
    NonUniformAxisTransform,
    SliceAlignment,
    WorldCoordinateSystem,
    classify_view,
)

A = SliceAlignment
KINDS = {"t": "time", "c": "channel"}

#: Every 2D view of a ``zyx`` world, and the 3D one.
VIEWS = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]


def _systems(data_names: str, world_names: str):
    data = CoordinateSystem(
        name="data",
        axes=tuple(Axis(name=n, axis_type=KINDS.get(n, "space")) for n in data_names),
    )
    world = CoordinateSystem(
        name="world",
        axes=tuple(Axis(name=n, axis_type=KINDS.get(n, "space")) for n in world_names),
    )
    return data, world


def _affine(linear, data_names: str = "zyx", world_names: str = "zyx"):
    linear = np.asarray(linear, dtype=np.float64)
    data, world = _systems(data_names, world_names)
    matrix = np.zeros((linear.shape[0] + 1, linear.shape[1] + 1))
    matrix[:-1, :-1] = linear
    matrix[-1, -1] = 1.0
    return AffineTransform.from_matrix(matrix, data, world)


def _rot(a: int, b: int, degrees: float, n: int = 3) -> np.ndarray:
    theta = np.deg2rad(degrees)
    m = np.eye(n)
    m[a, a] = m[b, b] = np.cos(theta)
    m[a, b] = -np.sin(theta)
    m[b, a] = np.sin(theta)
    return m


def _shear(row: int, column: int, amount: float = 0.5) -> np.ndarray:
    m = np.eye(3)
    m[row, column] = amount
    return m


#: name -> (linear, [case per view in VIEWS]).
TABLE = {
    "scale": (np.diag([2.0, 1.0, 1.5]), [A.AXIS_ALIGNED] * 4),
    "rot90_yx": (_rot(1, 2, 90.0).round(12), [A.AXIS_ALIGNED] * 4),
    "flip_x": (np.diag([1.0, 1.0, -1.0]), [A.AXIS_ALIGNED] * 4),
    "rot_yx": (_rot(1, 2, 30.0), [A.OBLIQUE, A.OBLIQUE, A.IN_PLANE, A.IN_PLANE]),
    # world x += 0.5 data y
    "shear_yx": (
        _shear(2, 1),
        [A.OBLIQUE, A.SLICE_SHIFT, A.IN_PLANE, A.IN_PLANE],
    ),
    # world x += 0.5 data z
    "shear_x_by_z": (
        _shear(2, 0),
        [A.OBLIQUE, A.IN_PLANE, A.SLICE_SHIFT, A.IN_PLANE],
    ),
    # world z += 0.5 data x
    "shear_z_by_x": (
        _shear(0, 2),
        [A.SLICE_SHIFT, A.IN_PLANE, A.OBLIQUE, A.IN_PLANE],
    ),
    "rot_zx": (_rot(0, 2, 30.0), [A.OBLIQUE, A.IN_PLANE, A.OBLIQUE, A.IN_PLANE]),
    "general": (
        _rot(0, 1, 20.0) @ _rot(1, 2, 25.0) @ _rot(0, 2, 15.0) @ np.diag([2.0, 1, 1]),
        [A.OBLIQUE, A.OBLIQUE, A.OBLIQUE, A.IN_PLANE],
    ),
}


@pytest.mark.parametrize("name", list(TABLE))
@pytest.mark.parametrize("view_index", range(len(VIEWS)))
def test_the_investigation_transforms_classify_per_view(name, view_index):
    linear, cases = TABLE[name]
    view = VIEWS[view_index]
    assert classify_view(_affine(linear), view).alignment is cases[view_index]


def test_the_partition_comes_from_the_sliced_rows():
    """``shear_x_by_z`` in a ``yx`` view: data z feeds world x, but the slice
    pins it, so it is collapsed -- which is what makes the case SLICE_SHIFT."""
    classification = classify_view(_affine(_shear(2, 0)), (1, 2))
    assert classification.collapsed_data_axes == (0,)
    assert classification.retained_data_axes == (1, 2)
    assert classification.sliced_world_axes == (0,)


def test_a_tilted_slice_collapses_every_axis_its_row_touches():
    classification = classify_view(_affine(_rot(0, 2, 30.0)), (1, 2))
    assert classification.alignment is A.OBLIQUE
    assert classification.collapsed_data_axes == (0, 2)
    assert classification.retained_data_axes == (1,)


def test_the_order_of_the_displayed_axes_does_not_matter():
    transform = _affine(_rot(1, 2, 30.0))
    assert classify_view(transform, (2, 1)) == classify_view(transform, (1, 2))


# -- broadcast ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("view", "case"),
    [((2, 3), A.AXIS_ALIGNED), ((0, 3), A.AXIS_ALIGNED), ((1, 2, 3), A.AXIS_ALIGNED)],
)
def test_a_store_broadcast_over_a_channel_axis(view, case):
    """A ``zyx`` store in a ``czyx`` world: ``c`` has an empty row, so it is
    broadcast, not tilted -- including in the ``(c, x)`` view of 3.1."""
    linear = np.zeros((4, 3))
    linear[1:, :] = np.eye(3)
    assert classify_view(_affine(linear, "zyx", "czyx"), view).alignment is case


def test_a_rotated_store_broadcast_over_a_channel_axis():
    linear = np.zeros((4, 3))
    linear[1:, :] = _rot(1, 2, 30.0)
    transform = _affine(linear, "zyx", "czyx")
    assert classify_view(transform, (1, 2, 3)).alignment is A.IN_PLANE
    assert classify_view(transform, (2, 3)).alignment is A.IN_PLANE


# -- a block container with a non-uniform time axis ----------------------------


@pytest.fixture
def labels_in_a_channel_world():
    """TZYX labels -> TCZYX world, broadcast over C, non-uniform T."""
    labels = DataCoordinateSystem(
        name="labels",
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
            Axis(name="c", axis_type="channel"),
            Axis(name="z", axis_type="space"),
            Axis(name="y", axis_type="space"),
            Axis(name="x", axis_type="space"),
        )
    )
    leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=(0.0, 1.0, 2.5, 4.0)),
        input_coordinate_system=CoordinateSystem(name="t", axes=(labels.axes[0],)).id,
        output_coordinate_system=CoordinateSystem(name="t", axes=(world.axes[0],)).id,
    )
    return ByDimensionTransform.from_axis_map(
        labels,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        broadcast_output_axes=("c",),
        axis_transforms={"t": leaf},
    )


def test_a_block_container_reports_its_pattern_structurally(labels_in_a_channel_world):
    expected = np.zeros((5, 4), dtype=bool)
    expected[0, 0] = True  # the non-uniform t leaf, dense within itself
    expected[2, 1] = expected[3, 2] = expected[4, 3] = True
    np.testing.assert_array_equal(labels_in_a_channel_world.axis_coupling(), expected)


def test_a_block_container_classifies_like_its_affine_twin(labels_in_a_channel_world):
    volume = classify_view(labels_in_a_channel_world, (2, 3, 4))
    assert volume.alignment is A.AXIS_ALIGNED
    assert volume.collapsed_data_axes == (0,)
    assert volume.retained_data_axes == (1, 2, 3)
    flat = classify_view(labels_in_a_channel_world, (3, 4))
    assert flat.alignment is A.AXIS_ALIGNED
    assert flat.collapsed_data_axes == (0, 1)
    assert flat.effective_transform is labels_in_a_channel_world


def test_a_non_uniform_leaf_couples_its_one_axis():
    leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=(0.0, 1.0, 3.0)),
        input_coordinate_system=uuid4(),
        output_coordinate_system=uuid4(),
    )
    np.testing.assert_array_equal(leaf.axis_coupling(), [[True]])


# -- drift ---------------------------------------------------------------------


@pytest.fixture
def drift():
    """``tzyx`` into ``tzyx``, world x += 0.3 data t."""
    linear = np.eye(4)
    linear[3, 0] = 0.3
    return _affine(linear, "tzyx", "tzyx")


@pytest.mark.parametrize(
    ("view", "case"),
    [
        ((2, 3), A.SLICE_SHIFT),  # 2D yx: scrubbing t moves the frame
        ((1, 2, 3), A.SLICE_SHIFT),  # 3D zyx, likewise
        ((0, 3), A.IN_PLANE),  # a (t, x) kymograph: the drift is a shear
        ((0, 2), A.OBLIQUE),  # (t, y) slices x, whose row reads t
    ],
)
def test_drift(drift, view, case):
    assert classify_view(drift, view).alignment is case


# -- D7: the tolerance -----------------------------------------------------------


def test_a_trig_built_quarter_turn_is_axis_aligned():
    """``cos(pi / 2)`` is 6e-17, not 0.  Exactly compared, this view would be
    OBLIQUE and draw nothing (design v2, 6.1)."""
    noisy = _rot(0, 2, 90.0)
    assert noisy[0, 0] != 0.0
    transform = _affine(noisy)
    classification = classify_view(transform, (1, 2))
    assert classification.alignment is A.AXIS_ALIGNED
    assert classification.effective_transform is not transform
    effective = np.asarray(classification.effective_transform.linear)
    np.testing.assert_array_equal(effective, noisy.round(12))
    # The user's transform is never modified.
    assert np.asarray(transform.linear)[0, 0] == noisy[0, 0]


@pytest.mark.parametrize(
    ("relative", "case"),
    [(0.5 * COUPLING_TOLERANCE, A.AXIS_ALIGNED), (2.0 * COUPLING_TOLERANCE, A.OBLIQUE)],
)
def test_the_tolerance_boundary(relative, case):
    """Relative to the smaller of the entry's row and column maxima."""
    linear = np.diag([4.0, 1.0, 2.0])
    linear[0, 2] = relative * min(4.0, 2.0)  # world z reads a little data x
    assert classify_view(_affine(linear), (1, 2)).alignment is case


def test_a_small_but_real_drift_is_kept():
    """Snapped only below 1e-6 of both the x scale and the t scale."""
    linear = np.eye(4)
    linear[3, 0] = 1e-4
    transform = _affine(linear, "tzyx", "tzyx")
    assert classify_view(transform, (2, 3)).alignment is A.SLICE_SHIFT


def test_the_effective_transform_is_the_same_object_when_nothing_snaps():
    transform = _affine(_rot(1, 2, 30.0))
    for view in VIEWS:
        assert classify_view(transform, view).effective_transform is transform


def test_a_container_snaps_inside_its_blocks(labels_in_a_channel_world):
    affine_block = labels_in_a_channel_world.blocks[1]
    exact = affine_block.transform
    matrix = np.array(exact.matrix)
    # Block rows are world (c, z, y, x); columns data (z, y, x).  World x
    # reads 1e-14 of data y: far below the tolerance.
    matrix[3, 1] = 1e-14
    noisy = AffineTransform(
        transform=matrix,
        input_coordinate_system=exact.input_coordinate_system,
        output_coordinate_system=exact.output_coordinate_system,
        broadcast_axes=exact.broadcast_axes,
    )
    container = ByDimensionTransform(
        blocks=(
            labels_in_a_channel_world.blocks[0],
            affine_block.model_copy(update={"transform": noisy}),
        ),
        input_coordinate_system=labels_in_a_channel_world.input_coordinate_system,
        output_coordinate_system=labels_in_a_channel_world.output_coordinate_system,
    )
    classification = classify_view(container, (3, 4))
    assert classification.alignment is A.AXIS_ALIGNED
    effective = classification.effective_transform
    assert effective is not container
    assert np.count_nonzero(np.asarray(effective.blocks[1].transform.linear)) == 3


# -- helpers and errors ------------------------------------------------------------


def test_data_axes_feeding_reads_rows():
    transform = _affine(_shear(2, 0))
    assert transform.data_axes_feeding((2,)) == (0, 2)
    assert transform.data_axes_feeding((1, 2)) == (0, 1, 2)
    assert transform.data_axes_feeding(()) == ()


def test_the_coupling_of_a_shear_does_not_raise_where_the_correspondence_does():
    transform = _affine(_shear(2, 1))
    with pytest.raises(ValueError, match="one output axis per input axis"):
        transform.axis_correspondence()
    expected = np.eye(3, dtype=bool)
    expected[2, 1] = True
    np.testing.assert_array_equal(transform.axis_coupling(), expected)


@pytest.mark.parametrize(("view", "match"), [((1, 1), "repeat"), ((3,), "3-axis")])
def test_a_bad_view_is_refused(view, match):
    with pytest.raises(ValueError, match=match):
        classify_view(_affine(np.eye(3)), view)
