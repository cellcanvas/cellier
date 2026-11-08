"""restrict() and to_affine().

The question the GPU boundary needs answered is not "is this transform
affine" but "with every collapsed axis pinned, is what remains affine".
These pin the arithmetic of the first half of that.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.transform import (
    AffineTransform,
    Axis,
    AxisCoordinates,
    DataCoordinateSystem,
    NonAffineTransformError,
    NonUniformAxisTransform,
    WorldCoordinateSystem,
)


def _space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


@pytest.fixture
def tzyx_to_tczyx():
    """A permuting, scaling, broadcasting transform -- the interesting shape."""
    data = DataCoordinateSystem(
        name="labels",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            _space("z"),
            _space("y"),
            _space("x"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            Axis(name="c", axis_type="channel"),
            _space("z"),
            _space("y"),
            _space("x"),
        )
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"t": 1.0, "z": 2.0, "y": 0.5, "x": 0.5},
        translation={"z": 3.0},
        broadcast_output_axes=("c",),
    )
    return data, world, transform


def test_restricting_agrees_with_mapping_the_full_coordinate(tzyx_to_tczyx):
    """The defining property: pin an axis, or map it and ignore it."""
    data, _, transform = tzyx_to_tczyx
    restricted = transform.restrict({0: 6.0}, data)

    full = transform.map_coordinates(np.array([6.0, 4.0, 8.0, 10.0]))
    reduced = restricted.map_coordinates(np.array([4.0, 8.0, 10.0]))
    assert reduced == pytest.approx(full)


def test_restricting_several_axes_at_once(tzyx_to_tczyx):
    data, _, transform = tzyx_to_tczyx
    restricted = transform.restrict({0: 6.0, 1: 4.0}, data)

    full = transform.map_coordinates(np.array([6.0, 4.0, 8.0, 10.0]))
    assert restricted.map_coordinates(np.array([8.0, 10.0])) == pytest.approx(full)


def test_restricting_by_name_needs_the_coordinate_system(tzyx_to_tczyx):
    data, _, transform = tzyx_to_tczyx
    by_name = transform.restrict({"t": 6.0}, data)
    by_index = transform.restrict({0: 6.0}, data)
    assert by_name.matrix == pytest.approx(by_index.matrix)


def test_a_name_without_a_system_raises(tzyx_to_tczyx):
    _, _, transform = tzyx_to_tczyx
    with pytest.raises(ValueError, match="no input_coordinate_system"):
        transform.restrict({"t": 6.0})


def test_restricting_nothing_is_the_identity(tzyx_to_tczyx):
    _, _, transform = tzyx_to_tczyx
    assert transform.restrict({}) is transform


def test_restricting_every_axis_leaves_a_constant(tzyx_to_tczyx):
    """Nothing left to vary: the result is a pure translation."""
    data, _, transform = tzyx_to_tczyx
    point = np.array([6.0, 4.0, 8.0, 10.0])
    restricted = transform.restrict(dict(enumerate(point)), data)

    assert restricted.input_ndim == 0
    assert restricted.translation == pytest.approx(transform.map_coordinates(point))


def test_the_dropped_axis_leaves_the_domain(tzyx_to_tczyx):
    data, _, transform = tzyx_to_tczyx
    restricted = transform.restrict({0: 6.0}, data)
    assert transform.input_ndim == 4
    assert restricted.input_ndim == 3
    assert restricted.output_ndim == transform.output_ndim


def test_restrict_does_not_round_or_clamp(tzyx_to_tczyx):
    """Exact values: rounding already happened one layer up.

    6.4 must stay 6.4 -- rounding here as well risks the two layers
    disagreeing about which plane a position selects.
    """
    data, _, transform = tzyx_to_tczyx
    restricted = transform.restrict({0: 6.4}, data)
    full = transform.map_coordinates(np.array([6.4, 0.0, 0.0, 0.0]))
    assert restricted.map_coordinates(np.array([0.0, 0.0, 0.0])) == pytest.approx(full)


def test_broadcast_axes_survive_restriction(tzyx_to_tczyx):
    """They name output axes, which restricting the input does not touch."""
    _, _, transform = tzyx_to_tczyx
    restricted = transform.restrict({0: 6.0})
    assert restricted.broadcast_axes == transform.broadcast_axes


def test_an_out_of_range_index_raises(tzyx_to_tczyx):
    _, _, transform = tzyx_to_tczyx
    with pytest.raises(ValueError, match="out of range"):
        transform.restrict({9: 1.0})


def test_an_unknown_name_raises(tzyx_to_tczyx):
    data, _, transform = tzyx_to_tczyx
    with pytest.raises((KeyError, ValueError)):
        transform.restrict({"nope": 1.0}, data)


# -- to_affine ---------------------------------------------------------------


def test_an_affine_transform_is_its_own_affine(tzyx_to_tczyx):
    _, _, transform = tzyx_to_tczyx
    assert transform.to_affine() is transform


def test_a_non_uniform_leaf_is_never_affine():
    data = DataCoordinateSystem(
        name="labels",
        datastore_id=uuid4(),
        axes=(Axis(name="t", axis_type="time", unit="frame"),),
    )
    world = WorldCoordinateSystem(
        axes=(Axis(name="t", axis_type="time", unit="second"),)
    )
    leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=(0.0, 1.0, 4.0)),
        input_coordinate_system=data.id,
        output_coordinate_system=world.id,
    )
    assert leaf.to_affine() is None
    with pytest.raises(NonAffineTransformError):
        leaf.restrict({0: 1.0})


def test_restrict_then_to_affine_is_the_gpu_boundary_shape(tzyx_to_tczyx):
    """What the render layer will do: pin the collapsed axes, ask for a matrix."""
    data, _, transform = tzyx_to_tczyx
    affine = transform.restrict({0: 6.0}, data).to_affine()
    assert affine is not None
    assert affine.matrix.shape == (6, 4)
