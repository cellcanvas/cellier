"""The affine boundary and the out-of-domain check.

Two things are pinned here: that a *sliced* non-uniform axis costs nothing
at the GPU boundary, and that a visual past its extent shows nothing rather
than pinning its last plane.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.render._spaces import affine_for_node, visual_covers_position
from cellier.transform import (
    AffineTransform,
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    CoordinateSystem,
    DataCoordinateSystem,
    NonAffineTransformError,
    NonUniformAxisTransform,
    WorldCoordinateSystem,
)

TIMES = tuple(float(t) for t in [0, 1, 2, 3, 4, 4.5, 5.0, 5.5, 6.0, 7, 8, 10, 12])


def _space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


@pytest.fixture
def scene():
    labels = DataCoordinateSystem(
        name="labels",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="frame"),
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
    return labels, world


@pytest.fixture
def nonuniform(scene):
    labels, world = scene
    leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=TIMES),
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


# -- the GPU boundary --------------------------------------------------------


def test_a_sliced_non_uniform_axis_still_yields_a_matrix(nonuniform):
    """The GPU still gets a matrix: T is collapsed, so the check never fires."""
    affine = affine_for_node(nonuniform, {0: 6.4})
    assert affine.matrix.shape == (6, 5)
    # The T row is now the constant the leaf evaluated to, and its column is
    # zero: the caller feeds that axis the same pinned value.
    assert affine.translation[0] == pytest.approx(5.2)
    assert affine.linear[:, 0] == pytest.approx(np.zeros(5))


def test_it_agrees_with_the_full_transform_on_the_pinned_plane(nonuniform):
    point = np.array([6.4, 3.0, 4.0, 5.0])
    affine = affine_for_node(nonuniform, {0: 6.4})
    assert affine.map_coordinates(point) == pytest.approx(
        nonuniform.map_coordinates(point)
    )


def test_displaying_a_non_uniform_axis_raises_and_names_it(nonuniform):
    """No 4x4 matrix expresses it, so the honest outcome is a refusal."""
    with pytest.raises(NonAffineTransformError, match="stop displaying axis 't'"):
        affine_for_node(nonuniform, {1: 3.0})


def test_an_affine_transform_passes_through_untouched(scene):
    """Nothing about an existing scene changes."""
    labels, world = scene
    affine = AffineTransform.from_axis_map(
        labels,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0},
        broadcast_output_axes=("c",),
    )
    assert affine_for_node(affine, {0: 6.0}) is affine


# -- the out-of-domain check -------------------------------------------------


def test_inside_the_extent_is_covered():
    assert visual_covers_position(((-0.5, 12.5), (-0.5, 31.5)), {0: 6.0})


def test_past_the_end_is_not_covered():
    """The motivating case: frames end, so there is nothing beyond them."""
    assert not visual_covers_position(((-0.5, 9.5),), {0: 11.0})


def test_before_the_start_is_not_covered():
    assert not visual_covers_position(((-0.5, 9.5),), {0: -2.0})


def test_the_edges_themselves_are_covered():
    assert visual_covers_position(((-0.5, 9.5),), {0: -0.5})
    assert visual_covers_position(((-0.5, 9.5),), {0: 9.5})


def test_an_empty_store_covers_nothing():
    assert not visual_covers_position(None, {0: 0.0})


def test_no_collapsed_axes_is_covered():
    assert visual_covers_position(((-0.5, 9.5),), {})


def test_a_position_with_no_preimage_is_not_covered():
    """nan means the transform reported no preimage on a bounded axis."""
    assert not visual_covers_position(((-0.5, 9.5),), {0: float("nan")})


def test_only_the_axes_it_is_given_are_checked():
    """Displayed axes are ranges, not positions, and are simply absent."""
    assert visual_covers_position(((-0.5, 2.5), (-0.5, 63.5)), {0: 1.0})
