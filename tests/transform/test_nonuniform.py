"""The non-uniform axis leaf.

The running example throughout is a table of thirteen frames whose world
times are unevenly spaced, densely sampled in the middle and sparse at the
end.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier._rounding import round_half_up_clamped
from cellier.render.visuals._slicing import round_world_to_voxel
from cellier.transform import (
    Axis,
    AxisAlignedBoundingBox,
    AxisCoordinates,
    DataCoordinateSystem,
    NonAffineTransformError,
    NonUniformAxisTransform,
    WorldCoordinateSystem,
)

TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0)


@pytest.fixture
def systems():
    data = DataCoordinateSystem(
        name="labels",
        datastore_id=uuid4(),
        axes=(Axis(name="t", axis_type="time", unit="frame"),),
    )
    world = WorldCoordinateSystem(
        axes=(Axis(name="t", axis_type="time", unit="second"),)
    )
    return data, world


@pytest.fixture
def leaf(systems):
    data, world = systems
    return NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=TIMES),
        input_coordinate_system=data.id,
        output_coordinate_system=world.id,
    )


# -- the table ---------------------------------------------------------------


def test_edges_default_to_half_a_gap():
    """The default that gives the end samples a full catchment.

    Asymmetric by construction: the first gap is 1 s and the last is 2 s.
    """
    assert AxisCoordinates(values=TIMES).resolved_edges == (-0.5, 13.0)


def test_default_edges_give_every_sample_a_full_catchment():
    """The point of extrapolating rather than stopping at the end samples."""
    coordinates = AxisCoordinates(values=TIMES)
    low, high = coordinates.resolved_edges
    table = np.asarray(TIMES)
    midpoints = (table[:-1] + table[1:]) / 2.0
    bounds = np.concatenate([[low], midpoints, [high]])
    widths = np.diff(bounds)
    # The first sample's catchment equals the first gap, and the last
    # equals the last gap -- neither is halved.
    assert widths[0] == pytest.approx(TIMES[1] - TIMES[0])
    assert widths[-1] == pytest.approx(TIMES[-1] - TIMES[-2])


def test_index_domain_matches_the_store_edge_convention():
    """A leaf reports its span the way a gridded store reports its own."""
    assert AxisCoordinates(values=TIMES).index_domain == (-0.5, 12.5)


def test_explicit_edges_are_kept():
    coordinates = AxisCoordinates(values=TIMES, edges=(-2.0, 20.0))
    assert coordinates.resolved_edges == (-2.0, 20.0)


def test_edges_on_the_end_samples_are_allowed():
    """The closed-domain choice: end samples get a half catchment."""
    assert AxisCoordinates(values=TIMES, edges=(0.0, 12.0)).resolved_edges == (
        0.0,
        12.0,
    )


def test_non_monotonic_values_raise():
    with pytest.raises(ValueError, match="strictly increasing"):
        AxisCoordinates(values=(0.0, 2.0, 1.0))


def test_repeated_values_raise():
    with pytest.raises(ValueError, match="strictly increasing"):
        AxisCoordinates(values=(0.0, 1.0, 1.0))


def test_edges_narrower_than_the_table_raise():
    """A narrower span would make samples unreachable -- silent data loss."""
    with pytest.raises(ValueError, match="must contain values"):
        AxisCoordinates(values=TIMES, edges=(2.0, 12.0))


def test_single_value_without_edges_raises():
    with pytest.raises(ValueError, match="no gap to extrapolate"):
        AxisCoordinates(values=(4.0,))


def test_single_value_with_edges_is_fine():
    assert AxisCoordinates(values=(4.0,), edges=(3.5, 4.5)).resolved_edges == (
        3.5,
        4.5,
    )


def test_empty_values_raise():
    with pytest.raises(ValueError, match="at least one value"):
        AxisCoordinates(values=())


# -- point mapping -----------------------------------------------------------


def test_exact_hit_at_every_table_entry(leaf):
    indices = np.arange(len(TIMES), dtype=float)
    assert leaf.map_coordinates(indices[:, np.newaxis]).ravel() == pytest.approx(
        np.asarray(TIMES)
    )


def test_round_trip_on_fractional_positions(leaf):
    positions = np.array([[0.25], [4.5], [6.4], [10.75]])
    world = leaf.map_coordinates(positions)
    assert leaf.imap_coordinates(world) == pytest.approx(positions)


def test_the_designs_worked_example(leaf):
    """5.2 s pulls back to frame 6.4."""
    assert leaf.imap_coordinates(np.array([5.2]))[0] == pytest.approx(6.4)
    assert leaf.map_coordinates(np.array([6.4]))[0] == pytest.approx(5.2)


def test_the_edges_map_to_the_ends_of_the_span(leaf):
    assert leaf.map_coordinates(np.array([-0.5]))[0] == pytest.approx(-0.5)
    assert leaf.map_coordinates(np.array([12.5]))[0] == pytest.approx(13.0)


def test_out_of_range_is_nan_not_clamped(leaf):
    """The headline change: past the data there is nothing, not the last frame."""
    assert np.isnan(leaf.imap_coordinates(np.array([15.0]))[0])
    assert np.isnan(leaf.imap_coordinates(np.array([-3.0]))[0])
    assert np.isnan(leaf.map_coordinates(np.array([20.0]))[0])
    assert np.isnan(leaf.map_coordinates(np.array([-1.0]))[0])


def test_a_mixed_array_maps_per_point(leaf):
    """Why nan rather than an exception: one call may span the boundary."""
    mapped = leaf.imap_coordinates(np.array([[5.2], [99.0]]))
    assert mapped[0, 0] == pytest.approx(6.4)
    assert np.isnan(mapped[1, 0])


def test_integer_input_is_not_truncated(leaf):
    """The wrapped GridInterpolation would return 4 here; this returns 4.5."""
    assert leaf.map_coordinates(np.array([[5]]))[0, 0] == pytest.approx(4.5)


def test_rank_is_preserved(leaf):
    assert leaf.map_coordinates(np.array([6.0])).shape == (1,)
    assert leaf.map_coordinates(np.array([[6.0], [7.0]])).shape == (2, 1)


# -- interpolation mode ------------------------------------------------------


def test_linear_and_nearest_differ_at_a_fractional_position(systems):
    """Test where the modes actually differ -- at a table entry they cannot."""
    data, world = systems
    common = {
        "coordinates": AxisCoordinates(values=TIMES),
        "input_coordinate_system": data.id,
        "output_coordinate_system": world.id,
    }
    linear = NonUniformAxisTransform(interpolation="linear", **common)
    nearest = NonUniformAxisTransform(interpolation="nearest", **common)

    assert linear.map_coordinates(np.array([6.4]))[0] == pytest.approx(5.2)
    assert nearest.map_coordinates(np.array([6.4]))[0] == pytest.approx(5.0)

    assert linear.imap_coordinates(np.array([5.2]))[0] == pytest.approx(6.4)
    assert nearest.imap_coordinates(np.array([5.2]))[0] == pytest.approx(6.0)


def test_the_modes_agree_at_every_table_entry(systems):
    """Which is why a test only at sample positions cannot tell them apart."""
    data, world = systems
    common = {
        "coordinates": AxisCoordinates(values=TIMES),
        "input_coordinate_system": data.id,
        "output_coordinate_system": world.id,
    }
    linear = NonUniformAxisTransform(interpolation="linear", **common)
    nearest = NonUniformAxisTransform(interpolation="nearest", **common)
    indices = np.arange(len(TIMES), dtype=float)[:, np.newaxis]
    assert linear.map_coordinates(indices) == pytest.approx(
        nearest.map_coordinates(indices)
    )


# -- interval semantics ------------------------------------------------------


def _box(system_id, low, high):
    return AxisAlignedBoundingBox(
        coordinate_system=system_id,
        min_coordinate=np.array([low]),
        max_coordinate=np.array([high]),
    )


def test_an_interval_maps_through_its_endpoints(leaf, systems):
    """Monotonic, so the endpoints are the answer -- exact, not conservative."""
    _, world = systems
    mapped = leaf.imap_bounding_box(_box(world.id, 4.7, 5.7))
    assert mapped.min_coordinate[0] == pytest.approx(5.4)
    assert mapped.max_coordinate[0] == pytest.approx(7.4)


def test_an_overhanging_interval_is_clipped_not_rejected(leaf, systems):
    """A trail window reaching past the first sample still starts at it.

    This is the point-versus-interval distinction: the same position that
    gives nan from imap_coordinates gives a clipped bound here.
    """
    _, world = systems
    mapped = leaf.imap_bounding_box(_box(world.id, -8.0, 2.0))
    assert mapped.min_coordinate[0] == pytest.approx(-0.5)
    assert mapped.max_coordinate[0] == pytest.approx(2.0)
    assert np.isnan(leaf.imap_coordinates(np.array([-8.0]))[0])


def test_an_unbounded_axis_stays_unbounded(leaf, systems):
    """Unbounded means 'not constrained', not 'reaches the edge of the data'."""
    _, world = systems
    mapped = leaf.imap_bounding_box(_box(world.id, -np.inf, np.inf))
    assert np.isneginf(mapped.min_coordinate[0])
    assert np.isposinf(mapped.max_coordinate[0])


def test_an_unbounded_side_stays_unbounded_in_both_directions(leaf, systems):
    """Each side keeps its own infinity; the finite side still maps exactly."""
    labels, world = systems
    back = leaf.imap_bounding_box(_box(world.id, -np.inf, 2.0))
    assert np.isneginf(back.min_coordinate[0])
    assert back.max_coordinate[0] == pytest.approx(2.0)
    forward = leaf.map_bounding_box(_box(labels.id, 2.0, np.inf), world)
    assert forward.min_coordinate[0] == pytest.approx(2.0)
    assert np.isposinf(forward.max_coordinate[0])


def test_a_wrong_rank_box_raises(leaf, systems):
    _, world = systems
    box = AxisAlignedBoundingBox(
        coordinate_system=world.id,
        min_coordinate=np.array([0.0, 0.0]),
        max_coordinate=np.array([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="1-D bounding box"):
        leaf.imap_bounding_box(box)


# -- the refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    ["map_direction", "imap_direction", "map_normal", "imap_normal"],
)
def test_vector_operations_refuse(leaf, operation):
    with pytest.raises(NonAffineTransformError, match="no single Jacobian"):
        getattr(leaf, operation)(np.array([1.0]))


def test_restrict_points_at_the_container(leaf):
    with pytest.raises(NonAffineTransformError, match="ByDimensionTransform"):
        leaf.restrict({"t": 5.2})


def test_never_affine_and_never_inverted(leaf):
    assert leaf.to_affine() is None
    assert leaf.inverse() is None
    assert leaf.transform.to_affine() is None


# -- equality ----------------------------------------------------------------


def test_two_leaves_over_the_same_table_are_equal(systems):
    """The wrapped transformnd classes compare by identity; this must not."""
    data, world = systems
    common = {
        "input_coordinate_system": data.id,
        "output_coordinate_system": world.id,
    }
    first = NonUniformAxisTransform(coordinates=AxisCoordinates(values=TIMES), **common)
    second = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=TIMES), **common
    )
    assert first == second
    assert hash(first) == hash(second)


def test_a_different_table_is_not_equal(systems):
    data, world = systems
    common = {
        "input_coordinate_system": data.id,
        "output_coordinate_system": world.id,
    }
    first = NonUniformAxisTransform(coordinates=AxisCoordinates(values=TIMES), **common)
    second = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=TIMES[:-1]), **common
    )
    assert first != second


def test_equality_is_total_against_a_foreign_type(leaf):
    """A raising or partial __eq__ would poison every containing model."""
    assert (leaf == "not a transform") is False


# -- nearest sample in world units --------------------------------------------


def test_linear_inverse_plus_round_half_up_is_nearest_in_world_units(leaf):
    """Rounding the linear inverse half-up picks the nearest frame in time.

    This is what lets the image path keep round_world_to_voxel unchanged on
    a non-uniform axis.
    """
    table = np.asarray(TIMES)
    for world_position in np.arange(TIMES[0], TIMES[-1] + 1e-9, 0.01):
        raw = leaf.imap_coordinates(np.array([world_position]))[0]
        via_inverse = round_world_to_voxel(float(raw), len(TIMES))
        distances = np.abs(table - world_position)
        nearest = int(np.flatnonzero(distances == distances.min())[-1])
        assert via_inverse == nearest, world_position


def test_midpoints_round_half_up_to_the_later_sample(leaf):
    table = np.asarray(TIMES)
    midpoints = (table[:-1] + table[1:]) / 2.0
    for expected, world_position in enumerate(midpoints, start=1):
        raw = leaf.imap_coordinates(np.array([world_position]))[0]
        assert round_world_to_voxel(float(raw), len(TIMES)) == expected


def test_the_shared_rounding_primitive_is_the_same_rule():
    """round_world_to_voxel became a wrapper; the two must not drift."""
    for raw in (-4.0, -0.5, 0.0, 3.49, 3.5, 12.4, 99.0):
        assert round_world_to_voxel(raw, 13) == round_half_up_clamped(raw, 13)
