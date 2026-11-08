"""Tests for cellier.transform._region."""

from functools import partial
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.transform import (
    Axis,
    AxisAlignedBoundingBox,
    ConvexRegion,
    CoordinateSystem,
    HalfSpace,
    WorldCoordinateSystem,
)
from cellier.transform import _geometry_ops as geometry_ops
from cellier.transform._geometry_ops import imap_half_spaces
from cellier.transform._region import half_spaces_from_arrays
from tests.transform._use_cases import CHANNEL_BROADCAST

space = partial(Axis, axis_type="space", unit="micrometer")


def tczyx() -> WorldCoordinateSystem:
    return WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            space(name="Z"),
            space(name="Y"),
            space(name="X"),
        )
    )


def tzyx() -> CoordinateSystem:
    return CoordinateSystem(
        name="data",
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            space(name="z"),
            space(name="y"),
            space(name="x"),
        ),
    )


# --- HalfSpace -------------------------------------------------------


def test_half_space_has_no_inside_field():
    """The sign of the normal carries the side; a slab is two half-spaces."""
    assert "inside" not in HalfSpace.model_fields
    assert "coordinate_system" not in HalfSpace.model_fields


def test_half_space_allows_a_zero_normal():
    """Contrast with Plane, which rejects it: D39 needs both outcomes."""
    vacuous = HalfSpace(normal=[0.0, 0.0], offset=5.0)
    infeasible = HalfSpace(normal=[0.0, 0.0], offset=-5.0)
    assert vacuous.is_vacuous() and not vacuous.is_infeasible()
    assert infeasible.is_infeasible() and not infeasible.is_vacuous()


def test_half_space_rejects_non_finite_values():
    with pytest.raises(ValidationError, match="finite"):
        HalfSpace(normal=[np.inf, 0.0], offset=1.0)
    with pytest.raises(ValidationError, match="finite"):
        HalfSpace(normal=[1.0, 0.0], offset=np.inf)
    with pytest.raises(ValidationError, match="finite"):
        HalfSpace(normal=[np.nan, 0.0], offset=1.0)


def test_half_space_equality_and_hash():
    first = HalfSpace(normal=[1.0, 0.0], offset=5.0)
    second = HalfSpace(normal=[1.0, 0.0], offset=5.0)
    assert first == second
    assert hash(first) == hash(second)
    assert (first == "not a half space") is False


def test_region_equality_delegates_to_half_spaces():
    """ConvexRegion needs nothing extra once HalfSpace has explicit eq."""
    coordinate_system = uuid4()
    half_spaces = (HalfSpace(normal=[1.0, 0.0], offset=5.0),)
    first = ConvexRegion(
        coordinate_system=coordinate_system, ndim=2, half_spaces=half_spaces
    )
    second = ConvexRegion(
        coordinate_system=coordinate_system,
        ndim=2,
        half_spaces=(HalfSpace(normal=[1.0, 0.0], offset=5.0),),
    )
    assert first == second


# --- ConvexRegion rank ------------------------------------------------


def test_unbounded_region_knows_its_rank():
    world = tczyx()
    region = ConvexRegion.unbounded(world)
    assert region.half_spaces == ()
    assert region.ndim == 5
    assert region.coordinate_system == world.id


def test_half_spaces_must_match_the_declared_rank():
    with pytest.raises(ValidationError, match="components"):
        ConvexRegion(
            coordinate_system=uuid4(),
            ndim=3,
            half_spaces=(HalfSpace(normal=[1.0, 0.0], offset=1.0),),
        )


def test_region_is_frozen():
    region = ConvexRegion.unbounded(tczyx())
    with pytest.raises(ValidationError):
        region.ndim = 2


# --- constructors -----------------------------------------------------


def test_from_axis_slabs_leaves_omitted_axes_unbounded():
    world = tczyx()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (2.0, 0.5)})
    bounds = region.bounding_box()
    assert np.array_equal(bounds.min_coordinate, [6.5, 1.5, -np.inf, -np.inf, -np.inf])
    assert np.array_equal(bounds.max_coordinate, [7.5, 2.5, np.inf, np.inf, np.inf])


def test_from_axis_slabs_bounds_a_displayed_axis_just_as_readily():
    """The region has no notion of displayed versus collapsed."""
    world = tczyx()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "Z": (4.0, 2.0)})
    bounds = region.bounding_box()
    assert bounds.min_coordinate[2] == 2.0
    assert bounds.max_coordinate[2] == 6.0


def test_from_axis_slabs_accepts_axis_ids():
    world = tczyx()
    region = ConvexRegion.from_axis_slabs(world, {world.axes[0].id: (7.0, 0.5)})
    assert region.bounding_box().min_coordinate[0] == 6.5


def test_from_axis_slabs_rejects_a_negative_half_thickness():
    with pytest.raises(ValueError, match="negative"):
        ConvexRegion.from_axis_slabs(tczyx(), {"T": (7.0, -1.0)})


def test_from_bounding_box_round_trips():
    world = tczyx()
    box = AxisAlignedBoundingBox(
        coordinate_system=world.id,
        min_coordinate=[0.0, 1.0, 2.0, 3.0, 4.0],
        max_coordinate=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    region = ConvexRegion.from_bounding_box(box)
    assert len(region.half_spaces) == 10
    assert region.bounding_box() == box


def test_from_bounding_box_skips_infinite_bounds():
    world = tczyx()
    box = AxisAlignedBoundingBox(
        coordinate_system=world.id,
        min_coordinate=[0.0, -np.inf, -np.inf, -np.inf, -np.inf],
        max_coordinate=[1.0, np.inf, np.inf, np.inf, np.inf],
    )
    region = ConvexRegion.from_bounding_box(box)
    assert len(region.half_spaces) == 2
    assert region.bounding_box() == box


def test_from_plane_slab_is_thickness_in_distance_units():
    world = tczyx()
    normal = np.array([0.0, 0.0, 1.0, 1.0, 0.0]) * 3.0  # deliberately non-unit
    region = ConvexRegion.from_plane_slab(world, normal, offset=0.0, half_thickness=1.0)
    # a point one unit along the unit normal is on the boundary
    unit = normal / np.linalg.norm(normal)
    assert region.contains(unit * 0.999)
    assert not region.contains(unit * 1.001)


def test_from_plane_slab_rejects_a_zero_normal():
    with pytest.raises(ValueError, match="zero vector"):
        ConvexRegion.from_plane_slab(tczyx(), np.zeros(5), 0.0, 1.0)


def test_intersection_concatenates_constraints():
    world = tczyx()
    first = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5)})
    second = ConvexRegion.from_axis_slabs(world, {"C": (2.0, 0.5)})
    combined = ConvexRegion.intersection(first, second)
    assert len(combined.half_spaces) == 4
    bounds = combined.bounding_box()
    assert np.array_equal(bounds.min_coordinate[:2], [6.5, 1.5])


def test_intersection_rejects_a_foreign_coordinate_system():
    first = ConvexRegion.unbounded(tczyx())
    second = ConvexRegion.unbounded(tczyx())
    with pytest.raises(ValueError, match="different coordinate systems"):
        ConvexRegion.intersection(first, second)


def test_intersection_requires_at_least_one_region():
    with pytest.raises(ValueError, match="at least one"):
        ConvexRegion.intersection()


# --- bounding_box: the fast path -------------------------------------


def test_axis_aligned_bounding_box_runs_no_lp(monkeypatch):
    """The fast path is 1000x, so assert the solver is never called."""

    def explode(*args, **kwargs):
        raise AssertionError("linprog must not be called on an axis-aligned region")

    monkeypatch.setattr(geometry_ops, "linprog", explode)
    region = ConvexRegion.from_axis_slabs(tczyx(), {"T": (7.0, 0.5), "C": (2.0, 0.5)})
    bounds = region.bounding_box()
    assert np.array_equal(bounds.min_coordinate, [6.5, 1.5, -np.inf, -np.inf, -np.inf])


def test_unbounded_region_runs_no_lp_either(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("linprog must not be called")

    monkeypatch.setattr(geometry_ops, "linprog", explode)
    bounds = ConvexRegion.unbounded(tczyx()).bounding_box()
    assert np.all(np.isneginf(bounds.min_coordinate))
    assert np.all(np.isposinf(bounds.max_coordinate))


def test_oblique_bounding_box_uses_the_lp_and_matches_it():
    world = tczyx()
    diagonal = np.array([0.0, 0.0, 1.0, 1.0, 0.0]) / np.sqrt(2)
    region = ConvexRegion.intersection(
        ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (2.0, 0.5)}),
        ConvexRegion.from_plane_slab(world, diagonal, 5.0, 0.5),
        ConvexRegion.from_axis_slabs(world, {"Z": (5.0, 5.0), "Y": (5.0, 5.0)}),
    )
    bounds = region.bounding_box()
    assert np.allclose(bounds.min_coordinate[:4], [6.5, 1.5, 0.0, 0.0])
    assert np.allclose(bounds.max_coordinate[:4], [7.5, 2.5, 7.7781746, 7.7781746])
    assert bounds.min_coordinate[4] == -np.inf
    assert bounds.max_coordinate[4] == np.inf


def test_oblique_bounding_box_really_calls_the_solver():
    """The LP is dead code until obliquity ships, so exercise it explicitly."""
    world = tczyx()
    diagonal = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    region = ConvexRegion.from_plane_slab(world, diagonal, 5.0, 0.5)
    calls = []
    original = geometry_ops.linprog

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    geometry_ops.linprog = counting
    try:
        region.bounding_box()
    finally:
        geometry_ops.linprog = original
    assert len(calls) == 2 * world.ndim


# --- contains, is_empty, simplify ------------------------------------


def test_contains_matches_the_constraints():
    world = tczyx()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5)})
    assert region.contains(np.array([7.0, 0.0, 0.0, 0.0, 0.0]))
    assert not region.contains(np.array([9.0, 0.0, 0.0, 0.0, 0.0]))
    stacked = np.array([[7.0, 0, 0, 0, 0], [9.0, 0, 0, 0, 0]])
    assert list(region.contains(stacked)) == [True, False]


def test_contains_on_an_unbounded_region_is_all_true():
    region = ConvexRegion.unbounded(tczyx())
    assert region.contains(np.zeros((4, 5))).all()


def test_contains_rejects_the_wrong_rank():
    with pytest.raises(ValueError, match="shape"):
        ConvexRegion.unbounded(tczyx()).contains(np.zeros(3))


def test_zero_thickness_region_is_not_special_cased():
    """contains() is not the voxel selection API; bounding_box() plus rounding is."""
    world = tczyx()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.0)})
    bounds = region.bounding_box()
    assert bounds.min_coordinate[0] == bounds.max_coordinate[0] == 7.0
    # the box includes T = 7 exactly, but a point one ulp away is outside
    assert region.contains(np.array([7.0, 0.0, 0.0, 0.0, 0.0]))
    assert not region.contains(np.array([np.nextafter(7.0, 8.0), 0.0, 0.0, 0.0, 0.0]))


def test_empty_region_is_detected_without_raising():
    world = tczyx()
    region = ConvexRegion(
        coordinate_system=world.id,
        ndim=5,
        half_spaces=(
            HalfSpace(normal=[1.0, 0, 0, 0, 0], offset=1.0),
            HalfSpace(normal=[-1.0, 0, 0, 0, 0], offset=-5.0),
        ),
    )
    assert region.is_empty() is True
    with pytest.raises(ValueError, match="empty region"):
        region.bounding_box()


def test_a_region_made_empty_by_an_infeasible_constraint_stays_empty():
    world = tczyx()
    region = ConvexRegion(
        coordinate_system=world.id,
        ndim=5,
        half_spaces=(HalfSpace(normal=np.zeros(5), offset=-5.0),),
    )
    assert region.is_empty()
    assert region.simplify().is_empty()


def test_simplify_drops_vacuous_constraints_but_not_infeasible_ones():
    world = tczyx()
    region = ConvexRegion(
        coordinate_system=world.id,
        ndim=5,
        half_spaces=(
            HalfSpace(normal=[1.0, 0, 0, 0, 0], offset=7.5),
            HalfSpace(normal=np.zeros(5), offset=5.0),
        ),
    )
    assert len(region.simplify().half_spaces) == 1
    assert not region.simplify().is_empty()


def test_simplify_returns_self_when_nothing_is_vacuous():
    region = ConvexRegion.from_axis_slabs(tczyx(), {"T": (7.0, 0.5)})
    assert region.simplify() is region


def _oblique_channel_broadcast_region(world, channel_slab):
    """An oblique channel_broadcast region, with a configurable C slab."""
    diagonal = np.array([0.0, 0.0, 1.0, 1.0, 0.0]) / np.sqrt(2)
    return ConvexRegion.intersection(
        ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": channel_slab}),
        ConvexRegion.from_plane_slab(world, diagonal, 5.0, 0.5),
        ConvexRegion.from_axis_slabs(world, {"Z": (5.0, 5.0), "Y": (5.0, 5.0)}),
    )


def _pull_into_data_space(region, data):
    linear, translation, _ = CHANNEL_BROADCAST
    normals, offsets = imap_half_spaces(
        region.normals, region.offsets, linear, translation
    )
    return ConvexRegion(
        coordinate_system=data.id,
        ndim=4,
        half_spaces=half_spaces_from_arrays(normals, offsets),
    )


def test_simplify_cuts_the_oblique_channel_broadcast_region_to_eight_constraints():
    """Design section 10.3's figure, for a C slab the data actually satisfies.

    Both constraints on the broadcast axis pull back to a zero normal with a
    non-negative offset, so both are vacuous and both are dropped.
    """
    world, data = tczyx(), tzyx()
    region = _oblique_channel_broadcast_region(world, channel_slab=(0.0, 0.5))
    assert len(region.half_spaces) == 10

    pulled = _pull_into_data_space(region, data)
    assert len(pulled.simplify().half_spaces) == 8
    assert not pulled.is_empty()


def test_a_channel_slab_off_the_data_makes_the_pulled_region_empty():
    """The other branch of D39, and where the probe's count came from.

    ``scripts/transform_region_probe.py`` selects ``C`` in ``[1.5, 2.5]``
    and drops every zero-normal row, reaching 8.  One of those rows is
    *infeasible*, not vacuous: the forward matrix places this data at
    ``C == 0``, so the selection matches nothing.  Dropping it would turn an
    empty region into a non-empty one, so ``simplify`` keeps it and the count
    is 9.
    """
    world, data = tczyx(), tzyx()
    region = _oblique_channel_broadcast_region(world, channel_slab=(2.0, 0.5))
    pulled = _pull_into_data_space(region, data)

    zero_normal = [h for h in pulled.half_spaces if not np.any(h.normal != 0.0)]
    assert len(zero_normal) == 2
    assert sum(h.is_vacuous() for h in zero_normal) == 1
    assert sum(h.is_infeasible() for h in zero_normal) == 1

    assert len(pulled.simplify().half_spaces) == 9
    assert pulled.is_empty()
    assert pulled.simplify().is_empty()
