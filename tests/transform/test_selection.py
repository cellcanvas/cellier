"""Tests for cellier.transform._selection (D43)."""

from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.transform import (
    AffineTransform,
    Axis,
    ConvexRegion,
    RegionSelection,
    RenderedCoordinateSystem,
    WorldCoordinateSystem,
)


def space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


def tczyx():
    return WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            space("Z"),
            space("Y"),
            space("X"),
        )
    )


def embedding(world, slices=None):
    """The D34 rendered -> world embedding, displaying ZYX."""
    slices = {"T": 7.0, "C": 2.0} if slices is None else slices
    rendered = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4())
    return AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={"Z": "Z", "Y": "Y", "X": "X"},
        constant_output_axes=slices,
    )


def selection(world=None, slices=None, slabs=None):
    world = world or tczyx()
    slabs = {"T": (7.0, 0.5), "C": (2.0, 0.5)} if slabs is None else slabs
    return RegionSelection(
        transform=embedding(world, slices),
        region=ConvexRegion.from_axis_slabs(world, slabs),
    )


def test_a_consistent_selection_validates():
    result = selection()
    assert result.transform.output_ndim == 5
    assert result.region.ndim == 5


def test_the_slice_position_is_the_translation_column_d35():
    result = selection()
    assert np.allclose(result.slice_position, [7.0, 2.0, 0.0, 0.0, 0.0])
    assert np.allclose(result.slice_position, result.transform.translation)


def test_a_slice_outside_its_own_region_is_rejected_d43():
    """The invariant that connects the two halves."""
    world = tczyx()
    with pytest.raises(ValidationError, match="would select nothing"):
        RegionSelection(
            transform=embedding(world, slices={"T": 99.0, "C": 2.0}),
            region=ConvexRegion.from_axis_slabs(
                world, {"T": (7.0, 0.5), "C": (2.0, 0.5)}
            ),
        )


def test_a_region_in_the_wrong_coordinate_system_is_rejected():
    world = tczyx()
    other = tczyx()
    with pytest.raises(ValidationError, match="output coordinate system"):
        RegionSelection(
            transform=embedding(world),
            region=ConvexRegion.unbounded(other),
        )


def test_a_region_of_the_wrong_rank_is_rejected():
    world = tczyx()
    mismatched = ConvexRegion(coordinate_system=world.id, ndim=3, half_spaces=())
    with pytest.raises(ValidationError, match="rank"):
        RegionSelection(transform=embedding(world), region=mismatched)


def test_an_unbounded_region_is_accepted():
    world = tczyx()
    assert RegionSelection(
        transform=embedding(world), region=ConvexRegion.unbounded(world)
    )


def test_a_zero_thickness_selection_is_accepted_d42():
    """The slice sits exactly on the boundary, which contains() includes."""
    assert selection(slabs={"T": (7.0, 0.0), "C": (2.0, 0.0)})


def test_an_oblique_selection_is_the_same_type_r4():
    world = tczyx()
    diagonal = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    result = RegionSelection(
        transform=embedding(world, slices={"T": 7.0, "C": 2.0}),
        region=ConvexRegion.intersection(
            ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (2.0, 0.5)}),
            ConvexRegion.from_plane_slab(world, diagonal, 0.0, 1.0),
        ),
    )
    assert isinstance(result.region, ConvexRegion)


def test_selection_is_frozen():
    result = selection()
    with pytest.raises(ValidationError):
        result.region = ConvexRegion.unbounded(tczyx())


def test_selection_round_trips():
    result = selection()
    back = RegionSelection.model_validate(result.model_dump())
    assert back.transform == result.transform
    assert back.region == result.region


def test_there_is_no_dims_editor_here_d43():
    """This phase fixes the artifact, not the editor that emits it."""
    import cellier.transform as package

    assert not hasattr(package, "DimsManager")
    assert not hasattr(RegionSelection, "from_indices")


def test_a_bounded_displayed_axis_is_accepted_r3():
    """R3 permits bounding a displayed axis; the invariant must allow it.

    D43 states the invariant as "the transform's translation satisfies
    every half-space".  The translation is zero on a displayed axis, so
    that literal reading would reject this selection even though every
    rendered point with ``Z`` in ``[11, 13]`` lands in the region.
    """
    world = tczyx()
    assert RegionSelection(
        transform=embedding(world),
        region=ConvexRegion.from_axis_slabs(
            world, {"T": (7.0, 0.5), "C": (2.0, 0.5), "Z": (12.0, 1.0)}
        ),
    )


def test_an_embedding_with_broadcast_axes_is_rejected_d8():
    """The validator resolves D8's drop rule to "nothing to drop".

    It holds axis **ids** and never the world ``CoordinateSystem``
    object, so it cannot resolve ``broadcast_axes`` to indices.  That is
    fine because a ``rendered -> world`` embedding records where the
    slice sits as ``constant_output_axes`` and never as a broadcast axis
    (D34/D35) -- but the assumption is checked rather than assumed.
    """
    world = tczyx()
    rendered = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4())
    broadcasting = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={"Z": "Z", "Y": "Y", "X": "X"},
        broadcast_output_axes=["T", "C"],
    )
    with pytest.raises(ValidationError, match="must not declare broadcast_axes"):
        RegionSelection(
            transform=broadcasting,
            region=ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5)}),
        )


def test_a_selection_that_reaches_nothing_is_still_rejected():
    """The generalisation must not weaken the check D43 was written for."""
    world = tczyx()
    with pytest.raises(ValidationError, match="would select nothing"):
        RegionSelection(
            transform=embedding(world, slices={"T": 99.0, "C": 2.0}),
            region=ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5)}),
        )


# -- with_planes_at_slice ----------------------------------------------

#: A slice position whose slab midpoint ``((p - 1) + (p + 1)) / 2`` is one ulp
#: above ``p``: the value the H4 hand check's slider landed on.
_ROUNDING_POSITION = 31.3796992481203


def test_the_slab_midpoint_misses_the_rounding_position():
    """The premise of the tests below: rebuilding ``p`` from bounds can miss."""
    p = _ROUNDING_POSITION
    assert ((p - 1.0) + (p + 1.0)) / 2.0 != p


def test_a_plane_at_the_slab_midpoint_selects_nothing():
    """Why the plane must come from the translation, not the bounds."""
    world = tczyx()
    p = _ROUNDING_POSITION
    box = selection(
        world, slices={"T": p, "C": 2.0}, slabs={"T": (p, 1.0)}
    ).region.bounding_box()
    midpoint = float(box.min_coordinate[0] + box.max_coordinate[0]) / 2.0
    with pytest.raises(ValidationError, match="would select nothing"):
        RegionSelection(
            transform=embedding(world, slices={"T": p, "C": 2.0}),
            region=ConvexRegion.from_axis_slabs(world, {"T": (midpoint, 0.0)}),
        )


def test_a_slab_narrows_to_a_plane_at_the_translation():
    world = tczyx()
    p = _ROUNDING_POSITION
    narrowed = selection(
        world, slices={"T": p, "C": 2.0}, slabs={"T": (p, 1.0)}
    ).with_planes_at_slice([0])
    box = narrowed.region.bounding_box()
    assert box.min_coordinate[0] == p
    assert box.max_coordinate[0] == p
    assert narrowed.slice_position[0] == p


def test_constraints_on_other_axes_are_kept():
    world = tczyx()
    original = selection(
        world,
        slices={"T": 7.0, "C": 2.0},
        slabs={"T": (7.0, 0.5), "C": (2.0, 0.5), "Z": (3.0, 1.0)},
    )
    narrowed = original.with_planes_at_slice([1])
    kept = [
        half_space
        for half_space in original.region.half_spaces
        if half_space.normal[1] == 0
    ]
    assert set(kept) <= set(narrowed.region.half_spaces)
    assert len(narrowed.region.half_spaces) == len(kept) + 2
    box = narrowed.region.bounding_box()
    assert box.min_coordinate[1] == box.max_coordinate[1] == 2.0
    assert (box.min_coordinate[2], box.max_coordinate[2]) == (2.0, 4.0)


def test_no_axes_returns_the_same_selection():
    original = selection()
    assert original.with_planes_at_slice([]) is original


def test_a_displayed_axis_has_no_plane_to_narrow_to():
    with pytest.raises(ValueError, match="is displayed"):
        selection().with_planes_at_slice([2])


def test_an_out_of_range_axis_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        selection().with_planes_at_slice([5])
