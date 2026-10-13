"""Tests for the axis-keyed orientation of a 3D camera fit."""

from uuid import uuid4

import pytest

from cellier.scene import spatial_axes, world_coordinate_system
from cellier.scene._fit_orientation import (
    DEFAULT_FIT_UP_3D,
    DEFAULT_FIT_VIEW_DIRECTION_3D,
    check_not_parallel,
    normalize_axis_vector,
    resolve_fit_orientation,
)

T, Z, Y, X = (uuid4() for _ in range(4))


def test_normalize_keys_by_axis_id_and_drops_zeros():
    world = world_coordinate_system(spatial_axes("z", "y", "x"))
    z, y, _x = (axis.id for axis in world.axes)
    assert normalize_axis_vector(world, {"z": -1, y: 2.0, "x": 0.0}, "up") == {
        z: -1.0,
        y: 2.0,
    }
    assert normalize_axis_vector(world, None, "up") is None


def test_normalize_refuses_an_axis_given_by_name_and_by_id():
    world = world_coordinate_system(spatial_axes("z", "y", "x"))
    with pytest.raises(ValueError, match="more than once"):
        normalize_axis_vector(world, {"z": 1.0, world.axes[0].id: 1.0}, "up")


def test_parallel_vectors_are_refused_in_either_sense():
    check_not_parallel({Z: 1.0}, {Y: 1.0})
    check_not_parallel({Z: 1.0}, None)
    with pytest.raises(ValueError, match="parallel"):
        check_not_parallel({Z: 1.0, Y: 1.0}, {Z: -3.0, Y: -3.0})


def test_vectors_are_placed_in_displayed_order():
    view, up = resolve_fit_orientation(
        [{Z: 1.0, X: -2.0}], [{Y: 3.0}], displayed_axis_ids=(X, Y, Z)
    )
    assert view == (-2.0, 0.0, 1.0)
    assert up == (0.0, 3.0, 0.0)


def test_no_vectors_gives_the_default():
    assert resolve_fit_orientation([None], [None], (Z, Y, X)) == (
        DEFAULT_FIT_VIEW_DIRECTION_3D,
        DEFAULT_FIT_UP_3D,
    )


def test_a_vector_on_an_axis_that_is_not_displayed_is_not_projected():
    view, up = resolve_fit_orientation(
        [{Z: 1.0, Y: -1.0, X: -1.0}], [{Y: 1.0}], displayed_axis_ids=(T, Y, X)
    )
    # The view has a z component, so all of it is dropped; up is on y, kept.
    assert view == DEFAULT_FIT_VIEW_DIRECTION_3D
    assert up == (0.0, 1.0, 0.0)


def test_the_first_usable_candidate_wins():
    view, up = resolve_fit_orientation(
        [{T: 1.0}, {X: 1.0}], [None, {Y: 1.0}], displayed_axis_ids=(Z, Y, X)
    )
    assert view == (0.0, 0.0, 1.0)
    assert up == (0.0, 1.0, 0.0)


def test_a_view_parallel_to_the_default_up_gives_the_default_pair():
    assert resolve_fit_orientation([{Z: -1.0}], [None], (Z, Y, X)) == (
        DEFAULT_FIT_VIEW_DIRECTION_3D,
        DEFAULT_FIT_UP_3D,
    )


def test_a_2d_display_is_refused():
    with pytest.raises(ValueError, match="three displayed axes"):
        resolve_fit_orientation([None], [None], (Y, X))
