"""``Plane.from_point_normal`` and the helpers added for clipping planes."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.transform import Axis, CoordinateSystem, Plane


def _system(names="tzyx") -> CoordinateSystem:
    return CoordinateSystem(
        name="data", axes=tuple(Axis(name=n, axis_type="space") for n in names)
    )


def test_from_point_normal_on_every_axis():
    system = _system("zyx")
    plane = Plane.from_point_normal(system, point=(1, 2, 3), normal=(0, 0, 2))
    assert plane.coordinate_system == system.id
    np.testing.assert_array_equal(plane.normal, [0, 0, 2])
    assert plane.offset == 6.0


def test_named_axes_leave_the_others_unconstrained():
    system = _system("tzyx")
    plane = Plane.from_point_normal(
        system, point=(40, 0, 0), normal=(1, 0, 0), axes=("z", "y", "x")
    )
    np.testing.assert_array_equal(plane.normal, [0, 1, 0, 0])
    assert plane.offset == 40.0


def test_axes_are_taken_in_the_order_given():
    system = _system("zyx")
    plane = Plane.from_point_normal(
        system, point=(7, 5), normal=(3, 2), axes=("x", "z")
    )
    np.testing.assert_array_equal(plane.normal, [2, 0, 3])
    assert plane.offset == 3 * 7 + 2 * 5


@pytest.mark.parametrize(
    ("point", "normal", "axes", "match"),
    [
        ((1, 2), (1, 0, 0), None, "same length"),
        ((1, 2), (1, 0), None, "pass axes="),
        ((1, 2, 3), (1, 0, 0), ("z", "y"), "names 2 axes"),
        ((1, 2), (1, 0), ("z", "z"), "more than once"),
    ],
)
def test_from_point_normal_refuses_mismatched_input(point, normal, axes, match):
    with pytest.raises(ValueError, match=match):
        Plane.from_point_normal(_system("zyx"), point, normal, axes)


def test_signed_distance_is_positive_along_the_normal():
    plane = Plane.from_point_normal(_system("zyx"), (0, 0, 10), (0, 0, 2))
    np.testing.assert_allclose(
        plane.signed_distance([[0, 0, 13], [5, 5, 10], [0, 0, 4]]), [3, 0, -6]
    )
    assert plane.signed_distance([0, 0, 11]) == pytest.approx(1.0)


def test_closest_point_lies_on_the_plane():
    plane = Plane.from_point_normal(_system("zyx"), (1, 2, 3), (1, 1, 0))
    closest = plane.closest_point_to((10, 0, 7))
    assert closest @ plane.normal == pytest.approx(plane.offset)
    # Moved along the normal only.
    np.testing.assert_allclose(
        np.cross(closest - (10, 0, 7), plane.normal), 0, atol=1e-12
    )


def test_the_normal_cannot_be_edited_in_place():
    plane = Plane.from_point_normal(_system("zyx"), (0, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="read-only"):
        plane.normal[0] = 1.0
