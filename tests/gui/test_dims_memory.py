"""The 2D/3D toggle's memory and its defaults (``cellier.gui._dims``).

``plans/displayed_axes_required.md`` sections 4.1 and 5.2.
"""

from __future__ import annotations

import pytest

from cellier.gui._dims import DisplayedAxesMemory, default_toggle_axes

# ---------------------------------------------------------------------------
# DisplayedAxesMemory
# ---------------------------------------------------------------------------


def test_the_defaults_are_used_until_a_mode_is_seen():
    memory = DisplayedAxesMemory((1, 2), (0, 1, 2))

    assert memory.axes_2d == (1, 2)
    assert memory.axes_3d == (0, 1, 2)
    assert memory.toggle_target((1, 2)) == (0, 1, 2)
    assert memory.toggle_target((0, 1, 2)) == (1, 2)


def test_a_round_trip_returns_to_the_2d_set_it_left():
    memory = DisplayedAxesMemory((1, 2), (0, 1, 2))
    memory.observe((0, 1))

    target = memory.toggle_target((0, 1))
    memory.observe(target)

    assert target == (0, 1, 2)
    assert memory.toggle_target(target) == (0, 1)


def test_the_3d_set_is_remembered_too():
    memory = DisplayedAxesMemory((2, 3), (1, 2, 3))
    memory.observe((0, 2, 3))
    memory.observe((2, 3))

    assert memory.toggle_target((2, 3)) == (0, 2, 3)


def test_the_order_of_a_remembered_set_is_kept():
    memory = DisplayedAxesMemory((1, 2), (0, 1, 2))
    memory.observe((2, 1))
    memory.observe((2, 0, 1))

    assert memory.toggle_target((0, 1, 2)) == (2, 1)
    assert memory.toggle_target((2, 1)) == (2, 0, 1)


def test_observing_twice_changes_nothing():
    memory = DisplayedAxesMemory((1, 2), (0, 1, 2))
    memory.observe((0, 1))
    memory.observe((0, 1))

    assert memory.axes_2d == (0, 1)
    assert memory.axes_3d == (0, 1, 2)


def test_a_remembered_2d_set_outside_the_3d_view_is_not_used():
    """Time must not come back on screen from a ``(z, y, x)`` view."""
    memory = DisplayedAxesMemory((2, 3), (1, 2, 3))
    memory.observe((0, 3))

    assert memory.toggle_target((1, 2, 3)) == (2, 3)
    # It is still remembered: nothing was observed to replace it.
    assert memory.axes_2d == (0, 3)


def test_the_last_two_of_the_3d_view_when_no_2d_set_fits():
    memory = DisplayedAxesMemory((2, 3), (1, 2, 3))
    memory.observe((2, 3))

    assert memory.toggle_target((0, 1, 2)) == (1, 2)
    assert memory.toggle_target((2, 0, 1)) == (1, 2)


def test_a_set_of_another_rank_is_ignored():
    memory = DisplayedAxesMemory((1, 2), (0, 1, 2))
    memory.observe(())
    memory.observe((0,))

    assert memory.axes_2d == (1, 2)
    assert memory.axes_3d == (0, 1, 2)


# ---------------------------------------------------------------------------
# default_toggle_axes
# ---------------------------------------------------------------------------

SPACE3 = ("space", "space", "space")
TZYX = ("time", "space", "space", "space")


@pytest.mark.parametrize(
    ("displayed", "axis_types", "expected"),
    [
        # 2D stated: add the last remaining space axis, result in world order.
        ((0, 1), SPACE3, ((0, 1), (0, 1, 2))),
        ((1, 2), SPACE3, ((1, 2), (0, 1, 2))),
        ((2, 1), SPACE3, ((2, 1), (0, 1, 2))),
        ((2, 3), TZYX, ((2, 3), (1, 2, 3))),
        # Several space axes remain: the last in world order (D13).
        ((2, 3), ("space", "space", "space", "space"), ((2, 3), (1, 2, 3))),
        ((0, 1), ("space", "space", "space", "space"), ((0, 1), (0, 1, 3))),
        ((1, 2), ("space", "space", "space", "time"), ((1, 2), (0, 1, 2))),
        # No space axis remains: the last remaining axis of any type (D8).
        ((1, 2), ("time", "space", "space"), ((1, 2), (0, 1, 2))),
        ((2, 3), ("time", "channel", "space", "space"), ((2, 3), (1, 2, 3))),
        # 3D stated: the last two in world order.
        ((0, 1, 2), SPACE3, ((1, 2), (0, 1, 2))),
        ((2, 0, 1), SPACE3, ((1, 2), (2, 0, 1))),
        ((0, 2, 3), TZYX, ((2, 3), (0, 2, 3))),
    ],
)
def test_default_toggle_axes(displayed, axis_types, expected):
    assert default_toggle_axes(displayed, axis_types) == expected


def test_a_two_axis_world_has_no_3d_set():
    assert default_toggle_axes((0, 1), ("space", "space")) is None


def test_a_displayed_set_of_another_rank_has_no_toggle():
    assert default_toggle_axes((), SPACE3) is None
