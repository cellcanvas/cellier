"""The graph trail's endpoint pull-back.

The trail window used to be converted from world units to data units by
dividing its width by the axis scale.  It is now converted by pulling the
window's two endpoints back through the transform.  The first test here is
the evidence that no existing (affine) scene changed; the rest pin the
behaviour that a scalar scale could not express at all.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.render._spaces import axis_scales
from cellier.render.visuals._graph_memory import _window_half_extents
from cellier.transform import (
    AffineTransform,
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    CoordinateSystem,
    DataCoordinateSystem,
    NonUniformAxisTransform,
    WorldCoordinateSystem,
)

TIMES = tuple(float(t) for t in [0, 1, 2, 3, 4, 4.5, 5.0, 5.5, 6.0, 7, 8, 10, 12])


def _space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


def _systems():
    data = DataCoordinateSystem(
        name="graph",
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
    return data, world


# -- 7c: the equivalence that makes the rewrite safe -------------------------


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0, 7.5])
@pytest.mark.parametrize("offset", [-3.0, 0.0, 11.25])
@pytest.mark.parametrize("window", [0.0, 0.5, 1.0, 4.0])
def test_on_an_affine_axis_the_pullback_equals_dividing_by_the_scale(
    scale, offset, window
):
    """`p - imap(w - b)` is algebraically `b / scale`.

    Asserted numerically across scales, translations and window widths --
    this is the evidence for "nothing in this plan changes the rendered
    appearance of an existing scene", since every existing transform is
    affine.
    """
    data, world = _systems()
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"t": scale, "z": 2.0, "y": 1.0, "x": 1.0},
        translation={"t": offset},
        broadcast_output_axes=("c",),
    )
    positions = {0: 4.0, 1: 3.0, 2: 4.0, 3: 5.0}

    before, after = _window_half_extents(transform, world, positions, 0, window, window)
    expected = window / axis_scales(transform)[0]

    assert before == pytest.approx(expected)
    assert after == pytest.approx(expected)


def test_an_asymmetric_affine_window_also_matches(scale=3.0):
    data, world = _systems()
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"t": scale, "z": 1.0, "y": 1.0, "x": 1.0},
        broadcast_output_axes=("c",),
    )
    positions = {0: 5.0, 1: 0.0, 2: 0.0, 3: 0.0}
    before, after = _window_half_extents(transform, world, positions, 0, 6.0, 1.5)
    assert before == pytest.approx(6.0 / scale)
    assert after == pytest.approx(1.5 / scale)


# -- the non-uniform behaviour a scalar scale cannot express -----------------


@pytest.fixture
def nonuniform():
    data, world = _systems()
    leaf = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=TIMES),
        input_coordinate_system=CoordinateSystem(name="t", axes=(data.axes[0],)).id,
        output_coordinate_system=CoordinateSystem(name="t", axes=(world.axes[0],)).id,
    )
    transform = ByDimensionTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        broadcast_output_axes=("c",),
        axis_transforms={"t": leaf},
    )
    return transform, world


def test_the_designs_default_half_thickness_case(nonuniform):
    """At t = 5.2 s a 0.5 s half-window gives extents (1.0, 1.0)."""
    transform, world = nonuniform
    positions = {0: 6.4, 1: 0.0, 2: 0.0, 3: 0.0}
    before, after = _window_half_extents(transform, world, positions, 0, 0.5, 0.5)
    assert (before, after) == (pytest.approx(1.0), pytest.approx(1.0))
    # The window is [5.4, 7.4] frames, which contains frames 6 and 7.
    assert 6.4 - before == pytest.approx(5.4)
    assert 6.4 + after == pytest.approx(7.4)


def test_a_symmetric_world_window_gives_asymmetric_data_extents(nonuniform):
    """At t = 8.0 s the same 0.5 s half-window gives (0.5, 0.25).

    The previous frame is 1 s back and the next is 2 s forward.  **No scalar
    scale can produce this shape at all**, which is the whole argument for
    the endpoint pull-back.
    """
    transform, world = nonuniform
    positions = {0: 10.0, 1: 0.0, 2: 0.0, 3: 0.0}
    before, after = _window_half_extents(transform, world, positions, 0, 0.5, 0.5)
    assert (before, after) == (pytest.approx(0.5), pytest.approx(0.25))


def test_a_two_second_trail_spans_a_varying_number_of_frames(nonuniform):
    """The headline check: constant physical length, varying node count."""
    transform, world = nonuniform

    # t = 5.2 s -> frames 4, 5, 6 (t = 4, 4.5, 5.0): three nodes.
    before, _ = _window_half_extents(
        transform, world, {0: 6.4, 1: 0.0, 2: 0.0, 3: 0.0}, 0, 2.0, 0.0
    )
    assert 6.4 - before == pytest.approx(3.2)

    # t = 10.0 s -> frames 10, 11 (t = 8, 10): two nodes.
    before, _ = _window_half_extents(
        transform, world, {0: 11.0, 1: 0.0, 2: 0.0, 3: 0.0}, 0, 2.0, 0.0
    )
    assert 11.0 - before == pytest.approx(10.0)


def test_a_window_reaching_off_the_start_clamps_rather_than_vanishing(nonuniform):
    """Interval semantics, not point semantics.

    A 2 s trail at t = 1.0 s reaches back to -1.0 s, which is outside the
    axis.  It must start at the first sample, not report no preimage.
    """
    transform, world = nonuniform
    positions = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    before, _after = _window_half_extents(transform, world, positions, 0, 2.0, 0.0)
    assert np.isfinite(before)
    assert 1.0 - before == pytest.approx(-0.5)  # the axis's own lower edge


def test_a_half_extent_is_never_negative(nonuniform):
    transform, world = nonuniform
    positions = {0: 12.0, 1: 0.0, 2: 0.0, 3: 0.0}
    before, after = _window_half_extents(transform, world, positions, 0, 0.0, 5.0)
    assert before >= 0.0
    assert after >= 0.0
