"""The node matrix stops being ``select_axes`` (design 3.6 site E, 3.9).

The golden baseline proves the composition agrees with ``select_axes`` on
every transform shipping today.  These tests pin *why* the composition is the
right one: the two cases where ``select_axes`` was silently wrong, and the two
things design 3.9 says are easy to get wrong on the way there.
"""

from __future__ import annotations

import pathlib
import re
from uuid import uuid4

import numpy as np
import pytest

from cellier.render._spaces import (
    axis_correspondence,
    build_render_spaces,
    node_matrix,
    pygfx_matrix,
    visual_to_data_transform,
)
from cellier.transform import AffineTransform, VisualCoordinateSystem
from tests._v2 import Context


def _spaces(ctx: Context, retained=None):
    """The systems a visual with these displayed axes would be placed with."""
    retained = sorted(ctx.displayed_axes) if retained is None else list(retained)
    visual_id = uuid4()
    visual = VisualCoordinateSystem.from_data(
        ctx.data, [ctx.data.axes[axis].id for axis in retained], visual_id
    )
    return build_render_spaces(
        ctx.data,
        visual,
        ctx.world,
        ctx.rendered,
        ctx.rendered_to_world,
        ctx.transform,
        retained,
    )


# ---------------------------------------------------------------------------
# Where select_axes was silently wrong
# ---------------------------------------------------------------------------


def test_the_composition_refuses_a_displayed_axis_with_no_data_behind_it():
    """Where ``select_axes`` invented a row, the composition raises.

    Design 3.8 finding 2, and the reason the call sites had to go.
    ``select_axes`` did not raise on a transform whose ranks disagreed: asked
    for "displayed" axes ``(2, 3, 4)`` of a 4-D ``tzyx`` transform it read the
    **homogeneous row** as if it were data and returned a plausible wrong
    matrix.  A second test asserted that misbehaviour directly until Phase 8
    deleted the method with v1; this is the half that outlives it.
    """
    from cellier.render.visuals._image import _displayed_submatrix

    ctx = Context(3, (1.0, 1.0, 1.0), displayed_axes=(0, 1, 2))
    with pytest.raises(ValueError, match="no data axis of this"):
        _displayed_submatrix(ctx.transform, (0, 1, 5))


def test_a_collapsed_axis_index_enters_the_translation():
    """For a block-diagonal transform it contributes nothing to the
    displayed rows, which is exactly why ``select_axes`` got away with
    dropping it.  It stops being free the moment there is a cross-term."""
    ctx = Context(3, (1.0, 1.0, 1.0), displayed_axes=(1, 2), slice_indices={0: 0.0})
    spaces = _spaces(ctx)
    without = node_matrix(spaces, ctx.transform, {0: 0.0})
    with_index = node_matrix(spaces, ctx.transform, {0: 7.0})
    # Diagonal transform: the collapsed axis has no cross-term, so nothing
    # reaches the displayed rows.
    np.testing.assert_array_equal(without, with_index)

    # Now shear world y by data z, the collapsed axis.
    matrix = np.eye(4)
    matrix[1, 0] = 0.5
    sheared = AffineTransform.from_matrix(matrix, ctx.data, ctx.world)
    spaces = build_render_spaces(
        ctx.data,
        VisualCoordinateSystem.from_data(
            ctx.data, [ctx.data.axes[a].id for a in (1, 2)], uuid4()
        ),
        ctx.world,
        ctx.rendered,
        ctx.rendered_to_world,
        ctx.transform,
        (1, 2),
    )
    at_zero = node_matrix(spaces, sheared, {0: 0.0})
    at_seven = node_matrix(spaces, sheared, {0: 7.0})
    assert not np.array_equal(at_zero, at_seven)
    # pygfx order is (x, y): the shear lands in y's translation, 0.5 * 7.
    assert at_seven[1, 3] == pytest.approx(3.5)
    assert at_zero[1, 3] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The two things design 3.9 says are easy to get wrong
# ---------------------------------------------------------------------------


def test_the_visual_system_is_ascending_not_displayed_order():
    """Design 3.14's invariant.  ``axis_selections`` is assembled per data
    axis ascending and numpy returns an ascending array, so the display
    permutation lives in the transform and never in the data."""
    ctx = Context(3, (2.0, 3.0, 4.0), displayed_axes=(2, 1, 0))
    spaces = _spaces(ctx)
    assert spaces.retained_axes == (0, 1, 2)
    assert spaces.visual.axis_names() == ("z", "y", "x")


def test_a_transposed_rendered_system_transposes_the_node_matrix():
    """A transpose costs no refetch: the visual space is unchanged and only
    the rendered system's axis order moves."""
    straight = Context(3, (2.0, 3.0, 4.0), displayed_axes=(0, 1, 2))
    swapped = Context(
        3,
        (2.0, 3.0, 4.0),
        displayed_axes=(0, 2, 1),
        data=straight.data,
        world=straight.world,
    )
    a = node_matrix(_spaces(straight), straight.transform, {})
    b = node_matrix(_spaces(swapped), swapped.transform, {})
    assert not np.array_equal(a, b)
    # A single swap is a reflection, not a rotation: det flips sign.
    assert np.linalg.det(a[:3, :3]) * np.linalg.det(b[:3, :3]) < 0


def test_the_node_matrix_is_narrowed_to_float32():
    """The model layer is float64 and float32 begins at the GPU."""
    ctx = Context(3, (2.0, 3.0, 4.0), displayed_axes=(0, 1, 2))
    assert ctx.transform.matrix.dtype == np.float64
    assert node_matrix(_spaces(ctx), ctx.transform, {}).dtype == np.float32


def test_an_unequal_rank_node_matrix_is_refused():
    """A node matrix maps a visual space onto a rendered space of the same
    rank; anything else means the visual kept an axis the canvas is not
    showing, which used to be read off the homogeneous row instead."""
    ctx = Context(3, (1.0, 1.0, 1.0), displayed_axes=(1, 2), slice_indices={0: 0.0})
    spaces = _spaces(ctx, retained=(0, 1, 2))
    composed = visual_to_data_transform(spaces, {})
    chained = composed.then(ctx.transform, spaces.data, spaces.world).then(
        spaces.world_to_rendered, spaces.world, spaces.rendered
    )
    with pytest.raises(ValueError, match="must be square"):
        pygfx_matrix(chained)


def test_every_collapsed_axis_must_state_where_it_sits():
    ctx = Context(3, (1.0, 1.0, 1.0), displayed_axes=(1, 2), slice_indices={0: 0.0})
    with pytest.raises(ValueError, match="no index in constants"):
        visual_to_data_transform(_spaces(ctx), {})


def test_a_windowed_request_moves_the_origin():
    """Forget the window origin and a brick is drawn at the corner of the
    volume instead of where it belongs."""
    ctx = Context(3, (1.0, 1.0, 1.0), displayed_axes=(0, 1, 2))
    spaces = _spaces(ctx)
    at_origin = node_matrix(spaces, ctx.transform, {})
    windowed = node_matrix(
        spaces, ctx.transform, {}, translation={0: 8.0, 1: 16.0, 2: 24.0}
    )
    # pygfx order reverses (z, y, x) to (x, y, z).
    np.testing.assert_allclose(windowed[:3, 3], [24.0, 16.0, 8.0])
    np.testing.assert_allclose(at_origin[:3, 3], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# The axis correspondence
# ---------------------------------------------------------------------------


def test_the_correspondence_is_read_back_off_the_matrix():
    """The matrix encodes it, so storing an axis_map alongside would be a
    second source of truth."""
    ctx = Context(4, (1.0, 2.0, 3.0, 4.0), displayed_axes=(1, 2, 3))
    assert axis_correspondence(ctx.transform) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_a_sheared_transform_has_no_axis_correspondence():
    ctx = Context(3, (1.0, 1.0, 1.0), displayed_axes=(0, 1, 2))
    matrix = np.eye(4)
    matrix[1, 0] = 0.5
    sheared = AffineTransform.from_matrix(matrix, ctx.data, ctx.world)
    with pytest.raises(ValueError, match="one output axis per input axis"):
        axis_correspondence(sheared)


def test_a_broadcast_axis_simply_has_no_entry():
    """A dataset with fewer axes than its world reaches no output axis along
    the ones it is broadcast over, and that is not an error."""
    from cellier.data._axes import default_data_to_world
    from tests._v2 import systems

    data, _ = systems(3, ("z", "y", "x"))
    _, world = systems(4, ("t", "z", "y", "x"))
    transform = default_data_to_world(data, world)
    assert axis_correspondence(transform) == {0: 1, 1: 2, 2: 3}
    assert len(transform.broadcast_axes) == 1


# ---------------------------------------------------------------------------
# The call sites are gone, not merely guarded
# ---------------------------------------------------------------------------


def test_no_render_visual_calls_select_axes_on_its_own_transform():
    """``select_axes`` on an unequal-rank transform returns silent garbage,
    so the plan requires every call site to be gone rather than guarded.
    The level transforms keep theirs -- those are square by construction and
    are the multiscale phase's to migrate."""
    root = pathlib.Path(__file__).resolve().parents[3]
    pattern = re.compile(r"self\._transform\.select_axes")
    offenders = [
        str(path.relative_to(root))
        for path in (root / "src" / "cellier" / "render").rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert offenders == []
