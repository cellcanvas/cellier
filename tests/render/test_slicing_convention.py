"""Tests for the unified round-half-up slice-index -> voxel convention.

Covers the shared helper (:func:`round_world_to_voxel`) and the invariant that
the in-memory and multiscale planning paths snap a world slice position to the
*same* voxel index at level 0 -- the guard that did not previously exist.

Phase 8 changed how that guard has to be written.  The two paths used to be
two different pieces of arithmetic -- ``_transform_slice_indices`` pulling a
world position back with ``imap_coordinates``, and
``_build_axis_selections_multiscale`` pushing it forward through a precomposed
``world -> level-k`` matrix -- and the point of comparing them was that they
could disagree.  Both are gone.  What remains is one pull-back
(``imap_region``) into one assembler (``axis_selections_from_box``), reached
by every image and label family, so the test now drives that pair and the
"both paths" claim is about the two *stores*, not two implementations.

These are the **labels** expectations, clamping included, and the assembler
every family plans through.  Image visuals run the design 3.2 rule before the
assembler (nearest sample within the thickness, nothing outside the data);
its tests are in ``tests/render/test_image_slicing_rule.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.visuals._slicing import (
    axis_selections_from_box,
    round_world_to_voxel,
)
from cellier.transform import AffineTransform, ConvexRegion
from tests._v2 import systems

# ── round_world_to_voxel ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, size, expected",
    [
        # Exact integers map to themselves.
        (0.0, 10, 0),
        (3.0, 10, 3),
        # Half-integer ties round toward +inf (round-half-up).
        (2.5, 10, 3),
        (0.5, 10, 1),
        (-0.5, 10, 0),  # -> 0 before clamp, stays 0
        # Below/above ties.
        (2.49, 10, 2),
        (2.51, 10, 3),
        # Floating-point jitter around a center.
        (2.4999999, 10, 2),
        (2.5000001, 10, 3),
        # Clamping to valid range.
        (-5.0, 10, 0),
        (100.0, 10, 9),
        (9.4, 10, 9),
    ],
)
def test_round_world_to_voxel(raw, size, expected):
    assert round_world_to_voxel(raw, size) == expected


def test_round_world_to_voxel_returns_python_int():
    result = round_world_to_voxel(np.float64(3.5), 10)
    assert isinstance(result, int)


# ── the one pull-back, and the one assembler ──────────────────────────────
#
# An in-memory store is a single-level pyramid, so "the in-memory answer" and
# "the multiscale answer at level 0" are the same two operations applied to
# the same numbers: pull the region back through ``data -> world``, then
# assemble.  Level ``k`` composes one further ``imap_region`` on top, which
# ``tests/v2/multiscale/test_level_regions.py`` covers.


def _placed(scale, translation=None):
    """A diagonal ``data -> world`` transform, with both its systems in hand."""
    ndim = len(scale)
    data, world = systems(ndim)
    offsets = tuple(translation) if translation is not None else (0.0,) * ndim
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={data.axes[i].id: world.axes[i].id for i in range(ndim)},
        scale={data.axes[i].id: float(scale[i]) for i in range(ndim)},
        translation={data.axes[i].id: float(offsets[i]) for i in range(ndim)},
        name="data_to_world",
    )
    return transform, world


def _plane_box(transform, world, positions):
    """The voxel-space box of a zero-thickness world plane per collapsed axis."""
    region = ConvexRegion.from_axis_slabs(
        world,
        {
            world.axes[axis].id: (float(position), 0.0)
            for axis, position in positions.items()
        },
    )
    return transform.imap_region(region, world).simplify().bounding_box()


@pytest.mark.parametrize(
    "scale, translation",
    [
        ((1.0, 1.0, 1.0), None),  # identity
        ((4.0, 1.0, 1.0), None),  # anisotropic z
        ((2.0, 2.0, 2.0), None),  # isotropic 2x
        ((3.0, 1.0, 1.0), (0.5, 0.0, 0.0)),  # scale + offset
    ],
)
@pytest.mark.parametrize("world_pos", [0, 1, 5, 7, 10, 11])
def test_the_pulled_back_plane_is_the_rounded_voxel(scale, translation, world_pos):
    """Axis 0 sliced, axes (1, 2) displayed.

    The assembler must give exactly ``round_world_to_voxel`` of the pulled-back
    position -- no second rounding rule anywhere on the path.
    """
    store_shape = (12, 8, 8)
    data_to_world, world = _placed(scale, translation)

    box = _plane_box(data_to_world, world, {0: world_pos})
    selections = axis_selections_from_box(box, store_shape)

    offset = 0.0 if translation is None else translation[0]
    expected = round_world_to_voxel((world_pos - offset) / scale[0], store_shape[0])
    assert selections[0] == expected
    assert selections[1:] == ((0, 8), (0, 8))


@pytest.mark.parametrize("world_pos", [3, 4, 5])
def test_half_integer_tie_rounds_up(world_pos):
    """A 2x scale maps odd world positions to half-integer voxel coords.

    world=3 -> data 1.5 -> 2; world=4 -> data 2.0 -> 2; world=5 -> data 2.5 -> 3.
    """
    store_shape = (12, 8, 8)
    data_to_world, world = _placed((2.0, 1.0, 1.0))

    box = _plane_box(data_to_world, world, {0: world_pos})
    selections = axis_selections_from_box(box, store_shape)

    assert selections[0] == round_world_to_voxel(world_pos / 2.0, store_shape[0])


def test_multiple_sliced_axes_agree_4d():
    """Two collapsed axes are rounded independently, each by its own scale."""
    store_shape = (6, 12, 8, 8)
    data_to_world, world = _placed((1.0, 2.0, 1.0, 1.0))

    box = _plane_box(data_to_world, world, {0: 3.0, 1: 5.0})
    selections = axis_selections_from_box(box, store_shape)

    assert selections[0] == round_world_to_voxel(3.0, store_shape[0])
    assert selections[1] == round_world_to_voxel(5.0 / 2.0, store_shape[1])
    assert selections[2:] == ((0, 8), (0, 8))
