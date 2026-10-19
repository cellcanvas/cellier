"""The one assembler that turns a voxel-space box into a datastore selection.

Design 3.7 step 4.  It replaces ``_build_axis_selections_memory`` and, when
the multiscale phase lands, ``_build_axis_selections_multiscale`` -- so the
same three branches decide what every image and label family fetches.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.visuals._slicing import (
    axis_selections_from_box,
    round_world_to_voxel,
)
from cellier.transform import AxisAlignedBoundingBox
from tests._v2 import systems


def _box(low, high):
    data, _ = systems(len(low))
    return AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=np.asarray(low, dtype=float),
        max_coordinate=np.asarray(high, dtype=float),
    )


def test_an_unbounded_axis_becomes_the_full_extent():
    """What a displayed axis looks like until the viewport crop lands, and
    exactly what the old assembler produced."""
    box = _box([-np.inf, -np.inf], [np.inf, np.inf])
    assert axis_selections_from_box(box, (20, 30)) == ((0, 20), (0, 30))


def test_a_collapsed_axis_becomes_one_plane_by_the_unchanged_rule():
    """``round_world_to_voxel`` is the one rule that had to survive this work
    untouched, and the assembler calls it rather than reimplementing it."""
    box = _box([1.5, -np.inf, -np.inf], [1.5, np.inf, np.inf])
    selections = axis_selections_from_box(box, (10, 20, 30))
    assert selections[0] == round_world_to_voxel(1.5, 10) == 2
    assert selections[1:] == ((0, 20), (0, 30))


def test_a_collapsed_axis_is_clamped_to_the_level_shape():
    box = _box([99.0, -np.inf], [99.0, np.inf])
    assert axis_selections_from_box(box, (10, 20))[0] == 9


def test_a_slab_covers_every_voxel_it_touches():
    """The third branch, reached when a thickness is asked for.  Voxel i spans
    ``[i - 0.5, i + 0.5)``, so a slab from 2.5 to 5.5 touches 3, 4, 5 and 6."""
    box = _box([2.5, -np.inf], [5.5, np.inf])
    assert axis_selections_from_box(box, (20, 20))[0] == (3, 7)


def test_a_slab_is_clamped_to_the_level_shape():
    box = _box([-5.0, -np.inf], [3.0, np.inf])
    assert axis_selections_from_box(box, (6, 20))[0] == (0, 4)


def test_a_slab_entirely_outside_the_data_is_empty_rather_than_inverted():
    box = _box([50.0, -np.inf], [60.0, np.inf])
    start, stop = axis_selections_from_box(box, (6, 20))[0]
    assert start == stop == 6


def test_a_half_bounded_axis_raises_and_names_the_shear():
    """D7.  A cross-term on a collapsed axis makes the preimage of the slice
    plane a slanted hyperplane, with no single voxel index on it.  The
    in-memory path had no guard and sliced silently wrong; this is a bug fix,
    not a new restriction."""
    box = _box([2.0, -np.inf], [np.inf, np.inf])
    with pytest.raises(ValueError, match="bounded on one side only"):
        axis_selections_from_box(box, (10, 20))


def test_the_error_points_at_the_guard_that_already_says_this():
    box = _box([-np.inf, -np.inf], [4.0, np.inf])
    with pytest.raises(ValueError, match="classify_view"):
        axis_selections_from_box(box, (10, 20))


def test_an_explicit_window_wins_over_the_box():
    """A multiscale brick's padded window comes from LOD and culling, not from
    the region; the region still decides the collapsed axes."""
    box = _box([1.0, -np.inf, -np.inf], [1.0, np.inf, np.inf])
    selections = axis_selections_from_box(
        box, (10, 20, 30), windows={1: (8, 12), 2: (16, 20)}
    )
    assert selections == (1, (8, 12), (16, 20))
