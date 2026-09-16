"""The image slicing rule: nearest sample within the thickness (design 3.2).

:func:`select_plane_within_slab` decides which one sample an image draws on a
sliced axis, or that it draws nothing.  At zero thickness inside the data it
must agree with :func:`round_world_to_voxel`, ties included; outside the data
it returns ``None`` where the labels assembler clamps.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.render.visuals._slicing import (
    round_world_to_voxel,
    select_plane_within_slab,
)
from cellier.transform import AxisCoordinates, NonUniformAxisTransform


def _affine(scale: float, offset: float):
    """``world = scale * index + offset`` and its inverse, as 1-D maps."""

    def index_to_world(indices):
        return scale * np.asarray(indices, dtype=float) + offset

    def world_to_index(values):
        return (np.asarray(values, dtype=float) - offset) / scale

    return index_to_world, world_to_index


def _select(position, half_thickness, size, scale=1.0, offset=0.0):
    index_to_world, world_to_index = _affine(scale, offset)
    return select_plane_within_slab(
        position, half_thickness, size, index_to_world, world_to_index
    )


_SIZE = 10


@pytest.mark.parametrize(
    "scale, offset",
    [(1.0, 0.0), (2.0, 0.0), (0.5, 0.25), (3.0, 10.0), (-1.0, 0.0), (-2.0, 5.0)],
)
def test_zero_thickness_matches_round_world_to_voxel_inside_the_data(scale, offset):
    """Every quarter-voxel position in ``[-0.5, size - 0.5)``, ties included."""
    for raw in np.arange(-0.5, _SIZE - 0.5, 0.25):
        position = scale * raw + offset
        expected = round_world_to_voxel(float(raw), _SIZE)
        assert _select(position, 0.0, _SIZE, scale, offset) == expected, raw


@pytest.mark.parametrize("scale", [-1.0, -2.0])
def test_a_negative_world_scale_still_breaks_ties_toward_the_higher_index(scale):
    # data 2.5 is the tie between samples 2 and 3.
    assert _select(scale * 2.5, 0.0, _SIZE, scale) == 3


@pytest.mark.parametrize(
    "raw",
    [-0.6, -5.0, _SIZE - 0.5, _SIZE, 100.0],
)
@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (2.0, 3.0), (-1.5, 0.0)])
def test_outside_the_data_draws_nothing(raw, scale, offset):
    """``[-0.5, size - 0.5)`` is half open: the upper edge is outside."""
    assert _select(scale * raw + offset, 0.0, _SIZE, scale, offset) is None


@pytest.mark.parametrize(
    "position, half_thickness, expected",
    [
        # Inside the data the nearest sample is the one containing the position.
        (4.2, 2.0, 4),
        (4.5, 1.0, 5),  # a tie still goes to the higher index
        # Past the end, the band reaching the end sample picks it.
        (12.0, 3.0, 9),
        (-2.0, 1.5, 0),  # band [-3.5, -0.5] touches sample 0's closed edge
        # Past the end, a band that stops short draws nothing.
        (-2.0, 1.4, None),
        (11.0, 1.5, None),  # band [9.5, 12.5] touches only sample 9's open edge
    ],
)
def test_a_slab_picks_the_nearest_overlapping_sample(
    position, half_thickness, expected
):
    assert _select(position, half_thickness, _SIZE) == expected


def test_an_empty_axis_draws_nothing():
    assert _select(0.0, 1.0, 0) is None


def _time_axis(values, interpolation="linear"):
    return NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=values),
        input_coordinate_system=uuid4(),
        output_coordinate_system=uuid4(),
        interpolation=interpolation,
    )


_TIMES = (0.0, 1.0, 2.5, 4.0, 10.0)


def _time_maps(leaf):
    def index_to_world(indices):
        return leaf.map_coordinates(np.asarray(indices, dtype=float)[:, None])[:, 0]

    def world_to_index(values):
        return leaf.imap_coordinates(np.asarray(values, dtype=float)[:, None])[:, 0]

    return index_to_world, world_to_index


@pytest.mark.parametrize(
    "position, expected",
    [
        (1.7, 1),  # nearer 1.0 than 2.5, though index-space rounding says 2
        (1.75, 2),  # the world tie between 1.0 and 2.5 goes to the higher index
        (6.9, 3),
        (7.0, 4),  # the world tie between 4.0 and 10.0
        (12.9, 4),
        (13.0, None),  # the upper outer edge is outside
        (-0.6, None),
    ],
)
def test_a_non_uniform_axis_picks_the_nearest_time(position, expected):
    index_to_world, world_to_index = _time_maps(_time_axis(_TIMES))
    assert (
        select_plane_within_slab(
            position, 0.0, len(_TIMES), index_to_world, world_to_index
        )
        == expected
    )


def test_a_non_uniform_axis_agrees_with_its_own_nearest_rule():
    """The transform's ``"nearest"`` inverse is the nearest time with ties up.

    It includes the upper outer edge, which the image rule excludes, so the
    sweep stops just short of it.
    """
    linear = _time_axis(_TIMES)
    nearest = _time_axis(_TIMES, interpolation="nearest")
    index_to_world, world_to_index = _time_maps(linear)
    low, high = linear.coordinates.resolved_edges
    positions = np.linspace(low, high, 523, endpoint=False)
    expected = nearest.imap_coordinates(positions[:, None])[:, 0]
    for position, index in zip(positions, expected, strict=True):
        chosen = select_plane_within_slab(
            float(position), 0.0, len(_TIMES), index_to_world, world_to_index
        )
        assert chosen == int(index), position
