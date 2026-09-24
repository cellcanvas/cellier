"""``cellier.render._level_mapping``: the level <-> data mapping helpers.

Checked against the brute-force reference (``_level_reference``) and against
hand-worked boxes for the three pyramid kinds the renderer supports: block
averaging ``t = (s - 1) / 2``, offset striding ``t = s // 2`` and plain
striding ``t = 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._level_mapping import (
    base_cell_range,
    brick_box_data,
    brick_centre_data,
    data_to_level,
    implied_power_of_two,
    level_extent_data,
    level_to_data,
    level_voxel_extent,
)
from tests.render._level_reference import data_to_level as reference_data_to_level
from tests.render._level_reference import sample_nearest

SCALE = np.array([2.0, 4.0, 3.0])
KINDS = {
    "block_average": (SCALE - 1.0) / 2.0,
    "offset_striding": np.floor(SCALE / 2.0),
    "plain_striding": np.zeros(3),
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_round_trip_and_reference(kind):
    translation = KINDS[kind]
    rng = np.random.default_rng(0)
    p = rng.uniform(-5.0, 50.0, size=(100, 3))
    u = data_to_level(p, SCALE, translation)
    np.testing.assert_allclose(level_to_data(u, SCALE, translation), p)
    np.testing.assert_allclose(u, reference_data_to_level(p, SCALE, translation))


@pytest.mark.parametrize("kind", list(KINDS))
def test_voxel_centre_is_its_data_position(kind):
    """Coarse voxel ``i`` is centred on ``s * i + t``; its extent is ``s`` wide."""
    translation = KINDS[kind]
    index = np.array([[0, 0, 0], [3, 1, 5]])
    low, high = level_voxel_extent(index, SCALE, translation)
    np.testing.assert_allclose((low + high) / 2.0, SCALE * index + translation)
    np.testing.assert_allclose(high - low, np.broadcast_to(SCALE, index.shape))


def test_block_average_voxel_covers_its_block():
    """Block averaging: coarse voxel ``i`` covers level-0 voxels ``s*i .. s*i+s-1``."""
    low, high = level_voxel_extent([1, 1, 1], SCALE, KINDS["block_average"])
    np.testing.assert_allclose(low, SCALE * 1 - 0.5)
    np.testing.assert_allclose(high, SCALE * 2 - 0.5)


def test_brick_boxes_tile_the_level():
    """Adjacent bricks share a face, and brick 0 starts at voxel 0's low edge."""
    translation = KINDS["offset_striding"]
    bricks = np.array([[0, 0, 0], [1, 2, 3]])
    low, high = brick_box_data(bricks, 8, SCALE, translation)
    np.testing.assert_allclose(
        low[0], level_voxel_extent([0, 0, 0], SCALE, translation)[0]
    )
    next_low, _ = brick_box_data(bricks + 1, 8, SCALE, translation)
    np.testing.assert_allclose(next_low, high)
    np.testing.assert_allclose(high - low, np.broadcast_to(8 * SCALE, bricks.shape))


def test_brick_box_per_axis_block_size():
    low, high = brick_box_data([[1, 1]], [4, 8], [2.0, 1.0], [0.5, 0.0])
    np.testing.assert_allclose(low, [[2.0 * 3.5 + 0.5, 7.5]])
    np.testing.assert_allclose(high, [[2.0 * 7.5 + 0.5, 15.5]])


def test_level_extent():
    low, high = level_extent_data([10, 4], [2.0, 4.0], [0.5, 1.5])
    np.testing.assert_allclose(low, [-0.5, -0.5])
    np.testing.assert_allclose(high, [2.0 * 9.5 + 0.5, 4.0 * 3.5 + 1.5])


def test_nearest_reference_on_an_offset_strided_level():
    """Offset striding: the level holds level-0 voxels ``s//2 + s*i``."""
    level0 = np.arange(24, dtype=np.int32)
    s, t = 4.0, 2.0
    level = level0[2::4]
    # Every level-0 voxel centre maps to the coarse voxel whose block holds it.
    p = np.arange(24, dtype=float)[:, None]
    got = sample_nearest(level, p, [s], [t])
    expected = level[np.clip(np.floor((np.arange(24) - t) / s + 0.5), 0, 5).astype(int)]
    np.testing.assert_array_equal(got, expected)
    # Each coarse voxel's own centre reads its own value.
    np.testing.assert_array_equal(
        sample_nearest(level, (s * np.arange(6) + t)[:, None], [s], [t]), level
    )


def test_brick_centre_is_the_box_midpoint():
    bricks = np.array([[0, 1, 2], [3, 0, 1]])
    low, high = brick_box_data(bricks, [8, 4, 2], SCALE, KINDS["offset_striding"])
    np.testing.assert_allclose(
        brick_centre_data(bricks, [8, 4, 2], SCALE, KINDS["offset_striding"]),
        (low + high) / 2.0,
    )


def test_implied_power_of_two_is_block_averaging():
    scale, translation = implied_power_of_two([1, 2, 3])
    np.testing.assert_allclose(scale, [1.0, 2.0, 4.0])
    np.testing.assert_allclose(translation, [0.0, 0.5, 1.5])


def test_base_cell_range():
    """Cell c covers data [c*bs - 0.5, (c+1)*bs - 0.5): 7.4 is still cell 0.

    Unclamped: -3.0 lies in cell -1.
    """
    start, stop = base_cell_range([7.4, -3.0], [7.6, 8.0], 8)
    np.testing.assert_array_equal(start, [0, -1])
    np.testing.assert_array_equal(stop, [2, 2])
    start, stop = base_cell_range([7.5], [15.4], 8)
    np.testing.assert_array_equal(start, [1])
    np.testing.assert_array_equal(stop, [2])


def test_planning_helpers_agree_on_centres_and_boxes():
    """build_*_grids, the sorts and the culls all place bricks the same way."""
    from cellier.render._frustum import compute_brick_aabb_corners
    from cellier.render._level_of_detail import build_level_grids, sort_arr_by_distance
    from cellier.render._level_of_detail_2d import build_tile_grids_2d, viewport_cull_2d
    from cellier.render.block_cache import BlockKey3D
    from cellier.render.lut_indirection import BlockLayout3D

    scales = [np.ones(3), np.array([2.0, 4.0, 3.0])]
    translations = [np.zeros(3), np.array([0.5, 2.0, 1.0])]
    layout = BlockLayout3D(volume_shape=(24, 32, 32), block_size=8)
    grids = build_level_grids(
        layout, 2, scales, translations, level_shapes=[(24, 32, 32), (8, 8, 16)]
    )
    arr, centres = grids[1]["arr"], grids[1]["centres"]
    index = arr[:, [3, 2, 1]]
    np.testing.assert_allclose(
        centres, brick_centre_data(index, 8, scales[1], translations[1])
    )
    # The sort ranks by the same centres: the nearest to a centre is that brick.
    for row in (0, len(arr) - 1):
        ranked = sort_arr_by_distance(arr, centres[row], 8, scales, translations)
        np.testing.assert_array_equal(ranked[0], arr[row])
    # The frustum's AABB is the brick box.
    key = BlockKey3D(
        level=2, g0=int(arr[-1, 1]), g1=int(arr[-1, 2]), g2=int(arr[-1, 3])
    )
    corners = compute_brick_aabb_corners(
        key, 8, np.stack(scales), np.stack(translations)
    )
    low, high = brick_box_data(index[-1], 8, scales[1], translations[1])
    np.testing.assert_allclose(corners.min(axis=0), low)
    np.testing.assert_allclose(corners.max(axis=0), high)

    # 2D: a viewport around one tile's centre keeps exactly the tiles whose
    # boxes overlap it.
    from cellier.render.lut_indirection._layout_2d import BlockLayout2D

    layout_2d = BlockLayout2D.from_shape(shape=(32, 32), block_size=8, overlap=1)
    s2, t2 = [np.ones(2), np.array([2.0, 4.0])], [np.zeros(2), np.array([0.5, 1.5])]
    tiles = build_tile_grids_2d(
        layout_2d,
        2,
        level_shapes=[(32, 32), (8, 16)],
        scale_vecs_shader=s2,
        translation_vecs_shader=t2,
    )[1]
    centre = tiles["centres"][0]
    kept, _ = viewport_cull_2d(
        tiles["arr"], 8, centre - 0.1, centre + 0.1, np.stack(s2), np.stack(t2)
    )
    np.testing.assert_array_equal(kept, tiles["arr"][:1])
