"""Tests for cellier.render._level_of_detail_2d (2D tile LOD selection).

Pure NumPy.  Mirrors the 3D LOD tests for the 2D tiled-image pipeline:
grid construction, zoom-based level selection, distance sorting, and the
viewport (2D frustum) cull.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._level_of_detail_2d import (
    build_tile_grids_2d,
    select_lod_2d,
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
)
from cellier.render.lut_indirection._layout_2d import BlockLayout2D

BLOCK_SIZE = 8


@pytest.fixture
def base_layout():
    return BlockLayout2D.from_shape((32, 32), block_size=BLOCK_SIZE, overlap=1)


@pytest.fixture
def level_shapes():
    return [(32, 32), (16, 16)]


# ---------------------------------------------------------------------------
# build_tile_grids_2d
# ---------------------------------------------------------------------------


def test_build_tile_grids_with_level_shapes(base_layout, level_shapes):
    grids = build_tile_grids_2d(base_layout, 2, level_shapes=level_shapes)
    assert len(grids) == 2
    # level 1: 32/8 = 4 -> 16 tiles; columns [level, gy, gx]
    assert grids[0]["arr"].shape == (16, 3)
    assert np.all(grids[0]["arr"][:, 0] == 1)
    # level 2: 16/8 = 2 -> 4 tiles
    assert grids[1]["arr"].shape == (4, 3)
    assert np.all(grids[1]["arr"][:, 0] == 2)


def test_build_tile_grids_power_of_two_fallback(base_layout):
    grids = build_tile_grids_2d(base_layout, 2)
    assert grids[0]["arr"].shape == (16, 3)  # 4x4
    assert grids[1]["arr"].shape == (4, 3)  # 2x2


def test_build_tile_grids_isotropic_centres(base_layout, level_shapes):
    grids = build_tile_grids_2d(base_layout, 2, level_shapes=level_shapes)
    arr = grids[0]["arr"]
    centres = grids[0]["centres"]
    gy, gx = arr[:, 1], arr[:, 2]
    # x <- W (gx), y <- H (gy).  Centre convention (plan v2, D1): tile g
    # covers level-0 voxels [8g, 8g + 8), centred on (g + 0.5) * 8 - 0.5.
    np.testing.assert_allclose(centres[:, 0], (gx + 0.5) * 8 - 0.5)
    np.testing.assert_allclose(centres[:, 1], (gy + 0.5) * 8 - 0.5)


def test_build_tile_grids_anisotropic_centres(base_layout, level_shapes):
    scale = [np.array([2.0, 3.0]), np.array([4.0, 6.0])]  # (x=W, y=H)
    translation = [np.array([10.0, 20.0]), np.zeros(2)]
    grids = build_tile_grids_2d(
        base_layout,
        2,
        level_shapes=level_shapes,
        scale_vecs_shader=scale,
        translation_vecs_shader=translation,
    )
    arr = grids[0]["arr"]
    centres = grids[0]["centres"]
    gy, gx = arr[:, 1], arr[:, 2]
    # s * ((g + 0.5) * 8 - 0.5) + t
    np.testing.assert_allclose(centres[:, 0], 2 * ((gx + 0.5) * 8 - 0.5) + 10.0)
    np.testing.assert_allclose(centres[:, 1], 3 * ((gy + 0.5) * 8 - 0.5) + 20.0)


# ---------------------------------------------------------------------------
# select_lod_2d
# ---------------------------------------------------------------------------


def test_select_lod_force_level_clamps(base_layout):
    grids = build_tile_grids_2d(base_layout, 2)
    # force below 1 clamps to 1
    low = select_lod_2d(grids, 2, 100.0, 100.0, force_level=0)
    assert np.all(low[:, 0] == 1)
    # force above n_levels clamps to n_levels
    high = select_lod_2d(grids, 2, 100.0, 100.0, force_level=5)
    assert np.all(high[:, 0] == 2)
    # returns a copy, not the cached array
    assert high is not grids[1]["arr"]


def test_select_lod_degenerate_inputs_return_finest(base_layout):
    grids = build_tile_grids_2d(base_layout, 2)
    for arr in (
        select_lod_2d(grids, 2, viewport_width_px=100.0, voxel_width=0.0),
        select_lod_2d(grids, 2, viewport_width_px=0.0, voxel_width=100.0),
    ):
        assert np.all(arr[:, 0] == 1)


def test_select_lod_log2_fallback(base_layout):
    grids = build_tile_grids_2d(base_layout, 2)
    # biased = voxel_width / viewport = 1 -> ideal = 1 -> level 1
    fine = select_lod_2d(grids, 2, viewport_width_px=100.0, voxel_width=100.0)
    assert np.all(fine[:, 0] == 1)
    # biased = 2 -> ideal = 1 + log2(2) = 2 -> level 2
    coarse = select_lod_2d(grids, 2, viewport_width_px=100.0, voxel_width=200.0)
    assert np.all(coarse[:, 0] == 2)


def test_select_lod_scale_factor_path(base_layout):
    grids = build_tile_grids_2d(base_layout, 2)
    lsf = [1.0, 2.0]  # geometric-mean threshold = sqrt(2) ~ 1.414
    # biased = 1 < 1.414 -> level 1
    fine = select_lod_2d(grids, 2, 100.0, 100.0, level_scale_factors=lsf)
    assert np.all(fine[:, 0] == 1)
    # biased = 2 >= 1.414 -> level 2
    coarse = select_lod_2d(grids, 2, 100.0, 200.0, level_scale_factors=lsf)
    assert np.all(coarse[:, 0] == 2)


# ---------------------------------------------------------------------------
# sort_tiles_by_distance_2d
# ---------------------------------------------------------------------------


def test_sort_tiles_empty():
    arr = np.empty((0, 3), dtype=np.int32)
    out = sort_tiles_by_distance_2d(arr, np.zeros(3), BLOCK_SIZE)
    assert out.shape == (0, 3)


def test_sort_tiles_nearest_first_isotropic():
    arr = np.array(
        [
            [1, 0, 4],
            [1, 0, 0],
            [1, 0, 2],
        ],
        dtype=np.int32,
    )
    out = sort_tiles_by_distance_2d(arr, np.zeros(3), BLOCK_SIZE)
    np.testing.assert_array_equal(out[:, 2], [0, 2, 4])  # gx ascending


def test_sort_tiles_anisotropic_reorders():
    arr = np.array(
        [
            [1, 3, 0],  # far along y
            [1, 0, 2],  # near along x
        ],
        dtype=np.int32,
    )
    cam = np.zeros(3)
    iso = sort_tiles_by_distance_2d(arr, cam, BLOCK_SIZE)
    scale = np.array([[1.0, 0.1]])  # squash y (H axis)
    translation = np.zeros((1, 2))
    aniso = sort_tiles_by_distance_2d(
        arr,
        cam,
        BLOCK_SIZE,
        level_scale_arr_shader=scale,
        level_translation_arr_shader=translation,
    )
    assert not np.array_equal(iso[0], aniso[0])


# ---------------------------------------------------------------------------
# viewport_cull_2d
# ---------------------------------------------------------------------------


def test_viewport_cull_empty():
    arr = np.empty((0, 3), dtype=np.int32)
    culled, n = viewport_cull_2d(arr, BLOCK_SIZE, np.zeros(2), np.array([16.0, 16.0]))
    assert len(culled) == 0
    assert n == 0


def test_viewport_cull_removes_outside_tiles():
    arr = np.array(
        [
            [1, 5, 5],  # x[40,48], y[40,48]: outside
            [1, 0, 0],  # x[0,8], y[0,8]
            [1, 1, 1],  # x[8,16], y[8,16]: on the edge
        ],
        dtype=np.int32,
    )
    culled, n = viewport_cull_2d(
        arr, BLOCK_SIZE, np.array([0.0, 0.0]), np.array([16.0, 16.0])
    )
    assert n == 1
    # Load order is kept.
    np.testing.assert_array_equal(culled, arr[1:])


def test_viewport_cull_nothing_removed():
    arr = np.array([[1, 0, 0]], dtype=np.int32)
    culled, n = viewport_cull_2d(
        arr, BLOCK_SIZE, np.array([0.0, 0.0]), np.array([100.0, 100.0])
    )
    assert n == 0
    np.testing.assert_array_equal(culled, arr)


def test_viewport_cull_anisotropic():
    # with a 10x x-scale, tile g1=1 spans x[80,160] and is culled by a small
    # viewport that would keep it under isotropic spacing
    arr = np.array([[1, 0, 1]], dtype=np.int32)
    scale = np.array([[10.0, 1.0]])
    translation = np.zeros((1, 2))
    culled, n = viewport_cull_2d(
        arr,
        BLOCK_SIZE,
        np.array([0.0, 0.0]),
        np.array([16.0, 16.0]),
        level_scale_arr_shader=scale,
        level_translation_arr_shader=translation,
    )
    assert n == 1
    assert len(culled) == 0
