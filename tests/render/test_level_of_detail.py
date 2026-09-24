"""Tests for cellier.render._level_of_detail (3D brick LOD selection).

Pure NumPy.  These lock down the DHW/XYZ axis conventions and the
per-level anisotropic scale/translation handling shared between
``build_level_grids`` and ``sort_arr_by_distance``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._level_of_detail import (
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.render.lut_indirection import BlockLayout3D

BLOCK_SIZE = 8


@pytest.fixture
def base_layout():
    return BlockLayout3D(volume_shape=(32, 32, 32), block_size=BLOCK_SIZE)


@pytest.fixture
def scale_vecs():
    # shader order (x=W, y=H, z=D); level 2 is 2x downsampled.
    return [np.ones(3), np.full(3, 2.0)]


@pytest.fixture
def translation_vecs():
    return [np.zeros(3), np.zeros(3)]


@pytest.fixture
def level_shapes():
    return [(32, 32, 32), (16, 16, 16)]


# ---------------------------------------------------------------------------
# build_level_grids
# ---------------------------------------------------------------------------


def test_build_level_grids_with_level_shapes(
    base_layout, scale_vecs, translation_vecs, level_shapes
):
    grids = build_level_grids(
        base_layout, 2, scale_vecs, translation_vecs, level_shapes=level_shapes
    )
    assert len(grids) == 2

    # level 1: 32/8 = 4 bricks per axis -> 64 rows
    g1 = grids[0]
    assert g1["arr"].shape == (64, 4)
    # columns are [level, gz, gy, gx]
    assert np.all(g1["arr"][:, 0] == 1)
    np.testing.assert_array_equal(g1["half_extents"], [4.0, 4.0, 4.0])

    # level 2: 16/8 = 2 bricks per axis -> 8 rows, half extents doubled
    g2 = grids[1]
    assert g2["arr"].shape == (8, 4)
    assert np.all(g2["arr"][:, 0] == 2)
    np.testing.assert_array_equal(g2["half_extents"], [8.0, 8.0, 8.0])


def test_build_level_grids_centres_formula_and_axis_order(
    base_layout, scale_vecs, translation_vecs, level_shapes
):
    translation_vecs = [np.array([100.0, 200.0, 300.0]), np.zeros(3)]
    grids = build_level_grids(
        base_layout, 2, scale_vecs, translation_vecs, level_shapes=level_shapes
    )
    arr = grids[0]["arr"]
    centres = grids[0]["centres"]
    gz, gy, gx = arr[:, 1], arr[:, 2], arr[:, 3]
    # centre[x] uses gx (W), centre[z] uses gz (D).  Centre convention (plan
    # v2, D1): brick g covers level voxels [8g, 8g + 8), centred on
    # s * ((g + 0.5) * 8 - 0.5) + t, here with s = 1.
    np.testing.assert_allclose(centres[:, 0], (gx + 0.5) * 8 - 0.5 + 100.0)
    np.testing.assert_allclose(centres[:, 1], (gy + 0.5) * 8 - 0.5 + 200.0)
    np.testing.assert_allclose(centres[:, 2], (gz + 0.5) * 8 - 0.5 + 300.0)


def test_build_level_grids_power_of_two_fallback_halves_dims(
    base_layout, scale_vecs, translation_vecs
):
    grids = build_level_grids(base_layout, 2, scale_vecs, translation_vecs)
    # no level_shapes -> grid dims halve per level: 4 -> 2
    assert grids[0]["arr"].shape == (64, 4)  # 4^3
    assert grids[1]["arr"].shape == (8, 4)  # 2^3


# ---------------------------------------------------------------------------
# select_levels_from_cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_levels", [1, 2, 3])
def test_select_levels_no_bands_all_finest(
    base_layout, scale_vecs, translation_vecs, n_levels
):
    # No thresholds and no base_layout -> every brick uses the finest level,
    # for any n_levels.  (Regression: n_levels >= 2 used to raise IndexError.)
    scale = [np.full(3, float(2**k)) for k in range(n_levels)]
    translation = [np.zeros(3)] * n_levels
    grids = build_level_grids(base_layout, n_levels, scale, translation)
    out = select_levels_from_cache(
        grids, n_levels, camera_pos=np.array([0.0, 0.0, 0.0])
    )
    assert np.all(out[:, 0] == 1)
    assert out.shape[0] == grids[0]["arr"].shape[0]


def test_select_levels_empty_thresholds_all_finest(
    base_layout, scale_vecs, translation_vecs
):
    # An explicit empty threshold list behaves the same as None.
    grids = build_level_grids(base_layout, 2, scale_vecs, translation_vecs)
    out = select_levels_from_cache(grids, 2, camera_pos=np.zeros(3), thresholds=[])
    assert np.all(out[:, 0] == 1)
    assert out.shape[0] == grids[0]["arr"].shape[0]


def test_select_levels_default_thresholds_match_world_diagonal(
    base_layout, scale_vecs, translation_vecs, level_shapes
):
    grids = build_level_grids(
        base_layout, 2, scale_vecs, translation_vecs, level_shapes=level_shapes
    )
    cam = np.array([50.0, 50.0, 50.0])
    from_default = select_levels_from_cache(
        grids, 2, cam, thresholds=None, base_layout=base_layout
    )
    # default thresholds are multiples of the finest level's world-space
    # diagonal, measured the same way the implementation does
    centres0 = grids[0]["centres"]
    half0 = grids[0]["half_extents"]
    diag = float(np.linalg.norm((centres0.max(0) + half0) - (centres0.min(0) - half0)))
    explicit = select_levels_from_cache(
        grids, 2, cam, thresholds=[diag * 1.0], base_layout=None
    )
    np.testing.assert_array_equal(from_default, explicit)


def test_select_levels_default_thresholds_anisotropic_use_world_space():
    # A squashed z scale must shrink the world-space diagonal relative to the
    # raw voxel diagonal -> the default thresholds differ.  This is what proves
    # the defaults live in world space, not voxel space.
    layout = BlockLayout3D(volume_shape=(32, 32, 32), block_size=BLOCK_SIZE)
    scale = [np.array([1.0, 1.0, 0.25]), np.array([2.0, 2.0, 0.5])]
    translation = [np.zeros(3), np.zeros(3)]
    level_shapes = [(32, 32, 32), (16, 16, 16)]
    grids = build_level_grids(layout, 2, scale, translation, level_shapes)

    # camera well outside the volume in x, so level-1 brick distances span a
    # range that straddles both the world (~46) and voxel (~55) thresholds
    cam = np.array([60.0, 20.0, 2.0])
    from_default = select_levels_from_cache(
        grids, 2, cam, thresholds=None, base_layout=layout
    )

    # voxel-diagonal thresholds (the OLD, unit-mismatched behaviour) differ
    voxel_diag = np.sqrt(sum(s**2 for s in layout.volume_shape))
    from_voxel = select_levels_from_cache(
        grids, 2, cam, thresholds=[voxel_diag], base_layout=None
    )
    assert not np.array_equal(from_default, from_voxel)

    # world-space diagonal thresholds match
    centres0 = grids[0]["centres"]
    half0 = grids[0]["half_extents"]
    world_diag = float(
        np.linalg.norm((centres0.max(0) + half0) - (centres0.min(0) - half0))
    )
    from_world = select_levels_from_cache(
        grids, 2, cam, thresholds=[world_diag], base_layout=None
    )
    np.testing.assert_array_equal(from_default, from_world)


def test_select_levels_near_camera_fine_far_coarse(
    base_layout, scale_vecs, translation_vecs, level_shapes
):
    grids = build_level_grids(
        base_layout, 2, scale_vecs, translation_vecs, level_shapes=level_shapes
    )
    cam = np.array([0.0, 0.0, 0.0])
    out = select_levels_from_cache(grids, 2, cam, thresholds=[30.0])

    # every output level is valid
    assert set(np.unique(out[:, 0])).issubset({1, 2})
    # rows are unique -> no brick selected twice
    assert len({tuple(r) for r in out}) == out.shape[0]

    # the brick nearest the camera (level 1, at grid origin) is present as fine
    fine_rows = out[out[:, 0] == 1]
    assert (fine_rows[:, 1:] == 0).all(axis=1).any()

    # level-1 survivors all have centre distance < 30 (uses centre distance)
    centres1 = grids[0]["centres"]
    for r in fine_rows:
        gz, gy, gx = r[1], r[2], r[3]
        idx = np.where(
            (grids[0]["arr"][:, 1] == gz)
            & (grids[0]["arr"][:, 2] == gy)
            & (grids[0]["arr"][:, 3] == gx)
        )[0][0]
        assert np.linalg.norm(centres1[idx] - cam) < 30.0


def test_select_levels_middle_band_uses_max_corner_and_centre(base_layout):
    # 3 levels so the middle level exercises the two-sided band:
    # (max_corner_dist >= thresholds[k-2]) & (centre dist < thresholds[k-1]).
    scale = [np.ones(3), np.full(3, 2.0), np.full(3, 4.0)]
    translation = [np.zeros(3)] * 3
    level_shapes = [(32, 32, 32), (16, 16, 16), (8, 8, 8)]
    grids = build_level_grids(base_layout, 3, scale, translation, level_shapes)

    cam = np.zeros(3)
    out = select_levels_from_cache(grids, 3, cam, thresholds=[10.0, 100.0])
    # the middle level must contribute rows -> the two-sided band ran
    assert np.any(out[:, 0] == 2)
    assert set(np.unique(out[:, 0])).issubset({1, 2, 3})


def test_select_levels_empty_when_no_band_matches(
    base_layout, scale_vecs, translation_vecs
):
    # 1 level, but a threshold pushes everything out of level 1's band and
    # there is no coarser level to catch them
    grids = build_level_grids(base_layout, 1, scale_vecs, translation_vecs)
    cam = np.array([1000.0, 1000.0, 1000.0])
    out = select_levels_from_cache(grids, 1, cam, thresholds=[1.0])
    assert out.shape == (0, 4)


# ---------------------------------------------------------------------------
# select_levels_arr_forced
# ---------------------------------------------------------------------------


def test_select_levels_forced_cache_is_zero_copy(
    base_layout, scale_vecs, translation_vecs
):
    grids = build_level_grids(base_layout, 2, scale_vecs, translation_vecs)
    out = select_levels_arr_forced(base_layout, force_level=2, level_grids=grids)
    # returns the cached array object directly (zero-copy)
    assert out is grids[1]["arr"]


def test_select_levels_forced_no_cache_matches_cache(
    base_layout, scale_vecs, translation_vecs
):
    # power-of-2 grids so the recompute matches the cached grid dims
    grids = build_level_grids(base_layout, 2, scale_vecs, translation_vecs)
    for force_level in (1, 2):
        cached = select_levels_arr_forced(
            base_layout, force_level=force_level, level_grids=grids
        )
        recomputed = select_levels_arr_forced(base_layout, force_level=force_level)
        np.testing.assert_array_equal(cached, recomputed)


def test_select_levels_forced_clamps(base_layout, scale_vecs, translation_vecs):
    """Out-of-range levels clamp, mirroring ``select_lod_2d``.

    ``force_level`` is 1-indexed, so ``0`` is out of range.  Without the lower
    clamp it indexed ``level_grids[-1]`` and silently returned the *coarsest*
    level -- the exact opposite of the finest level a caller passing ``0`` means.
    """
    grids = build_level_grids(base_layout, 2, scale_vecs, translation_vecs)

    # force below 1 clamps to 1 (finest), not to level_grids[-1] (coarsest)
    low = select_levels_arr_forced(base_layout, force_level=0, level_grids=grids)
    assert np.all(low[:, 0] == 1)

    # force above n_levels clamps to n_levels
    high = select_levels_arr_forced(base_layout, force_level=5, level_grids=grids)
    assert np.all(high[:, 0] == 2)


def test_select_levels_forced_no_cache_clamps(base_layout):
    """The uncached branch clamps too -- it used to raise on ``1 << -1``."""
    recomputed = select_levels_arr_forced(base_layout, force_level=0)
    assert np.all(recomputed[:, 0] == 1)


# ---------------------------------------------------------------------------
# sort_arr_by_distance
# ---------------------------------------------------------------------------


def test_sort_arr_empty_returns_empty():
    arr = np.empty((0, 4), dtype=np.int32)
    out = sort_arr_by_distance(arr, np.zeros(3), BLOCK_SIZE)
    assert out.shape == (0, 4)


def test_sort_arr_nearest_first_isotropic():
    # bricks at increasing distance from origin along x
    arr = np.array(
        [
            [1, 0, 0, 3],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.int32,
    )
    out = sort_arr_by_distance(arr, np.zeros(3), BLOCK_SIZE)
    # gx column ascending -> nearest first
    np.testing.assert_array_equal(out[:, 3], [0, 1, 3])


def test_sort_arr_centres_match_build_level_grids(
    base_layout, scale_vecs, translation_vecs, level_shapes
):
    grids = build_level_grids(
        base_layout, 2, scale_vecs, translation_vecs, level_shapes=level_shapes
    )
    arr = grids[0]["arr"]
    centres = grids[0]["centres"]
    cam = np.array([13.0, 27.0, 5.0])

    out = sort_arr_by_distance(
        arr,
        cam,
        BLOCK_SIZE,
        scale_vecs_shader=scale_vecs,
        translation_vecs_shader=translation_vecs,
    )
    # expected order from the grid's own centres (shared formula)
    dists = np.linalg.norm(centres - cam, axis=1)
    expected_order = np.argsort(dists, kind="stable")
    np.testing.assert_array_equal(out, arr[expected_order])


def test_sort_arr_anisotropic_differs_from_isotropic():
    # a strongly anisotropic z scale changes the ordering vs the isotropic
    # power-of-2 fallback
    arr = np.array(
        [
            [1, 3, 0, 0],  # far along z (data axis 0)
            [1, 0, 0, 2],  # closer along x under isotropic
        ],
        dtype=np.int32,
    )
    cam = np.zeros(3)
    iso = sort_arr_by_distance(arr, cam, BLOCK_SIZE)
    scale = [np.array([1.0, 1.0, 0.1])]  # squash z
    translation = [np.zeros(3)]
    aniso = sort_arr_by_distance(
        arr,
        cam,
        BLOCK_SIZE,
        scale_vecs_shader=scale,
        translation_vecs_shader=translation,
    )
    # under isotropic, the x=2 brick is nearer; under a squashed z, the z=3
    # brick becomes nearer -> different first row
    assert not np.array_equal(iso[0], aniso[0])


def test_sort_arr_stable_for_equidistant():
    # two bricks equidistant from camera keep input order (stable sort)
    arr = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    # tag them apart via level column to detect reordering
    arr[1, 0] = 1
    out = sort_arr_by_distance(arr, np.array([4.0, 4.0, 4.0]), BLOCK_SIZE)
    np.testing.assert_array_equal(out, arr)
