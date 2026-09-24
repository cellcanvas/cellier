"""ImageResidency3D and ImageResidency2D: the chunk scheduler's atlas adapters."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.block_cache import BlockCache3D, compute_block_cache_parameters_3d
from cellier.render.block_cache._block_cache_2d import BlockCache2D
from cellier.render.block_cache._cache_parameters_2d import (
    compute_block_cache_parameters_2d,
)
from cellier.render.block_cache._image_residency import (
    MAX_GRID,
    ImageResidency2D,
    ImageResidency3D,
    pack_keys,
    unpack_keys,
)
from cellier.render.block_cache._tile_manager_2d import BlockKey2D
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from cellier.render.scheduling import ChunkState, RegistryView, Residency, Tier

BLOCK = 4
# A 16^3 volume with 3 levels (2x per level, level 3 one 4^3 brick).
SHAPES = [(16, 16, 16), (8, 8, 8), (4, 4, 4)]
SCALES = np.array([[1.0] * 4, [1.0, 2.0, 2.0, 2.0], [1.0, 4.0, 4.0, 4.0]])
TRANSLATIONS = np.zeros((3, 4))


def _residency(**kwargs) -> ImageResidency3D:
    params = compute_block_cache_parameters_3d(
        block_size=BLOCK, gpu_budget_bytes=27 * (BLOCK + 2) ** 3 * 4, overlap=1
    )
    cache = BlockCache3D(params)
    lut = LutIndirectionManager3D(
        BlockLayout3D(volume_shape=SHAPES[0], block_size=BLOCK),
        n_levels=3,
        level_scale_vecs_data=[s[1:] for s in SCALES],
        level_shapes=SHAPES,
    )
    return ImageResidency3D(cache, lut, BLOCK, SCALES, TRANSLATIONS, **kwargs)


def _view(residency, rows, complete=False) -> RegistryView:
    """``(key, state, tier, slot, slice_id, wanted_gen)`` rows, key-sorted."""
    rows = sorted(rows)
    arr = np.array(rows, dtype=np.int64).reshape(-1, 6)
    return RegistryView.from_arrays(
        {
            "key": arr[:, 0],
            "state": arr[:, 1].astype(np.uint8),
            "tier": arr[:, 2].astype(np.uint8),
            "cls": np.zeros(len(rows), np.uint8),
            "rank": np.zeros(len(rows), np.int32),
            "slot": arr[:, 3].astype(np.int32),
            "slice_id": arr[:, 4].astype(np.int32),
            "wanted_gen": arr[:, 5].astype(np.int32),
        },
        generation=9,
        complete=complete,
    )


R, V, RC = int(ChunkState.RESIDENT), int(Tier.VISIBLE), int(Tier.RECENT)


def test_is_a_residency() -> None:
    residency = _residency()
    assert isinstance(residency, Residency)
    assert residency.n_slots == residency.block_cache.info.n_slots - 1


def test_keys_round_trip() -> None:
    levels = np.array([1, 3, 15])
    sids = np.array([0, 7, 65535])
    grids = np.array([[0, 1, 2], [4095, 0, 17], [3, 3, 3]])
    got = unpack_keys(pack_keys(levels, sids, grids))
    np.testing.assert_array_equal(got[0], levels)
    np.testing.assert_array_equal(got[1], sids)
    np.testing.assert_array_equal(got[2], grids)


@pytest.mark.parametrize(
    "levels, sids, grids",
    [([0], [0], [[0, 0, 0]]), ([16], [0], [[0, 0, 0]]), ([1], [0], [[MAX_GRID, 0, 0]])],
)
def test_keys_out_of_range_are_refused(levels, sids, grids) -> None:
    with pytest.raises(ValueError, match="out of range"):
        pack_keys(np.array(levels), np.array(sids), np.array(grids))


def test_interning_is_stable_per_atlas() -> None:
    residency = _residency()
    a = residency.intern((3, None, None, None))
    b = residency.intern(((2, 4), None, None, None))
    assert residency.intern((3, None, None, None)) == a
    assert a != b
    assert residency.slice_coord(a) == ((0, 3),)
    assert residency.slice_coord(b) == ((0, (2, 4)),)
    assert _residency().intern(((2, 4), None, None, None)) == 0  # its own table


def test_build_request_puts_windows_on_displayed_axes() -> None:
    residency = _residency()
    sid = residency.intern((5, None, None, None))
    keys = pack_keys(np.array([2]), np.array([sid]), np.array([[1, 0, 1]]))
    (request,) = residency.build_request(keys)
    assert request.scale_index == 1
    # Padded windows: g * block - overlap, size block + 2 * overlap.
    assert request.axis_selections == (5, (3, 9), (-1, 5), (3, 9))


def test_write_uploads_and_records_brick_max() -> None:
    seen: list[int] = []
    residency = _residency(on_write=lambda: seen.append(1))
    pbs = residency.block_cache.info.padded_block_size
    residency.write(3, 0, np.full((pbs, pbs, pbs), 2.5, np.float32))
    assert residency.brick_max[3] == 2.5
    sz, sy, sx = residency.slot_grid[3]
    atlas = residency.block_cache.cache_data
    assert atlas[sz * pbs, sy * pbs, sx * pbs] == 2.5
    assert residency.block_cache.tile_manager._slot_grid_pos(4) == tuple(
        residency.slot_grid[3]
    )
    assert seen == [1]


def test_rebuild_draw_paints_background_oldest_first_then_foreground() -> None:
    residency = _residency()
    old = residency.intern((1, None, None, None))
    newer = residency.intern((2, None, None, None))
    now = residency.intern((3, None, None, None))
    fine_old = int(pack_keys([1], [old], [[0, 0, 0]])[0])
    coarse_newer = int(pack_keys([3], [newer], [[0, 0, 0]])[0])
    coarse_now = int(pack_keys([3], [now], [[0, 0, 0]])[0])
    view = _view(
        residency,
        [
            (fine_old, R, RC, 1, old, 1),
            (coarse_newer, R, RC, 2, newer, 2),
            (coarse_now, 0, V, -1, now, 3),  # wanted, still loading
        ],
    )
    residency.rebuild_draw(view)
    lut = residency.lut_manager.lut_data
    # The newer slice's coarse brick covers the older slice's fine one.
    assert lut[0, 0, 0, 3] == 3
    drawn = residency.block_cache.tile_manager.tilemap
    assert sorted((key.level, key.slice_coord) for key in drawn) == [
        (1, ((0, 1),)),
        (3, ((0, 2),)),
    ]

    # Complete: the background leaves the LUT.
    view = _view(
        residency,
        [
            (fine_old, R, RC, 1, old, 1),
            (coarse_newer, R, RC, 2, newer, 2),
            (coarse_now, R, V, 3, now, 3),
        ],
        complete=True,
    )
    residency.rebuild_draw(view)
    tilemap = residency.block_cache.tile_manager.tilemap
    assert len(tilemap) == 1
    (key,) = tilemap
    assert (key.level, key.slice_coord) == (3, ((0, 3),))
    assert tilemap[key].index == 4  # scheduler slot 3 is atlas slot 4


def test_keys_in_region_maps_each_level_back_to_level_0() -> None:
    residency = _residency()
    t0 = residency.intern((0, None, None, None))
    t1 = residency.intern((1, None, None, None))
    keys = pack_keys(
        np.array([1, 1, 3, 1]),
        np.array([t0, t0, t0, t1]),
        np.array([[0, 0, 0], [3, 3, 3], [0, 0, 0], [0, 0, 0]]),
    )
    # A small region at t = 0 near the far corner of level 0.
    region = ((0.0, 1.0), (14.0, 15.0), (14.0, 15.0), (14.0, 15.0))
    hit = residency.keys_in_region(keys, [region])
    # The far level-1 brick and the level-3 brick (the whole volume) at t=0.
    assert hit.tolist() == [False, True, True, False]
    assert not residency.keys_in_region(keys, []).any()


# -- 2D ------------------------------------------------------------------------

SHAPES_2D = [(16, 16), (8, 8), (4, 4)]
SCALES_2D = np.array([[1.0] * 3, [1.0, 2.0, 2.0], [1.0, 4.0, 4.0]])


def _residency_2d(**kwargs) -> ImageResidency2D:
    params = compute_block_cache_parameters_2d(
        gpu_budget_bytes=16 * (BLOCK + 2) ** 2 * 4, block_size=BLOCK, overlap=1
    )
    cache = BlockCache2D(params)
    lut = LutIndirectionManager2D(
        BlockLayout2D.from_shape(shape=SHAPES_2D[0], block_size=BLOCK, overlap=1),
        n_levels=3,
        scale_vecs_data=[s[1:] for s in SCALES_2D],
        level_shapes=SHAPES_2D,
    )
    return ImageResidency2D(cache, lut, BLOCK, SCALES_2D, np.zeros((3, 3)), **kwargs)


def test_2d_is_a_residency() -> None:
    residency = _residency_2d()
    assert isinstance(residency, Residency)
    assert residency.n_slots == residency.block_cache.info.n_slots - 1
    tm = residency.block_cache.tile_manager
    assert tm._slot_grid_pos(5) == tuple(residency.slot_grid[4])


def test_2d_keys_leave_g2_zero() -> None:
    keys = pack_keys(np.array([2]), np.array([3]), np.array([[5, 6]]))
    _, _, grids = unpack_keys(keys)
    assert grids.tolist() == [[5, 6, 0]]


def test_2d_build_request_windows_the_displayed_axes() -> None:
    residency = _residency_2d()
    sid = residency.intern((7, None, None))
    keys = pack_keys(np.array([1]), np.array([sid]), np.array([[2, 0]]))
    (request,) = residency.build_request(keys)
    assert request.scale_index == 0
    assert request.axis_selections == (7, (7, 13), (-1, 5))


def test_2d_write_uploads_the_tile() -> None:
    seen: list[int] = []
    residency = _residency_2d(on_write=lambda: seen.append(1))
    pbs = residency.block_cache.info.padded_block_size
    residency.write(2, 0, np.full((pbs, pbs), 4.0, np.float32))
    sy, sx = residency.slot_grid[2]
    assert residency.block_cache.cache_data[sy * pbs, sx * pbs] == 4.0
    assert seen == [1]


def test_2d_background_is_clipped_to_the_viewport_until_complete() -> None:
    residency = _residency_2d()
    old = residency.intern((1, None, None))
    now = residency.intern((2, None, None))
    in_view = int(pack_keys([1], [old], [[0, 0]])[0])
    out_of_view = int(pack_keys([1], [old], [[3, 3]])[0])
    wanted = int(pack_keys([1], [now], [[3, 0]])[0])
    residency.viewport_cells = (0, 0, 2, 2)
    view = _view(
        residency,
        [
            (in_view, R, RC, 0, old, 1),
            (out_of_view, R, RC, 1, old, 1),
            (wanted, R, V, 2, now, 2),
        ],
    )
    residency.rebuild_draw(view)
    level = residency.lut_manager.lut_data[..., 2]
    assert level[0, 0] == 1  # background in view
    assert level[3, 3] == 0  # background out of view: clipped
    assert level[3, 0] == 1  # foreground is never clipped
    drawn = residency.block_cache.tile_manager.tilemap
    assert sorted((k.g0, k.g1, k.slice_coord) for k in drawn) == [
        (0, 0, ((0, 1),)),
        (3, 0, ((0, 2),)),
        (3, 3, ((0, 1),)),
    ]
    assert drawn[BlockKey2D(level=1, g0=3, g1=0, slice_coord=((0, 2),))].index == 3

    residency.rebuild_draw(
        _view(
            residency,
            [
                (in_view, R, RC, 0, old, 1),
                (out_of_view, R, RC, 1, old, 1),
                (wanted, R, V, 2, now, 2),
            ],
            complete=True,
        )
    )
    level = residency.lut_manager.lut_data[..., 2]
    assert np.count_nonzero(level) == 1  # a level-1 tile is one base cell
    assert len(residency.block_cache.tile_manager.tilemap) == 1


def test_2d_keys_in_region() -> None:
    residency = _residency_2d()
    t0 = residency.intern((0, None, None))
    keys = pack_keys(
        np.array([1, 1, 3]), np.array([t0, t0, t0]), np.array([[0, 0], [3, 3], [0, 0]])
    )
    region = ((0.0, 1.0), (14.0, 15.0), (14.0, 15.0))
    assert residency.keys_in_region(keys, [region]).tolist() == [False, True, True]


def test_2d_keys_in_region_uses_voxel_centres_on_a_translated_level() -> None:
    """Level 2 (scale 2) shifted by t = 0.25: centre convention (plan v2, D1).

    Brick 0's padded tile holds level voxels ``-1 .. BLOCK``; voxel ``BLOCK``
    is centred on ``2 * BLOCK + 0.25`` and covers up to ``2 * BLOCK + 1.25``.
    Level-0 voxel ``2 * BLOCK + 1`` (from ``2 * BLOCK + 0.5``) overlaps it;
    ``2 * BLOCK + 2`` (from ``2 * BLOCK + 1.5``) does not.  The edge-convention
    formula this replaced reached ``2 * BLOCK + 2.25`` and hit both.
    """
    translations = np.zeros((3, 3))
    translations[1] = (0.0, 0.25, 0.25)
    params = compute_block_cache_parameters_2d(
        gpu_budget_bytes=16 * (BLOCK + 2) ** 2 * 4, block_size=BLOCK, overlap=1
    )
    lut = LutIndirectionManager2D(
        BlockLayout2D.from_shape(shape=SHAPES_2D[0], block_size=BLOCK, overlap=1),
        n_levels=3,
        scale_vecs_data=[s[1:] for s in SCALES_2D],
        level_shapes=SHAPES_2D,
    )
    residency = ImageResidency2D(
        BlockCache2D(params), lut, BLOCK, SCALES_2D, translations
    )
    t0 = residency.intern((0, None, None))
    keys = pack_keys(np.array([2]), np.array([t0]), np.array([[0, 0]]))

    def hit(voxel: int) -> bool:
        region = ((0.0, 1.0), (voxel, voxel + 1.0), (voxel, voxel + 1.0))
        return bool(residency.keys_in_region(keys, [region])[0])

    assert hit(2 * BLOCK + 1)
    assert not hit(2 * BLOCK + 2)
