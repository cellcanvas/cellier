"""Tests for BlockCache3D: the atlas texture and its slot geometry.

Which brick occupies which slot is the chunk scheduler's business
(``tests/render/scheduling``); the atlas writes bricks and maps slots to
texture positions.
"""

import numpy as np
import pytest

from cellier.render.block_cache import (
    BlockCache3D,
    BlockKey3D,
    TileSlot,
    compute_block_cache_parameters_3d,
)

# Cache used throughout: block_size=4, overlap=1, grid_side=2, 8 slots (7 usable).
CACHE_INFO = compute_block_cache_parameters_3d(
    block_size=4, gpu_budget_bytes=8 * 6**3 * 4
)


def _slot(cache: BlockCache3D, index: int) -> TileSlot:
    return TileSlot(index=index, grid_pos=cache.tile_manager._slot_grid_pos(index))


def test_slot_grid_positions_cover_the_atlas_once() -> None:
    cache = BlockCache3D(CACHE_INFO)
    positions = {cache.tile_manager._slot_grid_pos(i) for i in range(8)}
    assert len(positions) == 8
    assert cache.tile_manager._slot_grid_pos(0) == (0, 0, 0)
    assert cache.tile_manager.n_data_slots == 7


def test_write_brick_fills_correct_slice() -> None:
    cache = BlockCache3D(CACHE_INFO)
    slot = _slot(cache, 5)

    pbs = cache.info.padded_block_size
    data = np.full((pbs, pbs, pbs), fill_value=7.0, dtype=np.float32)
    cache.write_brick(slot, data)

    sz, sy, sx = slot.grid_pos
    z0, y0, x0 = sz * pbs, sy * pbs, sx * pbs
    assert np.all(cache.cache_data[z0 : z0 + pbs, y0 : y0 + pbs, x0 : x0 + pbs] == 7.0)


def test_write_brick_does_not_touch_other_slots() -> None:
    cache = BlockCache3D(CACHE_INFO)
    pbs = cache.info.padded_block_size
    cache.write_brick(_slot(cache, 1), np.ones((pbs, pbs, pbs), dtype=np.float32))

    assert cache.cache_data.sum() == pytest.approx(float(pbs**3))


def test_n_resident_and_clear_follow_the_drawn_view() -> None:
    cache = BlockCache3D(CACHE_INFO)
    assert cache.n_resident == 0
    cache.tile_manager.tilemap = {
        BlockKey3D(level=1, g0=0, g1=0, g2=0): _slot(cache, 1)
    }
    assert cache.n_resident == 1
    cache.clear()
    assert cache.n_resident == 0
