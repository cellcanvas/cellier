"""LutIndirectionManager3D.paint: bricks from arrays into the LUT."""

import numpy as np

from cellier.render.block_cache import TileManager3D, compute_block_cache_parameters_3d
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D

CACHE_INFO = compute_block_cache_parameters_3d(
    block_size=4, gpu_budget_bytes=8 * 6**3 * 4
)
SLOTS = TileManager3D(CACHE_INFO)

# LUT layout: 4x4x4 finest grid (volume 4x4x4, block_size=1), 2 LOD levels.
BASE_LAYOUT = BlockLayout3D(volume_shape=(4, 4, 4), block_size=1)
N_LEVELS = 2


def _paint(lut, bricks, phases=None) -> None:
    """Paint ``(level, g0, g1, g2, atlas slot)`` rows, one phase by default."""
    rows = np.asarray(bricks, dtype=np.int64).reshape(-1, 5)
    grid_pos = np.array([SLOTS._slot_grid_pos(int(i)) for i in rows[:, 4]])
    lut.paint(
        rows[:, 0],
        rows[:, 1:4],
        grid_pos.reshape(-1, 3),
        np.arange(len(rows), dtype=np.float32) + 1,
        phases if phases is not None else [np.arange(len(rows))],
    )


def test_paint_nothing_is_all_zeros() -> None:
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    lut.lut_data[:] = 9
    _paint(lut, [])
    assert np.all(lut.lut_data == 0)
    assert np.all(lut.brick_max_data == 0)


def test_coarse_brick_fills_its_fine_cells() -> None:
    """A level-2 brick at (0, 0, 0) covers [0:2, 0:2, 0:2] of the finest grid,
    and every cell there carries level 2 and its slot's ``(sx, sy, sz)``."""
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    _paint(lut, [(2, 0, 0, 0, 1)])
    sz, sy, sx = SLOTS._slot_grid_pos(1)

    covered = lut.lut_data[0:2, 0:2, 0:2]
    assert np.all(covered[..., 0] == sx)
    assert np.all(covered[..., 1] == sy)
    assert np.all(covered[..., 2] == sz)
    assert np.all(covered[..., 3] == 2)
    assert np.all(lut.brick_max_data[0:2, 0:2, 0:2] == 1.0)


def test_uncovered_cells_remain_zero() -> None:
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    _paint(lut, [(2, 0, 0, 0, 1)])
    assert np.all(lut.lut_data[2:, :, :] == 0)
    assert np.all(lut.lut_data[:, 2:, :] == 0)
    assert np.all(lut.lut_data[:, :, 2:] == 0)


def test_fine_brick_overwrites_coarse_at_its_cell() -> None:
    """Within a phase coarser levels are painted first, whatever the order."""
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    _paint(lut, [(1, 0, 0, 0, 2), (2, 0, 0, 0, 1)])

    assert lut.lut_data[0, 0, 0, 3] == 1
    assert lut.lut_data[0, 0, 1, 3] == 2
    assert lut.lut_data[2, 0, 0, 3] == 0
    sz, sy, sx = SLOTS._slot_grid_pos(2)
    assert tuple(lut.lut_data[0, 0, 0, :3]) == (sx, sy, sz)


def test_a_later_phase_covers_an_earlier_one() -> None:
    """Foreground over background, even when the background is finer."""
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    fine_background, coarse_foreground = (1, 0, 0, 0, 2), (2, 0, 0, 0, 1)
    _paint(
        lut,
        [fine_background, coarse_foreground],
        phases=[np.array([0]), np.array([1])],
    )
    assert lut.lut_data[0, 0, 0, 3] == 2
