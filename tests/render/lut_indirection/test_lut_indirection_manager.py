"""LutIndirectionManager3D.paint: bricks from arrays into the LUT."""

import numpy as np
import pytest

from cellier.render.block_cache import TileManager3D, compute_block_cache_parameters_3d
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.lut_indirection._lut_paint import paint_lut

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


@pytest.mark.parametrize("loop_below", [0, 10**9], ids=["vectorised", "loop"])
def test_paint_writes_the_bytes_of_the_four_byte_fill(loop_below: int) -> None:
    """``paint`` fills through a ``uint32`` view; the tables are byte for byte
    what ``paint_lut`` writes into the four-byte table, tail bricks included."""
    shapes = [(46, 38, 50), (23, 19, 25), (12, 10, 13), (6, 5, 7)]
    scales = [np.asarray(shapes[0]) / np.asarray(shape) for shape in shapes]
    layout = BlockLayout3D(volume_shape=shapes[0], block_size=4)
    lut = LutIndirectionManager3D(
        layout, n_levels=len(shapes), level_scale_vecs_data=scales, level_shapes=shapes
    )
    rows = [
        (level, *grid)
        for level, count in enumerate(lut._counts, start=1)
        for grid in np.ndindex(*(int(c) for c in count))
    ]
    rng = np.random.default_rng(0)
    bricks = np.asarray(rows, dtype=np.int64)[rng.permutation(len(rows))]
    levels, grids = bricks[:, 0], bricks[:, 1:]
    slot_grid_pos = rng.integers(0, 256, size=(len(bricks), 3))
    brick_max = rng.random(len(bricks)).astype(np.float32)
    order = rng.permutation(len(bricks))
    phases = [order[: len(order) // 3], order[len(order) // 3 :]]

    expected = np.empty_like(lut.lut_data)
    expected_max = np.empty_like(lut.brick_max_data)
    paint_lut(
        expected,
        np.column_stack(
            [slot_grid_pos[:, 2], slot_grid_pos[:, 1], slot_grid_pos[:, 0], levels]
        ).astype(np.uint8),
        levels,
        grids,
        phases,
        lut._spans,
        lut._counts,
        extra=expected_max,
        extra_values=brick_max,
        loop_below=loop_below,
    )
    lut.paint(levels, grids, slot_grid_pos, brick_max, phases, loop_below=loop_below)

    assert set(np.unique(expected[..., 3]).tolist()) == {1, 2, 3, 4}
    assert lut.lut_data.tobytes() == expected.tobytes()
    assert lut.brick_max_data.tobytes() == expected_max.tobytes()
