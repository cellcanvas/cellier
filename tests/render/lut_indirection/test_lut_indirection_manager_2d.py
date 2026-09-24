"""Tests for the 2D LUT paint: painting order, and viewport clipping of
stale background tiles.

The residency adapter paints the background (earlier views) first, clipped
to the viewport, and the foreground (the current view) last, unclipped.
"""

import numpy as np

from cellier.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)

# 4x4 finest grid (block_size=1), single LOD level so each tile covers one cell.
BASE_LAYOUT = BlockLayout2D.from_shape(shape=(4, 4), block_size=1, overlap=1)
N_LEVELS = 1
VIEW = (0, 0, 2, 2)  # cells [0:2, 0:2] visible


def _paint(tiles, phases, clips):
    """*tiles*: rows ``(g0, g1, sy, sx)`` at level 1."""
    lut = LutIndirectionManager2D(BASE_LAYOUT, n_levels=N_LEVELS)
    tiles = np.asarray(tiles, dtype=np.int64).reshape(-1, 4)
    lut.paint(
        np.ones(len(tiles), np.int64),
        tiles[:, :2],
        tiles[:, 2:],
        [np.asarray(p, dtype=np.int64) for p in phases],
        clips,
    )
    return lut.lut_data


def test_background_tile_in_view_is_written() -> None:
    """An old-slice tile inside the viewport is kept as a placeholder."""
    lut = _paint([(0, 0, 0, 1)], [[0], []], [VIEW, None])
    assert lut[0, 0, 2] == 1  # level channel: referenced


def test_background_tile_out_of_view_is_clipped() -> None:
    """An old-slice tile outside the viewport is not referenced."""
    lut = _paint([(3, 3, 0, 1)], [[0], []], [VIEW, None])
    assert np.all(lut[..., 2] == 0)


def test_foreground_tile_out_of_view_is_not_clipped() -> None:
    """Current-slice (foreground) tiles are written regardless of viewport."""
    lut = _paint([(3, 3, 0, 1)], [[0]], [None])
    assert lut[3, 3, 2] == 1


def test_no_viewport_writes_background_everywhere() -> None:
    lut = _paint([(3, 3, 0, 1)], [[0], []], [None, None])
    assert lut[3, 3, 2] == 1


def test_foreground_overwrites_background_in_view() -> None:
    """Where both exist in view, the current-slice tile wins."""
    lut = _paint([(0, 0, 0, 1), (0, 0, 2, 3)], [[0], [1]], [VIEW, None])
    assert tuple(lut[0, 0]) == (3, 2, 1, 0)  # (sx, sy, level, 0)
    # Everything else is out of bounds.
    assert np.count_nonzero(lut[..., 2]) == 1
