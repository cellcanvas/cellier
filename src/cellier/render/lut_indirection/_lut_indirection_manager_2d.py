from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.logging import _GPU_LOGGER
from cellier.render.lut_indirection._cell_brick_rule import (
    brick_rule_issues,
    level_brick_counts,
    level_cell_spans,
)
from cellier.render.lut_indirection._lut_paint import LOOP_BELOW, paint_lut

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.render.lut_indirection._layout_2d import BlockLayout2D


class LutIndirectionManager2D:
    """Manages the 2D LUT indirection texture.

    The LUT maps each finest-level grid cell ``(gy, gx)`` to a cache
    slot ``(sx, sy)`` and a level indicator.  ``paint()`` rewrites the LUT
    from arrays of resident tiles (the image residency adapter calls it).

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels.
    scale_vecs_data : list[np.ndarray] or None
        Per-level scale vectors in data-axis order ``(sy, sx)`` — i.e.
        ``(scale_row, scale_col)`` for the tile grid.  When provided the
        LUT fill uses the actual per-axis downsampling factor instead of
        assuming a uniform ``2^(level-1)`` factor.  Pass
        ``ImageGeometry2D._scale_vecs_data`` here.
    level_shapes : list of tuple of int or None
        Per-level ``(H, W)`` shapes.  Gives each level's tile count, so the
        last tile owns the grid's tail cells (see ``_cell_brick_rule``).
        Without it the count is inferred from the grid.
    border : float or None
        The tile cache's padding in pixels.  When given together with the
        scales and shapes, levels whose cell -> tile rule needs more padding
        than this are logged as warnings.
    translation_vecs_data : list[np.ndarray] or None
        Per-level translations in level-0 voxels, ``(ty, tx)``; the budget
        warning then includes the translation term.  ``None`` assumes block
        averaging.
    sampling_margin : float
        How far past a sample position this path's shader reads, in level-k
        voxels, at the default ray density (plan v2, "Padding budgets"):
        subtracted from *border* for the warning.  Default 0.5 (linear).
    """

    def __init__(
        self,
        base_layout: BlockLayout2D,
        n_levels: int,
        scale_vecs_data: list[np.ndarray] | None = None,
        level_shapes: list | None = None,
        border: float | None = None,
        translation_vecs_data: list[np.ndarray] | None = None,
        sampling_margin: float = 0.5,
    ) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self._scale_vecs_data = scale_vecs_data
        self._level_shapes = level_shapes
        if border is not None:
            for issue in brick_rule_issues(
                scale_vecs_data,
                level_shapes,
                base_layout.block_size,
                border,
                sampling_margin=sampling_margin,
                translation_vecs_data=translation_vecs_data,
            ):
                _GPU_LOGGER.warning("brick_rule_padding  2d  %s", issue.describe())
        self.lut_data, self.lut_tex = build_lut_texture_2d(base_layout.grid_dims)
        # The one cell -> tile rule, shared with the shaders through the
        # block-scales buffer.  See _cell_brick_rule.
        self._spans = level_cell_spans(n_levels, 2, scale_vecs_data)
        self._counts = level_brick_counts(
            self._spans, base_layout.grid_dims, base_layout.block_size, level_shapes
        )

    def paint(
        self,
        levels: np.ndarray,
        grids: np.ndarray,
        slot_grid_pos: np.ndarray,
        phases: Sequence[np.ndarray],
        clips: Sequence[tuple[int, int, int, int] | None] | None = None,
        *,
        loop_below: int = LOOP_BELOW,
    ) -> None:
        """Rewrite the LUT and schedule its upload.

        Tiles are painted phase by phase, each phase coarsest to finest, so a
        finer tile covers the coarser fallback under it and a later phase
        covers an earlier one (design 5.8).

        Channel assignments: ``(sx, sy, level, 0)``; level 0 is out of bounds.

        Parameters
        ----------
        levels : np.ndarray
            ``(N,)`` 1-based level per tile.
        grids : np.ndarray
            ``(N, 2)`` tile grid position ``(g0, g1)`` at its level.
        slot_grid_pos : np.ndarray
            ``(N, 2)`` cache grid position ``(sy, sx)`` of each tile's slot.
        phases : sequence of np.ndarray
            Index arrays into the tiles, in painting order.
        clips : sequence of tuple or None
            Per phase, base-cell bounds ``(gy0, gx0, gy1, gx1)`` (half open)
            to clip that phase's writes to, or ``None``.  Stale background
            tiles are clipped to the viewport so out-of-view stale data is
            not referenced.
        loop_below : int
            See :func:`paint_lut`.
        """
        slot_grid_pos = np.asarray(slot_grid_pos, dtype=np.int64).reshape(-1, 2)
        levels = np.asarray(levels, dtype=np.int64)
        values = np.column_stack(
            [
                slot_grid_pos[:, 1],
                slot_grid_pos[:, 0],
                levels,
                np.zeros(len(levels), np.int64),
            ]
        ).astype(np.float32)
        paint_lut(
            self.lut_data,
            values,
            levels,
            np.asarray(grids, dtype=np.int64).reshape(-1, 2),
            phases,
            self._spans,
            self._counts,
            clips=clips,
            loop_below=loop_below,
        )
        self.lut_tex.update_range((0, 0, 0), self.lut_tex.size)


def build_lut_texture_2d(
    grid_dims: tuple[int, int],
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate CPU lut_data array and a gfx.Texture.

    Parameters
    ----------
    grid_dims : tuple[int, int]
        ``(gH, gW)`` -- finest level grid dimensions.

    Returns
    -------
    lut_data : np.ndarray
        Shape ``(gH, gW, 4)``, dtype float32.  Initialised to zeros
        (all tiles point to the reserved empty slot).
    lut_tex : gfx.Texture
        pygfx 2D texture wrapping ``lut_data``.
    """
    gh, gw = grid_dims
    lut_data = np.zeros((gh, gw, 4), dtype=np.float32)
    lut_tex = gfx.Texture(lut_data, dim=2)
    return lut_data, lut_tex
