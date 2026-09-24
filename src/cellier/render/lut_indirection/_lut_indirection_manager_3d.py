"""LUT-based indirection manager for bricked volume rendering."""

from __future__ import annotations

import logging
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

    from cellier.render.lut_indirection._layout_3d import BlockLayout3D


class LutIndirectionManager3D:
    """LUT indirection table for a bricked 3-D volume.

    Maintains a CPU-side uint8 array and a matching GPU RGBA8UI texture.
    Each voxel of the LUT corresponds to one brick position in the
    finest-level grid and encodes ``(sx, sy, sz, level)`` — the cache
    slot where that brick (or its best coarser fallback) currently lives.

    A companion ``brick_max_tex`` (R32Float, same grid dimensions) stores
    the maximum intensity per brick for MIP early-out.

    Notes
    -----
    Channel assignments follow the axis convention:

    - ``lut[d, h, w, 0]`` = tile_x = sx (cache W axis)
    - ``lut[d, h, w, 1]`` = tile_y = sy (cache H axis)
    - ``lut[d, h, w, 2]`` = tile_z = sz (cache D axis)
    - ``lut[d, h, w, 3]`` = level  (1 = finest; 0 = out-of-bounds)

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest LOAD level.  Determines the LUT grid
        dimensions ``(gD, gH, gW)``.
    n_levels : int
        Total number of level of detail levels.
    level_scale_vecs_data : list of ndarray, optional
        Per-level scale vectors in data order ``(sz, sy, sx)``.  Entry ``k``
        is the downscale factor of level ``k`` relative to the finest level.
        When provided, ``paint()`` uses the actual per-axis scales instead
        of the uniform ``2^(level-1)`` assumption.  Required for datasets
        where axes are downsampled at different rates (e.g. z-anisotropic
        microscopy data where z is never downsampled).
    level_shapes : list of tuple of int, optional
        Per-level voxel shapes in the same order.  Gives each level's brick
        count, so the last brick owns the grid's tail cells (see
        ``_cell_brick_rule``).  Without it the count is inferred from the grid.
    border : float, optional
        The brick cache's padding in voxels.  When given together with the
        scales and shapes, levels whose cell -> brick rule needs more padding
        than this are logged as warnings.
    level_translation_vecs_data : list of ndarray, optional
        Per-level translations in level-0 voxels, data order; the budget
        warning then includes the translation term.  ``None`` assumes block
        averaging.
    sampling_margin : float
        How far past a sample position this path's shader reads, in level-k
        voxels, at the default ray density (plan v2, "Padding budgets"):
        subtracted from *border* for the warning.  Default 0.5 (linear).

    Attributes
    ----------
    lut_data : np.ndarray
        CPU backing array, shape ``(gD, gH, gW, 4)``, dtype uint8.
        Channel layout: ``(tile_x=sx, tile_y=sy, tile_z=sz, level)``.
        All zeros = out-of-bounds (level 0, renders the null color).
    lut_tex : gfx.Texture
        GPU RGBA8UI 3-D texture wrapping ``lut_data``.
    brick_max_data : np.ndarray
        CPU backing array, shape ``(gD, gH, gW)``, dtype float32.
        Stores per-brick maximum intensity for MIP early-out.
    brick_max_tex : gfx.Texture
        GPU R32Float 3-D texture wrapping ``brick_max_data``.
    """

    def __init__(
        self,
        base_layout: BlockLayout3D,
        n_levels: int,
        level_scale_vecs_data: list | None = None,
        level_shapes: list | None = None,
        border: float | None = None,
        level_translation_vecs_data: list | None = None,
        sampling_margin: float = 0.5,
    ) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self._level_scale_vecs_data = level_scale_vecs_data
        self._level_shapes = level_shapes
        if border is not None:
            for issue in brick_rule_issues(
                level_scale_vecs_data,
                level_shapes,
                base_layout.block_size,
                border,
                sampling_margin=sampling_margin,
                translation_vecs_data=level_translation_vecs_data,
            ):
                _GPU_LOGGER.warning("brick_rule_padding  3d  %s", issue.describe())
        self.lut_data, self.lut_tex = build_lut_texture(base_layout)
        self.brick_max_data, self.brick_max_tex = build_brick_max_texture(base_layout)
        # The one cell -> brick rule, shared with the shaders through the
        # block-scales buffer.  See _cell_brick_rule.
        self._spans = level_cell_spans(n_levels, 3, level_scale_vecs_data)
        self._counts = level_brick_counts(
            self._spans, base_layout.grid_dims, base_layout.block_size, level_shapes
        )

    # ------------------------------------------------------------------
    # GPU writes
    # ------------------------------------------------------------------

    def paint(
        self,
        levels: np.ndarray,
        grids: np.ndarray,
        slot_grid_pos: np.ndarray,
        brick_max: np.ndarray,
        phases: Sequence[np.ndarray],
        *,
        loop_below: int = LOOP_BELOW,
    ) -> None:
        """Rewrite the LUT and brick-max tables, and schedule their upload.

        Bricks are painted phase by phase, each phase coarsest to finest, so
        a finer brick covers the coarser fallback under it and a later phase
        covers an earlier one (design 5.8).  The GPU upload is deferred to the
        next ``renderer.render()``.

        Parameters
        ----------
        levels : np.ndarray
            ``(N,)`` 1-based level per brick.
        grids : np.ndarray
            ``(N, 3)`` brick grid position ``(g0, g1, g2)`` at its level.
        slot_grid_pos : np.ndarray
            ``(N, 3)`` cache grid position ``(sz, sy, sx)`` of each brick's
            slot.
        brick_max : np.ndarray
            ``(N,)`` per-brick maximum, for the MIP early-out.
        phases : sequence of np.ndarray
            Index arrays into the bricks, in painting order.
        loop_below : int
            See :func:`paint_lut`.
        """
        values = np.column_stack(
            [slot_grid_pos[:, 2], slot_grid_pos[:, 1], slot_grid_pos[:, 0], levels]
        ).astype(np.uint8)
        paint_lut(
            self.lut_data,
            values,
            np.asarray(levels, dtype=np.int64),
            np.asarray(grids, dtype=np.int64).reshape(-1, 3),
            phases,
            self._spans,
            self._counts,
            extra=self.brick_max_data,
            extra_values=np.asarray(brick_max, dtype=np.float32),
            loop_below=loop_below,
        )
        if _GPU_LOGGER.isEnabledFor(logging.DEBUG):
            lut_levels, n_cells = np.unique(self.lut_data[..., 3], return_counts=True)
            _GPU_LOGGER.debug(
                "paint_lut  bricks=%d  lut_cells_by_level=%s",
                len(levels),
                {int(lv): int(n) for lv, n in zip(lut_levels, n_cells) if lv > 0},
            )
        self.lut_tex.update_range((0, 0, 0), self.lut_tex.size)
        self.brick_max_tex.update_range((0, 0, 0), self.brick_max_tex.size)


def build_lut_texture(base_layout: BlockLayout3D) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the LUT texture (zeroed = all out-of-bounds).

    Notes
    -----
    Channel assignments follow the axis convention:

    - ``lut[d, h, w, 0]`` = tile_x = sx (cache W axis)
    - ``lut[d, h, w, 1]`` = tile_y = sy (cache H axis)
    - ``lut[d, h, w, 2]`` = tile_z = sz (cache D axis)
    - ``lut[d, h, w, 3]`` = level  (1 = finest; 0 = out-of-bounds)

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.

    Returns
    -------
    lut_data : np.ndarray
        Backing uint8 array of shape ``(gD, gH, gW, 4)``.
    lut_tex : gfx.Texture
        RGBA8UI 3D texture.
    """
    gd, gh, gw = base_layout.grid_dims
    lut_data = np.zeros((gd, gh, gw, 4), dtype=np.uint8)
    lut_tex = gfx.Texture(lut_data, dim=3, format="rgba8uint")
    return lut_data, lut_tex


def build_brick_max_texture(
    base_layout: BlockLayout3D,
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the per-brick max intensity texture (zeroed).

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.

    Returns
    -------
    brick_max_data : np.ndarray
        Backing float32 array of shape ``(gD, gH, gW)``.
    brick_max_tex : gfx.Texture
        R32Float 3D texture.
    """
    gd, gh, gw = base_layout.grid_dims
    brick_max_data = np.zeros((gd, gh, gw), dtype=np.float32)
    brick_max_tex = gfx.Texture(brick_max_data, dim=3, format="r32float")
    return brick_max_data, brick_max_tex
