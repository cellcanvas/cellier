"""LOD level selection for tiled 2D image rendering.

The 2D LOD selector uses zoom level (pixels-per-world-unit) to choose
the coarsest level where each tile still has sufficient resolution.

For orthographic cameras, all tiles are at the same effective "distance",
so a single global LOD level is selected based on the zoom ratio.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from cellier.render._level_mapping import (
    brick_box_data,
    brick_centre_data,
    implied_power_of_two,
)

if TYPE_CHECKING:
    from cellier.render.lut_indirection._layout_2d import BlockLayout2D


def build_tile_grids_2d(
    base_layout: BlockLayout2D,
    n_levels: int,
    level_shapes: list[tuple[int, int]] | None = None,
    scale_vecs_shader: list[np.ndarray] | None = None,
    translation_vecs_shader: list[np.ndarray] | None = None,
) -> list[dict]:
    """Precompute static per-level coarse tile grids.  Called once at startup.

    For level k (1-indexed):
    - Coarse grid dims are computed from ``level_shapes`` when provided.
    - World-space tile centres incorporate per-axis scale and translation.

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels.
    level_shapes : list[tuple[int, int]] or None
        ``(H, W)`` shape per level (data order).
    scale_vecs_shader : list[np.ndarray] or None
        ``(2,)`` per level in shader order ``(x=W, y=H)``.
    translation_vecs_shader : list[np.ndarray] or None
        ``(2,)`` per level in shader order ``(x=W, y=H)``.

    Returns
    -------
    grids : list[dict]
        One dict per level (index 0 = level 1).  Each dict contains:

        ``arr`` : ndarray, shape (M_k, 3), dtype int32
            ``[level, gy_c, gx_c]`` for every coarse tile.
        ``centres`` : ndarray, shape (M_k, 2), dtype float64
            World-space ``(x, y)`` centre of each coarse tile.
    """
    bs = base_layout.block_size
    gh, gw = base_layout.grid_dims

    grids = []
    for level in range(1, n_levels + 1):
        k = level - 1  # 0-indexed

        if level_shapes is not None:
            # data order: (H=axis0, W=axis1)
            h_k, w_k = level_shapes[k]
            cgh = (h_k + bs - 1) // bs
            cgw = (w_k + bs - 1) // bs
        else:
            scale = 1 << k
            cgh = (gh + scale - 1) // scale
            cgw = (gw + scale - 1) // scale

        gy_c, gx_c = np.meshgrid(
            np.arange(cgh, dtype=np.int32),
            np.arange(cgw, dtype=np.int32),
            indexing="ij",
        )
        gy_c = gy_c.ravel()
        gx_c = gx_c.ravel()

        lvl_col = np.full(len(gy_c), level, dtype=np.int32)
        arr = np.stack([lvl_col, gy_c, gx_c], axis=1)  # (M_k, 3)

        if scale_vecs_shader is not None and translation_vecs_shader is not None:
            sv = np.asarray(scale_vecs_shader[k], dtype=np.float64)[:2]  # (W, H)
            tv = np.asarray(translation_vecs_shader[k], dtype=np.float64)[:2]
        else:
            sv, tv = implied_power_of_two(level)
        # Centre convention (plan v2, D1), shader order (x=W, y=H).
        centres = brick_centre_data(np.stack([gx_c, gy_c], axis=1), bs, sv, tv)

        grids.append({"arr": arr, "centres": centres})

    return grids


def select_lod_2d(
    level_grids: list[dict],
    n_levels: int,
    viewport_width_px: float,
    voxel_width: float,
    lod_bias: float = 1.0,
    force_level: int | None = None,
    level_scale_factors: list[float] | None = None,
) -> np.ndarray:
    """Select LOD level based on the camera zoom.

    Uses a single global LOD level for all tiles based on the current
    zoom ratio.  When ``level_scale_factors`` are provided, searches
    over actual scale factors instead of assuming power-of-2.

    Parameters
    ----------
    level_grids : list[dict]
        Precomputed output of ``build_tile_grids_2d``.
    n_levels : int
        Number of LOD levels.
    viewport_width_px : float
        Viewport width in logical pixels.
    voxel_width : float
        Visible width in level-0 voxels.  The caller is responsible for
        converting from world units using the visual's world-to-data
        transform before passing this value.
    lod_bias : float
        Multiplicative bias. 1.0 = neutral (default).
    force_level : int or None
        If set, bypass zoom selection and use this level.
    level_scale_factors : list[float] or None
        Effective isotropic zoom factor per level (geometric mean).
        ``level_scale_factors[0]`` is always 1.0 (finest).

    Returns
    -------
    arr : ndarray, shape (M, 3), dtype int32
        ``[level, gy_c, gx_c]`` rows for all selected tiles.
    """
    if force_level is not None:
        level = min(max(force_level, 1), n_levels)
        return level_grids[level - 1]["arr"].copy()

    if voxel_width <= 0 or viewport_width_px <= 0:
        return level_grids[0]["arr"].copy()

    # Level-0 voxels per screen pixel.
    screen_pixel_size = voxel_width / viewport_width_px
    biased = screen_pixel_size * max(lod_bias, 1e-6)

    if level_scale_factors is not None:
        # Transition thresholds at the geometric mean of consecutive
        # scale factors.  This matches the old log2 formula for
        # power-of-2 data and generalises to arbitrary scales.
        selected_level = 1
        for k in range(len(level_scale_factors) - 1):
            threshold = math.sqrt(level_scale_factors[k] * level_scale_factors[k + 1])
            if biased >= threshold:
                selected_level = k + 2  # bump to next level (1-indexed)
            else:
                break
        selected_level = max(1, min(n_levels, selected_level))
    else:
        # Fallback: log2 formula (power-of-2 assumption).
        ideal = 1.0 + math.log2(max(biased, 1e-6))
        selected_level = max(1, min(n_levels, round(ideal)))

    return level_grids[selected_level - 1]["arr"].copy()


def _tile_transforms(
    arr: np.ndarray,
    level_scale_arr_shader: np.ndarray | None,
    level_translation_arr_shader: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(M, 2)`` scale and translation, shader order ``(x=W, y=H)``."""
    levels = arr[:, 0].astype(np.int64)
    if level_scale_arr_shader is not None and level_translation_arr_shader is not None:
        scale = np.asarray(level_scale_arr_shader, dtype=np.float64)[levels - 1, :2]
        translation = np.asarray(level_translation_arr_shader, dtype=np.float64)[
            levels - 1, :2
        ]
        return scale, translation
    s, t = implied_power_of_two(levels)
    return s[:, None], t[:, None]


def sort_tiles_by_distance_2d(
    arr: np.ndarray,
    camera_pos: np.ndarray,
    block_size: int,
    level_scale_arr_shader: np.ndarray | None = None,
    level_translation_arr_shader: np.ndarray | None = None,
) -> np.ndarray:
    """Sort tiles nearest-to-camera-center first.

    Parameters
    ----------
    arr : ndarray, shape (M, 3), dtype int32
        ``[level, gy_c, gx_c]`` rows.
    camera_pos : ndarray, shape (3,)
        Camera world-space position ``(x, y, z)``.
    block_size : int
        Tile side length in pixels.
    level_scale_arr_shader : ndarray, shape (n_levels, 2) or None
        Per-level scale factors in shader order ``(x=W, y=H)``.
    level_translation_arr_shader : ndarray, shape (n_levels, 2) or None
        Per-level translation in shader order ``(x=W, y=H)``.

    Returns
    -------
    sorted_arr : ndarray
        Same shape, sorted by Euclidean distance from camera center.
    """
    if len(arr) == 0:
        return arr

    scale, translation = _tile_transforms(
        arr, level_scale_arr_shader, level_translation_arr_shader
    )
    centres = brick_centre_data(arr[:, [2, 1]], block_size, scale, translation)

    dx = centres[:, 0] - float(camera_pos[0])
    dy = centres[:, 1] - float(camera_pos[1])
    dist_sq = dx * dx + dy * dy

    order = np.argsort(dist_sq)
    return arr[order]


def viewport_cull_2d(
    arr: np.ndarray,
    block_size: int,
    view_min: np.ndarray,
    view_max: np.ndarray,
    level_scale_arr_shader: np.ndarray | None = None,
    level_translation_arr_shader: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Remove tiles that lie entirely outside the viewport.

    Parameters
    ----------
    arr : ndarray
        ``(M, 3)`` rows ``(level, gy, gx)``, in load order.
    block_size : int
        Tile side length in data pixels at finest level.
    view_min : ndarray, shape (2,)
        Viewport AABB minimum ``(x, y)`` in level-0 data space.
    view_max : ndarray, shape (2,)
        Viewport AABB maximum ``(x, y)`` in level-0 data space.
    level_scale_arr_shader : ndarray, shape (n_levels, 2) or None
        Per-level scale in shader order ``(x=W, y=H)``.
    level_translation_arr_shader : ndarray, shape (n_levels, 2) or None
        Per-level translation in shader order ``(x=W, y=H)``.

    Returns
    -------
    culled : ndarray
        The rows of ``arr`` that overlap the viewport, order preserved.
    n_culled : int
        Number of tiles removed.
    """
    if len(arr) == 0:
        return arr, 0

    scale, translation = _tile_transforms(
        arr, level_scale_arr_shader, level_translation_arr_shader
    )
    low, high = brick_box_data(arr[:, [2, 1]], block_size, scale, translation)

    visible = (
        (high[:, 0] > view_min[0])
        & (low[:, 0] < view_max[0])
        & (high[:, 1] > view_min[1])
        & (low[:, 1] < view_max[1])
    )
    return arr[visible], len(arr) - int(np.sum(visible))
