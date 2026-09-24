"""The coarse backstop the multiscale planners load ahead of the target.

Design v3, 5.9.  A backstop is one pyramid level, loaded before every target
read (the scheduler's ``BACKSTOP`` class), so a view is blurry rather than
blank -- or showing the previous slice -- while the target loads.  All four
planners (image and labels, 2D and 3D) call these helpers, which work on the
same level grids and sort / cull functions as their target planning.

What the settings mean is :class:`~cellier.visuals.ProgressiveLoadingConfig`'s
business; this module turns them into brick arrays in load order.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from cellier.render._frustum import bricks_in_frustum_arr
from cellier.render._level_of_detail import sort_arr_by_distance
from cellier.render._level_of_detail_2d import (
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
)

if TYPE_CHECKING:
    from cellier.visuals._loading import ProgressiveLoadingConfig


def backstop_level(loading: ProgressiveLoadingConfig, n_levels: int) -> int:
    """The 1-based backstop level, clamped to the pyramid like ``force_level``."""
    if loading.backstop_level is None:
        return n_levels
    return min(max(int(loading.backstop_level), 1), n_levels)


def backstop_cap(loading: ProgressiveLoadingConfig, n_slots: int) -> int:
    """Slots the backstop may hold: ``backstop_max_slot_fraction`` of *n_slots*."""
    return math.floor(loading.backstop_max_slot_fraction * n_slots)


def backstop_bricks_3d(
    level_grids: list[dict],
    level: int,
    camera_pos_data: np.ndarray,
    block_size: int,
    scale_arr_shader: np.ndarray,
    translation_arr_shader: np.ndarray,
    frustum_planes: np.ndarray | None = None,
) -> np.ndarray:
    """The backstop level's bricks, nearest the camera first.

    Parameters
    ----------
    level_grids : list[dict]
        The planner's precomputed level grids (``build_level_grids``).
    level : int
        1-based backstop level (:func:`backstop_level`).
    camera_pos_data : np.ndarray
        Camera position in level-0 data space, shader order, as the target's
        distance sort takes it.
    block_size : int
        Brick side in voxels.
    scale_arr_shader, translation_arr_shader : np.ndarray
        Per-level brick placement, as the target's sort and cull take them.
    frustum_planes : np.ndarray or None
        Cull to these planes (``backstop_extent="view"``), or ``None`` for
        the whole volume (``"full"``).

    Returns
    -------
    np.ndarray
        ``(N, 4)`` rows ``[level, g0, g1, g2]``.
    """
    arr = level_grids[level - 1]["arr"]
    arr = sort_arr_by_distance(
        arr,
        camera_pos_data,
        block_size,
        scale_vecs_shader=scale_arr_shader,
        translation_vecs_shader=translation_arr_shader,
    )
    if frustum_planes is not None:
        arr, _ = bricks_in_frustum_arr(
            arr,
            block_size,
            frustum_planes,
            level_scale_arr_shader=scale_arr_shader,
            level_translation_arr_shader=translation_arr_shader,
        )
    return np.asarray(arr, dtype=np.int64).reshape(-1, 4)


def backstop_tiles_2d(
    level_grids: list[dict],
    level: int,
    camera_pos: np.ndarray,
    block_size: int,
    scale_arr_shader: np.ndarray,
    translation_arr_shader: np.ndarray,
    view_min: np.ndarray | None = None,
    view_max: np.ndarray | None = None,
) -> np.ndarray:
    """The backstop level's tiles, nearest the canvas centre first.

    Parameters
    ----------
    level_grids : list[dict]
        The planner's precomputed tile grids (``build_tile_grids_2d``).
    level : int
        1-based backstop level (:func:`backstop_level`).
    camera_pos : np.ndarray
        ``(x, y, 0)``: the canvas centre in level-0 data space.
    block_size : int
        Tile side in pixels.
    scale_arr_shader, translation_arr_shader : np.ndarray
        Per-level tile placement, shader order ``(x, y)``.
    view_min, view_max : np.ndarray or None
        The viewport in level-0 data space, ``(x, y)``.  When given
        (``backstop_extent="view"``), tiles are culled to it plus one
        backstop tile of margin; ``None`` keeps the whole slice.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` rows ``[level, g0, g1]``.
    """
    arr = level_grids[level - 1]["arr"]
    arr = sort_tiles_by_distance_2d(
        arr,
        camera_pos,
        block_size,
        level_scale_arr_shader=scale_arr_shader,
        level_translation_arr_shader=translation_arr_shader,
    )
    if view_min is not None and view_max is not None:
        margin = block_size * np.asarray(scale_arr_shader[level - 1][:2], float)
        arr, _ = viewport_cull_2d(
            arr,
            block_size,
            np.asarray(view_min, float) - margin,
            np.asarray(view_max, float) + margin,
            level_scale_arr_shader=scale_arr_shader,
            level_translation_arr_shader=translation_arr_shader,
        )
    return np.asarray(arr, dtype=np.int64).reshape(-1, 3)
