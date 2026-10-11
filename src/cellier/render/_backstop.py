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

import itertools
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
    from collections.abc import Sequence

    from cellier.visuals._loading import ProgressiveLoadingConfig


def backstop_level(loading: ProgressiveLoadingConfig, n_levels: int) -> int:
    """The 1-based backstop level, clamped to the pyramid like ``force_level``."""
    if loading.backstop_level is None:
        return n_levels
    return min(max(int(loading.backstop_level), 1), n_levels)


def backstop_cap(loading: ProgressiveLoadingConfig, n_slots: int) -> int:
    """Slots the backstop may hold: ``backstop_max_slot_fraction`` of *n_slots*."""
    return math.floor(loading.backstop_max_slot_fraction * n_slots)


def intermediate_cap(loading: ProgressiveLoadingConfig, n_slots: int) -> int:
    """Slots all the intermediate stages together may hold."""
    return math.floor(loading.intermediate_max_slot_fraction * n_slots)


def intermediate_bricks(
    target_arr: np.ndarray,
    offset: int,
    backstop_level: int,
    grid: Sequence[int],
    spans: Sequence[Sequence[int]],
    counts: Sequence[Sequence[int]],
) -> np.ndarray:
    """The intermediate stage of a target plan: its ancestors *offset* levels up.

    ``plans/progressive_intermediate_level_design_v2.md``, section 3.  For a
    target brick at level ``L`` its ancestors are the bricks of level
    ``L + offset`` that own one of its base cells, by the shared cell -> brick
    rule (:mod:`cellier.render.lut_indirection._cell_brick_rule`), so the
    stage covers exactly what the lookup table will look up.  Where the
    coarser level's span is a whole multiple of the finer one's that is one
    brick; where the spans do not nest it is a range of up to a few along an
    axis, and all of them are taken.

    Arithmetic only: 2D and 3D, any level mix.  Keys another block of the
    plan already holds are not removed; that is :func:`desired_bricks`' work.

    Parameters
    ----------
    target_arr : np.ndarray
        ``(N, 1 + ndim)`` rows ``[level, g...]``, levels 1-based, in load
        order, as planned and before any truncation.
    offset : int
        Levels coarser than each target brick's own level; at least 1.
    backstop_level : int
        1-based.  A brick whose ancestors would be at or beyond it has none:
        dropped, not clamped, since the backstop already draws there.
    grid : sequence of int
        The base grid's dimensions (one cell per finest-level brick).
    spans, counts : sequence of sequence of int
        Per level (entry ``k`` is level ``k + 1``): base cells per brick and
        brick count per axis, from ``level_cell_spans`` and
        ``level_brick_counts``.

    Returns
    -------
    np.ndarray
        ``(M, 1 + ndim)`` ``int64`` rows.  Every ancestor once, at the place
        of the first target brick that has it, so the stage stays in load
        order.  One brick's ancestors are in ascending grid order, the first
        axis slowest.
    """
    grid_arr = np.asarray(grid, dtype=np.int64)
    ndim = len(grid_arr)
    target_arr = np.asarray(target_arr, dtype=np.int64).reshape(-1, 1 + ndim)
    target_arr = target_arr[target_arr[:, 0] + offset < backstop_level]
    if not len(target_arr):
        return np.empty((0, 1 + ndim), dtype=np.int64)
    spans_arr = np.asarray(spans, dtype=np.int64)
    counts_arr = np.asarray(counts, dtype=np.int64)
    index = target_arr[:, 0] - 1
    g = target_arr[:, 1:]
    span, count = spans_arr[index], counts_arr[index]
    up_span, up_count = spans_arr[index + offset], counts_arr[index + offset]
    # The base cells each brick owns (``cell_range``); the last brick of a
    # level owns the grid's tail.  A brick that owns no cell stands for the
    # nearest one.
    start = np.minimum(g * span, grid_arr)
    stop = np.where(g >= count - 1, grid_arr, np.minimum((g + 1) * span, grid_arr))
    empty = start >= stop
    start = np.where(empty, np.minimum(start, grid_arr - 1), start)
    stop = np.where(empty, start + 1, stop)
    # Their owners at the coarser level (``brick_for_cell``): along an axis,
    # the range from the owner of the first cell to the owner of the last.
    low = np.minimum(start // up_span, up_count - 1)
    high = np.minimum((stop - 1) // up_span, up_count - 1)
    # Step through the longest range any brick has, clamped to each brick's
    # own (one step on a pyramid whose spans nest).
    longest = (high - low).max(axis=0) + 1
    up_level = target_arr[:, 0] + offset
    rows = [
        np.column_stack(
            [
                up_level,
                *(
                    np.minimum(low[:, axis] + step, high[:, axis])
                    for axis, step in enumerate(steps)
                ),
            ]
        )
        for steps in itertools.product(*(range(int(n)) for n in longest))
    ]
    # Interleaved, so each brick's ancestors stay at its place in the order.
    stacked = np.stack(rows, axis=1).reshape(-1, 1 + ndim)
    _, first = np.unique(stacked, axis=0, return_index=True)
    return stacked[np.sort(first)]


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
