"""Brick planning for the ``"plane"`` render mode (plane rendering design v3, 6).

A multiscale visual in plane mode reads only the bricks its render planes
cross, at levels picked by a rule of the planes' own:

- **Culling rows** (6.2).  Each plane becomes half-spaces in level-0 data
  coordinates, the space the frustum and clipping rows are in: a thin slab
  either side of the plane, and one row per bounded side of its extent.
  A brick is kept when a plane reaches the part of the volume the brick
  *owns* in the lookup table, which is the brick a sample there reads
  (``owned_cell_boxes``).  The planes are a union, so they are applied as
  a mask before the level bands, not appended to the row list.
- **The level rule** (6.3).  A level's voxel is measured in the plane: its
  largest world size over every in-plane direction.  A brick draws level
  ``k`` where a screen pixel at its distance, times the bias, is between
  the voxels of levels ``k - 1`` and ``k``: the 2D rule's transition,
  applied per brick.
- **Orthographic** (6.3): the whole visual takes one level by the same
  measure, and a plan that does not fit the atlas takes the next coarser
  level until it does.

``plan_plane_bricks`` is the one planner both multiscale families call in
plane mode.  In a volume mode they keep their own planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cellier.render._frustum import bricks_in_frustum_arr
from cellier.render._level_of_detail import (
    cull_mask,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.render.lut_indirection._cell_brick_rule import (
    level_brick_counts,
    level_cell_spans,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.render._render_planes import ReducedPlanes

#: The slab kept either side of a plane, in level-0 voxels.  A sample at
#: the plane reads the brick that owns the place it is in (and, with linear
#: sampling, its neighbours from that brick's own padding), so the bricks
#: the plane passes through are enough.  The slab only covers the
#: difference between float64 here and float32 in the shader for a plane
#: lying on a brick face, where both sides are then kept.
PLANE_REACH_VOXELS = 1e-3


@dataclass(frozen=True)
class PlanePlanning:
    """A visual's render planes, as the brick planner takes them.

    Parameters
    ----------
    row_sets : list[np.ndarray]
        One ``(N_i, 4)`` array of culling rows per plane: ``n . p + d >= 0``
        in level-0 data coordinates, pygfx order.
    voxel_size : np.ndarray
        ``(n_levels,)``: the world size of one voxel of each level in the
        planes (6.3's ``size(k)``); with several planes, the smallest.
    metric : np.ndarray
        ``(3,)``: world units per level-0 voxel along each data axis, pygfx
        order.  Turns a distance in level-0 voxels into a world distance.
    """

    row_sets: list[np.ndarray]
    voxel_size: np.ndarray
    metric: np.ndarray


def owned_cell_boxes(geo) -> list[dict]:
    """Per level, the box of the volume each brick owns in the lookup table.

    The shaders find a brick through the LUT: a position is in one base
    cell (a level-0 brick), and the cell names the level-k brick that owns
    it, by the one rule of ``_cell_brick_rule``.  The box of the cells a
    brick owns is therefore where that brick is read from.  On a pyramid
    with integer level ratios and block-averaged translations it is the
    brick's own box; otherwise the two differ by up to a voxel of the
    level, and a cull on the brick's own box would drop a brick the plane
    is drawn from.

    Cached on *geo* for as long as its level grids are the same.

    Parameters
    ----------
    geo : MultiscaleBrickLayout3D
        The visual's brick layout.

    Returns
    -------
    list[dict]
        One per level, shaped like a level of ``build_level_grids``:
        ``arr``, ``centres`` and ``half_extents`` (both ``(M, 3)``, level-0
        data coordinates, pygfx order), ``centre_abs_max``, and ``owns``
        (``(M,)`` bool: the brick owns at least one cell).
    """
    cached = getattr(geo, "_owned_cell_boxes", None)
    if cached is not None and cached[0] is geo._level_grids:
        return cached[1]
    block = int(geo.block_size)
    grid_dims = np.asarray(geo.base_layout.grid_dims, dtype=np.int64)
    extent = np.asarray(geo.level_shapes[0], dtype=np.float64)
    spans = level_cell_spans(geo.n_levels, 3, geo._scale_vecs_data)
    counts = level_brick_counts(spans, grid_dims, block, geo.level_shapes)
    boxes = []
    for level, grid in enumerate(geo._level_grids):
        brick = grid["arr"][:, 1:4].astype(np.int64)  # (gz, gy, gx)
        span = np.asarray(spans[level], dtype=np.int64)
        count = np.asarray(counts[level], dtype=np.int64)
        start = np.minimum(brick * span, grid_dims)
        stop = np.where(
            brick >= count - 1, grid_dims, np.minimum((brick + 1) * span, grid_dims)
        )
        # Voxel i is centred on i: cells [start, stop) cover this interval.
        low = start * block - 0.5
        high = np.minimum(stop * block, extent) - 0.5
        centres = ((low + high) / 2.0)[:, ::-1]
        boxes.append(
            {
                "arr": grid["arr"],
                "centres": centres,
                "half_extents": ((high - low) / 2.0)[:, ::-1],
                "centre_abs_max": np.abs(centres).max(axis=0),
                "owns": (stop > start).all(axis=1),
            }
        )
    geo._owned_cell_boxes = (geo._level_grids, boxes)
    return boxes


def data_to_world_affine(
    to_level0: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """``world = linear @ data + offset`` from the visual's world-to-data map.

    Parameters
    ----------
    to_level0 : callable
        Maps ``(N, 3)`` points of the scene's rendered space (pygfx order)
        to level-0 data coordinates of the displayed axes (pygfx order).
        Affine on the displayed axes.

    Returns
    -------
    linear : np.ndarray
        ``(3, 3)``.
    offset : np.ndarray
        ``(3,)``.
    """
    probes = np.concatenate([np.zeros((1, 3)), np.eye(3)])
    mapped = np.asarray(to_level0(probes), dtype=np.float64)
    world_to_data = (mapped[1:] - mapped[0]).T
    linear = np.linalg.inv(world_to_data)
    return linear, -linear @ mapped[0]


def plane_voxel_sizes(
    reduced: ReducedPlanes, linear: np.ndarray, level_scales: np.ndarray
) -> np.ndarray:
    """The world size of one voxel of each level, measured in the planes.

    For a plane with in-plane axes ``B`` (3 x 2) and a level with per-axis
    voxel size ``s`` (level-0 voxels), a unit in-plane direction ``B c``
    crosses ``|diag(1 / s) W B c|`` voxels per world unit, ``W`` being
    ``world -> data``.  A voxel is longest along the ``c`` that makes that
    least: ``1 / sqrt(smallest eigenvalue of B^T W^T diag(1 / s^2) W B)``.
    Over every in-plane direction, so the answer does not change when the
    frame is spun about the normal.

    Parameters
    ----------
    reduced : ReducedPlanes
        The planes the view draws; at least one.
    linear : np.ndarray
        ``(3, 3)`` ``data -> world`` linear block, pygfx order.
    level_scales : np.ndarray
        ``(n_levels, 3)``: each level's voxel in level-0 voxels, pygfx order.

    Returns
    -------
    np.ndarray
        ``(n_levels,)``.  With several planes, the smallest over them: no
        plane is drawn coarser than asked.
    """
    world_to_data = np.linalg.inv(linear)
    scales = np.asarray(level_scales, dtype=np.float64)
    sizes = np.full(len(scales), np.inf)
    for axis_0, axis_1 in zip(reduced.axis_0, reduced.axis_1, strict=True):
        in_data = world_to_data @ np.stack([axis_0, axis_1], axis=1)  # (3, 2)
        for level, scale in enumerate(scales):
            voxels = in_data / scale[:, None]
            smallest = float(np.linalg.eigvalsh(voxels.T @ voxels)[0])
            size = np.inf if smallest <= 0.0 else 1.0 / np.sqrt(smallest)
            sizes[level] = min(sizes[level], size)
    return sizes


def _half_spaces(reduced: ReducedPlanes, index: int, reach_world: float) -> list:
    """Plane *index*'s region as world half-spaces ``n . w + c >= 0``."""
    origin = reduced.origin[index]
    axis_0, axis_1 = reduced.axis_0[index], reduced.axis_1[index]
    normal = np.cross(axis_0, axis_1)
    offset = float(normal @ origin)
    rows = [(normal, reach_world - offset), (-normal, reach_world + offset)]
    extent, bounded = reduced.extent[index], reduced.bounded[index]
    for axis, low, high in ((axis_0, 0, 1), (axis_1, 2, 3)):
        along = float(axis @ origin)
        if bounded[low]:
            rows.append((axis, -along - float(extent[low])))
        if bounded[high]:
            rows.append((-axis, along + float(extent[high])))
    return rows


def plane_row_sets(
    reduced: ReducedPlanes, linear: np.ndarray, offset: np.ndarray
) -> list[np.ndarray]:
    """Each plane as culling rows (design 6.2).

    A world half-space ``n . w + c >= 0`` is, with ``w = A d + b``, the
    data half-space ``(A^T n) . d + (n . b + c) >= 0``: the normal maps as a
    covector.

    Parameters
    ----------
    reduced : ReducedPlanes
        The planes the view draws.
    linear, offset : np.ndarray
        ``data -> world``, pygfx order.

    Returns
    -------
    list[np.ndarray]
        One ``(N, 4)`` array per plane: two rows for the slab, one for each
        bounded side of the extent.
    """
    normals = reduced.normal
    # A level-0 voxel's box, in world, spans |A| 1 along each world axis.
    voxel_world = np.abs(linear).sum(axis=1)
    out = []
    for index in range(len(reduced)):
        reach = PLANE_REACH_VOXELS * float(np.abs(normals[index]) @ voxel_world)
        out.append(
            np.array(
                [
                    (*(linear.T @ n), float(n @ offset) + c)
                    for n, c in _half_spaces(reduced, index, reach)
                ],
                dtype=np.float64,
            )
        )
    return out


def build_plane_planning(
    reduced: ReducedPlanes,
    to_level0: Callable[[np.ndarray], np.ndarray],
    level_scales: np.ndarray,
) -> PlanePlanning:
    """What the brick planner needs of the planes one view draws.

    Parameters
    ----------
    reduced : ReducedPlanes
        The planes the view draws, in pygfx order.  May be empty: the plan
        is then empty.
    to_level0 : callable
        The visual's rendered-space to level-0 data map, pygfx order.
    level_scales : np.ndarray
        ``(n_levels, 3)``: each level's voxel in level-0 voxels, pygfx order.

    Returns
    -------
    PlanePlanning
    """
    linear, offset = data_to_world_affine(to_level0)
    scales = np.asarray(level_scales, dtype=np.float64)
    if len(reduced) == 0:
        return PlanePlanning(
            row_sets=[],
            voxel_size=np.full(len(scales), np.inf),
            metric=np.linalg.norm(linear, axis=0),
        )
    return PlanePlanning(
        row_sets=plane_row_sets(reduced, linear, offset),
        voxel_size=plane_voxel_sizes(reduced, linear, scales),
        metric=np.linalg.norm(linear, axis=0),
    )


def plane_thresholds(
    voxel_size: np.ndarray, fov_y_rad: float, screen_height_px: float, bias: float
) -> list[float]:
    """World distances at which a plane's bricks change level (design 6.3).

    A screen pixel at distance ``d`` covers ``d / focal`` world units.
    Level ``k + 1`` takes over where that, times the bias, reaches the
    geometric mean of the two levels' voxels.

    Returns
    -------
    list[float]
        ``n_levels - 1`` distances, as ``select_levels_from_cache`` takes
        them.
    """
    focal = (screen_height_px / 2.0) / np.tan(fov_y_rad / 2.0)
    bias = max(bias, 1e-6)
    return [
        float(np.sqrt(voxel_size[k] * voxel_size[k + 1])) * focal / bias
        for k in range(len(voxel_size) - 1)
    ]


def plane_level_orthographic(
    voxel_size: np.ndarray,
    view_height_world: float,
    screen_height_px: float,
    bias: float,
) -> int | None:
    """The one level an orthographic view draws a plane at (design 6.3).

    Returns
    -------
    int or None
        1-indexed, as ``select_levels_arr_forced`` takes it; ``None`` for a
        degenerate view.
    """
    if view_height_world <= 0 or screen_height_px <= 0:
        return None
    biased = view_height_world / screen_height_px * max(bias, 1e-6)
    level = 1
    for k in range(len(voxel_size) - 1):
        if biased < float(np.sqrt(voxel_size[k] * voxel_size[k + 1])):
            break
        level = k + 2
    return level


def plane_cull_masks(
    geo, planning: PlanePlanning, clip_rows: np.ndarray | None
) -> list[np.ndarray]:
    """Per level, the bricks a plane is drawn from that no clipping row drops.

    A brick is kept when a plane reaches the box of the cells it owns
    (:func:`owned_cell_boxes`) and its own box reaches the kept side of
    every clipping row, as in a volume mode.
    """
    masks = []
    for grid, owned in zip(geo._level_grids, owned_cell_boxes(geo), strict=True):
        keep = owned["owns"] & cull_mask(owned, None, plane_row_sets=planning.row_sets)
        if clip_rows is not None:
            keep &= cull_mask(grid, clip_rows)
        masks.append(keep)
    return masks


def plan_plane_bricks(
    geo,
    planning: PlanePlanning,
    *,
    clip_rows: np.ndarray | None,
    camera_pos_data: np.ndarray,
    frustum_planes: np.ndarray | None,
    fov_y_rad: float,
    screen_height_px: float,
    bias: float,
    force_level: int | None,
    view_height_world: float = 0.0,
    max_bricks: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Plan the bricks a visual's render planes draw, nearest in world first.

    Cull to the planes and the clipping rows, then the plane's level bands
    (or the forced or orthographic level), the distance sort and the
    frustum test (design 6.1).

    Parameters
    ----------
    geo : MultiscaleBrickLayout3D
        The visual's brick layout: level grids, scales, block size.
    planning : PlanePlanning
        From :func:`build_plane_planning`.
    clip_rows : np.ndarray or None
        The visual's clipping planes as culling rows.
    camera_pos_data : np.ndarray
        The camera in level-0 data coordinates, pygfx order.
    frustum_planes : np.ndarray or None
        The frustum as culling rows, or ``None`` not to cull to it.
    fov_y_rad, screen_height_px, bias :
        The view and the settled bias.
    force_level : int or None
        1-indexed level that replaces the rule.
    view_height_world : float
        The visible world height of an orthographic view (``fov_y_rad`` 0).
    max_bricks : int or None
        The room the atlas has for the target.  An orthographic plan over
        it takes the next coarser level until it fits (D-P48); a
        perspective plan is left to the truncation that follows.

    Returns
    -------
    arr : np.ndarray
        ``(N, 4)`` rows ``[level, g0, g1, g2]``.
    stats : dict
        ``n_ranked`` (bricks left by the cull, summed over the levels the
        rule could use), ``n_selected`` (before the frustum test),
        ``level`` (the one level of a forced or orthographic plan, else
        ``None``) and ``levels_stepped`` (coarser steps taken to fit).
    """
    grids = geo._level_grids
    n_levels = geo.n_levels
    keep = plane_cull_masks(geo, planning, clip_rows)
    rows = frustum_planes
    if clip_rows is not None:
        rows = clip_rows if rows is None else np.concatenate([rows, clip_rows])

    def finish(arr: np.ndarray) -> tuple[np.ndarray, int]:
        arr = sort_arr_by_distance(
            arr,
            camera_pos_data,
            geo.block_size,
            scale_vecs_shader=geo._scale_arr_shader,
            translation_vecs_shader=geo._translation_arr_shader,
            # Nearest in world units, as the level rule measures: this is
            # the order bricks load in and what a full atlas keeps.
            metric=planning.metric,
        )
        selected = len(arr)
        if rows is not None:
            arr, _ = bricks_in_frustum_arr(
                arr,
                geo.block_size,
                rows,
                level_scale_arr_shader=geo._scale_arr_shader,
                level_translation_arr_shader=geo._translation_arr_shader,
            )
        return arr, selected

    def at_level(level: int) -> tuple[np.ndarray, int]:
        return finish(
            select_levels_arr_forced(geo.base_layout, level, grids, keep=keep)
        )

    stats = {"n_ranked": 0, "n_selected": 0, "level": None, "levels_stepped": 0}
    orthographic = force_level is None and fov_y_rad <= 0
    level = force_level
    if orthographic:
        level = plane_level_orthographic(
            planning.voxel_size, view_height_world, screen_height_px, bias
        )
    if level is not None:
        level = min(max(int(level), 1), n_levels)
        arr, selected = at_level(level)
        while (
            orthographic
            and max_bricks is not None
            and len(arr) > max_bricks
            and level < n_levels
        ):
            level += 1
            stats["levels_stepped"] += 1
            arr, selected = at_level(level)
        stats.update(
            n_ranked=int(keep[level - 1].sum()), n_selected=selected, level=level
        )
        return arr, stats

    thresholds = None
    if fov_y_rad > 0:
        thresholds = plane_thresholds(
            planning.voxel_size, fov_y_rad, screen_height_px, bias
        )
    arr, selected = finish(
        select_levels_from_cache(
            grids,
            n_levels,
            camera_pos_data,
            thresholds=thresholds,
            base_layout=geo.base_layout,
            keep=keep,
            metric=planning.metric if thresholds is not None else None,
        )
    )
    stats.update(n_ranked=int(sum(int(k.sum()) for k in keep)), n_selected=selected)
    return arr, stats
