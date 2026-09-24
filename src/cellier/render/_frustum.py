"""Frustum geometry and conservative AABB visibility tests.

Contains pure math functions only — no pygfx camera objects or scene
helpers.  Camera extraction and wireframe construction belong in
application-level code.

Axis-order note
---------------
BlockKey3D stores (gz, gy, gx) in DHW / numpy order, but world
coordinates are (x, y, z).  The conversion used throughout is:

    world_x = gx * block_world
    world_y = gy * block_world
    world_z = gz * block_world
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from cellier.render._level_mapping import brick_box_data, implied_power_of_two
from cellier.render._level_of_detail import CORNER_OFFSETS

if TYPE_CHECKING:
    from cellier.render.block_cache import BlockKey3D

# ---------------------------------------------------------------------------
# Plane computation
# ---------------------------------------------------------------------------


def _compute_plane_parameters(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Return (4,) plane coefficients [a, b, c, d].

    Normal points toward the side where the three points wind
    counter-clockwise.  A point ``p`` is *inside* when
    ``dot(p, [a,b,c]) + d >= 0``.
    """
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 0.0])
    normal = normal / norm
    d = -np.dot(normal, p0)
    return np.array([normal[0], normal[1], normal[2], d])


def frustum_planes_from_corners(corners: np.ndarray) -> np.ndarray:
    """Compute the 6 frustum half-space planes from frustum corners.

    Parameters
    ----------
    corners : ndarray, shape (2, 4, 3)
        ``corners[0]`` = near plane, ``corners[1]`` = far plane.
        Within each plane: (left-bottom, right-bottom, right-top,
        left-top).

    Returns
    -------
    planes : ndarray, shape (6, 4)
        Plane coefficients ordered: near, far, left, right, top, bottom.
        A point ``p`` is *inside* the frustum when
        ``dot(plane[:3], p) + plane[3] >= 0`` for all 6 planes.
        Normals all point *inward*.
    """
    n = corners[0]  # near: lb, rb, rt, lt
    f = corners[1]  # far:  lb, rb, rt, lt

    planes = np.empty((6, 4), dtype=np.float64)
    planes[0] = _compute_plane_parameters(n[0], n[2], n[1])  # near
    planes[1] = _compute_plane_parameters(f[0], f[1], f[2])  # far
    planes[2] = _compute_plane_parameters(n[3], n[0], f[0])  # left
    planes[3] = _compute_plane_parameters(n[1], n[2], f[2])  # right
    planes[4] = _compute_plane_parameters(n[2], n[3], f[3])  # top
    planes[5] = _compute_plane_parameters(n[0], n[1], f[1])  # bottom
    return planes


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------


def compute_brick_aabb_corners(
    brick_key: BlockKey3D,
    block_size: int,
    level_scale_arr_shader: np.ndarray | None = None,
    level_translation_arr_shader: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 8 world-space AABB corners for a brick.

    Parameters
    ----------
    brick_key : BlockKey3D
        Grid address of the brick (level, gz, gy, gx).
    block_size : int
        Brick side length in voxels at level 1.
    level_scale_arr_shader : ndarray, shape (n_levels, 3) or None
        Per-level scale in shader order ``(x=W, y=H, z=D)``.
    level_translation_arr_shader : ndarray, shape (n_levels, 3) or None
        Per-level translation in shader order.

    Returns
    -------
    corners : ndarray, shape (8, 3)
    """
    k = brick_key.level - 1
    if level_scale_arr_shader is not None and level_translation_arr_shader is not None:
        sv = np.asarray(level_scale_arr_shader[k], dtype=np.float64)  # (W, H, D)
        tv = np.asarray(level_translation_arr_shader[k], dtype=np.float64)
    else:
        sv, tv = implied_power_of_two(brick_key.level)
    # Centre convention (plan v2, D1), shader order (x=W, y=H, z=D).
    low, high = brick_box_data(
        [brick_key.g2, brick_key.g1, brick_key.g0], block_size, sv, tv
    )
    return low + CORNER_OFFSETS * (high - low)  # (8, 3)


# ---------------------------------------------------------------------------
# Frustum culling — array pipeline (primary hot path)
# ---------------------------------------------------------------------------


def bricks_in_frustum_arr(
    arr: np.ndarray,
    block_size: int,
    frustum_planes: np.ndarray,
    level_scale_arr_shader: np.ndarray | None = None,
    level_translation_arr_shader: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Conservative AABB frustum test over a brick array.

    Operates entirely on numpy arrays — no BlockKey3D objects.

    Parameters
    ----------
    arr : ndarray, shape (M, 4), dtype int32
        Columns: ``[level, gz_c, gy_c, gx_c]``.
    block_size : int
        Level-1 brick side length in voxels.
    frustum_planes : ndarray, shape (6, 4)
        Inward-pointing half-space planes.
    level_scale_arr_shader : ndarray, shape (n_levels, 3) or None
        Per-level scale in shader order ``(x=W, y=H, z=D)``.
    level_translation_arr_shader : ndarray, shape (n_levels, 3) or None
        Per-level translation in shader order.

    Returns
    -------
    visible_arr : ndarray, shape (K, 4)
        Subset of rows that pass the frustum test.
    timings : dict
        Wall-clock timings in milliseconds.
    """
    M = len(arr)
    if M == 0:
        return arr, {
            "build_corners_ms": 0.0,
            "einsum_ms": 0.0,
            "mask_ms": 0.0,
        }

    levels = arr[:, 0].astype(np.int64)

    t0 = time.perf_counter()

    if level_scale_arr_shader is not None and level_translation_arr_shader is not None:
        scale = np.asarray(level_scale_arr_shader, dtype=np.float64)[levels - 1]
        translation = np.asarray(level_translation_arr_shader, dtype=np.float64)[
            levels - 1
        ]
    else:
        s_, t_ = implied_power_of_two(levels)
        scale, translation = s_[:, None], t_[:, None]
    # Centre convention (plan v2, D1): (M, 3) boxes in shader order (W, H, D).
    brick_mins, brick_maxs = brick_box_data(
        arr[:, [3, 2, 1]], block_size, scale, translation
    )
    # (M, 8, 3) -- per-axis brick widths broadcast over corners.
    all_corners = (
        brick_mins[:, np.newaxis, :]
        + CORNER_OFFSETS[np.newaxis, :, :] * (brick_maxs - brick_mins)[:, np.newaxis, :]
    )

    build_corners_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dists = (
        np.einsum("ijk,lk->ijl", all_corners, frustum_planes[:, :3])
        + frustum_planes[:, 3]
    )
    einsum_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    visible_mask = (dists.max(axis=1) >= 0.0).all(axis=1)  # (M,)
    mask_ms = (time.perf_counter() - t0) * 1000

    timings = {
        "build_corners_ms": build_corners_ms,
        "einsum_ms": einsum_ms,
        "mask_ms": mask_ms,
    }
    return arr[visible_mask], timings


# ---------------------------------------------------------------------------
# Frustum culling — dict pipeline (for compatibility / debugging)
# ---------------------------------------------------------------------------


def bricks_in_frustum(
    brick_keys: set[BlockKey3D] | dict[BlockKey3D, int],
    block_size: int,
    frustum_planes: np.ndarray,
) -> tuple[dict[BlockKey3D, int], dict]:
    """Conservative AABB frustum test over a set or dict of brick keys.

    Parameters
    ----------
    brick_keys : set or dict[BlockKey3D, int]
        Candidate bricks.  If a dict, the values are preserved.
    block_size : int
        Level-1 brick side length in voxels.
    frustum_planes : ndarray, shape (6, 4)
        Inward-pointing half-space planes.

    Returns
    -------
    visible : dict[BlockKey3D, int]
        Subset that passes the frustum test.
    timings : dict
    """
    if isinstance(brick_keys, dict):
        keys_list = list(brick_keys.keys())
        values = brick_keys
    else:
        keys_list = list(brick_keys)
        values = dict.fromkeys(keys_list, 0)

    n = len(keys_list)
    if n == 0:
        return {}, {"build_corners_ms": 0.0, "einsum_ms": 0.0, "mask_ms": 0.0}

    t0 = time.perf_counter()
    index = np.array([[k.g2, k.g1, k.g0] for k in keys_list], dtype=np.float64)
    s_, t_ = implied_power_of_two(np.array([k.level for k in keys_list]))
    brick_mins, brick_maxs = brick_box_data(index, block_size, s_[:, None], t_[:, None])
    all_corners = (
        brick_mins[:, np.newaxis, :]
        + CORNER_OFFSETS[np.newaxis, :, :] * (brick_maxs - brick_mins)[:, np.newaxis, :]
    )
    build_corners_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dists = (
        np.einsum("ijk,lk->ijl", all_corners, frustum_planes[:, :3])
        + frustum_planes[:, 3]
    )
    einsum_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    any_inside = dists.max(axis=1) >= 0.0
    visible_mask = any_inside.all(axis=1)
    mask_ms = (time.perf_counter() - t0) * 1000

    visible = {keys_list[i]: values[keys_list[i]] for i in range(n) if visible_mask[i]}
    timings = {
        "build_corners_ms": build_corners_ms,
        "einsum_ms": einsum_ms,
        "mask_ms": mask_ms,
    }
    return visible, timings
