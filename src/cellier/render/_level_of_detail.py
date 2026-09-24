"""LOD level selection for bricked volume rendering.

Two-level speedup over a naive enumeration approach.

Optimisation 1 — enumerate coarse grids directly, no deduplication
-------------------------------------------------------------------
For each LOD level k enumerate *only that level's coarse grid*
(e.g. 512 bricks for L3 on a 32³ base grid), filter by its distance
band, and concatenate.  No two levels can produce the same coarse
brick, so no deduplication is ever needed.

Optimisation 2 — precomputed, cached grid data
-----------------------------------------------
The coarse grid index arrays and world-space brick centres depend only
on ``block_size`` and ``grid_dims``, which are fixed at startup.  They
are computed once by ``build_level_grids()`` and cached.  The hot path
(``select_levels_from_cache``) does no allocation — only distance
computation and boolean masking on the cached arrays.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from cellier.render._level_mapping import brick_centre_data, implied_power_of_two

if TYPE_CHECKING:
    from cellier.render.lut_indirection import BlockLayout3D

# Pre-computed (8, 3) offset table for AABB corner construction.
# Reused by the vectorised frustum helper in _frustum.py.
CORNER_OFFSETS = np.array(
    list(itertools.product([0.0, 1.0], repeat=3)), dtype=np.float64
)  # (8, 3)


# ---------------------------------------------------------------------------
# Startup: build per-level coarse grid cache
# ---------------------------------------------------------------------------


def build_level_grids(
    base_layout: BlockLayout3D,
    n_levels: int,
    scale_vecs_shader: list[np.ndarray],
    translation_vecs_shader: list[np.ndarray],
    level_shapes: list[tuple[int, ...]] | None = None,
) -> list[dict]:
    """Precompute static per-level coarse grid arrays.  Called once at startup.

    For level k (1-indexed):

    - The coarse grid dims are computed from ``level_shapes`` (actual
      voxel counts per level) instead of a power-of-2 assumption.
    - World-space brick centres incorporate per-axis scale and
      translation from the level transforms.

    Parameters
    ----------
    base_layout : BlockLayout3D
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels.
    scale_vecs_shader : list[np.ndarray]
        ``(3,)`` per level in shader order ``(x=W, y=H, z=D)``.
    translation_vecs_shader : list[np.ndarray]
        ``(3,)`` per level in shader order ``(x=W, y=H, z=D)``.
    level_shapes : list[tuple[int, ...]] or None
        ``(D, H, W)`` shape per level (data order).  When ``None``,
        coarse grid dims are derived from ``base_layout`` using
        power-of-2 downsampling.

    Returns
    -------
    grids : list[dict]
        One dict per level (index 0 = level 1).  Each dict contains:

        ``arr`` : ndarray, shape (M_k, 4), dtype int32
            ``[level, gz_c, gy_c, gx_c]`` for every coarse brick.
        ``centres`` : ndarray, shape (M_k, 3), dtype float64
            World-space ``(x, y, z)`` centre of each coarse brick.
        ``half_extents`` : ndarray, shape (3,), dtype float64
            Half-brick-width per axis in world space ``(x, y, z)``.
    """
    bs = base_layout.block_size
    gd, gh, gw = base_layout.grid_dims

    grids = []
    for level in range(1, n_levels + 1):
        k = level - 1  # 0-indexed

        # Coarse grid dimensions from actual level shapes.
        if level_shapes is not None:
            # data order: (D=axis0, H=axis1, W=axis2)
            d_k, h_k, w_k = level_shapes[k]
            cgd = (d_k + bs - 1) // bs  # z grid dim
            cgh = (h_k + bs - 1) // bs  # y grid dim
            cgw = (w_k + bs - 1) // bs  # x grid dim
        else:
            scale = 1 << k
            cgd = (gd + scale - 1) // scale
            cgh = (gh + scale - 1) // scale
            cgw = (gw + scale - 1) // scale

        gz_c, gy_c, gx_c = np.meshgrid(
            np.arange(cgd, dtype=np.int32),
            np.arange(cgh, dtype=np.int32),
            np.arange(cgw, dtype=np.int32),
            indexing="ij",
        )
        gz_c = gz_c.ravel()
        gy_c = gy_c.ravel()
        gx_c = gx_c.ravel()

        lvl_col = np.full(len(gz_c), level, dtype=np.int32)
        arr = np.stack([lvl_col, gz_c, gy_c, gx_c], axis=1)  # (M_k, 4)

        sv = np.asarray(scale_vecs_shader[k], dtype=np.float64)  # (W, H, D)
        tv = np.asarray(translation_vecs_shader[k], dtype=np.float64)
        # Centre convention (plan v2, D1): shader order (x=W, y=H, z=D).
        centres = brick_centre_data(np.stack([gx_c, gy_c, gz_c], axis=1), bs, sv, tv)

        half_extents = (bs * sv / 2.0).astype(np.float64)
        grids.append({"arr": arr, "centres": centres, "half_extents": half_extents})

    return grids


# ---------------------------------------------------------------------------
# Hot path: per-update LOD selection using the precomputed cache
# ---------------------------------------------------------------------------


def select_levels_from_cache(
    level_grids: list[dict],
    n_levels: int,
    camera_pos: np.ndarray,
    thresholds: list[float] | None = None,
    base_layout: BlockLayout3D | None = None,
) -> np.ndarray:
    """Select LOD levels using precomputed coarse grid data.

    For each level k, compute distances from the camera to the
    precomputed coarse brick centres, apply the distance band that
    belongs to level k, and collect survivors.  No new coordinate
    arrays are allocated; only distance computation and boolean masking
    on the cached arrays.

    Because the bands partition the distance axis, no brick can appear
    in more than one level — no deduplication needed.

    Parameters
    ----------
    level_grids : list[dict]
        Precomputed output of ``build_level_grids``.
    n_levels : int
        Number of LOD levels.
    camera_pos : np.ndarray
        Camera world-space position ``(x, y, z)``.
    thresholds : list[float] or None
        LOD cutoff distances.  ``thresholds[i]`` is the distance beyond
        which level ``i+2`` is preferred over level ``i+1``.
        If None (or empty), and no ``base_layout`` is supplied to derive
        defaults, every brick is assigned the finest level -- for any
        ``n_levels``.
    base_layout : BlockLayout3D or None
        Enables default thresholds when ``thresholds`` is None.  The
        default cutoffs are multiples of the finest level's world-space
        diagonal (measured from ``level_grids``), so they honour
        anisotropic scale/translation.  When None, ``thresholds`` stays
        empty and all bricks fall to the finest level.

    Returns
    -------
    arr : ndarray, shape (M, 4), dtype int32
        ``[level, gz_c, gy_c, gx_c]`` rows for all selected bricks,
        not yet sorted — call ``sort_arr_by_distance`` next.
    """
    cam = np.asarray(camera_pos, dtype=np.float64)

    if thresholds is None:
        if base_layout is not None:
            # World-space diagonal from the finest level's precomputed grid, so
            # the default bands live in the same units as the distances below
            # (``dist = norm(centres - cam)``).  ``centres`` / ``half_extents``
            # already bake in per-level scale and translation, which the raw
            # voxel ``volume_shape`` diagonal ignored on anisotropic data.
            centres0 = level_grids[0]["centres"]  # (M, 3) world-space
            half0 = level_grids[0]["half_extents"]  # (3,) world-space
            world_min = centres0.min(axis=0) - half0
            world_max = centres0.max(axis=0) + half0
            diag = float(np.linalg.norm(world_max - world_min))
            thresholds = [diag * (i + 1) for i in range(n_levels - 1)]
        else:
            thresholds = []

    # No LOD bands -> every brick uses the finest level.  This guards the
    # ``n_levels >= 2`` case, where the per-level loop below would otherwise
    # index an empty threshold list; it also handles an explicit ``[]``.
    if not thresholds:
        return level_grids[0]["arr"]

    parts: list[np.ndarray] = []

    for level in range(1, n_levels + 1):
        grid = level_grids[level - 1]
        centres = grid["centres"]  # (M_k, 3) — precomputed, no alloc
        arr_k = grid["arr"]  # (M_k, 4)

        diff = centres - cam
        dist = np.sqrt((diff * diff).sum(axis=1))

        if level > 1:
            abs_d = np.abs(diff)
            max_corner_dist = np.sqrt(((abs_d + grid["half_extents"]) ** 2).sum(axis=1))
        else:
            max_corner_dist = dist

        if level == 1:
            # ``thresholds`` is guaranteed non-empty here (empty was handled by
            # the early return above).
            mask = dist < thresholds[0]
        elif level == n_levels:
            mask = max_corner_dist >= thresholds[level - 2]
        else:
            mask = (max_corner_dist >= thresholds[level - 2]) & (
                dist < thresholds[level - 1]
            )

        if mask.any():
            parts.append(arr_k[mask])

    if not parts:
        return np.empty((0, 4), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def select_levels_arr_forced(
    base_layout: BlockLayout3D,
    force_level: int,
    level_grids: list[dict] | None = None,
) -> np.ndarray:
    """Return the full coarse grid for a forced single LOD level.

    Uses the precomputed cache when available; falls back to computing
    from scratch otherwise.

    Parameters
    ----------
    base_layout : BlockLayout3D
        The block layout of the finest (level 1) resolution.
    force_level : int
        1-indexed level to force: 1 is the finest level.  Note this does *not*
        match the 0-indexed naming used for the levels themselves elsewhere
        (``level_grids[0]`` is the finest, as is a store's ``s0``), so a caller
        meaning "the finest level" may reasonably reach for ``0``.  Values are
        clamped to the valid range rather than rejected, matching
        ``select_lod_2d``: without the lower clamp ``force_level=0`` would index
        ``level_grids[-1]`` and silently return the *coarsest* level, and the
        uncached branch below would raise on ``1 << -1``.
    level_grids : list[dict] or None
        If provided, returns ``level_grids[level - 1]["arr"]``
        directly (zero-copy view).
    """
    level = max(force_level, 1)

    if level_grids is not None:
        level = min(level, len(level_grids))
        return level_grids[level - 1]["arr"]

    # The uncached branch has no ``n_levels`` to clamp against, so only the
    # lower bound applies here.
    gd, gh, gw = base_layout.grid_dims
    scale = 1 << (level - 1)
    cgd = (gd + scale - 1) // scale
    cgh = (gh + scale - 1) // scale
    cgw = (gw + scale - 1) // scale

    gz_c, gy_c, gx_c = np.meshgrid(
        np.arange(cgd, dtype=np.int32),
        np.arange(cgh, dtype=np.int32),
        np.arange(cgw, dtype=np.int32),
        indexing="ij",
    )
    gz_c = gz_c.ravel()
    gy_c = gy_c.ravel()
    gx_c = gx_c.ravel()
    lvl = np.full(len(gz_c), level, dtype=np.int32)
    return np.stack([lvl, gz_c, gy_c, gx_c], axis=1)


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def sort_arr_by_distance(
    arr: np.ndarray,
    camera_pos: np.ndarray,
    block_size: int,
    scale_vecs_shader: np.ndarray | list[np.ndarray] | None = None,
    translation_vecs_shader: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    """Sort brick rows nearest-to-camera first.

    Brick centres are computed in level-0 data space, shader order ``(x, y,
    z)``, from the per-level scale and translation vectors
    (``cellier.render._level_mapping.brick_centre_data``), so anisotropic
    datasets sort correctly.  ``camera_pos`` must be in the same space.

    Without ``scale_vecs_shader`` / ``translation_vecs_shader`` it assumes an
    isotropic power-of-two block-averaged pyramid (``implied_power_of_two``),
    which does not handle anisotropic voxel spacing.

    Parameters
    ----------
    arr : ndarray, shape (M, 4), dtype int32
        The array of bricks to sort — columns ``[level, gz, gy, gx]``.
    camera_pos : np.ndarray
        Camera position in world XYZ (shader) space, shape ``(3,)``.
    block_size : int
        Brick edge length in finest-level voxels.
    scale_vecs_shader : list of ndarray, optional
        Per-level scale vectors in shader order ``(sx, sy, sz)``.
        Index 0 = level 1 (finest).
    translation_vecs_shader : list of ndarray, optional
        Per-level translation vectors in shader order ``(tx, ty, tz)``.
        Index 0 = level 1 (finest).

    Returns
    -------
    sorted_arr : ndarray, shape (M, 4)
    """
    if len(arr) == 0:
        return arr

    cam = np.asarray(camera_pos, dtype=np.float64)
    levels = arr[:, 0].astype(np.int64)
    index = arr[:, [3, 2, 1]]  # (gx, gy, gz): shader order

    if scale_vecs_shader is not None and translation_vecs_shader is not None:
        # The same centres as build_level_grids.
        scale = np.asarray(scale_vecs_shader, dtype=np.float64)[levels - 1]
        translation = np.asarray(translation_vecs_shader, dtype=np.float64)[levels - 1]
    else:
        s, t = implied_power_of_two(levels)
        scale, translation = s[:, None], t[:, None]
    centres = brick_centre_data(index, block_size, scale, translation)

    distances = np.sqrt(((centres - cam[:3]) ** 2).sum(axis=1))
    order = np.argsort(distances, kind="stable")
    return arr[order]


# ---------------------------------------------------------------------------
# Convert to BrickKey dict
# ---------------------------------------------------------------------------
