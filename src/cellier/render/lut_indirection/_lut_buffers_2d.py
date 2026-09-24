"""LUT parameter and block-scales uniform buffers for the 2D image block shader.

All vec2 values use ``(x=W, y=H)`` axis convention to match WGSL.
Uniform structs are padded to vec4 boundaries per WGSL alignment rules.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from pygfx.resources import Buffer

from cellier.render.lut_indirection._cell_brick_rule import (
    UNBOUNDED_BRICK_COUNT,
    level_brick_counts,
    level_cell_spans,
)

if TYPE_CHECKING:
    from cellier.render.block_cache._cache_parameters_2d import (
        BlockCacheParameters2D,
    )
    from cellier.render.lut_indirection._layout_2d import BlockLayout2D

# ---- LUT params --------------------------------------------------------

LUT_PARAMS_DTYPE = np.dtype(
    [
        ("block_size_x", "<f4"),
        ("block_size_y", "<f4"),
        ("overlap", "<f4"),
        ("_pad0", "<f4"),
        ("cache_size_x", "<f4"),
        ("cache_size_y", "<f4"),
        ("_pad1", "<f4"),
        ("_pad2", "<f4"),
        ("lut_size_x", "<f4"),
        ("lut_size_y", "<f4"),
        ("_pad3", "<f4"),
        ("_pad4", "<f4"),
        ("vol_size_x", "<f4"),
        ("vol_size_y", "<f4"),
        ("_pad5", "<f4"),
        ("_pad6", "<f4"),
    ]
)


def build_lut_params_buffer_2d(
    base_layout: BlockLayout2D,
    cache_info: BlockCacheParameters2D,
) -> Buffer:
    """Build the LUT parameter uniform buffer.

    Spatial uniforms are emitted in voxel-space (pixel units), not
    proxy-grid units. This keeps shader sampling aligned with the true
    image footprint when geometry uses real ``(W, H)`` extents.

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest (level 1) resolution.
    cache_info : BlockCacheParameters2D
        Cache sizing metadata.

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_lut_params`` in the shader.
    """
    gh, gw = base_layout.grid_dims
    bs = float(base_layout.block_size)
    ov = float(cache_info.overlap)
    gs = float(cache_info.grid_side)
    h, w = base_layout.volume_shape
    padded_block = bs + 2.0 * ov
    cache_extent = gs * padded_block

    # NOTE: Must be 0-d array, not shape (1,).
    data = np.zeros((), dtype=LUT_PARAMS_DTYPE)
    data["block_size_x"] = bs
    data["block_size_y"] = bs
    data["overlap"] = ov
    data["cache_size_x"] = cache_extent
    data["cache_size_y"] = cache_extent
    data["lut_size_x"] = float(gw)
    data["lut_size_y"] = float(gh)
    data["vol_size_x"] = float(w)
    data["vol_size_y"] = float(h)

    return Buffer(data, force_contiguous=True)


# ---- Block scales -------------------------------------------------------

MAX_LEVELS = 10

# Each level is a separate vec4 field: the float scale, then the integer
# cell -> tile rule (base cells per tile, tile count) as exact small floats,
# then the level's translation in level-0 voxels (plan v2, D1).
BLOCK_SCALES_DTYPE = np.dtype(
    [(f"scale_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)]
    + [(f"span_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)]
    + [(f"bricks_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)]
    + [(f"offset_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)]
)


def build_block_scales_buffer_2d(
    level_scale_vecs_data: list[np.ndarray] | None = None,
    n_levels: int | None = None,
    level_shapes: list[tuple[int, int]] | None = None,
    block_size: int | None = None,
    level_translation_vecs_data: list[np.ndarray] | None = None,
) -> Buffer:
    """Build the block-scales uniform buffer.

    Level 0 is reserved (all zeros -- renders black).
    Level k (1-indexed, 1 = finest) gets ``1 / scale_factor`` per axis.

    The buffer also carries the cell -> tile rule from ``_cell_brick_rule``:
    ``span_k`` is base cells per level-k tile and ``bricks_k`` the level-k
    tile count.  The LUT writer uses the same numbers, which keeps the
    shader's tile lookup in step with the LUT on pyramids whose level ratio is
    not an integer.

    Input vectors are in data-axis order ``(sy, sx)``; shader
    fields ``[0], [1]`` are ``(x=W, y=H)`` so the assignment reverses.

    Parameters
    ----------
    level_scale_vecs_data : list[np.ndarray] or None
        Per-level scale vectors in 2D data order ``(sy, sx)``.
    n_levels : int or None
        Fallback: number of LOD levels for power-of-2 scales.
    level_shapes : list of tuple of int or None
        Per-level ``(H, W)`` shapes.  Without them the tile counts are left
        unbounded, which matches the LUT writer's own fallback.
    block_size : int or None
        Tile side length in pixels.  Required with ``level_shapes``.
    level_translation_vecs_data : list[np.ndarray] or None
        Per-level translation in level-0 voxels, 2D data order ``(ty, tx)``,
        parallel to *level_scale_vecs_data*.  Stored as ``offset_k`` in
        shader order ``(x, y)``, not inverted.  ``None`` stores zeros.

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_block_scales`` in the shader.
    """
    data = np.zeros((), dtype=BLOCK_SCALES_DTYPE)
    for k in range(MAX_LEVELS):
        data[f"span_{k}"][:2] = 1.0
        data[f"bricks_{k}"][:2] = float(UNBOUNDED_BRICK_COUNT)

    if level_scale_vecs_data is not None:
        n = min(len(level_scale_vecs_data), MAX_LEVELS - 1)
        for k in range(1, n + 1):
            sv = level_scale_vecs_data[k - 1]  # data order: [sy, sx]
            # shader x = W = data axis 1 (sx)
            data[f"scale_{k}"][0] = 1.0 / float(sv[1])
            # shader y = H = data axis 0 (sy)
            data[f"scale_{k}"][1] = 1.0 / float(sv[0])
            data[f"scale_{k}"][2] = 0.0  # unused z
            if level_translation_vecs_data is not None:
                ty, tx = (float(v) for v in level_translation_vecs_data[k - 1])
                data[f"offset_{k}"][:2] = (tx, ty)
    elif n_levels is not None:
        n = min(n_levels, MAX_LEVELS - 1)
        for k in range(1, n + 1):
            s = 1.0 / (2 ** (k - 1))
            data[f"scale_{k}"][0] = s  # x
            data[f"scale_{k}"][1] = s  # y
            data[f"scale_{k}"][2] = 0.0  # unused z
    else:
        n = 0

    spans = level_cell_spans(n, 2, level_scale_vecs_data)
    counts = None
    if level_shapes is not None and block_size is not None:
        grid_dims = tuple(math.ceil(int(size) / block_size) for size in level_shapes[0])
        counts = level_brick_counts(spans, grid_dims, block_size, level_shapes)
    for k in range(1, n + 1):
        # Shader order (x, y) is the reverse of data order (y, x).
        data[f"span_{k}"][:2] = spans[k - 1][::-1]
        if counts is not None:
            data[f"bricks_{k}"][:2] = counts[k - 1][::-1]

    return Buffer(data, force_contiguous=True)
