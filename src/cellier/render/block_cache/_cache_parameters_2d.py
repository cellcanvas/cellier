from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx

#: Ghost border of 2D image and labels tiles, in level-k pixels (plan v2,
#: D5).  Two pixels absorb the translation term and the cell -> tile rule on
#: non-integer pyramids (worst in the repo: 1.34) for linear (1.5 allowed)
#: and nearest (2.0) sampling; one pixel left the 2D image only 0.5.
TILE_BORDER_2D = 2


@dataclass(frozen=True)
class BlockCacheParameters2D:
    """Metadata about the allocated 2D cache texture.

    Attributes
    ----------
    grid_side : int
        Number of tile slots per cache axis.
    n_slots : int
        Total number of slots (``grid_side ** 2``).
    padded_block_size : int
        ``block_size + 2 * overlap`` -- actual pixels per slot axis.
    overlap : int
        Number of border pixels duplicated on each side.
    """

    grid_side: int
    n_slots: int
    padded_block_size: int
    overlap: int


def compute_block_cache_parameters_2d(
    gpu_budget_bytes: int,
    block_size: int,
    overlap: int = TILE_BORDER_2D,
    bytes_per_pixel: int = 4,
) -> BlockCacheParameters2D:
    """Compute cache dimensions that fit within the GPU memory budget.

    This is for 2D blocks.

    Parameters
    ----------
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    block_size : int
        Tile side length in pixels (logical, no overlap).
    overlap : int
        Border pixels duplicated on each side.
    bytes_per_pixel : int
        Bytes per pixel (4 for float32).

    Returns
    -------
    info : BlockCacheParameters2D
        Cache sizing metadata.
    """
    padded = block_size + 2 * overlap
    bytes_per_tile = padded * padded * bytes_per_pixel
    max_slots = gpu_budget_bytes // bytes_per_tile
    # 2D grid: grid_side^2 = n_slots
    grid_side = math.isqrt(max_slots)
    # Ensure at least 2 (slot 0 is reserved).
    grid_side = max(grid_side, 2)
    n_slots = grid_side * grid_side

    return BlockCacheParameters2D(
        grid_side=grid_side,
        n_slots=n_slots,
        padded_block_size=padded,
        overlap=overlap,
    )


_FORMAT_MAP_2D = {np.float32: None, np.int32: "1xi4"}


def build_cache_texture_2d(
    cache_info: BlockCacheParameters2D,
    dtype=None,
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the fixed-size 2D cache texture (zeroed).

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata.
    dtype : np.dtype or None
        Data type for the cache texture. Defaults to float32.
        Pass ``np.int32`` for label caches.

    Returns
    -------
    cache_data : np.ndarray
        The backing array ``(cH, cW)``.
    cache_tex : gfx.Texture
        pygfx 2D texture wrapping ``cache_data``.
    """
    if dtype is None:
        dtype = np.float32
    dtype = np.dtype(dtype)
    pbs = cache_info.padded_block_size
    gs = cache_info.grid_side
    cache_shape = (gs * pbs, gs * pbs)
    cache_data = np.zeros(cache_shape, dtype=dtype)
    fmt = _FORMAT_MAP_2D.get(dtype.type)
    if fmt is not None:
        cache_tex = gfx.Texture(cache_data, dim=2, format=fmt)
    else:
        cache_tex = gfx.Texture(cache_data, dim=2)
    return cache_data, cache_tex
