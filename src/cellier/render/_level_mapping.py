"""The one definition of where a pyramid level sits in data space.

Every multiscale path -- CPU planning, LUT painting, the shaders, picking --
maps between a level's voxels and the data (level-0 voxel) space through the
store's ``level -> data`` transform, a per-axis scale ``s`` and translation
``t`` in level-0 voxel units.  This module states that mapping once.

Convention: **level-0 centre coordinates** (OME-Zarr).  Level-0 voxel ``i``
is centred on data coordinate ``p = i`` and covers ``[i - 0.5, i + 0.5]``.
Level-k voxel ``i`` is centred on level coordinate ``u = i``.

==========================  ===============================================
quantity                    definition
==========================  ===============================================
level -> data               ``p = s * u + t``
data -> level               ``u = (p - t) / s``
level-k voxel ``i``         ``u`` in ``[i - 0.5, i + 0.5]``
brick ``b`` of the level    voxels ``[b*bs, (b+1)*bs)``;
                            ``u`` in ``[b*bs - 0.5, (b+1)*bs - 0.5]``
brick ``b`` in data         ``[s (b*bs - 0.5) + t, s ((b+1)*bs - 0.5) + t]``
nearest sample (labels)     voxel ``floor(u + 0.5)``
linear sample (image)       texel coordinate ``u + 0.5``
==========================  ===============================================

All functions are per axis and broadcast with NumPy: ``scale`` and
``translation`` are scalars or ``(ndim,)`` arrays in the same axis order as
the coordinates.  See ``plans/multiscale_level_transform_v2.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def level_to_data(u: ArrayLike, scale: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Level coordinates to data (level-0 voxel) coordinates: ``s * u + t``."""
    return np.asarray(scale, dtype=np.float64) * np.asarray(
        u, dtype=np.float64
    ) + np.asarray(translation, dtype=np.float64)


def data_to_level(p: ArrayLike, scale: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Data (level-0 voxel) coordinates to level coordinates: ``(p - t) / s``."""
    return (
        np.asarray(p, dtype=np.float64) - np.asarray(translation, dtype=np.float64)
    ) / np.asarray(scale, dtype=np.float64)


def level_voxel_extent(
    index: ArrayLike, scale: ArrayLike, translation: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Data-space ``(low, high)`` of level voxel *index*: ``u`` in ``i -+ 0.5``."""
    index = np.asarray(index, dtype=np.float64)
    return (
        level_to_data(index - 0.5, scale, translation),
        level_to_data(index + 0.5, scale, translation),
    )


def brick_box_data(
    brick_index: ArrayLike,
    block_size: ArrayLike,
    scale: ArrayLike,
    translation: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Data-space ``(low, high)`` corners of level bricks.

    Brick ``b`` holds level voxels ``[b*bs, (b+1)*bs)``, so it covers ``u``
    in ``[b*bs - 0.5, (b+1)*bs - 0.5]`` and data
    ``[s (b*bs - 0.5) + t, s ((b+1)*bs - 0.5) + t]``.

    Parameters
    ----------
    brick_index : array-like
        Brick grid indices, e.g. ``(N, ndim)``.
    block_size : array-like
        Brick side in level voxels, scalar or per axis.
    scale, translation : array-like
        The level's ``level -> data`` scale and translation, per axis.

    Returns
    -------
    low, high : ndarray
        Same shape as *brick_index* (after broadcasting).  The box is
        unclipped: the last brick of a level can extend past the level's
        extent.
    """
    first = np.asarray(brick_index, dtype=np.float64) * np.asarray(
        block_size, dtype=np.float64
    )
    return (
        level_to_data(first - 0.5, scale, translation),
        level_to_data(
            first + np.asarray(block_size, dtype=np.float64) - 0.5, scale, translation
        ),
    )


def level_extent_data(
    shape: ArrayLike, scale: ArrayLike, translation: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Data-space ``(low, high)`` a level of *shape* covers.

    Voxels ``0 .. N - 1`` cover ``u`` in ``[-0.5, N - 0.5]``.
    """
    shape = np.asarray(shape, dtype=np.float64)
    return (
        level_to_data(np.full_like(shape, -0.5), scale, translation),
        level_to_data(shape - 0.5, scale, translation),
    )


def brick_centre_data(
    brick_index: ArrayLike,
    block_size: ArrayLike,
    scale: ArrayLike,
    translation: ArrayLike,
) -> np.ndarray:
    """Data-space centre of level bricks: the midpoint of :func:`brick_box_data`.

    ``s * ((b + 0.5) * bs - 0.5) + t``.
    """
    first = np.asarray(brick_index, dtype=np.float64) * np.asarray(
        block_size, dtype=np.float64
    )
    half = np.asarray(block_size, dtype=np.float64) / 2.0
    return level_to_data(first + half - 0.5, scale, translation)


def implied_power_of_two(levels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Scale and translation assumed for 1-based *levels* without transforms.

    Planning helpers accept missing per-level vectors (tests, legacy
    callers); they then assume an isotropic power-of-two block-averaged
    pyramid: ``s = 2 ** (level - 1)`` and ``t = (s - 1) / 2``.
    """
    scale = np.power(2.0, np.asarray(levels, dtype=np.float64) - 1.0)
    return scale, (scale - 1.0) / 2.0


def base_cell_range(
    low: ArrayLike, high: ArrayLike, block_size: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Half-open range of base-grid cells a data-space interval overlaps.

    Base cell ``c`` holds level-0 voxels ``[c*bs, (c+1)*bs)``, so it covers
    data ``[c*bs - 0.5, (c+1)*bs - 0.5)``.  Unclamped.

    Returns
    -------
    start, stop : ndarray of int64
        ``floor((low + 0.5) / bs)`` and ``ceil((high + 0.5) / bs)``.
    """
    bs = np.asarray(block_size, dtype=np.float64)
    start = np.floor((np.asarray(low, dtype=np.float64) + 0.5) / bs)
    stop = np.ceil((np.asarray(high, dtype=np.float64) + 0.5) / bs)
    return start.astype(np.int64), stop.astype(np.int64)
