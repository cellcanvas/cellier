"""Brute-force reference sampler for multiscale level placement.

The oracle the shader-agreement tests compare against.  It implements the
mapping in ``plans/multiscale_level_transform_v2.md`` directly -- on purpose
not through ``cellier.render._level_mapping`` -- so a mistake in the helpers
cannot hide in both places.

    u = (p - t) / s      data (level-0 voxel) -> level coordinates
    nearest              voxel ``floor(u + 0.5)``
    linear               voxel centres at integer ``u``
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def data_to_level(p, scale, translation) -> np.ndarray:
    """``u = (p - t) / s`` per axis; *p* is ``(N, ndim)``."""
    return (np.asarray(p, float) - np.asarray(translation, float)) / np.asarray(
        scale, float
    )


def sample_nearest(level_array, p, scale, translation, outside=0) -> np.ndarray:
    """Nearest-neighbour value of *level_array* at data points *p*.

    Voxel ``floor(u + 0.5)``; points outside the level get *outside*.
    """
    u = data_to_level(p, scale, translation)
    idx = np.floor(u + 0.5).astype(np.int64)
    shape = np.asarray(level_array.shape)
    inside = ((idx >= 0) & (idx < shape)).all(axis=1)
    out = np.full(len(idx), outside, dtype=level_array.dtype)
    out[inside] = level_array[tuple(idx[inside].T)]
    return out


def sample_linear(level_array, p, scale, translation) -> np.ndarray:
    """Linear value at *p*; ``map_coordinates`` puts voxel centres at ``u``.

    Edges repeat the outermost voxel (``mode="nearest"``).
    """
    u = data_to_level(p, scale, translation)
    return ndimage.map_coordinates(
        np.asarray(level_array, dtype=np.float64), u.T, order=1, mode="nearest"
    )
