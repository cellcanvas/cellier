"""Sparse dirty-brick tracking for the multiscale paint controller."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BrickKey(NamedTuple):
    """Identifier for a level-k brick by its grid coordinates.

    Parameters
    ----------
    level : int
        LOD level.  ``WriteLayer`` only tracks ``level == 0``.
    grid_coords : tuple[int, ...]
        One grid index per *spatial* axis, in data-array axis order.
        Length matches the painted store's ndim (2 for 2-D, 3 for 3-D).
    """

    level: int
    grid_coords: tuple[int, ...]


class WriteLayer:
    """Sparse in-memory tracker of dirty level-0 bricks for a paint session.

    Records *which* bricks have been touched; the painted voxel data
    itself lives in the `WriteBuffer`.  N-dim agnostic — used
    unchanged for 2-D and 3-D paint sessions.

    Parameters
    ----------
    data_store_id :
        UUID of the painted store.  Recorded but unused by ``WriteLayer``
        itself; useful for logs and assertions.
    block_size :
        Side length of one level-0 brick in voxels.  Must match the
        ``render_config.block_size`` of the visual being painted, since
        the paint controller relies on ``WriteLayer`` and the renderer
        agreeing on brick boundaries.
    """

    def __init__(self, data_store_id, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size!r}")
        self.data_store_id = data_store_id
        self.block_size = int(block_size)
        self._dirty: set[BrickKey] = set()

    def mark_dirty(self, key: BrickKey) -> None:
        """Mark a single brick as dirty."""
        self._dirty.add(key)

    def is_dirty(self, key: BrickKey) -> bool:
        """Return True if *key* is currently marked dirty."""
        return key in self._dirty

    def dirty_keys(self) -> set[BrickKey]:
        """Return a copy of the current dirty-brick set."""
        return set(self._dirty)

    def clear(self) -> None:
        """Drop all dirty-brick entries.  Called at session end."""
        self._dirty.clear()

    def voxel_to_brick_key(self, voxel_idx: np.ndarray) -> BrickKey:
        """Map a level-0 voxel index to its enclosing level-0 `BrickKey`.

        Parameters
        ----------
        voxel_idx :
            Shape ``(ndim,)``, integer.  In data-array axis order (e.g.
            ``(z, y, x)`` for a 3-D store).

        Returns
        -------
        BrickKey
            ``level=0`` and ``grid_coords[i] = voxel_idx[i] // block_size``.
        """
        return BrickKey(
            level=0,
            grid_coords=tuple(int(v) // self.block_size for v in voxel_idx),
        )

    def voxels_to_brick_keys(self, voxel_indices: np.ndarray) -> set[BrickKey]:
        """Vectorised `voxel_to_brick_key` over an ``(N, ndim)`` array.

        Returns the unique set of bricks the voxels fall into.
        """
        if voxel_indices.size == 0:
            return set()
        grid = (voxel_indices.astype(np.int64) // self.block_size).astype(np.int64)
        unique_grids = {tuple(int(g) for g in row) for row in grid}
        return {BrickKey(level=0, grid_coords=g) for g in unique_grids}
