"""What each LUT cell of a mask can read, for the 2D band's skip (masked image 7.4).

A MaskedImage's 2D band searches for the mask's contour along a fan of rays;
most of those steps run through cells that cannot hold one.  Each LUT cell
therefore carries four numbers, uploaded as an ``rgba32float`` texture the
LUT's shape:

- the min and max of its brick (over the padded block's voxels inside the
  array, a voxel of halo past the brick on every side): recorded per atlas
  slot when the brick is written, then gathered per cell when the LUT is
  painted, so the cells of a coarse brick share one range;
- the level the LUT draws the cell at, so a cell drawn at another level
  than the range was taken at never skips;
- its clearance: the gap, in level-0 voxels on every axis, to the nearest
  cell not wholly on its side of the threshold (a chessboard distance over
  the LUT grid).

In an aligned 2D view the reach of a pixel is flat along the plane's
normal, and the cells above and below the slice are not loaded, so the
clearance is measured within each layer of cells across that axis
(:attr:`CellRanges.layer_axis`); otherwise over the whole grid.

The clearance depends on the threshold.  A threshold change zeroes it at once
(no pixel then exits early, which is exact) and :meth:`CellRanges.refresh`
computes it again, once a frame.  An unloaded cell has no range and is never
wholly on a side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx
from scipy import ndimage as ndi

from cellier.render.block_cache._image_residency import unpack_keys

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["CellRanges"]

#: A clearance that every reach fits in: the whole grid is on one side.
_EVERYWHERE = 1.0e9


class CellRanges:
    """The per-cell ``(min, max, level, clearance)`` table of a mask's atlas.

    Parameters
    ----------
    level_shapes : sequence of tuple[int, int, int]
        Each level's spatial shape, data order ``(z, y, x)``, finest first.
    block_size : int
        Brick side in voxels.
    overlap : int
        The padded block's halo, voxels a side.
    n_slots : int
        Scheduler slots of the atlas.
    grid_side : int
        Slots a side of the atlas's slot grid (atlas slot ``s + 1`` is
        scheduler slot ``s``).
    lut_shape : tuple[int, int, int]
        The LUT's cells, ``(z, y, x)``.
    """

    def __init__(
        self,
        level_shapes: Sequence[tuple[int, int, int]],
        block_size: int,
        overlap: int,
        n_slots: int,
        grid_side: int,
        lut_shape: tuple[int, int, int],
    ) -> None:
        self.level_shapes = [
            tuple(int(n) for n in shape[-3:]) for shape in level_shapes
        ]
        self.block_size = int(block_size)
        self.overlap = int(overlap)
        self.grid_side = int(grid_side)
        #: Per scheduler slot, the range its brick wrote.
        self.brick_min = np.zeros(int(n_slots), dtype=np.float32)
        self.brick_max = np.zeros(int(n_slots), dtype=np.float32)
        self.data = np.zeros((*lut_shape, 4), dtype=np.float32)
        self.texture = gfx.Texture(self.data, dim=3)
        #: The threshold the clearance is for.
        self.threshold = 0.0
        #: The data axis (``z, y, x`` order) a 2D plane is normal to, or None.
        self.layer_axis: int | None = None
        self._stale = False

    # -- bricks ------------------------------------------------------------------

    def record(self, slot: int, key: int, data: np.ndarray) -> None:
        """Keep the range of the padded brick *data* written to *slot*.

        Only voxels inside the level's array count: the zero padding past
        its edge is not the mask's (the shader tests the edge itself).
        """
        levels, _, grids = unpack_keys(np.asarray([key], dtype=np.int64))
        level = int(levels[0])
        shape = self.level_shapes[min(level, len(self.level_shapes)) - 1]
        data = np.asarray(data)
        data = data.reshape(data.shape[-3:])
        halo = 1
        index = []
        for axis in range(3):
            start = int(grids[0, axis]) * self.block_size - self.overlap
            lo = max(self.overlap - halo, -start)
            hi = min(self.overlap + self.block_size + halo, shape[axis] - start)
            index.append(slice(lo, max(hi, lo)))
        part = data[tuple(index)]
        if part.size == 0:
            self.brick_min[slot] = self.brick_max[slot] = 0.0
            return
        self.brick_min[slot] = float(part.min())
        self.brick_max[slot] = float(part.max())

    # -- cells -------------------------------------------------------------------

    def repaint(self, lut_data: np.ndarray) -> None:
        """Gather each cell's range from the bricks the LUT now draws.

        Called after every LUT paint, with the LUT's ``(z, y, x, 4)`` data:
        channels 0 to 2 the slot's grid position ``(sx, sy, sz)``, 3 the
        level (0: no brick).
        """
        lut = np.asarray(lut_data)
        level = lut[..., 3].astype(np.int64)
        loaded = level > 0
        side = self.grid_side
        flat = (
            lut[..., 2].astype(np.int64) * side * side
            + lut[..., 1].astype(np.int64) * side
            + lut[..., 0].astype(np.int64)
        )
        slots = np.clip(flat - 1, 0, len(self.brick_min) - 1)
        self.data[..., 0] = np.where(loaded, self.brick_min[slots], 0.0)
        self.data[..., 1] = np.where(loaded, self.brick_max[slots], 0.0)
        self.data[..., 2] = level
        self._clearance()

    def set_threshold(self, threshold: float) -> None:
        """Adopt a new threshold: the clearance is zeroed until :meth:`refresh`."""
        threshold = float(threshold)
        if threshold == self.threshold:
            return
        self.threshold = threshold
        self.data[..., 3] = 0.0
        self._stale = True
        self.texture.update_full()

    def set_layer_axis(self, axis: int | None) -> None:
        """Measure the clearance within layers across *axis* (data order), or not."""
        axis = None if axis is None else int(axis)
        if axis == self.layer_axis:
            return
        self.layer_axis = axis
        self.data[..., 3] = 0.0
        self._stale = True
        self.texture.update_full()

    def refresh(self) -> None:
        """Compute the clearance again if the threshold changed since."""
        if self._stale:
            self._clearance()

    def _clearance(self) -> None:
        iso = self.threshold
        low, high, level = self.data[..., 0], self.data[..., 1], self.data[..., 2]
        side = np.where(low >= iso, 1, np.where(high < iso, -1, 0))
        side[level == 0] = 0
        clearance = np.zeros(side.shape, dtype=np.float32)
        metric: str | np.ndarray = "chessboard"
        if self.layer_axis is not None:
            # Neighbours within the layer only: each layer is measured alone.
            metric = np.zeros((3, 3, 3), dtype=bool)
            index = [slice(None)] * 3
            index[self.layer_axis] = 1
            metric[tuple(index)] = True
        for value in (-1, 1):
            same = side == value
            if same.all():
                clearance[:] = _EVERYWHERE
            elif same.any():
                # Chessboard cells to the nearest other cell: n is n - 1 between.
                cells = ndi.distance_transform_cdt(same, metric=metric)
                # A layer with no other cell is reported as -1: wholly clear.
                cells = np.where(cells < 0, np.float32(_EVERYWHERE), cells)
                clearance[same] = (cells[same] - 1) * float(self.block_size)
        self.data[..., 3] = clearance
        self._stale = False
        self.texture.update_full()
