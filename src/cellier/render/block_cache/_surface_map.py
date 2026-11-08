"""The mask's coarse level, for ``target_bricks="surface"`` (masked image 8.3).

A MaskedImage that loads only the bricks near its surface has to know where
the surface is before it plans its target.  It learns that from the mask's
backstop: every backstop brick the atlas writes is copied into a
:class:`SurfaceMap`, the backstop level as one array per slice (the map
*is* the coarse level, filled as its bricks land).  The plan then keeps

- the mask's target bricks that touch a coarse cell crossing the threshold
  (the min and max over the cell and its 26 neighbours straddle it: a margin of
  one coarse cell), and
- the value's target bricks within ``surface_thickness`` of one, mapped into
  the mask's grid.

A coarse cell no brick has filled is counted as crossing, so a cell the map
does not know keeps its bricks: the filter only ever drops a brick it can see
is far from the surface.  A thin feature the coarse level averages below the
threshold is invisible to it, and is never loaded.

Every query is a box sum over a summed-area table of the crossing cells, so
a plan's filter is a handful of vectorised array operations whatever the
number of bricks; none of numpy's GIL-holding calls (``np.unique``,
weighted ``np.bincount``, ``np.repeat``) is used.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

__all__ = [
    "SurfaceMap",
    "crossing_cells",
    "mask_bricks_near_surface",
    "value_bricks_near_surface",
]

#: Slices a map keeps, newest last.  A coarse level is small (a 512 cubed
#: mask at 16x is 32 cubed, 128 KiB), so a time series scrubbed back and forth
#: keeps its recent time points without reading their backstops again.
MAX_SLICES = 16


class SurfaceMap:
    """The mask's backstop level, one array per slice, filled brick by brick.

    Parameters
    ----------
    level_shapes : sequence of tuple[int, int, int]
        Each level's shape over the three fetched data axes, finest first.
    block_size : int
        Brick side in voxels.
    overlap : int
        The padded block's halo, voxels a side.

    Attributes
    ----------
    level : int or None
        The 1-based level the map keeps (the backstop's); set by the plan.
        ``None`` keeps nothing.
    """

    def __init__(
        self,
        level_shapes: Sequence[tuple[int, int, int]],
        block_size: int,
        overlap: int,
    ) -> None:
        self.level_shapes = [tuple(int(n) for n in shape) for shape in level_shapes]
        self.block_size = int(block_size)
        self.overlap = int(overlap)
        self.level: int | None = None
        # selection -> (values, known); oldest first.
        self._slices: OrderedDict[Hashable, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        #: Slices whose backstop the scheduler reported done (resident or
        #: given up): the map is as full as it will get.
        self._settled: set[Hashable] = set()

    # -- filling ---------------------------------------------------------------

    def set_level(self, level: int) -> None:
        """Keep *level* from now on; a change of level forgets every slice."""
        level = int(level)
        if level != self.level:
            self.level = level
            self.clear()

    def clear(self) -> None:
        """Forget every slice (the store changed)."""
        self._slices.clear()
        self._settled.clear()

    @property
    def shape(self) -> tuple[int, int, int] | None:
        """The kept level's shape, or ``None`` with no level."""
        if self.level is None:
            return None
        return self.level_shapes[min(self.level, len(self.level_shapes)) - 1]

    def _arrays(self, selection: Hashable) -> tuple[np.ndarray, np.ndarray]:
        arrays = self._slices.get(selection)
        if arrays is None:
            shape = self.shape
            arrays = (np.zeros(shape, np.float32), np.zeros(shape, bool))
            self._slices[selection] = arrays
            while len(self._slices) > MAX_SLICES:
                old, _ = self._slices.popitem(last=False)
                self._settled.discard(old)
        else:
            self._slices.move_to_end(selection)
        return arrays

    def record(
        self, selection: Hashable, level: int, grid: Sequence[int], data: Any
    ) -> None:
        """Copy a written brick's voxels in, when it is at the kept level.

        Parameters
        ----------
        selection : hashable
            The brick's slice (its key's interned selection).
        level : int
            The brick's 1-based level.
        grid : sequence of int
            Its brick grid index over the three fetched axes.
        data : array-like
            The padded block as written to the atlas.
        """
        if self.level is None or int(level) != self.level:
            return
        values, known = self._arrays(selection)
        block = np.asarray(data)
        block = block.reshape(block.shape[-3:])
        target, source = [], []
        for axis in range(3):
            start = int(grid[axis]) * self.block_size
            stop = min(start + self.block_size, values.shape[axis])
            if stop <= start:
                return
            target.append(slice(start, stop))
            source.append(slice(self.overlap, self.overlap + stop - start))
        values[tuple(target)] = block[tuple(source)]
        known[tuple(target)] = True

    def settle(self, selection: Hashable) -> None:
        """Mark *selection*'s backstop done: plan from what the map holds."""
        self._settled.add(selection)

    # -- reading ---------------------------------------------------------------

    def ready(self, selection: Hashable) -> bool:
        """Whether a target can be planned for *selection*.

        True once every coarse cell is known, or once the backstop is
        reported done (a backstop truncated or given up leaves cells
        unknown, and they count as crossing).
        """
        if selection in self._settled:
            return True
        arrays = self._slices.get(selection)
        return arrays is not None and bool(arrays[1].all())

    def crossing(self, selection: Hashable, threshold: float) -> np.ndarray | None:
        """The coarse cells crossing *threshold* (unknown cells count), or None."""
        arrays = self._slices.get(selection)
        if arrays is None:
            return None
        values, known = arrays
        return crossing_cells(values, known, threshold)


def _neighbourhood(array: np.ndarray, reduce) -> np.ndarray:
    """*reduce* over each voxel's 3x3x3 block, edge voxels repeated.

    *reduce* is ``np.minimum`` or ``np.maximum``; separable, one axis at a
    time.
    """
    out = array
    for axis in range(3):
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        padded = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        lo = np.take(padded, np.arange(0, n), axis=axis)
        mid = np.take(padded, np.arange(1, n + 1), axis=axis)
        hi = np.take(padded, np.arange(2, n + 2), axis=axis)
        out = reduce(reduce(lo, mid), hi)
    return out


def crossing_cells(
    values: np.ndarray, known: np.ndarray, threshold: float
) -> np.ndarray:
    """The coarse cells the surface at *threshold* may pass through.

    A cell crosses where the min and max over it and its 26 neighbours
    straddle the threshold, or where any of them is unknown.
    """
    low = _neighbourhood(values, np.minimum)
    high = _neighbourhood(values, np.maximum)
    unknown = _neighbourhood(~known, np.maximum)
    return ((low <= threshold) & (high >= threshold)) | unknown


def _summed(cells: np.ndarray) -> np.ndarray:
    """A summed-area table with a zero row in front on every axis."""
    table = np.zeros(tuple(n + 1 for n in cells.shape), dtype=np.int64)
    table[1:, 1:, 1:] = cells.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return table


def _any_in_boxes(table: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Whether each inclusive cell box ``[lo, hi]`` holds a set cell.

    *lo* and *hi* are ``(N, 3)`` cell indices; boxes are clipped to the grid
    and an empty one holds nothing.
    """
    shape = np.asarray(table.shape) - 1
    a = np.clip(lo, 0, shape)
    b = np.clip(hi + 1, 0, shape)
    empty = (b <= a).any(axis=1)
    a0, a1, a2 = a.T
    b0, b1, b2 = b.T
    total = (
        table[b0, b1, b2]
        - table[a0, b1, b2]
        - table[b0, a1, b2]
        - table[b0, b1, a2]
        + table[a0, a1, b2]
        + table[a0, b1, a2]
        + table[b0, a1, a2]
        - table[a0, a1, a2]
    )
    return (total > 0) & ~empty


def _brick_boxes_level0(
    arr: np.ndarray,
    block_size: int,
    level_shapes: Sequence[tuple[int, int, int]],
    scales: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Each planned brick's extent in level-0 voxel coordinates, ``(lo, hi)``.

    *arr* rows are ``(level, g0, g1, g2)``; a brick spans its voxels' edges,
    ``[g * B - 0.5, min((g + 1) * B, n) - 0.5]`` at its level.
    """
    levels = arr[:, 0].astype(np.int64) - 1
    grid = arr[:, 1:4].astype(np.float64)
    shapes = np.asarray(level_shapes, dtype=np.float64)[levels]
    lo_k = grid * block_size - 0.5
    hi_k = np.minimum((grid + 1) * block_size, shapes) - 0.5
    s = np.asarray(scales, dtype=np.float64)[levels]
    t = np.asarray(translations, dtype=np.float64)[levels]
    return s * lo_k + t, s * hi_k + t


#: Slack for a box edge that lands on a cell boundary up to rounding.
_EDGE_EPS = 1e-6


def _cells(lo0, hi0, coarse_scale, coarse_translation):
    """Level-0 extents to the inclusive coarse cells they overlap.

    Coarse cell ``c`` spans ``[c - 0.5, c + 0.5)`` in its own voxels.  An
    edge on a cell boundary touches only the cell it bounds from inside: a
    level-0 brick of a whole number of coarse cells covers just those.
    """
    lo_c = (lo0 - coarse_translation) / coarse_scale
    hi_c = (hi0 - coarse_translation) / coarse_scale
    lo = np.floor(lo_c + 0.5 + _EDGE_EPS).astype(np.int64)
    hi = np.ceil(hi_c + 0.5 - _EDGE_EPS).astype(np.int64) - 1
    return lo, np.maximum(hi, lo)


def mask_bricks_near_surface(
    arr: np.ndarray,
    crossing: np.ndarray,
    *,
    block_size: int,
    level_shapes: Sequence[tuple[int, int, int]],
    scales: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    coarse_level: int,
) -> np.ndarray:
    """The rows of the mask's brick plan *arr* that touch a crossing cell.

    Parameters
    ----------
    arr : np.ndarray
        ``(N, 4)`` planned bricks ``(level, g0, g1, g2)``, in plan order.
    crossing : np.ndarray
        :func:`crossing_cells` of the coarse level.
    block_size : int
        Brick side in voxels.
    level_shapes : sequence of tuple
        The mask's level shapes over the fetched axes.
    scales, translations : sequence of np.ndarray
        Per level, level-k voxel to level-0 voxel over the fetched axes.
    coarse_level : int
        The 1-based level *crossing* is at.

    Returns
    -------
    np.ndarray
        The kept rows, in plan order.
    """
    if len(arr) == 0:
        return arr
    lo0, hi0 = _brick_boxes_level0(arr, block_size, level_shapes, scales, translations)
    k = coarse_level - 1
    lo, hi = _cells(lo0, hi0, np.asarray(scales[k]), np.asarray(translations[k]))
    return arr[_any_in_boxes(_summed(crossing), lo, hi)]


def value_bricks_near_surface(
    arr: np.ndarray,
    crossing: np.ndarray,
    *,
    block_size: int,
    level_shapes: Sequence[tuple[int, int, int]],
    scales: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    value_to_mask: np.ndarray,
    mask_scale: np.ndarray,
    mask_translation: np.ndarray,
    reach_voxels: np.ndarray,
) -> np.ndarray:
    """The rows of the value's brick plan *arr* within reach of a crossing cell.

    Parameters
    ----------
    arr : np.ndarray
        ``(N, 4)`` planned value bricks ``(level, g0, g1, g2)``.
    crossing : np.ndarray
        :func:`crossing_cells` of the mask's coarse level.
    block_size : int
        The value's brick side in voxels.
    level_shapes, scales, translations
        The value's levels, as for :func:`mask_bricks_near_surface`.
    value_to_mask : np.ndarray
        ``(4, 4)``: value level-0 voxels to mask level-0 voxels over the
        fetched axes.
    mask_scale, mask_translation : np.ndarray
        The mask's coarse level, coarse voxel to mask level-0 voxel.
    reach_voxels : np.ndarray
        How far from the surface the value is read, in mask level-0 voxels
        per axis: a value brick is kept within this of a crossing cell.

    Returns
    -------
    np.ndarray
        The kept rows, in plan order.
    """
    if len(arr) == 0:
        return arr
    lo0, hi0 = _brick_boxes_level0(arr, block_size, level_shapes, scales, translations)
    # The box's eight corners into the mask's grid; their bounding box.
    corners = np.stack(
        [np.where(np.asarray(bits, bool), hi0, lo0) for bits in np.ndindex(2, 2, 2)],
        axis=1,
    )  # (N, 8, 3)
    matrix = np.asarray(value_to_mask, dtype=np.float64)
    mapped = corners @ matrix[:3, :3].T + matrix[:3, 3]
    lo_m = mapped.min(axis=1) - reach_voxels
    hi_m = mapped.max(axis=1) + reach_voxels
    lo, hi = _cells(lo_m, hi_m, np.asarray(mask_scale), np.asarray(mask_translation))
    return arr[_any_in_boxes(_summed(crossing), lo, hi)]
