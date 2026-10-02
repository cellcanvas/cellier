"""A per-axis index over item bounds, for slab and plane queries.

``plans/mesh_refactor_v3.md`` S4.  An item is anything with an extent along
one axis: a mesh face (its three vertices' extremes), a point (``min ==
max``), a line segment (its two endpoints' extremes).  The index answers,
without a pass over every item:

- **contained**: the items wholly inside ``[lo, hi]``;
- **overlapping**: the items that touch ``[lo, hi]``.

Layout: the items sorted by their minimum (``sorted_min`` and ``order``, 8
bytes per item with ``float32`` coordinates).  An item's maximum is not
stored; it is asked of the owner for the items a query narrows down to.
Items much longer than the rest (more than ``k`` times the median extent)
are kept in a short "long" list that every query scans, so one outlier does
not widen every search window.

An index is immutable once built, so readers take no lock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

#: Items longer than this many median extents go in the long list.
LONG_ITEM_FACTOR: float = 8.0


def _at_least(value: float, dtype: np.dtype) -> np.generic:
    """The smallest *dtype* value that is ``>= value``."""
    cast = dtype.type(value)
    if float(cast) < value:
        cast = np.nextafter(cast, dtype.type(np.inf))
    return cast


def _at_most(value: float, dtype: np.dtype) -> np.generic:
    """The largest *dtype* value that is ``<= value``."""
    cast = dtype.type(value)
    if float(cast) > value:
        cast = np.nextafter(cast, dtype.type(-np.inf))
    return cast


class AxisBoundsIndex:
    """Items sorted by their minimum along one axis.

    Build one with :meth:`build`.  The comparisons of every query are exact:
    a bound that is not representable in the coordinates' dtype is moved to
    the nearest representable value on the side that keeps the answer the
    same.

    Attributes
    ----------
    sorted_min : np.ndarray
        The short items' minima, ascending.
    order : np.ndarray
        ``int32`` item ids, in the order of ``sorted_min``.
    longest : float
        The largest extent among the short items (``E`` in the plan).
    long_ids, long_min, long_max : np.ndarray
        The long items, and their bounds.
    """

    __slots__ = (
        "_item_max",
        "long_ids",
        "long_max",
        "long_min",
        "longest",
        "order",
        "sorted_min",
    )

    def __init__(
        self,
        sorted_min: np.ndarray,
        order: np.ndarray,
        longest: float,
        long_ids: np.ndarray,
        long_min: np.ndarray,
        long_max: np.ndarray,
        item_max: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.sorted_min = sorted_min
        self.order = order
        self.longest = longest
        self.long_ids = long_ids
        self.long_min = long_min
        self.long_max = long_max
        self._item_max = item_max

    @classmethod
    def build(
        cls,
        item_min: np.ndarray,
        item_max: np.ndarray,
        item_max_of: Callable[[np.ndarray], np.ndarray],
        *,
        long_factor: float = LONG_ITEM_FACTOR,
    ) -> AxisBoundsIndex:
        """Build the index from every item's bounds along the axis.

        Parameters
        ----------
        item_min, item_max : np.ndarray
            ``(n_items,)`` bounds, in the coordinates' own float dtype.  Only
            read here; the index keeps the sorted minima and nothing of
            ``item_max``.
        item_max_of : Callable[[np.ndarray], np.ndarray]
            ``ids -> max`` for a subset of items, in the same dtype.  Called
            by queries, possibly from several threads.
        long_factor : float
            Items with an extent over this many medians go in the long list.

        Returns
        -------
        AxisBoundsIndex
        """
        item_min = np.asarray(item_min)
        item_max = np.asarray(item_max)
        extent = item_max - item_min
        if len(extent):
            is_long = extent > long_factor * float(np.median(extent))
        else:
            is_long = np.zeros(0, dtype=bool)
        short = np.flatnonzero(~is_long).astype(np.int32)
        order = short[np.argsort(item_min[short], kind="stable")]
        long_ids = np.flatnonzero(is_long).astype(np.int32)
        longest = 0.0
        if len(short):
            # The subtraction above rounds; one step up keeps the overlap
            # window wide enough for the item it was rounded down on.
            widest = extent[short].max()
            longest = float(np.nextafter(widest, widest.dtype.type(np.inf)))
        return cls(
            sorted_min=np.ascontiguousarray(item_min[order]),
            order=order,
            longest=longest,
            long_ids=long_ids,
            long_min=item_min[long_ids],
            long_max=item_max[long_ids],
            item_max=item_max_of,
        )

    @property
    def nbytes(self) -> int:
        """Bytes the index holds."""
        return sum(
            array.nbytes
            for array in (
                self.sorted_min,
                self.order,
                self.long_ids,
                self.long_min,
                self.long_max,
            )
        )

    def _window(self, lo: float, hi: float) -> tuple[int, int]:
        """The run of ``sorted_min`` with ``lo <= min <= hi``."""
        dtype = self.sorted_min.dtype
        start = 0
        stop = len(self.sorted_min)
        if lo > -np.inf:
            start = int(np.searchsorted(self.sorted_min, _at_least(lo, dtype), "left"))
        if hi < np.inf:
            stop = int(np.searchsorted(self.sorted_min, _at_most(hi, dtype), "right"))
        return start, max(start, stop)

    def count_min_between(self, lo: float, hi: float) -> int:
        """How many items :meth:`min_between` would return, at most.

        The short items in the window plus every long item: cheap, and
        enough to choose the most selective of several axes.
        """
        start, stop = self._window(lo, hi)
        return (stop - start) + len(self.long_ids)

    def min_between(self, lo: float, hi: float) -> np.ndarray:
        """Ids of the items whose minimum lies in ``[lo, hi]``.

        Every contained item is among them; the caller applies its own test
        for the maximum (and any other axis).
        """
        start, stop = self._window(lo, hi)
        ids = self.order[start:stop]
        if len(self.long_ids):
            keep = np.ones(len(self.long_ids), dtype=bool)
            dtype = self.long_min.dtype
            if lo > -np.inf:
                keep &= self.long_min >= _at_least(lo, dtype)
            if hi < np.inf:
                keep &= self.long_min <= _at_most(hi, dtype)
            if keep.any():
                ids = np.concatenate([ids, self.long_ids[keep]])
        return ids

    def contained(self, lo: float, hi: float) -> np.ndarray:
        """Ids of the items with ``lo <= min`` and ``max <= hi``."""
        ids = self.min_between(lo, hi)
        if hi == np.inf or not len(ids):
            return ids
        upper = self._item_max(ids)
        return ids[upper <= _at_most(hi, upper.dtype)]

    def overlapping(self, lo: float, hi: float) -> np.ndarray:
        """Ids of the items with ``min <= hi`` and ``max >= lo``.

        A short item that reaches ``lo`` starts no lower than ``lo -
        longest``, so the search window is ``[lo - longest, hi]``.
        """
        start, stop = self._window(lo - self.longest, hi)
        ids = self.order[start:stop]
        if lo > -np.inf and len(ids):
            upper = self._item_max(ids)
            ids = ids[upper >= _at_least(lo, upper.dtype)]
        if len(self.long_ids):
            keep = np.ones(len(self.long_ids), dtype=bool)
            dtype = self.long_min.dtype
            if hi < np.inf:
                keep &= self.long_min <= _at_most(hi, dtype)
            if lo > -np.inf:
                keep &= self.long_max >= _at_least(lo, dtype)
            if keep.any():
                ids = np.concatenate([ids, self.long_ids[keep]])
        return ids
