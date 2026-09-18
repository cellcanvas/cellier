"""What a data store says when its data changes (``plans/store_change_events.md``).

A store announces a change on its ``data_changed`` signal with a
:class:`StoreChange`.  There are two kinds, kept separate because their
consumers differ:

* ``"extent"`` -- the data may now occupy a different region of data space
  (new geometry positions, image data of a new shape, a store that grew).
  Everything derived from ``axis_extents`` is stale.
* ``"contents"`` -- same region, different values (a paint stroke, new
  colours, a frame streamed into the existing extent).  Only what was drawn
  from the data is stale.

When unsure, announce ``"extent"``: it costs an extent recompute, while the
opposite mistake leaves extent-derived state stale.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

StoreChangeKind = Literal["extent", "contents"]
"""The two kinds of store change."""

STORE_CHANGE_KINDS: tuple[StoreChangeKind, ...] = ("extent", "contents")

DataRegion = tuple[tuple[float, float], ...]
"""A changed region: one ``(start, stop)`` per data axis, in level-0 data
coordinates, ``stop`` exclusive.

The store knows where data landed; it does not know the render layer's brick
grid, so it speaks in its own coordinates and the render layer converts
(``plans/streaming_acquisition_design.md`` 8b).
"""


class StoreChange(NamedTuple):
    """One announced change to a data store.

    Attributes
    ----------
    kind : "extent" or "contents"
        What the change can invalidate; see the module docstring.
    regions : tuple[DataRegion, ...] or None
        Where the data changed, or ``None`` for "anywhere".  Advisory: a
        consumer that cannot use regions treats the whole store as changed.
    """

    kind: StoreChangeKind
    regions: tuple[DataRegion, ...] | None = None
