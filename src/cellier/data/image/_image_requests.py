"""ChunkRequest — immutable request object for a single padded brick read.

Lives here as a standalone module so the data stores and the render layer
can import it without a circular dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID


class ChunkRequest(NamedTuple):
    """A request for one padded brick / chunk of data.

    All coordinates encode the *padded* region including overlap.  They
    may be negative or exceed the store bounds; ``get_data()`` on the
    ``MultiscaleZarrDataStore`` is responsible for clamping and
    zero-padding so the returned array always has the full requested
    shape.

    Parameters
    ----------
    chunk_request_id :
        Unique ID for this individual chunk.
    slice_request_id :
        Shared ID for all chunks that belong to the same planning event.
    scale_index :
        0-based index into ``MultiscaleZarrDataStore`` levels (0 = finest).
    axis_selections :
        Per-axis selection in data axis order.  Each element is either:
        - ``int`` → axis is sliced (single plane, already scaled to this level)
        - ``(start, stop)`` → axis is displayed (windowed range; may extend
          outside bounds)
    """

    chunk_request_id: UUID
    slice_request_id: UUID
    scale_index: int
    axis_selections: tuple[int | tuple[int, int], ...]
