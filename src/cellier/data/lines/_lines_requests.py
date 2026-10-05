# src/cellier/v2/data/lines/_lines_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.transform import ConvexRegion


class LinesSliceRequest(NamedTuple):
    """Request for one slab-filtered slice of line-segment data.

    The first three fields satisfy the AsyncSlicer logging contract:
    ``slice_request_id`` is the task key; ``chunk_request_id`` and
    ``scale_index`` appear in INFO/DEBUG log lines.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.  Used by
        AsyncSlicer as the dict key for the running task — REQUIRED.
    chunk_request_id : UUID
        Per-request ID.  For lines (never tiled) this equals
        ``slice_request_id``.
    scale_index : int
        Always 0 — no LOD levels.  Present for slicer logging compat.
    displayed_axes : tuple[int, ...]
        Axis indices rendered in the canvas (2 for 2D, 3 for 3D).
    retained_axes : tuple[int, ...]
        The **data** axes this visual's geometry keeps, ascending.

        Not the same list as ``displayed_axes``, which indexes the **world**:
        a ``zyx`` store in a ``czyx`` world retains ``(0, 1, 2)`` while the
        world displays ``(1, 2, 3)``, and a transform that permutes its axes
        retains a different set again.  Indexing the position array with
        world axes raises on the first and silently uploads the wrong columns
        on the second.
    region : ConvexRegion
        The selected region, already pulled back into **data** coordinates
        (design 3.12).  It is the whole filter -- one ``contains`` call
        instead of a per-axis mask loop.

        Until v1 was retired this was optional, and a request without one
        fell back to comparing a **world** position from ``slice_indices``
        against **data** coordinates -- the latent bug D4 exists to fix: on a
        2 um z spacing it could show a vertex 24 um off the slice plane and
        hide the ones that are on it.  There is no second path now (R8.3).
    clip_planes : tuple[tuple[tuple[float, ...], float], ...]
        Clipping planes the read applies on the CPU, as ``(normal, offset)``
        pairs in data coordinates, kept where ``normal . p >= offset``.
        Set only where the shader cannot clip: geometry flattened along an
        axis a plane has a component on (clipping planes design 5.2).
        Empty otherwise.
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    retained_axes: tuple[int, ...]
    region: ConvexRegion
    clip_planes: tuple = ()


@dataclass(frozen=True)
class LinesData:
    """Slab-filtered line-segment data returned by LinesMemoryStore.get_data().

    Parameters
    ----------
    request_id : UUID
        Echo of LinesSliceRequest.slice_request_id.
    positions : np.ndarray
        (n_vertices, n_displayed_dims) float32.  n_vertices is always even;
        pair ``(2n, 2n+1)`` defines segment ``n``.  Projected onto displayed
        axes; padded to 3D in the render layer.
    colors : np.ndarray | None
        (n_vertices, 4) float32 RGBA, index-matched to positions.
        None when the store carries no per-vertex colors.
    color_mode : str
        ``"uniform"`` or ``"vertex"``.  Ignored when colors is None.
    is_empty : bool
        True when the slab filter produced zero surviving segments and a
        placeholder geometry was returned.
    original_edge_indices : np.ndarray | None
        (n_segments,) int array mapping each rendered segment to its edge
        index in the store's full segment array.  This is the slab filter's
        surviving-segment list; in a full 3-D view it is the identity
        ``arange(n_segments)``.  Used by the render layer to translate a
        pick's rendered-buffer edge index into the original edge index.
        None on the empty-placeholder path.
    """

    request_id: UUID
    positions: np.ndarray
    colors: np.ndarray | None
    color_mode: str = "uniform"
    is_empty: bool = False
    original_edge_indices: np.ndarray | None = None

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return f"positions={self.positions.shape}"
