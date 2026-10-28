# src/cellier/v2/data/points/_points_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.transform import ConvexRegion


class PointsSliceRequest(NamedTuple):
    """Request for one proximity-filtered slice of points data.

    The first three fields satisfy the AsyncSlicer logging contract:
    ``slice_request_id`` is the task key; ``chunk_request_id`` and
    ``scale_index`` appear in INFO/DEBUG log lines.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.  Used by
        AsyncSlicer as the dict key for the running task — REQUIRED.
    chunk_request_id : UUID
        Per-request ID.  For points (never tiled) this equals
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
        2 um z spacing it could show a point 24 um off the slice plane and
        hide the three that are on it.  There is no second path now (R8.3).
    fold_axes : tuple[int, ...]
        Collapsed data axes whose columns the store also returns, for the
        render layer's per-element fold (full affine design v2, D11).  Empty
        in every view where no collapsed axis reaches a displayed one.
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    retained_axes: tuple[int, ...]
    region: ConvexRegion
    fold_axes: tuple[int, ...] = ()


@dataclass(frozen=True)
class PointsData:
    """Proximity-filtered points data returned by PointsMemoryStore.get_data().

    Parameters
    ----------
    request_id : UUID
        Echo of PointsSliceRequest.slice_request_id.
    positions : np.ndarray
        (n_points, n_displayed_dims) float32.  Projected onto displayed
        axes; padded to 3D in the render layer.
    colors : np.ndarray | None
        (n_points, 4) float32 RGBA, index-matched to positions.
        None when the store carries no per-point colors.
    sizes : np.ndarray | None
        (n_points,) float32 per-point sizes.
        None when the store carries no per-point sizes.
    color_mode : str
        ``"uniform"`` or ``"vertex"``.  Ignored when colors is None.
    size_mode : str
        ``"uniform"`` or ``"vertex"``.  Ignored when sizes is None.
    is_empty : bool
        True when the proximity filter produced zero surviving points
        and a placeholder geometry was returned.
    original_indices : np.ndarray | None
        (n_points,) int array mapping each row of ``positions`` back to its
        index in the store's full point array.  This is the proximity
        filter's surviving-index list; in a full 3-D view it is the identity
        ``arange(N)``.  Used by the render layer to translate a pick's
        rendered-buffer vertex index into the original point index.  None on
        the empty-placeholder path.
    fold_positions : np.ndarray | None
        ``(n, len(fold_axes))``: the surviving points on the request's
        ``fold_axes``.  None when it asked for none.
    """

    request_id: UUID
    positions: np.ndarray
    colors: np.ndarray | None
    sizes: np.ndarray | None
    color_mode: str = "uniform"
    size_mode: str = "uniform"
    is_empty: bool = False
    original_indices: np.ndarray | None = None
    fold_positions: np.ndarray | None = None

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return f"positions={self.positions.shape}"
