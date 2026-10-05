# src/cellier/data/graph/_graph_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np


class GraphSliceRequest(NamedTuple):
    """Request for one slab-filtered slice of spatial-graph data.

    Mirrors ``PointsSliceRequest`` with one change: the scalar ``thickness``
    becomes a per-axis asymmetric extent, so a trail window on any sliced
    axis is expressible without a second request type.

    The first three fields satisfy the AsyncSlicer logging contract:
    ``slice_request_id`` is the task key; ``chunk_request_id`` and
    ``scale_index`` appear in INFO/DEBUG log lines.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.  Used by
        AsyncSlicer as the dict key for the running task -- REQUIRED.
    chunk_request_id : UUID
        Per-request ID.  For graphs (never tiled) this equals
        ``slice_request_id``.
    scale_index : int
        Always 0 -- no LOD levels.  Present for slicer logging compat.
    displayed_axes : tuple[int, ...]
        Axis indices rendered in the canvas (2 for 2D, 3 for 3D).
    slice_positions : dict[int, float]
        Collapsed axis -> the slab's centre, in **data** coordinates.

        Design 3.12 calls the graph "a per-family policy on top of the
        region", and the policy is real: the window is asymmetric -- a trail
        extends backwards in time -- and the fade is a signed distance
        measured from the slice, neither of which ``contains`` can express.
        So this family keeps its slab and only the *space* the numbers are in
        changed: the centre is the region pulled back through the visual's
        transform, and ``extents`` / ``fades`` are divided by their axis's
        world-units-per-data-unit, so the store compares data against data.

        Named ``slice_indices`` until Phase 8, which was doubly misleading:
        they are positions rather than indices, and by then they were in data
        rather than world coordinates.
    retained_axes : tuple[int, ...]
        The **data** axes this visual's geometry keeps, ascending.

        Not the same list as ``displayed_axes``, which indexes the **world**:
        a ``zyx`` store in a ``czyx`` world retains ``(0, 1, 2)`` while the
        world displays ``(1, 2, 3)``, and a transform that permutes its axes
        retains a different set again.  Indexing the position array with
        world axes raises on the first and silently uploads the wrong columns
        on the second.
    extents : dict[int, tuple[float, float]]
        Axis -> ``(before, after)`` half-extents of the slab on that axis.
        Carries an entry for *every* sliced axis; axes with no
        ``TrailConfig`` get ``(0.5, 0.5)``, which is exactly the symmetric
        slab points and lines use, so a graph with no trail slices
        identically to them.
    fades : dict[int, tuple[float, float, float]]
        Axis -> ``(fade_before, fade_after, min_alpha)``.  Carries entries
        only for trail axes with ``fade=True``.  Empty means no fade is
        computed and ``GraphData.node_alpha`` / ``edge_alpha`` are None.
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
    slice_positions: dict[int, float]
    extents: dict[int, tuple[float, float]]
    fades: dict[int, tuple[float, float, float]]
    retained_axes: tuple[int, ...]
    clip_planes: tuple = ()


@dataclass(frozen=True)
class GraphData:
    """Slab-filtered graph data returned by ``GraphMemoryStore.get_data()``.

    Nodes and edges travel together in one object so they can never be
    committed to the GPU out of sync (D2).

    ``node_alpha`` / ``edge_alpha`` are deliberately *separate* from the
    colour arrays.  The store computes the trail fade because it is a
    function of the data and the slice position; the render layer composes
    it with the base colour because that is where the appearance lives.
    Keeping them apart avoids handing an appearance colour down into the
    store, and lets the fade ride its own 1-float buffer (D19).

    Parameters
    ----------
    request_id : UUID
        Echo of ``GraphSliceRequest.slice_request_id``.
    node_positions : np.ndarray
        (n, n_displayed) float32.  Projected onto displayed axes; padded
        to 3D in the render layer.
    node_colors : np.ndarray | None
        (n, 4) float32 RGBA, row-matched to ``node_positions``.
        None when the store carries no per-node colours.
    node_sizes : np.ndarray | None
        (n,) float32 per-node sizes, or None.
    node_alpha : np.ndarray | None
        (n,) float32 trail-fade multiplier in [0, 1], or None when no
        trail axis has fading enabled.
    node_color_mode : str
        ``"uniform"`` or ``"vertex"``.  Describes what the *store* carries;
        the render layer honours the appearance's declaration (D20).
    node_size_mode : str
        ``"uniform"`` or ``"vertex"``.
    original_node_rows : np.ndarray | None
        (n,) int array mapping each rendered node back to its row in the
        store's full position array.  None on the empty-placeholder path.
    edge_positions : np.ndarray
        (2 * e, n_displayed) float32 in ``LineSegmentMaterial`` vertex-pair
        layout: pair ``(2n, 2n + 1)`` defines segment ``n``.
    edge_colors : np.ndarray | None
        (2 * e, 4) float32 RGBA, row-matched to ``edge_positions``.  A
        per-*edge* colour in the store is expanded to two vertices here.
    edge_alpha : np.ndarray | None
        (2 * e,) float32 per-vertex fade, or None.  Row-aligned with
        ``edge_positions`` by construction, so ``LineSegmentMaterial``
        interpolates the fade along each segment for free.
    edge_color_mode : str
        ``"uniform"`` or ``"vertex"``.
    edge_endpoint_rows : np.ndarray | None
        (e, 2) int array of the endpoint *node rows* of each rendered edge.
        Present on both slice strategies and used to build an edge pick
        payload's endpoint ids.  None on the empty-placeholder path.
    original_edge_rows : np.ndarray | None
        (e,) int array mapping each rendered edge to its row in the store's
        full edge array.  Populated by the ``"mask"`` strategy only: the
        ``"roi"`` strategy's index returns endpoint pairs rather than edge
        rows, and translating the whole candidate set every reslice is
        exactly the per-frame ``searchsorted`` D18 exists to avoid.  A pick
        on an ROI-sliced edge resolves its single pair to a row on demand.
    nodes_empty : bool
        True when the node filter produced zero rows and placeholder
        geometry was returned.
    edges_empty : bool
        True when the edge filter produced zero rows.  Independent of
        ``nodes_empty``: isolated nodes give edges-empty-but-nodes-present,
        and both sub-nodes get the placeholder treatment separately.
    """

    request_id: UUID
    node_positions: np.ndarray
    #: (2 * e, n_displayed) -- always present; a placeholder pair when empty.
    edge_positions: np.ndarray

    # Nodes
    node_colors: np.ndarray | None = None
    node_sizes: np.ndarray | None = None
    node_alpha: np.ndarray | None = None
    node_color_mode: str = "uniform"
    node_size_mode: str = "uniform"
    original_node_rows: np.ndarray | None = None

    # Edges, in LineSegmentMaterial vertex-pair layout
    edge_colors: np.ndarray | None = None
    edge_alpha: np.ndarray | None = None
    edge_color_mode: str = "uniform"
    edge_endpoint_rows: np.ndarray | None = None
    original_edge_rows: np.ndarray | None = None

    nodes_empty: bool = False
    edges_empty: bool = False

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return (
            f"nodes={self.node_positions.shape} "
            f"edges={self.edge_positions.shape[0] // 2}"
        )
