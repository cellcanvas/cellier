# src/cellier/data/graph/_graph_memory_store.py
from __future__ import annotations

import asyncio
from typing import Any, Literal

import numpy as np
from pydantic import (
    ConfigDict,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from cellier.data._base_data_store import BaseDataStore
from cellier.data.graph._graph_requests import GraphData, GraphSliceRequest
from cellier.transform import AffineTransform

#: Placeholder vertex counts for an empty slice.  pygfx forbids empty
#: geometry buffers, so a single invisible node / a single degenerate segment
#: keeps the sub-node valid.  LineSegmentMaterial needs an even vertex count,
#: so the edge placeholder is two vertices.  The column count is
#: ``len(displayed_axes)``, not the store's ndim: a 4-D tzyx graph displays
#: 2 or 3 of its axes, and indexing a fixed 3-column placeholder by data-axis
#: index would run off the end.
_PLACEHOLDER_N_NODES = 1
_PLACEHOLDER_N_EDGE_VERTICES = 2

#: Guard against a zero falloff distance in the trail-fade divide.
_FALLOFF_EPS = 1e-6

#: Slab half-extent used for a sliced axis carrying no TrailConfig.  Matches
#: the hardcoded thickness in the points and lines request builders, so a
#: graph with no trail slices identically to them.
_DEFAULT_EXTENT = (0.5, 0.5)

_GRAPH_EXTRA_MSG = (
    "requires the 'graph' extra: pip install 'cellier[graph]' "
    "(installs geff and spatial-graph)"
)


def _spatial_graph_module() -> Any:
    """Import ``spatial_graph``, raising a message that names the extra."""
    try:
        import spatial_graph
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(f"Spatial graph indexing {_GRAPH_EXTRA_MSG}") from e
    return spatial_graph


def _spatial_graph_available() -> bool:
    """True when ``spatial_graph`` can be imported."""
    try:
        import spatial_graph  # noqa: F401
    except ImportError:
        return False
    return True


class GraphMemoryStore(BaseDataStore):
    """In-memory spatial-graph data store backed by numpy arrays.

    One store backs one graph visual and returns nodes and edges together in
    a single ``GraphData`` per slice, so the two can never be committed to
    the GPU out of sync (D2).

    Field layout
    ------------
    The flat numpy arrays are the **source of truth** and are what
    serializes; the ``spatial_graph`` index is a derived, lazily built
    private attribute reachable through :attr:`graph` (D3).  A store used
    only with ``slice_strategy="mask"`` never builds an index at all, which
    is what makes ``spatial-graph`` optional at *runtime* and not merely at
    import time.

    Positions are stored in *data-axis order*: column 0 is axis 0 (t or z),
    the last column is x.  The render layer applies the ``[:, [2, 1, 0]]``
    reversal before uploading to pygfx.

    ``edges`` holds endpoint **row indices**, not original node ids; the
    conversion happens once, at construction.  ``node_ids`` is read only
    when building a pick payload, never on the per-frame path (D18).

    All reads are synchronous (data is in CPU RAM); ``get_data`` is declared
    ``async`` to satisfy the AsyncSlicer contract and to provide a single
    cancellation checkpoint.

    Parameters
    ----------
    positions : np.ndarray
        (n_nodes, ndim) float32 array in data-axis order.
    edges : np.ndarray
        (n_edges, 2) int32 array of endpoint *row indices* into positions.
    node_ids : np.ndarray | None
        Original node ids (arbitrary dtype -- geff files may carry uint64 or
        strings).  Defaults to ``arange(n_nodes)``.  Consulted only per pick.
    node_colors : np.ndarray | None
        (n_nodes, 4) float32 RGBA, row-matched to positions.
    node_sizes : np.ndarray | None
        (n_nodes,) float32 per-node sizes.
    edge_colors : np.ndarray | None
        (n_edges, 4) float32 RGBA, row-matched to edges.  Expanded to two
        vertices per edge at slice time.
    transform : AffineTransform | None
        Data-to-world transform derived from a geff file's per-axis
        ``scale`` / ``offset`` (D23).  ``None`` for a store built from raw
        arrays.  ``Controller.add_graph`` uses it as the visual's default
        transform when its own ``transform`` argument is None.
    directed : bool
        Whether the graph is directed.  Decides which ``spatial_graph``
        class the lazy index builds, and nothing else: the slice path is
        direction-agnostic.
    slice_strategy : str
        ``"mask"`` (default) or ``"roi"``, chosen explicitly by the caller
        (D17).  The store never inspects the data to pick one.  ``"roi"``
        without ``spatial-graph`` importable is an error at validation, not
        a silent fallback.
    name : str
        Human-readable label.
    """

    store_type: Literal["graph_memory"] = "graph_memory"
    name: str = "graph_memory_store"

    positions: np.ndarray
    edges: np.ndarray
    node_ids: np.ndarray | None = None
    node_colors: np.ndarray | None = None
    node_sizes: np.ndarray | None = None
    edge_colors: np.ndarray | None = None

    transform: AffineTransform | None = None
    directed: bool = False

    slice_strategy: Literal["mask", "roi"] = "mask"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Derived, never serialized.  Built on first access; see the properties.
    _graph: Any = PrivateAttr(default=None)
    _edge_span: np.ndarray | None = PrivateAttr(default=None)
    _edge_row_lookup: np.ndarray | None = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"positions must be a 2-D array, got shape {arr.shape}")
        return np.ascontiguousarray(arr)

    @field_validator("edges", mode="before")
    @classmethod
    def _coerce_edges(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.int32)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"edges must have shape (n_edges, 2), got {arr.shape}")
        return np.ascontiguousarray(arr)

    @field_validator("node_ids", mode="before")
    @classmethod
    def _coerce_node_ids(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        # Deliberately un-coerced dtype: geff ids may be uint64 or strings.
        return np.ascontiguousarray(np.asarray(v))

    @field_validator("node_colors", "node_sizes", "edge_colors", mode="before")
    @classmethod
    def _coerce_float32(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @model_validator(mode="after")
    def _check_consistency(self) -> GraphMemoryStore:
        n_nodes = self.positions.shape[0]
        if self.edges.shape[0] and n_nodes:
            hi = int(self.edges.max())
            lo = int(self.edges.min())
            if lo < 0 or hi >= n_nodes:
                raise ValueError(
                    f"edges reference node rows outside [0, {n_nodes}); "
                    f"got range [{lo}, {hi}]. Pass row indices, not node ids."
                )
        if self.node_ids is not None and self.node_ids.shape[0] != n_nodes:
            raise ValueError(
                f"node_ids has {self.node_ids.shape[0]} entries but there are "
                f"{n_nodes} nodes."
            )
        if self.slice_strategy == "roi" and not _spatial_graph_available():
            raise ValueError(
                f"slice_strategy='roi' {_GRAPH_EXTRA_MSG}. "
                "Use slice_strategy='mask' to slice without an index."
            )
        return self

    # ------------------------------------------------------------------
    # Serializers
    # ------------------------------------------------------------------

    @field_serializer("positions", "edges")
    def _serialize_array(self, arr: np.ndarray, _info: Any) -> list:
        return arr.tolist()

    @field_serializer("node_ids", "node_colors", "node_sizes", "edge_colors")
    def _serialize_optional_array(
        self, arr: np.ndarray | None, _info: Any
    ) -> list | None:
        return arr.tolist() if arr is not None else None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        edges: np.ndarray,
        *,
        node_ids: np.ndarray | None = None,
        node_colors: np.ndarray | None = None,
        node_sizes: np.ndarray | None = None,
        edge_colors: np.ndarray | None = None,
        transform: AffineTransform | None = None,
        directed: bool = False,
        slice_strategy: Literal["mask", "roi"] = "mask",
        name: str = "graph_memory_store",
    ) -> GraphMemoryStore:
        """Build a store from raw arrays -- the dependency-free path.

        No spatial index is constructed here, and none is constructed at all
        unless :attr:`graph` is touched or ``slice_strategy="roi"``.

        Parameters
        ----------
        positions : np.ndarray
            (n_nodes, ndim) node positions in data-axis order.
        edges : np.ndarray
            (n_edges, 2) endpoint *row indices*.
        node_ids : np.ndarray | None
            Original node ids; defaults to ``arange(n_nodes)``.
        node_colors, node_sizes, edge_colors : np.ndarray | None
            Optional per-element appearance arrays.
        transform : AffineTransform | None
            Data-to-world transform, normally left None for raw arrays.
        directed : bool
            Whether the graph is directed.
        slice_strategy : str
            ``"mask"`` or ``"roi"`` (D17).
        name : str
            Human-readable label.

        Returns
        -------
        GraphMemoryStore
        """
        return cls(
            positions=positions,
            edges=edges,
            node_ids=node_ids,
            node_colors=node_colors,
            node_sizes=node_sizes,
            edge_colors=edge_colors,
            transform=transform,
            directed=directed,
            slice_strategy=slice_strategy,
            name=name,
        )

    @classmethod
    def from_geff(
        cls,
        path: Any,
        *,
        axis_names: list[str] | None = None,
        node_color_prop: str | None = None,
        node_size_prop: str | None = None,
        directed: bool | None = None,
        slice_strategy: Literal["mask", "roi"] = "mask",
        name: str = "graph_memory_store",
    ) -> GraphMemoryStore:
        """Build a store from a geff file.

        There is deliberately **no** ``transform`` parameter (D23): the
        file's axes are the sole source of the store's transform at
        construction.  ``Controller.add_graph(transform=...)`` still wins
        when passed explicitly, as it does for every other visual.

        See :mod:`cellier.data.graph._geff_io` for the reader details.
        """
        from cellier.data.graph._geff_io import read_geff

        payload = read_geff(
            path,
            axis_names=axis_names,
            node_color_prop=node_color_prop,
            node_size_prop=node_size_prop,
        )
        store = cls(
            positions=payload.positions,
            edges=payload.edges,
            node_ids=payload.node_ids,
            node_colors=payload.node_colors,
            node_sizes=payload.node_sizes,
            transform=payload.transform,
            directed=payload.directed if directed is None else directed,
            slice_strategy=slice_strategy,
            name=name,
        )
        store._axes = payload.axes
        store._node_props = payload.node_props
        store._edge_props = payload.edge_props
        return store

    # geff metadata retained for downstream display; nothing consumes it yet.
    _axes: list = PrivateAttr(default_factory=list)
    _node_props: dict = PrivateAttr(default_factory=dict)
    _edge_props: dict = PrivateAttr(default_factory=dict)

    @property
    def axes(self) -> list:
        """The geff ``Axis`` objects in file order; empty for raw arrays."""
        return self._axes

    @property
    def node_props(self) -> dict:
        """Non-position node properties retained from a geff file."""
        return self._node_props

    @property
    def edge_props(self) -> dict:
        """Edge properties retained from a geff file."""
        return self._edge_props

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions per node."""
        return self.positions.shape[1]

    @property
    def n_nodes(self) -> int:
        """Total number of nodes in the store."""
        return self.positions.shape[0]

    @property
    def n_edges(self) -> int:
        """Total number of edges in the store."""
        return self.edges.shape[0]

    @property
    def node_color_mode(self) -> str:
        """``"vertex"`` when per-node colours are present, else ``"uniform"``.

        Descriptive only: it says what the store carries, and never decides
        what the material does (D20).
        """
        return "vertex" if self.node_colors is not None else "uniform"

    @property
    def node_size_mode(self) -> str:
        """``"vertex"`` when per-node sizes are present, else ``"uniform"``."""
        return "vertex" if self.node_sizes is not None else "uniform"

    @property
    def edge_color_mode(self) -> str:
        """``"vertex"`` when per-edge colours are present, else ``"uniform"``."""
        return "vertex" if self.edge_colors is not None else "uniform"

    @property
    def graph(self) -> Any:
        """The spatial index, built on first use over ROW indices (D18).

        Node ids in the returned ``SpatialGraph`` / ``SpatialDiGraph`` are
        ``arange(n_nodes)``, *not* the caller's ``node_ids``, so queries
        return buffer rows directly and nothing on the per-frame path pays
        an id-to-row translation.  Reaching past the public query wrappers
        therefore sees row indices.

        Raises
        ------
        ImportError
            When ``spatial-graph`` is not installed.
        """
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    @property
    def edge_span(self) -> np.ndarray:
        """Per-axis 99th percentile of ``abs(pos[u, ax] - pos[v, ax])``.

        Compare against the data extent on the axis being sliced: well
        under ~15% of the extent means ``slice_strategy="roi"`` will pay
        off.  Purely informational -- nothing reads this to choose a
        strategy.

        Computed on first access and cached, so a store that never asks
        never pays the O(E) pass.  99th percentile rather than max so a
        handful of outlier edges (a track skipping frames through an
        occlusion) does not misrepresent an otherwise local graph.
        """
        if self._edge_span is None:
            if self.n_edges == 0:
                self._edge_span = np.zeros(self.ndim, dtype=np.float32)
            else:
                deltas = np.abs(
                    self.positions[self.edges[:, 0]] - self.positions[self.edges[:, 1]]
                )
                self._edge_span = np.percentile(deltas, 99.0, axis=0).astype(np.float32)
        return self._edge_span

    def _build_graph(self) -> Any:
        """Construct the ``spatial_graph`` index over row indices."""
        sg = _spatial_graph_module()
        cls = sg.SpatialDiGraph if self.directed else sg.SpatialGraph
        kwargs: dict[str, Any] = {}
        if self.directed:
            kwargs["directed"] = True
        graph = cls(
            ndims=self.ndim,
            node_dtype="uint64",
            node_attr_dtypes={"position": f"float32[{self.ndim}]"},
            edge_attr_dtypes={},
            position_attr="position",
            **kwargs,
        )
        graph.add_nodes(
            np.arange(self.n_nodes, dtype=np.uint64),
            position=np.ascontiguousarray(self.positions, dtype=np.float32),
        )
        if self.n_edges:
            graph.add_edges(np.ascontiguousarray(self.edges, dtype=np.uint64))
        return graph

    # ------------------------------------------------------------------
    # Id / row translation -- user-facing, never on the per-frame path
    # ------------------------------------------------------------------

    def ids_for_rows(self, rows: Any) -> np.ndarray:
        """Translate node row indices to the store's original node ids.

        Parameters
        ----------
        rows : array-like of int
            Row indices into ``positions``.

        Returns
        -------
        np.ndarray
            Original node ids, in the store's own id dtype.
        """
        rows = np.asarray(rows, dtype=np.intp)
        if self.node_ids is None:
            return rows
        return self.node_ids[rows]

    def rows_for_ids(self, ids: Any) -> np.ndarray:
        """Translate original node ids to row indices into ``positions``.

        Requires ``node_ids`` to be sorted, which it is for every geff file
        and for the ``arange`` default.  Called on an explicit user request,
        never per frame (D18).

        Parameters
        ----------
        ids : array-like
            Original node ids.

        Returns
        -------
        np.ndarray
            Row indices, ``intp``.
        """
        ids = np.asarray(ids)
        if self.node_ids is None:
            return ids.astype(np.intp)
        return np.searchsorted(self.node_ids, ids).astype(np.intp)

    def edge_row_for_endpoints(self, source_row: int, target_row: int) -> int:
        """Return the store edge row joining two node rows, or ``-1``.

        Used only to give an ROI-sliced edge pick the same ``edge_index``
        a mask-sliced one carries.  The ROI index returns endpoint pairs
        rather than edge rows, and translating a whole candidate set every
        reslice is exactly the per-frame ``searchsorted`` D18 removes -- so
        the translation happens here, for one edge, on a pick.

        Backed by a sorted key array built on first call and cached.

        Parameters
        ----------
        source_row, target_row : int
            Endpoint row indices.

        Returns
        -------
        int
            Row into ``edges``, or -1 when no such edge exists.
        """
        if self.n_edges == 0:
            return -1
        if self._edge_row_lookup is None:
            keys = self._edge_keys(self.edges[:, 0], self.edges[:, 1])
            order = np.argsort(keys, kind="stable")
            self._edge_row_lookup = np.stack([keys[order], order]).astype(np.int64)
        keys, rows = self._edge_row_lookup
        # An undirected store may hold the pair either way round.
        candidates = [self._edge_keys(source_row, target_row)]
        if not self.directed:
            candidates.append(self._edge_keys(target_row, source_row))
        for key in candidates:
            idx = int(np.searchsorted(keys, key))
            if idx < keys.shape[0] and keys[idx] == key:
                return int(rows[idx])
        return -1

    def _edge_keys(self, source: Any, target: Any) -> Any:
        """Pack an endpoint row pair into one sortable int64 key."""
        stride = np.int64(self.n_nodes + 1)
        return np.asarray(source, dtype=np.int64) * stride + np.asarray(
            target, dtype=np.int64
        )

    # ------------------------------------------------------------------
    # Public spatial query passthroughs
    # ------------------------------------------------------------------

    def query_nodes_in_roi(self, roi_min: Any, roi_max: Any) -> np.ndarray:
        """Return the original ids of nodes inside an axis-aligned box.

        Parameters
        ----------
        roi_min, roi_max : array-like of float
            Per-axis lower and upper bounds, in data-axis order.

        Returns
        -------
        np.ndarray
            Original node ids (see ``ids_for_rows``).
        """
        roi = self._make_roi_from_bounds(roi_min, roi_max)
        rows = np.asarray(self.graph.query_nodes_in_roi(roi)).astype(np.intp)
        return self.ids_for_rows(rows)

    def query_nearest_nodes(self, points: Any, k: int = 1) -> np.ndarray:
        """Return the original ids of the ``k`` nodes nearest each point."""
        query = np.ascontiguousarray(points, dtype=self.positions.dtype)
        rows = np.asarray(self.graph.query_nearest_nodes(query, k)).astype(np.intp)
        return self.ids_for_rows(rows)

    def query_nearest_edges(self, points: Any, k: int = 1) -> np.ndarray:
        """Return endpoint id pairs for the ``k`` edges nearest each point."""
        query = np.ascontiguousarray(points, dtype=self.positions.dtype)
        pairs = np.asarray(self.graph.query_nearest_edges(query, k)).astype(np.intp)
        return self.ids_for_rows(pairs.reshape(-1)).reshape(pairs.shape)

    def _make_roi_from_bounds(self, roi_min: Any, roi_max: Any) -> np.ndarray:
        """Stack per-axis bounds into the dtype ``spatial_graph`` demands.

        The binding does no coercion: a float64 ROI against float32
        positions raises ``ValueError: Buffer dtype mismatch``.
        """
        dtype = self.positions.dtype
        return np.stack(
            [
                np.asarray(roi_min, dtype=dtype),
                np.asarray(roi_max, dtype=dtype),
            ]
        )

    # ------------------------------------------------------------------
    # Async data access -- one checkpoint for cancellability
    # ------------------------------------------------------------------

    async def get_data(self, request: GraphSliceRequest) -> GraphData:
        """Return slab-filtered graph data for *request*.

        Checkpoint
        ----------
        A  After the node and edge masks are built but before the gathers
           that build the GPU buffers, matching the position points and
           lines use.  If CancelledError fires there the callback is never
           called, so stale geometry never reaches the GPU.

        Inclusion rule
        --------------
        A node survives when it is inside the slab on every non-displayed
        axis.  An edge survives when **either** endpoint does (D5), and it
        draws in full to the out-of-slab endpoint's projected position.
        That far endpoint contributes a line vertex but no node marker
        (D6) -- which is the only rule that renders anything for tracking
        data, where every edge spans two timepoints.

        Parameters
        ----------
        request : GraphSliceRequest
            Built by ``GFXGraphMemoryVisual.build_slice_request[_2d]``.

        Returns
        -------
        GraphData
            Filtered, projected node and edge data ready for GPU upload.
        """
        if self.slice_strategy == "roi":
            node_rows, edge_endpoints, edge_rows = self._select_roi(request)
        else:
            node_rows, edge_endpoints, edge_rows = self._select_mask(request)

        # -- Checkpoint A ---------------------------------------------
        await asyncio.sleep(0)

        return self._gather(request, node_rows, edge_endpoints, edge_rows)

    # ------------------------------------------------------------------
    # Selection -- the two strategies, required by test to agree exactly
    # ------------------------------------------------------------------

    def _window_mask(self, request: GraphSliceRequest) -> np.ndarray:
        """Boolean (n_nodes,) mask of nodes inside the slab on every axis."""
        mask = np.ones(self.n_nodes, dtype=bool)
        for axis, idx in request.slice_indices.items():
            before, after = request.extents.get(axis, _DEFAULT_EXTENT)
            coord = self.positions[:, axis]
            mask &= (coord >= idx - before) & (coord <= idx + after)
        return mask

    def _select_mask(
        self, request: GraphSliceRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The reference selection: two vectorized passes over the arrays.

        No optional dependency, O(E) always and never worse -- which is why
        it is the default and the correctness reference (D17).
        """
        node_mask = self._window_mask(request)
        node_rows = np.flatnonzero(node_mask)

        if self.n_edges:
            # Two fancy-index gathers over (E,) and an OR.  This is the whole
            # of D5; there is no special case for a trail axis, which is D13.
            edge_mask = node_mask[self.edges[:, 0]] | node_mask[self.edges[:, 1]]
            edge_rows = np.flatnonzero(edge_mask)
            edge_endpoints = self.edges[edge_rows]
        else:
            edge_rows = np.zeros(0, dtype=np.intp)
            edge_endpoints = np.zeros((0, 2), dtype=np.int32)

        return node_rows, edge_endpoints, edge_rows

    def _select_roi(
        self, request: GraphSliceRequest
    ) -> tuple[np.ndarray, np.ndarray, None]:
        """Index-accelerated selection over the same slab (D17).

        The ROI is the trail window on the sliced axes and unbounded on the
        displayed ones, so for an axis-aligned slab the query box *is* the
        exact region -- there is no coarse/exact split on the node side.

        ``query_edges_in_roi`` is a different matter: it reports every edge
        whose segment AABB intersects the box, which includes edges crossing
        the slab with *neither* endpoint inside.  The O(k) refinement below
        restores D5 exactly and is not redundant.

        Returns ``None`` for the edge rows: the index yields endpoint pairs,
        and translating the whole candidate set here is the per-frame
        ``searchsorted`` D18 exists to avoid.  A pick resolves its one edge
        through ``edge_row_for_endpoints``.
        """
        dtype = self.positions.dtype
        roi_min = np.full(self.ndim, -np.inf, dtype=dtype)
        roi_max = np.full(self.ndim, np.inf, dtype=dtype)
        for axis, idx in request.slice_indices.items():
            before, after = request.extents.get(axis, _DEFAULT_EXTENT)
            roi_min[axis] = idx - before
            roi_max[axis] = idx + after
        roi = np.stack([roi_min, roi_max])

        graph = self.graph
        node_rows = np.asarray(graph.query_nodes_in_roi(roi)).astype(np.intp)
        node_rows.sort()

        candidates = np.asarray(graph.query_edges_in_roi(roi))
        if candidates.shape[0]:
            candidates = candidates.astype(np.intp).reshape(-1, 2)
            keep = self._endpoint_in_window(
                candidates[:, 0], request
            ) | self._endpoint_in_window(candidates[:, 1], request)
            candidates = candidates[keep]
        else:
            candidates = np.zeros((0, 2), dtype=np.intp)

        return node_rows, candidates, None

    def _endpoint_in_window(
        self, rows: np.ndarray, request: GraphSliceRequest
    ) -> np.ndarray:
        """Slab test for a set of node rows -- the ROI refinement, O(k)."""
        keep = np.ones(rows.shape[0], dtype=bool)
        for axis, idx in request.slice_indices.items():
            before, after = request.extents.get(axis, _DEFAULT_EXTENT)
            coord = self.positions[rows, axis]
            keep &= (coord >= idx - before) & (coord <= idx + after)
        return keep

    # ------------------------------------------------------------------
    # Trail alpha and the gathers
    # ------------------------------------------------------------------

    def _trail_alpha(
        self, request: GraphSliceRequest, rows: np.ndarray, in_window: np.ndarray
    ) -> np.ndarray:
        """Per-node fade multiplier for *rows*, given their in-window flags.

        Signed distance from the current slice index with independent
        falloff per side.  Two properties worth stating:

        - In-window nodes are clamped *up* to ``min_alpha``, so a trail
          never disappears before the window ends.
        - Out-of-window nodes get exactly 0.0.  They only ever appear as
          the far vertex of a dangling edge (D13), so the edge fades
          smoothly to nothing at the window boundary rather than being cut
          square -- the gradient is free, because LineSegmentMaterial
          interpolates along the segment.
        """
        alpha = np.ones(rows.shape[0], dtype=np.float32)
        min_alpha = 0.0
        for axis, (fade_before, fade_after, axis_min_alpha) in request.fades.items():
            min_alpha = max(min_alpha, axis_min_alpha)
            d = self.positions[rows, axis] - request.slice_indices[axis]
            falloff = np.where(d < 0.0, fade_before, fade_after)
            a = 1.0 - np.abs(d) / np.maximum(falloff, _FALLOFF_EPS)
            alpha *= np.clip(a, 0.0, 1.0).astype(np.float32)
        return np.where(in_window, np.maximum(alpha, min_alpha), 0.0).astype(np.float32)

    def _gather(
        self,
        request: GraphSliceRequest,
        node_rows: np.ndarray,
        edge_endpoints: np.ndarray,
        edge_rows: np.ndarray | None,
    ) -> GraphData:
        """Build the GPU-ready buffers from the selected rows."""
        displayed = list(request.displayed_axes)
        fading = bool(request.fades)

        nodes_empty = node_rows.shape[0] == 0
        edges_empty = edge_endpoints.shape[0] == 0

        if nodes_empty:
            node_positions = np.zeros(
                (_PLACEHOLDER_N_NODES, len(displayed)), dtype=np.float32
            )
            node_colors = node_sizes = node_alpha = None
            original_node_rows = None
        else:
            node_positions = self.positions[node_rows][:, displayed]
            node_colors = (
                self.node_colors[node_rows] if self.node_colors is not None else None
            )
            node_sizes = (
                self.node_sizes[node_rows] if self.node_sizes is not None else None
            )
            node_alpha = (
                self._trail_alpha(
                    request, node_rows, np.ones(node_rows.shape[0], dtype=bool)
                )
                if fading
                else None
            )
            original_node_rows = node_rows

        if edges_empty:
            edge_positions = np.zeros(
                (_PLACEHOLDER_N_EDGE_VERTICES, len(displayed)), dtype=np.float32
            )
            edge_colors = edge_alpha = None
            edge_endpoint_rows = None
            original_edge_rows = None
        else:
            # (e, 2).reshape(-1) interleaves to [e0.a, e0.b, e1.a, e1.b, ...],
            # exactly the vertex-pair layout LineSegmentMaterial wants.
            vertex_rows = edge_endpoints.reshape(-1).astype(np.intp)
            edge_positions = self.positions[vertex_rows][:, displayed]
            edge_colors = (
                np.repeat(self.edge_colors[edge_rows], 2, axis=0)
                if (self.edge_colors is not None and edge_rows is not None)
                else None
            )
            if fading:
                # Alpha is gathered by the same vertex_rows index as the
                # positions, so the two buffers are row-aligned by
                # construction.  Out-of-window endpoints get 0.0.
                edge_alpha = self._trail_alpha(
                    request, vertex_rows, self._endpoint_in_window(vertex_rows, request)
                )
            else:
                edge_alpha = None
            edge_endpoint_rows = edge_endpoints
            original_edge_rows = edge_rows

        return GraphData(
            request_id=request.slice_request_id,
            node_positions=np.ascontiguousarray(node_positions, dtype=np.float32),
            edge_positions=np.ascontiguousarray(edge_positions, dtype=np.float32),
            node_colors=node_colors,
            node_sizes=node_sizes,
            node_alpha=node_alpha,
            node_color_mode=self.node_color_mode,
            node_size_mode=self.node_size_mode,
            original_node_rows=original_node_rows,
            edge_colors=edge_colors,
            edge_alpha=edge_alpha,
            edge_color_mode=(
                self.edge_color_mode if edge_colors is not None else "uniform"
            ),
            edge_endpoint_rows=edge_endpoint_rows,
            original_edge_rows=original_edge_rows,
            nodes_empty=nodes_empty,
            edges_empty=edges_empty,
        )
