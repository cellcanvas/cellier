# src/cellier/render/visuals/_graph_memory.py
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.graph._graph_requests import GraphSliceRequest
from cellier.render.shaders._alpha_modulated import (
    AlphaLineSegmentMaterial,
    AlphaPointsMaterial,
)

if TYPE_CHECKING:
    from cellier._state import DimsState
    from cellier.data.graph._graph_requests import GraphData
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        PickWriteChangedEvent,
        TrailChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.transform import AffineTransform
    from cellier.visuals._graph_memory import GraphAppearance, GraphVisual

# Placeholder geometry -- pygfx forbids empty geometry buffers.  One
# invisible node; one degenerate segment (LineSegmentMaterial needs an even
# vertex count, so two vertices is the minimum).
_PLACEHOLDER_NODE_POSITIONS = np.zeros((1, 3), dtype=np.float32)
_PLACEHOLDER_EDGE_POSITIONS = np.zeros((2, 3), dtype=np.float32)

#: Half-extent used for a sliced axis carrying no TrailConfig.  Matches the
#: hardcoded thickness in the points and lines request builders, so a graph
#: with no trail slices identically to them.
_DEFAULT_EXTENT = (0.5, 0.5)


def _pygfx_matrix(transform: AffineTransform) -> np.ndarray:
    """Embed a 2-D or 3-D AffineTransform into a 4x4 pygfx matrix.

    Reverses data axis order (z, y, x) -> pygfx (x, y, z).
    Identical to the helpers in _points_memory.py and _lines_memory.py --
    keep in sync or extract to a shared utility.
    """
    nd = transform.ndim
    src = transform.matrix
    swap = list(reversed(range(nd)))
    m = np.eye(4, dtype=np.float32)
    for dst_i, src_i in enumerate(swap):
        for dst_j, src_j in enumerate(swap):
            m[dst_i, dst_j] = src[src_i, src_j]
        m[dst_i, 3] = src[src_i, nd]
    return m


def _node_id_for_row(data_store, row: int):
    """Translate a store node row to its original id (D14).

    Falls back to the row itself when no store is available, which keeps
    the payload well-formed in unit tests that drive the visual directly.
    """
    if data_store is None:
        return row
    return data_store.ids_for_rows([row])[0]


def _build_node_material(appearance: GraphAppearance) -> AlphaPointsMaterial:
    """Construct the node material from a GraphAppearance.

    ``depth_compare`` comes from ``node_depth_compare``, which defaults to
    ``"<="`` so a node coplanar with an edge wins the tie and reads on top
    (D22).  Do not "simplify" this to a ``render_order`` assignment: with
    both children coplanar, insertion order and ``render_order`` were both
    measured to do nothing, because the depth test decides and the
    first-drawn fragment wins under ``"<"``.
    """
    material = AlphaPointsMaterial(
        size=appearance.node_size,
        size_space=appearance.node_size_space,
        color=appearance.node_color,
        color_mode=appearance.node_color_mode,
        size_mode=appearance.node_size_mode,
        opacity=appearance.opacity,
        depth_test=appearance.depth_test,
        depth_write=appearance.depth_write,
        depth_compare=appearance.node_depth_compare,
    )
    material.alpha_mode = appearance.transparency_mode
    return material


def _build_edge_material(appearance: GraphAppearance) -> AlphaLineSegmentMaterial:
    """Construct the edge material from a GraphAppearance."""
    material = AlphaLineSegmentMaterial(
        thickness=appearance.edge_thickness,
        thickness_space=appearance.edge_thickness_space,
        color=appearance.edge_color,
        color_mode=appearance.edge_color_mode,
        opacity=appearance.opacity,
        depth_test=appearance.depth_test,
        depth_write=appearance.depth_write,
        depth_compare=appearance.edge_depth_compare,
    )
    material.alpha_mode = appearance.transparency_mode
    return material


class GFXGraphMemoryVisual:
    """Render-layer visual for one GraphVisual: a pygfx compound (D1).

    A ``gfx.Group`` holds a ``gfx.Points`` (nodes) and a ``gfx.Line`` with
    ``LineSegmentMaterial`` (edges).  The group carries the node matrix and
    the children inherit it, so the transform is applied once rather than
    per sub-node.

    Single node for both 2D and 3D, exactly like points and lines:
    ``get_node_for_dims`` always returns the group, ``SceneManager.swap_node``
    no-ops on ``old_node is new_node``, and the reslice that follows does
    the real work.  The group is what gets registered in ``_active_nodes``,
    and ``SceneManager.get_visual_id_for_node`` already walks the parent
    chain from a picked leaf, so the compound needs no scene-manager
    changes.

    Draw order
    ----------
    **Nodes are drawn over edges by the depth compare, not by ordering
    (D22).**  With both children coplanar at ``z = 0``, insertion order
    does not decide it and neither does ``render_order``; the depth test
    does, and the first-drawn fragment wins because ``depth_compare`` is
    ``"<"``.  Nodes win because ``GraphAppearance.node_depth_compare``
    defaults to ``"<="``.

    Coordinate convention
    ---------------------
    The store delivers positions in data-axis order.  In ``_commit``:

    - **3D path** -- ``positions[:, [2, 1, 0]]`` reverses data ``(z, y, x)``
      to pygfx ``(x, y, z)``.
    - **2D path** -- zero-pad the ``(n, 2)`` displayed-plane coords to 3D,
      then swap ``[1, 0, 2]`` so stored ``(row, col)`` becomes pygfx
      ``(x=col, y=row, z=0)``.

    **Both** the node buffer and the edge-vertex buffer go through it.

    Trail alpha
    -----------
    An all-ones alpha buffer is uploaded when no trail has ``fade=True``.
    Swapping to a stock material to avoid the binding would force a
    pipeline rebuild on every fade toggle.

    Parameters
    ----------
    visual_model : GraphVisual
        Associated model-layer visual.
    render_modes : set[str]
        ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    transform : AffineTransform
        Data-to-world transform. Must cover all data axes.
    """

    #: In-memory visuals are cheap to reslice and must never be cancelled.
    cancellable: bool = False

    def __init__(
        self,
        visual_model: GraphVisual,
        render_modes: set[str],
        transform: AffineTransform,
    ) -> None:
        invalid = render_modes - {"2d", "3d"}
        if invalid or not render_modes:
            raise ValueError(
                f"render_modes must be a non-empty subset of {{'2d','3d'}}, "
                f"got {render_modes!r}"
            )

        self.visual_model_id: UUID = visual_model.id
        self.render_modes: set[str] = render_modes
        self._transform: AffineTransform = transform
        self._last_displayed_axes: tuple[int, ...] | None = None

        self._aabb_enabled: bool = visual_model.aabb.enabled
        self._aabb_color: str = visual_model.aabb.color
        self._aabb_line_width: float = visual_model.aabb.line_width
        self._aabb_line: gfx.Line | None = None

        appearance = visual_model.appearance
        self._node_material = _build_node_material(appearance)
        self._edge_material = _build_edge_material(appearance)
        self._node_material.pick_write = (
            visual_model.pick_write and appearance.node_pick_write
        )
        self._edge_material.pick_write = (
            visual_model.pick_write and appearance.edge_pick_write
        )
        self._pick_write: bool = visual_model.pick_write

        self._empty_node_material = AlphaPointsMaterial(color=(0, 0, 0, 0), opacity=0.0)
        self._empty_edge_material = AlphaLineSegmentMaterial(
            color=(0, 0, 0, 0), opacity=0.0
        )

        # Declared colour modes, honoured verbatim: the render layer never
        # infers them from the data and never writes them back (D20).
        self._node_color_mode: str = appearance.node_color_mode
        self._edge_color_mode: str = appearance.edge_color_mode
        self._node_size_mode: str = appearance.node_size_mode

        self._nodes_empty: bool = True
        self._edges_empty: bool = True

        # A rendered vertex index is a row in the *slice*, not in the store.
        # These maps translate it back; None means identity / placeholder.
        self._original_node_rows: np.ndarray | None = None
        self._original_edge_rows: np.ndarray | None = None
        self._edge_endpoint_rows: np.ndarray | None = None

        # Snapshot of the model's trail, refreshed by on_trail_changed.
        self._trail: dict = dict(visual_model.trail)

        # Warn-once bookkeeping for a trail on a displayed axis (D21).  This
        # is deliberately NOT the warnings module's own dedup, which keys on
        # (message, category, module, lineno) and is process-global -- two
        # visuals with the same bad axis would emit only one warning.
        self._warned_displayed_axes: set[int] = set()

        self.node_points = gfx.Points(
            gfx.Geometry(
                positions=_PLACEHOLDER_NODE_POSITIONS.copy(),
                alphas=np.ones(1, dtype=np.float32),
            ),
            self._empty_node_material,
        )
        self.node_edges = gfx.Line(
            gfx.Geometry(
                positions=_PLACEHOLDER_EDGE_POSITIONS.copy(),
                alphas=np.ones(2, dtype=np.float32),
            ),
            self._empty_edge_material,
        )
        self.node_points.visible = appearance.node_visible
        self.node_edges.visible = appearance.edge_visible

        self.node = gfx.Group()
        self.node.add(self.node_edges)
        self.node.add(self.node_points)
        self.node.render_order = appearance.render_order
        self.node.visible = appearance.visible

        # Both attributes point to the same node; swap_node's identity guard
        # makes dim-toggling a no-op for this visual.
        self.node_2d = self.node
        self.node_3d = self.node

    # ------------------------------------------------------------------
    # LOD
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 -- single-resolution in-memory store."""
        return 1

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group:
        """Return the node for the given displayed axes -- always the group."""
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return self.node

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        return True

    def get_node(self, mode: str) -> gfx.Group:
        return self.node

    def build_node(
        self, mode, visual_model, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    def rebuild_node_geometry(
        self, mode, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    def on_stacked_axes_changed(self, stacked_axes: tuple[int, ...]) -> None:
        pass

    # ------------------------------------------------------------------
    # Node matrix
    # ------------------------------------------------------------------

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        sub = self._transform.select_axes(displayed_axes)
        self.node.local.matrix = _pygfx_matrix(sub)

    # ------------------------------------------------------------------
    # Pick index translation
    # ------------------------------------------------------------------

    def node_row_for_vertex(self, vertex_index: int) -> int:
        """Map a pick vertex index to the store's node row.

        In a sliced view the rendered buffer holds only the nodes that
        survived the window, so ``vertex_index`` is a row in that subset.
        """
        idx_map = self._original_node_rows
        if idx_map is None or not (0 <= vertex_index < len(idx_map)):
            return vertex_index
        return int(idx_map[vertex_index])

    def edge_row_for_vertex(self, vertex_index: int) -> int:
        """Map a pick vertex index to the store's edge row, or ``-1``.

        ``LineSegmentMaterial`` lays out one explicit vertex pair per edge,
        so the rendered edge index is the integer half.  Under the ``roi``
        slice strategy ``original_edge_rows`` is None -- the index returns
        endpoint pairs, not rows -- and the caller resolves the row from
        the endpoints instead.
        """
        rendered = vertex_index // 2
        idx_map = self._original_edge_rows
        if idx_map is None:
            return -1
        if not (0 <= rendered < len(idx_map)):
            return -1
        return int(idx_map[rendered])

    def endpoint_rows_for_vertex(self, vertex_index: int) -> tuple[int, int] | None:
        """Return the store node rows of the picked edge's two endpoints."""
        rendered = vertex_index // 2
        endpoints = self._edge_endpoint_rows
        if endpoints is None or not (0 <= rendered < len(endpoints)):
            return None
        source, target = endpoints[rendered]
        return int(source), int(target)

    def decode_pick(self, hit_object, pick_info: dict, data_store=None):
        """Return a typed pick payload for whichever child was hit.

        Picks report the store's **original node id**, not the render-buffer
        row (D14): a rendered vertex index is a row in the *slice*, and even
        the store row is not what a caller needs to look up attributes or
        index back into a geff file.

        Parameters
        ----------
        hit_object : gfx.WorldObject
            The picked object -- one of this visual's two children, or
            something else entirely.
        pick_info : dict
            The pygfx pick payload.
        data_store : GraphMemoryStore | None
            The backing store, supplied by the render manager, which already
            holds it.  Used only to translate rows to original ids; never
            read on the per-frame path.  With ``None`` the payload falls
            back to reporting rows.

        Returns
        -------
        GraphNodePickInfo | GraphEdgePickInfo | None
            ``None`` when the hit object is neither child, or when the
            payload carries no vertex index.
        """
        from cellier.events._events import GraphEdgePickInfo, GraphNodePickInfo

        index = pick_info.get("vertex_index")
        if index is None:
            return None

        if hit_object is self.node_points:
            row = self.node_row_for_vertex(int(index))
            return GraphNodePickInfo(
                node_id=_node_id_for_row(data_store, row), node_row=row
            )

        if hit_object is self.node_edges:
            endpoints = self.endpoint_rows_for_vertex(int(index))
            if endpoints is None:
                return None
            source_row, target_row = endpoints
            edge_row = self.edge_row_for_vertex(int(index))
            if edge_row < 0 and data_store is not None:
                # The ROI strategy returns endpoint pairs rather than edge
                # rows, so the row is resolved here, for this one edge --
                # never for a whole candidate set per frame (D18).
                edge_row = data_store.edge_row_for_endpoints(source_row, target_row)
            return GraphEdgePickInfo(
                edge_index=edge_row,
                source_node_id=_node_id_for_row(data_store, source_row),
                target_node_id=_node_id_for_row(data_store, target_row),
            )

        return None

    # ------------------------------------------------------------------
    # Slice request building
    # ------------------------------------------------------------------

    def _build_request(self, dims_state: DimsState) -> GraphSliceRequest:
        """Assemble one request, resolving the trail into extents and fades.

        A ``TrailConfig`` on a *displayed* axis produces no extent: a window
        on an axis you are looking down has no meaning, and the config
        becomes live again the moment the view changes back.  That is
        legitimate, but indistinguishable at slice time from a typo, so it
        warns once per (visual, axis) -- see ``_warned_displayed_axes``.
        """
        sliced = dims_state.selection.slice_indices
        displayed = set(dims_state.selection.displayed_axes)

        extents: dict[int, tuple[float, float]] = {}
        fades: dict[int, tuple[float, float, float]] = {}

        for axis in sliced:
            config = self._trail.get(axis)
            if config is None:
                extents[axis] = _DEFAULT_EXTENT
                continue
            extents[axis] = (config.before, config.after)
            if config.fade:
                fades[axis] = (
                    config.resolved_fade_before,
                    config.resolved_fade_after,
                    config.min_alpha,
                )

        for axis in self._trail:
            if axis in displayed and axis not in self._warned_displayed_axes:
                self._warned_displayed_axes.add(axis)
                warnings.warn(
                    f"Trail configured on axis {axis}, which is currently "
                    f"displayed; a window on an axis you are looking down has "
                    f"no effect. It will apply again when axis {axis} becomes "
                    f"a sliced axis. Displayed axes are "
                    f"{tuple(dims_state.selection.displayed_axes)}.",
                    stacklevel=2,
                )

        shared_id = uuid4()
        return GraphSliceRequest(
            slice_request_id=shared_id,
            chunk_request_id=shared_id,
            scale_index=0,
            displayed_axes=dims_state.selection.displayed_axes,
            slice_indices=dict(sliced),
            extents=extents,
            fades=fades,
        )

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float = 1.0,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[GraphSliceRequest]:
        """3-D planning path -- returns one GraphSliceRequest."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        return [self._build_request(dims_state)]

    def build_slice_request_2d(
        self,
        camera_pos_world: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[GraphSliceRequest]:
        """2-D planning path -- returns one GraphSliceRequest."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        return [self._build_request(dims_state)]

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pygfx_positions(positions: np.ndarray) -> np.ndarray:
        """Reorder store positions into pygfx ``(x, y, z)``.

        The single axis-reordering site for this visual.  Both the node
        buffer and the edge-vertex buffer go through it.
        """
        n_points, n_dims = positions.shape
        if n_dims == 2:
            # 2D path -- pad the displayed-plane coords with z=0, then swap
            # so stored (row, col) becomes pygfx (x=col, y=row, z=0).
            zeros = np.zeros((n_points, 1), dtype=np.float32)
            padded = np.concatenate([positions, zeros], axis=1)
            return np.ascontiguousarray(padded[:, [1, 0, 2]])
        # 3D path -- reverse (z, y, x) data order to (x, y, z) pygfx order.
        return np.ascontiguousarray(positions)[:, [2, 1, 0]]

    def _commit(self, graph_data: GraphData, is_2d: bool) -> None:
        """Upload one GraphData into the two sub-nodes.

        Nodes and edges are committed together and their materials swapped
        independently, so an empty edge set never blanks the nodes.

        ``color_mode`` and ``size_mode`` are written from the *appearance*,
        never from the data (D20): a declared ``"vertex"`` with nothing in
        the store to back it is a misconfiguration and raises here rather
        than silently falling back to uniform.
        """
        self._commit_nodes(graph_data)
        self._commit_edges(graph_data)

    def _commit_nodes(self, graph_data: GraphData) -> None:
        positions = self._to_pygfx_positions(graph_data.node_positions)
        n_points = positions.shape[0]

        colors = graph_data.node_colors
        if self._node_color_mode == "vertex" and colors is None:
            if not graph_data.nodes_empty:
                raise ValueError(
                    f"Visual {self.visual_model_id}: appearance declares "
                    "node_color_mode='vertex' but the graph store carries no "
                    "per-node colors. Set node_color_mode='uniform', or give "
                    "the store node_colors."
                )

        geom_kwargs: dict = {
            "positions": positions,
            "alphas": self._alpha_buffer(graph_data.node_alpha, n_points),
        }
        if colors is not None:
            geom_kwargs["colors"] = np.ascontiguousarray(colors)
        sizes = graph_data.node_sizes
        if self._node_size_mode == "vertex" and sizes is None:
            if not graph_data.nodes_empty:
                raise ValueError(
                    f"Visual {self.visual_model_id}: appearance declares "
                    "node_size_mode='vertex' but the graph store carries no "
                    "per-node sizes. Set node_size_mode='uniform', or give "
                    "the store node_sizes."
                )
        if sizes is not None:
            geom_kwargs["sizes"] = np.ascontiguousarray(sizes)

        self.node_points.geometry = gfx.Geometry(**geom_kwargs)

        target = (
            self._empty_node_material if graph_data.nodes_empty else self._node_material
        )
        if self.node_points.material is not target:
            self.node_points.material = target

        self._nodes_empty = graph_data.nodes_empty
        self._original_node_rows = (
            None if graph_data.nodes_empty else graph_data.original_node_rows
        )

    def _commit_edges(self, graph_data: GraphData) -> None:
        positions = self._to_pygfx_positions(graph_data.edge_positions)
        n_vertices = positions.shape[0]

        colors = graph_data.edge_colors
        if self._edge_color_mode == "vertex" and colors is None:
            if not graph_data.edges_empty:
                raise ValueError(
                    f"Visual {self.visual_model_id}: appearance declares "
                    "edge_color_mode='vertex' but the graph store carries no "
                    "per-edge colors. Set edge_color_mode='uniform', or give "
                    "the store edge_colors."
                )

        geom_kwargs: dict = {
            "positions": positions,
            "alphas": self._alpha_buffer(graph_data.edge_alpha, n_vertices),
        }
        if colors is not None:
            geom_kwargs["colors"] = np.ascontiguousarray(colors)

        self.node_edges.geometry = gfx.Geometry(**geom_kwargs)

        target = (
            self._empty_edge_material if graph_data.edges_empty else self._edge_material
        )
        if self.node_edges.material is not target:
            self.node_edges.material = target

        self._edges_empty = graph_data.edges_empty
        self._original_edge_rows = (
            None if graph_data.edges_empty else graph_data.original_edge_rows
        )
        self._edge_endpoint_rows = (
            None if graph_data.edges_empty else graph_data.edge_endpoint_rows
        )

    @staticmethod
    def _alpha_buffer(alpha: np.ndarray | None, n: int) -> np.ndarray:
        """Return the per-element alpha buffer, all-ones when no fade is on.

        Uploading ones rather than swapping materials to avoid the binding
        keeps one pipeline: a material swap would force a rebuild on every
        fade toggle.
        """
        if alpha is None:
            return np.ones(n, dtype=np.float32)
        return np.ascontiguousarray(alpha, dtype=np.float32)

    def on_data_ready(self, batch: list[tuple[GraphSliceRequest, GraphData]]) -> None:
        """3-D callback -- called on the main thread by SliceCoordinator."""
        if not batch:
            return
        _, data = batch[0]
        self._commit(data, is_2d=False)

    def on_data_ready_2d(
        self, batch: list[tuple[GraphSliceRequest, GraphData]]
    ) -> None:
        """2-D callback -- called on the main thread by SliceCoordinator."""
        if not batch:
            return
        _, data = batch[0]
        self._commit(data, is_2d=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        self._transform = event.transform
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_trail_changed(self, event: TrailChangedEvent) -> None:
        """Refresh the trail snapshot and reset the warn-once set (D21).

        Clearing the whole set rather than just the changed axis's entry is
        deliberate: it is less machinery, and re-warning about a still-
        broken axis right after the user edits that same trail dict is
        useful rather than noisy.  The set is otherwise *sticky across
        every dims change*, which is what makes flipping 2D/3D repeatedly
        silent.
        """
        self._trail = dict(event.trail)
        self._warned_displayed_axes.clear()

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply appearance field changes to the live materials.

        ``node_size_space`` / ``edge_thickness_space`` are applied live.
        They were previously documented as constructor-only, which is not
        true of the pinned pygfx: assigning either on a live material
        re-renders.
        """
        name = event.field_name
        val = event.new_value

        if name == "node_color":
            self._node_material.color = val
        elif name == "node_size":
            self._node_material.size = val
        elif name == "node_color_mode":
            self._node_material.color_mode = val
            self._node_color_mode = val
        elif name == "node_size_mode":
            self._node_material.size_mode = val
            self._node_size_mode = val
        elif name == "node_size_space":
            self._node_material.size_space = val
        elif name == "node_visible":
            self.node_points.visible = val
        elif name == "node_pick_write":
            self._node_material.pick_write = self._pick_write and val
        elif name == "node_depth_compare":
            self._node_material.depth_compare = val
        elif name == "edge_color":
            self._edge_material.color = val
        elif name == "edge_thickness":
            self._edge_material.thickness = val
        elif name == "edge_thickness_space":
            self._edge_material.thickness_space = val
        elif name == "edge_color_mode":
            self._edge_material.color_mode = val
            self._edge_color_mode = val
        elif name == "edge_visible":
            self.node_edges.visible = val
        elif name == "edge_pick_write":
            self._edge_material.pick_write = self._pick_write and val
        elif name == "edge_depth_compare":
            self._edge_material.depth_compare = val
        elif name == "opacity":
            self._node_material.opacity = val
            self._edge_material.opacity = val
        elif name == "depth_test":
            self._node_material.depth_test = val
            self._edge_material.depth_test = val
        elif name == "depth_write":
            self._node_material.depth_write = val
            self._edge_material.depth_write = val
        elif name == "transparency_mode":
            self._node_material.alpha_mode = val
            self._edge_material.alpha_mode = val
        elif name == "render_order":
            self.node.render_order = val

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        self.node.visible = event.visible

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        self._pick_write = event.pick_write
        self._node_material.pick_write = event.pick_write
        self._edge_material.pick_write = event.pick_write

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Store AABB param changes; apply to the line node if it exists."""
        if event.field_name == "enabled":
            self._aabb_enabled = event.new_value
            if self._aabb_line is not None:
                self._aabb_line.visible = event.new_value
        elif event.field_name == "color":
            self._aabb_color = event.new_value
            if self._aabb_line is not None:
                self._aabb_line.material.color = event.new_value
        elif event.field_name == "line_width":
            self._aabb_line_width = event.new_value
            if self._aabb_line is not None:
                self._aabb_line.material.thickness = event.new_value

    # ------------------------------------------------------------------
    # No-op cancellation stubs (no brick cache)
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        """No-op -- in-memory visuals have no reserved GPU brick slots."""

    def cancel_pending_2d(self) -> None:
        """No-op -- in-memory visuals have no reserved GPU brick slots."""

    def tick(self) -> None:
        """Called once per rendered frame. No per-frame state to advance."""
