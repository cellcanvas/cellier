# src/cellier/v2/render/visuals/_lines_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.lines._lines_requests import LinesSliceRequest
from cellier.render.shaders._alpha_modulated import AlphaLineSegmentMaterial

if TYPE_CHECKING:
    from cellier._state import DimsState
    from cellier.data.lines._lines_requests import LinesData
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        PickWriteChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.transform import AffineTransform
    from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual

# Placeholder geometry — one degenerate segment (both vertices at the origin).
# LineSegmentMaterial requires an even vertex count; two vertices is the minimum.
_PLACEHOLDER_POSITIONS = np.zeros((2, 3), dtype=np.float32)


def _pygfx_matrix(transform: AffineTransform) -> np.ndarray:
    """Embed a 2-D or 3-D AffineTransform into a 4x4 pygfx matrix.

    Reverses data axis order (z, y, x) → pygfx (x, y, z).
    Identical to the helper in _points_memory.py — keep in sync or
    extract to a shared utility.
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


def _build_material(appearance: LinesMemoryAppearance) -> AlphaLineSegmentMaterial:
    """Construct the line material from a LinesMemoryAppearance.

    ``AlphaLineSegmentMaterial`` rather than ``gfx.LineSegmentMaterial`` so
    every in-memory visual shares one pipeline shape with the graph visual.
    It reads an all-ones ``alphas`` buffer here and is otherwise identical;
    see ``render/shaders/_alpha_modulated.py``.
    """
    mat = AlphaLineSegmentMaterial(
        thickness=appearance.thickness,
        thickness_space=appearance.thickness_space,
        color=appearance.color,
        color_mode=appearance.color_mode,
        opacity=appearance.opacity,
        depth_test=appearance.depth_test,
        depth_write=appearance.depth_write,
        depth_compare=appearance.depth_compare,
    )
    mat.alpha_mode = appearance.transparency_mode
    return mat


class GFXLinesMemoryVisual:
    """Render-layer visual for one LinesVisual backed by in-memory lines data.

    Uses a single ``gfx.Line`` node with ``gfx.LineSegmentMaterial`` for
    both 2D and 3D modes.  ``get_node_for_dims`` always returns ``self.node``;
    SceneManager.swap_node no-ops because ``old_node is new_node``.  The
    subsequent reslice updates the geometry and swaps the material between
    ``_material`` and ``_empty_material``.

    Segment representation
    ----------------------
    Vertices arrive as even-length arrays.  Pair ``(2n, 2n+1)`` is segment
    ``n``.  ``gfx.LineSegmentMaterial`` draws them as independent disconnected
    segments, not a connected polyline.

    Coordinate convention
    ---------------------
    The store delivers positions in data-axis order: column 0 is axis 0 (z),
    column 1 is axis 1 (y), column 2 is axis 2 (x).

    In ``_commit``:
    - **3D path** — ``positions[:, [2, 1, 0]]`` reverses to pygfx ``(x, y, z)``
      order.  This is the single axis-reversal site; do not reorder elsewhere.
    - **2D path** — incoming ``(N, 2)`` positions (already projected by the
      store) are padded with a zero column.  The node matrix handles
      orientation; no column reversal is applied.

    Parameters
    ----------
    visual_model : LinesVisual
        Associated model-layer visual.
    render_modes : set[str]
        ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    transform : AffineTransform
        Data-to-world transform. Must cover all data axes.
    """

    #: In-memory visuals are cheap to reslice and must never be cancelled.
    #: SliceCoordinator.submit() reads this flag before calling cancel_visual.
    cancellable: bool = False

    def __init__(
        self,
        visual_model: LinesVisual,
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
        self._material = _build_material(appearance)
        # Plumb the model's pick_write flag into the pygfx material so the
        # pick buffer records this visual; pygfx materials default to False.
        self._material.pick_write = visual_model.pick_write
        self._empty_material = AlphaLineSegmentMaterial(color=(0, 0, 0, 0), opacity=0.0)

        # The declared color_mode, honoured verbatim.  It is a caller
        # declaration of where RGB comes from and is never inferred from
        # the data nor written back at commit time (D20).
        self._color_mode: str = appearance.color_mode
        self._is_empty: bool = True

        # Maps each rendered segment to its edge index in the store's full
        # segment array.  In a sliced view the rendered buffer is a filtered
        # subset, so the pick's rendered edge index is NOT the original edge
        # index; this map (the slab's surviving segments) translates it.
        # None means identity (no filtering / placeholder).
        self._original_edge_indices: np.ndarray | None = None

        geom = gfx.Geometry(
            positions=_PLACEHOLDER_POSITIONS.copy(),
            alphas=np.ones(2, dtype=np.float32),
        )
        self.node = gfx.Line(geom, self._empty_material)
        self.node.render_order = appearance.render_order

        # Both attributes point to the same node.
        # SceneManager.swap_node's old_node is new_node guard makes dim-toggling
        # a no-op for this visual.
        self.node_2d = self.node
        self.node_3d = self.node

    # ------------------------------------------------------------------
    # LOD
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution in-memory store."""
        return 1

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def edge_index_for_vertex(self, vertex_index: int) -> int:
        """Map a pygfx pick vertex index to this visual's edge index.

        ``gfx.LineSegmentMaterial`` lays out one explicit vertex pair per edge:
        edge ``k`` occupies positions ``2k`` and ``2k + 1``.  The pick buffer
        reports the hit vertex index, so the rendered edge index is the integer
        half.  In a sliced view the rendered buffer holds only the segments that
        survived the slab filter, so that rendered edge index is translated back
        to the store's original edge index via ``_original_edge_indices`` (None
        means identity / no filtering).

        Parameters
        ----------
        vertex_index : int
            ``vertex_index`` from the pygfx pick payload.

        Returns
        -------
        int
            Index of the edge (vertex pair) in the store's full segment array.
        """
        rendered_edge = vertex_index // 2
        idx_map = self._original_edge_indices
        if idx_map is None or not (0 <= rendered_edge < len(idx_map)):
            return rendered_edge
        return int(idx_map[rendered_edge])

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Line:
        """Return the node for the given displayed axes — always self.node.

        SceneManager.swap_node will no-op because the returned node is
        the same object as the currently active node.  The reslice that
        follows updates the geometry and material.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            New displayed axes.

        Returns
        -------
        gfx.Line
            Always self.node.
        """
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return self.node

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        return True

    def get_node(self, mode: str) -> gfx.Line:
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
    # Slice request building
    # ------------------------------------------------------------------

    def _build_request(self, dims_state: DimsState) -> LinesSliceRequest:
        shared_id = uuid4()
        return LinesSliceRequest(
            slice_request_id=shared_id,
            chunk_request_id=shared_id,
            scale_index=0,
            displayed_axes=dims_state.selection.displayed_axes,
            slice_indices=dict(dims_state.selection.slice_indices),
            thickness=0.5,
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
    ) -> list[LinesSliceRequest]:
        """3-D planning path — returns one LinesSliceRequest."""
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
    ) -> list[LinesSliceRequest]:
        """2-D planning path — returns one LinesSliceRequest."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        return [self._build_request(dims_state)]

    # ------------------------------------------------------------------
    # Commit — shared logic for both callbacks
    # ------------------------------------------------------------------

    def _commit(self, lines_data: LinesData, is_2d: bool) -> None:
        """Upload LinesData to self.node.

        Pads 2D positions to 3D with z=0.
        Applies axis reversal for 3D data to match pygfx coordinate order.
        Swaps material between _material and _empty_material as needed.

        ``color_mode`` is written from the *appearance*, never from the
        data (D20).  A declared ``"vertex"`` with no colours in the store
        raises here rather than silently falling back to uniform, which is
        the exact class of behaviour D20 removes.

        Coordinate convention (DO NOT reorder positions elsewhere):
        - 3D path: positions[:, [2, 1, 0]] reverses (z, y, x) → (x, y, z)
          for pygfx.
        - 2D path: zero-pad to 3D; reverse columns [1, 0, 2]
        """
        positions = lines_data.positions
        n_verts = positions.shape[0]
        n_dims = positions.shape[1]

        if n_dims == 2:
            # 2D path — pad displayed-plane coords with z=0.
            # Node matrix handles axis orientation; no reversal needed.
            zeros = np.zeros((n_verts, 1), dtype=np.float32)
            pos3d = np.concatenate([positions, zeros], axis=1)[:, [1, 0, 2]]

        else:
            # 3D path — reverse (z, y, x) data order to (x, y, z) pygfx order.
            # This is the single axis-reversal site for lines. Do not reorder
            # positions anywhere else in the pipeline.
            pos3d = np.ascontiguousarray(positions)[:, [2, 1, 0]]

        colors = lines_data.colors
        if self._color_mode == "vertex" and colors is None and not lines_data.is_empty:
            raise ValueError(
                f"Visual {self.visual_model_id}: appearance declares "
                "color_mode='vertex' but the lines store carries no "
                "per-vertex colors. Set color_mode='uniform', or give the "
                "store colors."
            )

        # An all-ones alpha buffer: the material multiplies it in, so this
        # is visually identical to the stock material.
        geom_kwargs: dict = {
            "positions": pos3d,
            "alphas": np.ones(n_verts, dtype=np.float32),
        }
        if colors is not None:
            geom_kwargs["colors"] = np.ascontiguousarray(colors)

        self.node.geometry = gfx.Geometry(**geom_kwargs)

        # Select material.
        target = self._empty_material if lines_data.is_empty else self._material
        if self.node.material is not target:
            self.node.material = target

        self._is_empty = lines_data.is_empty
        # Keep the pick edge map in lockstep with the uploaded buffer.
        self._original_edge_indices = (
            None if lines_data.is_empty else lines_data.original_edge_indices
        )

    def on_data_ready(self, batch: list[tuple[LinesSliceRequest, LinesData]]) -> None:
        """3-D callback — called on the main thread by SliceCoordinator."""
        if not batch:
            return
        _, data = batch[0]
        self._commit(data, is_2d=False)

    def on_data_ready_2d(
        self, batch: list[tuple[LinesSliceRequest, LinesData]]
    ) -> None:
        """2-D callback — called on the main thread by SliceCoordinator."""
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

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply appearance field changes to the live material.

        ``color_mode`` is applied straight to the material; nothing in
        ``_commit`` overwrites it (D20).

        ``thickness_space`` is applied live.  It was previously documented
        here as constructor-only, which is not true of the pinned pygfx.
        """
        name = event.field_name
        val = event.new_value
        if name == "color":
            self._material.color = val
        elif name == "color_mode":
            self._material.color_mode = val
            self._color_mode = val
        elif name == "thickness_space":
            self._material.thickness_space = val
        elif name == "opacity":
            self._material.opacity = val
        elif name == "thickness":
            self._material.thickness = val
        elif name == "depth_test":
            self._material.depth_test = val
        elif name == "depth_write":
            self._material.depth_write = val
        elif name == "depth_compare":
            self._material.depth_compare = val
        elif name == "transparency_mode":
            self._material.alpha_mode = val
        elif name == "render_order":
            self.node.render_order = val

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        self.node.visible = event.visible

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        self._material.pick_write = event.pick_write

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Store AABB param changes; apply to line node if it exists."""
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
        """No-op — in-memory visuals have no reserved GPU brick slots."""

    def cancel_pending_2d(self) -> None:
        """No-op — in-memory visuals have no reserved GPU brick slots."""

    def tick(self) -> None:
        """Called once per rendered frame. No per-frame state to advance.

        If any per-frame state, implement it here (e.g., temporal jitter seed).
        """
        pass
