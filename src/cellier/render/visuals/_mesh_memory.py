# src/cellier/v2/render/visuals/_mesh_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.mesh._mesh_requests import MeshSliceRequest

if TYPE_CHECKING:
    from cellier._state import DimsState
    from cellier.data.mesh._mesh_requests import MeshData
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        PickWriteChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.transform import AffineTransform
    from cellier.visuals._mesh_memory import MeshVisual

_SIDE_MAP = {"both": "both", "front": "front", "back": "back"}

_PLACEHOLDER_POSITIONS = np.zeros((3, 3), dtype=np.float32)
_PLACEHOLDER_INDICES = np.array([[0, 1, 2]], dtype=np.int32)
_PLACEHOLDER_NORMALS = np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32)


def _apply_alpha_mode(material: gfx.MeshAbstractMaterial, opacity: float) -> None:
    """Set alpha_mode based on opacity."""
    transparent = float(opacity) < 1.0 - 1e-6
    material.alpha_mode = "blend" if transparent else "solid"


def _pygfx_matrix(transform: AffineTransform) -> np.ndarray:
    """Embed a 2-D or 3-D AffineTransform into a 4x4 pygfx matrix.

    Reverses data axis order (z, y, x) → pygfx (x, y, z).
    Identical to the helper in _image_memory.py — keep in sync or
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


def _build_material_3d(appearance) -> gfx.MeshAbstractMaterial:
    side = _SIDE_MAP.get(appearance.side, "both")
    if appearance.appearance_type == "flat":
        material = gfx.MeshBasicMaterial(
            color=appearance.color,
            color_mode=appearance.color_mode,
            wireframe=appearance.wireframe,
            wireframe_thickness=appearance.wireframe_thickness,
            opacity=appearance.opacity,
            side=side,
        )
    else:  # phong
        material = gfx.MeshPhongMaterial(
            color=appearance.color,
            color_mode=appearance.color_mode,
            shininess=appearance.shininess,
            opacity=appearance.opacity,
            flat_shading=appearance.flat_shading,
            side=side,
        )
    if appearance.transparency_mode != "blend":
        material.alpha_mode = appearance.transparency_mode
    else:
        _apply_alpha_mode(material, appearance.opacity)
    material.depth_test = appearance.depth_test
    material.depth_write = appearance.depth_write
    material.depth_compare = appearance.depth_compare
    return material


def _build_material_2d(appearance) -> gfx.MeshBasicMaterial:
    """Flat unlit material for the 2D cross-section path.

    Always both-sided in 2D to avoid winding confusion after projection.
    """
    material = gfx.MeshBasicMaterial(
        color=appearance.color,
        color_mode=appearance.color_mode,
        wireframe=False,
        opacity=appearance.opacity,
        side="both",
    )
    material.depth_test = appearance.depth_test
    material.depth_write = appearance.depth_write
    material.depth_compare = appearance.depth_compare
    return material


class GFXMeshMemoryVisual:
    """Render-layer visual for one MeshVisual backed by in-memory mesh data.

    Uses a single gfx.Mesh node for both 2D and 3D.
    get_node_for_dims always returns self.node; SceneManager.swap_node
    no-ops because old_node is new_node.  The subsequent reslice
    updates geometry and swaps the material (3D material in on_data_ready,
    flat 2D material in on_data_ready_2d).

    Parameters
    ----------
    visual_model : MeshVisual
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
        visual_model: MeshVisual,
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
        self._material_3d = _build_material_3d(appearance)
        self._material_2d = _build_material_2d(appearance)
        # Plumb the model's pick_write flag into both pygfx materials so the
        # pick buffer records this visual; pygfx materials default to False.
        self._material_3d.pick_write = visual_model.pick_write
        self._material_2d.pick_write = visual_model.pick_write
        self._empty_material = gfx.MeshBasicMaterial(color=(0, 0, 0, 0), opacity=0.0)
        # The declared color_mode, honoured verbatim.  It is a caller
        # declaration of where RGB comes from and is never inferred from
        # the data nor written back at commit time (D20).
        self._color_mode: str = appearance.color_mode
        self._is_empty: bool = True

        # Maps each rendered face to its face index in the store's full face
        # array.  In a sliced view the rendered buffer holds only the faces
        # that survived the slab filter, so pygfx's face_index is NOT the
        # original face index; this map (the slab's surviving faces) translates
        # it.  None means identity (no filtering / placeholder).
        self._original_face_indices: np.ndarray | None = None

        geom = gfx.Geometry(
            positions=_PLACEHOLDER_POSITIONS.copy(),
            indices=_PLACEHOLDER_INDICES.copy(),
            normals=_PLACEHOLDER_NORMALS.copy(),
        )
        self.node = gfx.Mesh(geom, self._empty_material)
        self.node.render_order = appearance.render_order

        # Both attributes point to the same node.
        # swap_node's old_node is new_node guard makes dim-toggling a no-op.
        self.node_2d: gfx.Mesh | None = self.node if "2d" in render_modes else None
        self.node_3d: gfx.Mesh | None = self.node if "3d" in render_modes else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution in-memory store."""
        return 1

    # ------------------------------------------------------------------
    # Cancellation stubs
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        """No-op — no GPU brick slots to release."""

    def cancel_pending_2d(self) -> None:
        """No-op."""

    # ------------------------------------------------------------------
    # get_node_for_dims  (implements the protocol from the refactor)
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Mesh:
        """Always return self.node; eagerly update the node matrix.

        SceneManager.swap_node will no-op because the returned node is
        the same object as the currently active node.  The reslice that
        follows updates the geometry and material.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            New displayed axes.

        Returns
        -------
        gfx.Mesh
            Always self.node.
        """
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return self.node

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        return True

    def get_node(self, mode: str) -> gfx.Mesh:
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

    def _build_request(self, dims_state: DimsState) -> MeshSliceRequest:
        shared_id = uuid4()
        return MeshSliceRequest(
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
    ) -> list[MeshSliceRequest]:
        """3-D planning path — returns one MeshSliceRequest."""
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
    ) -> list[MeshSliceRequest]:
        """2-D planning path — returns one MeshSliceRequest."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        return [self._build_request(dims_state)]

    # ------------------------------------------------------------------
    # Commit — shared logic for both callbacks
    # ------------------------------------------------------------------

    def _commit(self, mesh_data: MeshData, is_2d: bool) -> None:
        """Upload MeshData to self.node.

        Pads 2D positions/normals to 3D with z=0.
        Swaps material between 3D, 2D, and empty variants as needed.

        ``color_mode`` is written from the *appearance*, never from the
        data (D20).  Two ways it can disagree with the store, both raised
        rather than silently corrected:

        - declaring ``"vertex"`` or ``"face"`` when the store carries no
          colours at all;
        - declaring ``"vertex"`` against per-face colours, or the reverse.
          pygfx does not validate this -- it renders the mismatched buffer
          without complaining -- so an unchecked mismatch is a wrong
          picture with no signal.
        """
        positions = mesh_data.positions
        normals = mesh_data.normals
        n_verts = positions.shape[0]
        n_dims = positions.shape[1]

        # Pad 2D → 3D.
        if n_dims == 2:
            zeros = np.zeros((n_verts, 1), dtype=np.float32)
            pos3d = np.concatenate([positions, zeros], axis=1)[:, [1, 0, 2]]
            nor3d = np.concatenate([normals, zeros], axis=1)[:, [1, 0, 2]]

        else:
            # switch the order from xyz to zyx to match the image
            pos3d = np.ascontiguousarray(positions)[:, [2, 1, 0]]
            nor3d = np.ascontiguousarray(normals)[:, [2, 1, 0]]

        # Build geometry.
        colors = mesh_data.colors
        geom_kwargs: dict = {
            "positions": pos3d,
            "indices": np.ascontiguousarray(mesh_data.indices),
            "normals": nor3d,
        }
        if self._color_mode != "uniform" and not mesh_data.is_empty:
            if colors is None:
                raise ValueError(
                    f"Visual {self.visual_model_id}: appearance declares "
                    f"color_mode='{self._color_mode}' but the mesh store "
                    "carries no colors. Set color_mode='uniform', or give "
                    "the store colors and a colors_layout."
                )
            if mesh_data.color_mode != self._color_mode:
                raise ValueError(
                    f"Visual {self.visual_model_id}: appearance declares "
                    f"color_mode='{self._color_mode}' but the store's "
                    f"colors_layout is '{mesh_data.color_mode}'. pygfx will "
                    "render the mismatched buffer without complaining, so "
                    "this is refused rather than corrected."
                )
        if colors is not None:
            geom_kwargs["colors"] = np.ascontiguousarray(colors)
        self.node.geometry = gfx.Geometry(**geom_kwargs)

        # Select material.
        if mesh_data.is_empty:
            target = self._empty_material
        elif is_2d:
            target = self._material_2d
        else:
            target = self._material_3d

        if self.node.material is not target:
            self.node.material = target

        self._is_empty = mesh_data.is_empty
        # Keep the pick face map in lockstep with the uploaded buffer.
        self._original_face_indices = (
            None if mesh_data.is_empty else mesh_data.original_face_indices
        )

    def face_index_for_pick(self, face_index: int) -> int:
        """Map a pygfx pick face index to the store's original face index.

        In a sliced view the rendered geometry holds only the faces that
        survived the slab filter, so ``face_index`` is a row in that subset,
        not the original face index.  ``_original_face_indices`` (the slab's
        surviving-face list) recovers the original index; when it is None
        (no filtering / placeholder) the index passes through unchanged.

        Parameters
        ----------
        face_index : int
            ``face_index`` from the pygfx pick payload.

        Returns
        -------
        int
            Index into the store's full face array.
        """
        idx_map = self._original_face_indices
        if idx_map is None or not (0 <= face_index < len(idx_map)):
            return face_index
        return int(idx_map[face_index])

    def on_data_ready(self, batch: list[tuple[MeshSliceRequest, MeshData]]) -> None:
        """3-D callback — called on the main thread by SliceCoordinator."""
        if not batch:
            return
        _, data = batch[0]
        self._commit(data, is_2d=False)

    def on_data_ready_2d(self, batch: list[tuple[MeshSliceRequest, MeshData]]) -> None:
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
        """Apply appearance field changes to live materials.

        ``color_mode`` is applied straight to both materials; nothing in
        ``_commit`` overwrites it (D20).
        """
        name = event.field_name
        val = event.new_value
        for mat in (self._material_3d, self._material_2d):
            if name == "color":
                mat.color = val
            elif name == "color_mode":
                mat.color_mode = val
                self._color_mode = val
            elif name == "opacity":
                mat.opacity = val
            elif name == "side":
                mat.side = val
        if name == "opacity":
            if self._material_3d.alpha_mode in ("blend", "solid"):
                _apply_alpha_mode(self._material_3d, float(val))
        elif name == "transparency_mode":
            if val != "blend":
                self._material_3d.alpha_mode = val
                self._material_2d.alpha_mode = val
            else:
                _apply_alpha_mode(self._material_3d, float(self._material_3d.opacity))
                _apply_alpha_mode(self._material_2d, float(self._material_2d.opacity))
        if name == "depth_test":
            self._material_3d.depth_test = val
            self._material_2d.depth_test = val
        elif name == "depth_write":
            self._material_3d.depth_write = val
            self._material_2d.depth_write = val
        elif name == "depth_compare":
            self._material_3d.depth_compare = val
            self._material_2d.depth_compare = val
        # Flat-only fields.
        if name == "wireframe" and hasattr(self._material_3d, "wireframe"):
            self._material_3d.wireframe = val
        elif name == "wireframe_thickness" and hasattr(
            self._material_3d, "wireframe_thickness"
        ):
            self._material_3d.wireframe_thickness = val
        # Phong-only fields.
        if name == "shininess" and hasattr(self._material_3d, "shininess"):
            self._material_3d.shininess = val
        elif name == "flat_shading" and hasattr(self._material_3d, "flat_shading"):
            self._material_3d.flat_shading = val
        if name == "render_order":
            self.node.render_order = val

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        self.node.visible = event.visible

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        self._material_3d.pick_write = event.pick_write
        self._material_2d.pick_write = event.pick_write

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

    def tick(self) -> None:
        """Called once per rendered frame. No per-frame state to advance.

        If any per-frame state, implement it here (e.g., temporal jitter seed).
        """
        pass
