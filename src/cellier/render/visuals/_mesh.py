"""The render-layer mesh visual: level children loaded by the chunk scheduler.

``plans/mesh_refactor_v3.md`` 5.3 and 5.8.  ``GFXMeshVisual`` loads through
``ChunkScheduler`` (``chunked = True``): a plan is a
:class:`~cellier.render.scheduling.DesiredSet` of whole levels, the read runs
in an executor (``MeshMemoryStore.get_data``), and
:class:`~cellier.render._level_residency.LevelResidency` decides what is
uploaded and what may be drawn.

One display rule holds everywhere: **the mesh never draws a position other
than the slider's.**  While a new position loads, the mesh is not drawn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.mesh._mesh_requests import (
    MeshSectionData,
    MeshSectionRequest,
    MeshSliceRequest,
)
from cellier.render._clipping import GeometryClippingMixin
from cellier.render._level_residency import LevelResidency
from cellier.render._spaces import (
    RenderSpaces,
    data_slice_positions,
    geometry_data_region,
    geometry_section_region,
    node_matrix,
)
from cellier.render.scheduling import PlanMode
from cellier.render.visuals._aabb import box_wireframe_positions, make_aabb_line

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from cellier._state import DimsState
    from cellier.data.mesh._mesh_requests import MeshData
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        MeshPickInfo,
        MeshSectionChangedEvent,
        PickWriteChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.scheduling import DesiredSet
    from cellier.transform import BaseTransform
    from cellier.visuals._mesh_memory import BaseMeshVisual

_SIDE_MAP = {"both": "both", "front": "front", "back": "back"}

_PLACEHOLDER_INDICES = np.array([[0, 1, 2]], dtype=np.int32)
_PLACEHOLDER_NORMALS = np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32)

#: The level every mesh has; a single-level mesh has no other.
FINE_LEVEL = 0

#: The depth rule of a 2D section, whatever the appearance's
#: ``depth_compare`` (which applies to the 3D surface).  Every mesh's cut
#: lies in the slice plane, so every fragment ties on depth.  Under "<" the
#: first-drawn fragment wins the tie: a higher ``render_order`` is hidden,
#: and an outline (drawn after every fill) is hidden by any fill.  Under
#: "<=" the last-drawn wins, so the order is ``render_order``, then the
#: order the meshes were added, with outlines over fills.
SECTION_DEPTH_COMPARE = "<="


def _apply_alpha_mode(material: gfx.MeshAbstractMaterial, opacity: float) -> None:
    """Set alpha_mode based on opacity."""
    transparent = float(opacity) < 1.0 - 1e-6
    material.alpha_mode = "blend" if transparent else "solid"


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
    """Flat unlit material for the 2D path.

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
    material.depth_compare = SECTION_DEPTH_COMPARE
    return material


def _build_material_outline(appearance, width: float) -> gfx.LineSegmentMaterial:
    """The section outline: line segments, a fixed width in screen pixels."""
    material = gfx.LineSegmentMaterial(
        color=appearance.color,
        color_mode="uniform",
        thickness=float(width),
        thickness_space="screen",
        opacity=appearance.opacity,
    )
    material.depth_test = appearance.depth_test
    material.depth_write = appearance.depth_write
    material.depth_compare = SECTION_DEPTH_COMPARE
    return material


def _placeholder_geometry(corner: np.ndarray) -> gfx.Geometry:
    """A degenerate triangle at *corner*: what a level that holds nothing has.

    pygfx's scene bounding box counts hidden nodes, so the placeholder sits
    inside the store's extent (M5) and not at the origin, where it would
    stretch a camera fit.
    """
    return gfx.Geometry(
        positions=np.tile(np.asarray(corner, dtype=np.float32), (3, 1)),
        indices=_PLACEHOLDER_INDICES.copy(),
        normals=_PLACEHOLDER_NORMALS.copy(),
    )


def _set_known_bounds(node: gfx.WorldObject, bounds: np.ndarray | None) -> None:
    """Hand pygfx the bounds the read already computed.

    pygfx caches a world object's geometry bounds against the positions
    buffer's revision, and computes them with a pass over every vertex on the
    UI thread (a camera fit, the ambient occlusion radius).  Filling the
    cache skips that pass.  It is private pygfx state: when the attributes
    are not there, pygfx computes the bounds itself, as before.
    """
    if bounds is None or not hasattr(node, "_bounds_geometry_rev"):
        return
    try:
        from pygfx.utils.bounds import Bounds
    except ImportError:
        return
    node._bounds_geometry = Bounds(np.asarray(bounds, dtype=np.float64), None)
    node._bounds_geometry_rev = node.geometry.positions.rev


def _placeholder_segment(corner: np.ndarray) -> gfx.Geometry:
    """A zero-length segment at *corner*: an outline that holds nothing."""
    return gfx.Geometry(positions=np.tile(np.asarray(corner, dtype=np.float32), (2, 1)))


class _LevelNodes:
    """The scene-graph children of one resident level.

    ``mesh_3d`` draws the level in a 3D view.  ``group_2d`` holds what a 2D
    view draws: ``fill`` (the area the section encloses, or whole flattened
    faces when nothing cuts) and ``outline`` (the section's curve, in screen
    pixels, drawn over the fill).  All start, and go back to, a placeholder;
    the level's keys are in ``LevelResidency``, not here.
    """

    def __init__(
        self,
        level: int,
        material_3d: gfx.Material,
        material_2d: gfx.Material,
        material_outline: gfx.Material,
        render_order: float,
        corner: np.ndarray,
    ) -> None:
        self.level = level
        self.mesh_3d = gfx.Mesh(_placeholder_geometry(corner), material_3d)
        self.fill = gfx.Mesh(_placeholder_geometry(corner), material_2d)
        self.outline = gfx.Line(_placeholder_segment(corner), material_outline)
        self.group_2d = gfx.Group()
        self.group_2d.add(self.fill)
        self.group_2d.add(self.outline)
        self.set_render_order(render_order)
        self.mesh_3d.visible = False
        self.group_2d.visible = False
        self.fill.visible = False
        self.outline.visible = False
        #: ``"2d"`` or ``"3d"``: which child holds data; ``None`` for neither.
        self.holds: str | None = None
        self.is_empty: bool = True
        # Maps each drawn 3D face to its face index in the level.  ``None``
        # is the identity (every face of the level is drawn, in order).
        self.original_face_indices: np.ndarray | None = None
        # The level's face behind each fill triangle (-1: a cap triangle)
        # and each outline segment.  ``None`` is the identity.
        self.fill_face_ids: np.ndarray | None = None
        self.outline_face_ids: np.ndarray | None = None

    def set_render_order(self, render_order: float) -> None:
        """The outline is drawn after the fill it borders."""
        self.mesh_3d.render_order = render_order
        self.fill.render_order = render_order
        self.outline.render_order = render_order + 1

    def upload(
        self, data: MeshData | MeshSectionData, is_2d: bool, corner: np.ndarray
    ) -> None:
        """Put *data* on the children for its dimensionality; no array copies."""
        self.release(corner)
        self.holds = "2d" if is_2d else "3d"
        if data.is_empty:
            return
        self.is_empty = False
        if isinstance(data, MeshSectionData):
            if len(data.fill_indices):
                arrays = {
                    "positions": data.fill_positions,
                    "indices": data.fill_indices,
                }
                if data.fill_colors is not None:
                    arrays["colors"] = data.fill_colors
                self.fill.geometry = gfx.Geometry(**arrays)
                _set_known_bounds(self.fill, data.fill_bounds)
                self.fill_face_ids = data.fill_face_ids
                self.fill.visible = True
            if len(data.outline_face_ids):
                arrays = {"positions": data.outline_positions}
                if data.outline_colors is not None:
                    arrays["colors"] = data.outline_colors
                self.outline.geometry = gfx.Geometry(**arrays)
                _set_known_bounds(self.outline, data.outline_bounds)
                self.outline_face_ids = data.outline_face_ids
                self.outline.visible = True
            return
        arrays = {"positions": data.positions, "indices": data.indices}
        if data.normals is not None:
            arrays["normals"] = data.normals
        if data.colors is not None:
            arrays["colors"] = data.colors
        target = self.fill if is_2d else self.mesh_3d
        target.geometry = gfx.Geometry(**arrays)
        _set_known_bounds(target, data.bounds)
        if is_2d:
            self.fill_face_ids = data.original_face_indices
            self.fill.visible = True
        else:
            self.original_face_indices = data.original_face_indices

    def release(self, corner: np.ndarray) -> None:
        """Back to the placeholder; the level's arrays are let go."""
        if not self.is_empty:
            self.mesh_3d.geometry = _placeholder_geometry(corner)
            self.fill.geometry = _placeholder_geometry(corner)
            self.outline.geometry = _placeholder_segment(corner)
        self.holds = None
        self.is_empty = True
        self.fill.visible = False
        self.outline.visible = False
        self.original_face_indices = None
        self.fill_face_ids = None
        self.outline_face_ids = None

    def move_placeholder(self, corner: np.ndarray) -> None:
        """Re-seat an unused child's placeholder (the extent changed)."""
        if self.is_empty:
            self.mesh_3d.geometry = _placeholder_geometry(corner)
            self.fill.geometry = _placeholder_geometry(corner)
            self.outline.geometry = _placeholder_segment(corner)

    def show(self, drawn: bool) -> None:
        """Draw this level, or not; an empty result draws nothing.

        Within ``group_2d`` a part that is switched off or has nothing in it
        stays hidden (set at upload).
        """
        visible = drawn and not self.is_empty
        self.mesh_3d.visible = visible and self.holds == "3d"
        self.group_2d.visible = visible and self.holds == "2d"


class GFXMeshVisual(GeometryClippingMixin):
    """Render-layer visual for a mesh, loaded by the chunk scheduler.

    The scene graph::

        node_3d (gfx.Group)            node_2d (gfx.Group)
          level mesh_3d (gfx.Mesh)       level group_2d (gfx.Group)
          AABB line                        fill (gfx.Mesh)
                                           outline (gfx.Line)
                                         AABB line

    with one set of level children per resident level: the fine level
    always, and a coarse level for a multiscale mesh.  At most one level is
    drawn at a time.

    ``get_node_for_dims`` returns the node for the current dimensionality,
    so the scene manager swaps them on a 2D/3D toggle.  The user's
    visibility is on the two nodes; the display rule (plan L3) sets the
    level children's.

    Parameters
    ----------
    visual_model : MeshVisual or MultiscaleMeshVisual
        Associated model-layer visual.
    render_modes : set[str]
        ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    transform : BaseTransform
        Data-to-world transform. Must cover all data axes.
    axis_extents : Sequence[tuple[float, float]] or None
        The store's level-0 extent per data axis.  It sizes the bounding
        box and places the placeholders, so a camera fit frames the store's
        whole extent whether or not the mesh is loaded.  ``None`` when the
        store is empty.
    coarse_level : int or None
        The store level (0-based) kept resident beside the fine level, for a
        multiscale mesh.  ``None`` (the default) is a single-level mesh.
    """

    #: Loads through the chunk scheduler (``is_chunked_visual``).
    chunked: bool = True

    def __init__(
        self,
        visual_model: BaseMeshVisual,
        render_modes: set[str],
        transform: BaseTransform,
        axis_extents: Sequence[tuple[float, float]] | None = None,
        coarse_level: int | None = None,
    ) -> None:
        invalid = render_modes - {"2d", "3d"}
        if invalid or not render_modes:
            raise ValueError(
                f"render_modes must be a non-empty subset of {{'2d','3d'}}, "
                f"got {render_modes!r}"
            )
        if coarse_level is not None and int(coarse_level) <= FINE_LEVEL:
            raise ValueError(
                f"coarse_level must be a level coarser than {FINE_LEVEL}, "
                f"got {coarse_level}."
            )

        self.visual_model_id: UUID = visual_model.id
        self.render_modes: set[str] = render_modes
        self._transform: BaseTransform | None = transform
        # The systems this visual's geometry is placed with, pushed by
        # the controller.  ``None`` until the scene has a canvas.
        self._spaces: RenderSpaces | None = None
        # Where the collapsed axes sit in **data** units, from the last
        # planned request.  The node matrix reads it (design 3.9).
        self._last_data_positions: dict[int, float] = {}
        self._last_displayed_axes: tuple[int, ...] | None = None
        self._axis_extents: tuple[tuple[float, float], ...] | None = (
            None if axis_extents is None else tuple(axis_extents)
        )

        appearance = visual_model.appearance
        self._material_3d = _build_material_3d(appearance)
        self._material_2d = _build_material_2d(appearance)
        # Plumb the model's pick_write flag into both pygfx materials so the
        # pick buffer records this visual; pygfx materials default to False.
        # How a 2D view draws the mesh (the model's ``section`` config).
        section = visual_model.section
        self._section_mode: str = section.mode
        self._section_outline: bool = section.outline
        self._section_fill: bool = section.fill
        self._material_outline = _build_material_outline(
            appearance, section.outline_width
        )
        self._material_3d.pick_write = visual_model.pick_write
        self._material_2d.pick_write = visual_model.pick_write
        self._material_outline.pick_write = visual_model.pick_write
        # The declared color_mode, honoured verbatim.  It is a caller
        # declaration of where RGB comes from and is never inferred from
        # the data nor written back at commit time (D20).
        self._color_mode: str = appearance.color_mode
        self._render_order = appearance.render_order

        # Level children, built once and keyed by store level.  A
        # single-level mesh has the fine level only; materials are shared
        # across levels.
        resident = [FINE_LEVEL] if coarse_level is None else [FINE_LEVEL, coarse_level]
        self._levels: dict[int, _LevelNodes] = {
            int(level): _LevelNodes(
                int(level),
                self._material_3d,
                self._material_2d,
                self._material_outline,
                self._render_order,
                np.zeros(3),
            )
            for level in resident
        }
        self._residency = LevelResidency(
            len(self._levels),
            upload=self._upload_level,
            release=self._release_level,
            on_change=self._apply_display,
            fine_level=FINE_LEVEL,
            keeps_previous=self._only_clip_changed,
        )
        # Whether the 2D children hold a section (per-vertex colours) or
        # whole faces drawn flat.
        self._section_drawn: bool = False
        self._sync_2d_color_modes()
        # Level-of-detail settings (a multiscale mesh); ``None`` otherwise.
        self._lod = getattr(visual_model, "lod", None)
        # What each canvas last drew, as ``(level, write serial)``.
        self._drawn: dict[UUID, tuple[int, int] | None] = {}

        self._aabb_enabled: bool = visual_model.aabb.enabled
        self._aabb_color: str = visual_model.aabb.color
        self._aabb_line_width: float = visual_model.aabb.line_width
        # False until the store's extent is known in the local frame.
        self._aabb_has_bounds: bool = False

        self.node_3d: gfx.Group | None = None
        self.node_2d: gfx.Group | None = None
        self._aabb_line_3d: gfx.Line | None = None
        self._aabb_line_2d: gfx.Line | None = None
        if "3d" in render_modes:
            self.node_3d = gfx.Group()
            self._aabb_line_3d = make_aabb_line(self._aabb_color, self._aabb_line_width)
            self.node_3d.add(self._aabb_line_3d)
            for nodes in self._levels.values():
                self.node_3d.add(nodes.mesh_3d)
        if "2d" in render_modes:
            self.node_2d = gfx.Group()
            self._aabb_line_2d = make_aabb_line(self._aabb_color, self._aabb_line_width)
            self.node_2d.add(self._aabb_line_2d)
            for nodes in self._levels.values():
                self.node_2d.add(nodes.group_2d)
        for node in (self.node_2d, self.node_3d):
            if node is not None:
                node.visible = bool(appearance.visible)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Resident levels: 1 for a single-level mesh, 2 for a multiscale one."""
        return len(self._levels)

    @property
    def resident_levels(self) -> tuple[int, ...]:
        """The store levels kept resident, finest first."""
        return tuple(sorted(self._levels))

    def drawn_level(self) -> int | None:
        """The store level being drawn now, or ``None`` when nothing is."""
        for nodes in self._levels.values():
            if nodes.mesh_3d.visible or nodes.group_2d.visible:
                return nodes.level
        return None

    @property
    def _is_2d(self) -> bool:
        """Whether the visual is placed in a 2D view now."""
        if self._spaces is not None:
            return len(self._spaces.retained_axes) == 2
        displayed = self._last_displayed_axes
        if displayed is None:
            return "3d" not in self.render_modes
        return len(displayed) == 2

    @property
    def _aabb_line(self) -> gfx.Line | None:
        """The bounding-box line of the current dimensionality."""
        return self._aabb_line_2d if self._is_2d else self._aabb_line_3d

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Return the node for the dimensionality; update its matrix.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            New displayed axes.

        Returns
        -------
        gfx.Group or None
            ``node_2d`` for two displayed axes, ``node_3d`` for three;
            ``None`` when that mode is not in ``render_modes``.
        """
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return self.node_2d if len(displayed_axes) == 2 else self.node_3d

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        return mode in self.render_modes

    def get_node(self, mode: str) -> gfx.Group | None:
        return self.node_2d if mode == "2d" else self.node_3d

    def build_node(
        self, mode, visual_model, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    def rebuild_node_geometry(
        self, mode, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    # ------------------------------------------------------------------
    # Node matrix
    # ------------------------------------------------------------------

    def set_render_spaces(self, spaces: RenderSpaces | None) -> None:
        """Receive the coordinate systems this visual is placed with.

        Pushed by the controller when ``displayed_axes`` changes -- a pure
        reorder included -- and when the first canvas gives the scene a
        rendered system.
        """
        self._spaces = spaces
        if spaces is not None and self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)
        self._refresh_aabb()

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Place the node with the composition of design 3.9.

        ``visual -> data -> world -> rendered``.  A no-op until the
        controller supplies the systems.
        """
        self._last_displayed_axes = displayed_axes
        if self._spaces is None or self._transform is None:
            return
        node = self.node_2d if len(displayed_axes) == 2 else self.node_3d
        if node is None:
            return
        node.local.matrix = node_matrix(
            self._spaces, self._transform, self._collapsed_origin()
        )
        self._apply_clipping_planes()

    def _clip_targets(self):
        """Every material, at the slice the node is drawn at (design 4.6)."""
        constants = self._collapsed_origin() if self._spaces is not None else {}
        yield (
            (
                self._material_3d,
                self._material_2d,
                self._material_outline,
            ),
            constants,
        )

    def _data_region(self, selection):
        """The selection in this visual's data coordinates (design 3.12)."""
        if selection is None or self._spaces is None or self._transform is None:
            return None
        return geometry_data_region(
            selection, self._transform, self._spaces.world, self._spaces.data
        )

    def _collapsed_origin(self) -> dict[int, float]:
        """Where the dropped data axes sit, for the node matrix (design 3.9).

        Taken from the selection where there is one -- the plane's depth
        matters to a 3-D rendered system -- and the origin otherwise.  For a
        transform without a cross-term it reaches no displayed row either way.
        """
        if self._last_data_positions:
            return {
                axis: float(self._last_data_positions.get(axis, 0.0))
                for axis in self._spaces.collapsed_axes
            }
        return dict.fromkeys(self._spaces.collapsed_axes, 0.0)

    # ------------------------------------------------------------------
    # Planning (L1, L4)
    # ------------------------------------------------------------------

    def _build_request(self, dims_state: DimsState, selection=None) -> MeshSliceRequest:
        """The store request of the fine level; other levels differ in level.

        In a 3D view the region carries the whole selection: the scene's
        thickness as the user set it (none is a plane), with a discrete axis
        anchored at the sample the slider selects
        (``geometry_data_region``), and faces are kept whole.

        In a 2D view the constraint along a spatial, continuous axis is not
        a filter: it is the plane the mesh is cut with (X1,
        ``geometry_section_region``).  Cut mode cuts at the slice position,
        whatever the thickness; slab mode cuts at the slab's two faces.
        """
        if selection is None or self._spaces is None or self._transform is None:
            raise RuntimeError(
                "This visual has no region to plan from: either it has not "
                "been placed in a world or the reslicing request carried no "
                "selection."
            )
        section = None
        if len(self._spaces.retained_axes) == 2:
            region, planes = geometry_section_region(
                selection, self._transform, self._spaces.world, self._spaces.data
            )
            if planes is not None:
                slab = self._section_mode == "slab" and planes.high > planes.low
                section = MeshSectionRequest(
                    normal=planes.normal,
                    offsets=(planes.low, planes.high) if slab else (planes.position,),
                    mode="slab" if slab else "cut",
                    outline=self._section_outline,
                    fill=self._section_fill,
                )
        else:
            region = self._data_region(selection)
        self._last_data_positions = data_slice_positions(
            selection.region, self._transform, self._spaces.world
        )
        # The clip line follows the slice (clipping planes design 4.1).  A
        # slab section is flattened, so the read clips it (design 5.2); a
        # cut section and a 3D mesh are clipped by the shader.
        self._slab_section = section is not None and section.mode == "slab"
        clip_planes = self._section_clip_planes(self._begin_request_clipping(), section)
        shared_id = uuid4()
        retained = tuple(self._spaces.retained_axes)
        return MeshSliceRequest(
            slice_request_id=shared_id,
            chunk_request_id=shared_id,
            scale_index=FINE_LEVEL,
            displayed_axes=dims_state.selection.displayed_axes,
            retained_axes=retained,
            region=region,
            # pygfx draws (x, y, z); the data is ascending (z, y, x).
            output_axes=tuple(reversed(retained)),
            section=section,
            clip_planes=clip_planes,
        )

    @staticmethod
    def _only_clip_changed(held: Hashable, planned: Hashable) -> bool:
        """Whether two request keys differ in their clipping planes alone.

        Then the held result is in the right place and only behind, so it
        stays on screen until the new one is uploaded (D31).  Any other
        difference (the slice, the section, the axes) hides it, as before.
        """
        return held[:-1] == planned[:-1] and held[-1] != planned[-1]

    def _wants_cpu_clip(self) -> bool:
        """Only a slab section is flattened; see ``GeometryClippingMixin``."""
        return getattr(self, "_slab_section", False) and super()._wants_cpu_clip()

    def _section_clip_planes(self, planes: tuple, section) -> tuple:
        """The read's planes, reduced to the axes the section kernel sees.

        The kernel works on the two displayed axes and the section axis.
        Any other collapsed axis (time, say) is substituted at the position
        this request draws.
        """
        if not planes or section is None:
            return ()
        section_normal = np.asarray(section.normal)
        out = []
        for normal, offset in planes:
            normal = list(normal)
            for axis in self._spaces.collapsed_axes:
                if section_normal[axis] == 0.0 and normal[axis] != 0.0:
                    offset -= normal[axis] * float(
                        self._last_data_positions.get(axis, 0.0)
                    )
                    normal[axis] = 0.0
            out.append((tuple(normal), float(offset)))
        return tuple(out)

    @staticmethod
    def request_key(request: MeshSliceRequest) -> Hashable:
        """Everything a read depends on except the store's contents (L1).

        ``(retained axes, filter region, section key)``; the level is added
        by ``LevelResidency``.  A transform change, a displayed-axes change
        and a 2D/3D switch each change the key or need no read, so only a
        store change invalidates.  The key does not depend on the camera.

        The section key is the request's plane or planes with the mode and
        the parts.  In cut mode the plane is the slice position, so a
        thickness change on the cut axis is the same key and reads nothing.
        """
        region = request.region
        return (
            tuple(request.retained_axes),
            (int(region.ndim), tuple(region.half_spaces)),
            request.section,
            request.clip_planes,
        )

    def plan(
        self,
        request: ReslicingRequest,
        config: VisualRenderConfig | None = None,
        mode: PlanMode = PlanMode.FULL,
    ) -> list[DesiredSet]:
        """Describe what this view needs: one desired set of whole levels.

        A single-level mesh plans its one level whatever the mode.  A
        multiscale mesh plans the coarse and the fine level together: the
        coarse read is the backstop, so it is issued first and never waits
        for a fine read, and the fine level replaces it on screen when it
        has loaded.  Every level is asked for with the same region.

        Parameters
        ----------
        request : ReslicingRequest
            The view; its camera is not used.
        config : VisualRenderConfig or None
            Unused: a mesh has no LOD or culling settings.
        mode : PlanMode
            ``BACKSTOP_ONLY`` plans the coarse level only (a dims scrub of a
            multiscale mesh).

        Returns
        -------
        list[DesiredSet]
            One set, for this visual's one cache.
        """
        displayed = request.dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        fine = self._build_request(request.dims_state, request.selection)
        key = self.request_key(fine)
        levels = sorted(self._levels)
        requests = {level: fine._replace(scale_index=level) for level in levels}
        coarse = [level for level in levels if level != FINE_LEVEL]
        asked = coarse if (mode is PlanMode.BACKSTOP_ONLY and coarse) else levels
        return [self._residency.desired(dict.fromkeys(levels, key), asked, requests)]

    def residencies(self) -> dict[int, LevelResidency]:
        """``cache_id -> adapter``: one cache per visual."""
        return {self._residency.cache_id: self._residency}

    # ------------------------------------------------------------------
    # Upload and display (L2, L3)
    # ------------------------------------------------------------------

    def _placeholder_corner(self) -> np.ndarray:
        """The extent's minimum corner in the node's frame; else the origin."""
        box = self._local_extent()
        return np.zeros(3) if box is None else box[0]

    def _local_extent(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The store's extent on the retained axes, in upload column order."""
        if self._axis_extents is None or self._spaces is None:
            return None
        output_axes = tuple(reversed(self._spaces.retained_axes))
        if any(axis >= len(self._axis_extents) for axis in output_axes):
            return None
        low, high = np.zeros(3), np.zeros(3)
        for column, axis in enumerate(output_axes):
            low[column], high[column] = self._axis_extents[axis]
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            return None
        return low, high

    def _check_colors(self, data: MeshData | MeshSectionData) -> None:
        """Refuse a colour layout the appearance does not declare.

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
        if self._color_mode == "uniform" or data.is_empty:
            return
        if isinstance(data, MeshSectionData):
            has_colors = data.fill_colors is not None or data.outline_colors is not None
        else:
            has_colors = data.colors is not None
        if not has_colors:
            raise ValueError(
                f"Visual {self.visual_model_id}: appearance declares "
                f"color_mode='{self._color_mode}' but the mesh store "
                "carries no colors. Set color_mode='uniform', or give "
                "the store colors and a colors_layout."
            )
        if data.color_mode != self._color_mode:
            raise ValueError(
                f"Visual {self.visual_model_id}: appearance declares "
                f"color_mode='{self._color_mode}' but the store's "
                f"colors_layout is '{data.color_mode}'. pygfx will "
                "render the mismatched buffer without complaining, so "
                "this is refused rather than corrected."
            )

    def _upload_level(
        self, level: int, request_key: Hashable, data: MeshData | MeshSectionData
    ) -> None:
        """``LevelResidency`` callback: a planned result is resident.

        The commit on the UI thread: a ``gfx.Geometry`` from the arrays the
        read prepared, and bookkeeping.  No array is copied.
        """
        self._check_colors(data)
        is_2d = len(request_key[0]) == 2
        self._levels[level].upload(data, is_2d, self._placeholder_corner())
        if is_2d:
            self._section_drawn = isinstance(data, MeshSectionData)
            self._sync_2d_color_modes()

    def _sync_2d_color_modes(self) -> None:
        """Set where the 2D materials take their colour from.

        A section's colours are per vertex whatever the store's layout (a
        cut point has an interpolated colour, a cap triangle no face), so a
        mesh declared ``"face"`` draws its section per vertex.  Whole faces
        drawn flat keep the declared mode and have no outline.
        """
        if self._color_mode == "uniform":
            fill = outline = "uniform"
        elif self._section_drawn:
            fill = outline = "vertex"
        else:
            fill, outline = self._color_mode, "uniform"
        self._material_2d.color_mode = fill
        self._material_outline.color_mode = outline

    def _release_level(self, level: int) -> None:
        """``LevelResidency`` callback: the level holds nothing any more."""
        self._levels[level].release(self._placeholder_corner())

    def _apply_display(self, prefer_coarse: bool = False) -> tuple[int, int] | None:
        """Apply the display rule (L3): draw one drawable level, or nothing."""
        level = self._residency.level_to_draw(prefer_coarse=prefer_coarse)
        for nodes in self._levels.values():
            nodes.show(nodes.level == level)
        return self._residency.drawn_serial(level)

    def prepare_draw(
        self, canvas_id: UUID, camera_moving: bool, dims_scrubbing: bool
    ) -> bool:
        """Choose what this canvas draws, just before it renders.

        Parameters
        ----------
        canvas_id : UUID
            The canvas about to render.
        camera_moving : bool
            Whether that canvas's camera is in motion.
        dims_scrubbing : bool
            Whether the scene's dims are being scrubbed.

        Returns
        -------
        bool
            Whether what this canvas draws of the visual changed since its
            last frame.  The canvas then discards its accumulation history
            in the same frame.
        """
        drawn = self._apply_display(self._prefers_coarse(camera_moving, dims_scrubbing))
        changed = self._drawn.get(canvas_id) != drawn
        self._drawn[canvas_id] = drawn
        return changed

    @property
    def awaits_data(self) -> bool:
        """Whether the mesh draws nothing only because its read has not landed.

        The display rule hides a level that does not hold the newest plan's
        key.  A canvas may hold its frame for a short time while this is
        true (``RenderManagerConfig.draw_hold_ms``), so a read that lands
        within it does not show as a blank frame.
        """
        nodes = [node for node in (self.node_2d, self.node_3d) if node is not None]
        return self._residency.awaiting and any(node.visible for node in nodes)

    def forget_canvas(self, canvas_id: UUID) -> None:
        """Drop the record of what a removed canvas drew."""
        self._drawn.pop(canvas_id, None)

    def set_lod(self, lod) -> None:
        """Adopt new level-of-detail settings; read by the next frame."""
        self._lod = lod

    def _prefers_coarse(self, camera_moving: bool, dims_scrubbing: bool) -> bool:
        """Whether the coarse level is preferred now; a single level has none.

        During a dims scrub with ``lod.dims_drag_draw == "coarse"`` (D23),
        and, in a 3D view, while this canvas's camera moves with
        ``lod.camera_motion == "coarse"`` (I2).  A 2D view never switches
        level on camera motion.  It is a preference: when the coarse level
        is not drawable the fine one is drawn, and the reverse.
        """
        if self._lod is None or len(self._levels) < 2:
            return False
        if dims_scrubbing and self._lod.dims_drag_draw == "coarse":
            return True
        return bool(
            camera_moving and self._lod.camera_motion == "coarse" and not self._is_2d
        )

    def decode_pick(
        self, hit_object: gfx.WorldObject, pick_info: dict
    ) -> MeshPickInfo | None:
        """Translate a pick on one of this visual's children.

        Parameters
        ----------
        hit_object : gfx.WorldObject
            The picked world object.
        pick_info : dict
            The pygfx pick payload.

        Returns
        -------
        MeshPickInfo or None
            What was drawn there (plan 5.10):

            - a 3D face, or a 2D face lying in the slice plane:
              ``part="face"`` and the face in its level's numbering;
            - the section's fill: ``part="fill"``, no face;
            - the section's outline: ``part="outline"`` and the face the
              plane crosses there.

            ``None`` when the payload names no element.
        """
        from cellier.events._events import MeshPickInfo

        for nodes in self._levels.values():
            level = nodes.level
            if hit_object is nodes.outline:
                vertex = pick_info.get("vertex_index")
                if vertex is None:
                    return None
                face = self._mapped(nodes.outline_face_ids, int(vertex) // 2)
                return MeshPickInfo(face_index=face, part="outline", level=level)
            if hit_object is not nodes.mesh_3d and hit_object is not nodes.fill:
                continue
            index = pick_info.get("face_index")
            if index is None:
                return None
            if hit_object is nodes.mesh_3d:
                face = self._mapped(nodes.original_face_indices, int(index))
                return MeshPickInfo(face_index=face, part="face", level=level)
            face = self._mapped(nodes.fill_face_ids, int(index))
            if face < 0:
                return MeshPickInfo(face_index=None, part="fill", level=level)
            return MeshPickInfo(face_index=face, part="face", level=level)
        return None

    @staticmethod
    def _mapped(id_map: np.ndarray | None, index: int) -> int:
        """*index* through a drawn-element-to-face map; ``None`` is identity."""
        if id_map is None or not (0 <= index < len(id_map)):
            return index
        return int(id_map[index])

    @staticmethod
    def _face_index(nodes: _LevelNodes, face_index: int) -> int:
        return GFXMeshVisual._mapped(nodes.original_face_indices, face_index)

    def face_index_for_pick(self, face_index: int) -> int:
        """Map a pygfx pick face index to the fine level's face index.

        In a sliced view the rendered geometry holds only the faces that
        survived the slab filter, so ``face_index`` is a row in that subset.
        The level's surviving-face list recovers the original index; when
        every face is drawn the index passes through unchanged.

        Parameters
        ----------
        face_index : int
            ``face_index`` from the pygfx pick payload.

        Returns
        -------
        int
            Index into the level's full face array.
        """
        return self._face_index(self._levels[FINE_LEVEL], face_index)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        self._transform = event.transform
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply appearance field changes to live materials.

        ``color_mode`` is applied straight to both materials; nothing at
        upload time overwrites it (D20).
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
        # The outline shares the mesh's appearance (D3).
        if name == "color":
            self._material_outline.color = val
        elif name == "opacity":
            self._material_outline.opacity = val
        elif name == "color_mode":
            self._sync_2d_color_modes()
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
        if name in ("depth_test", "depth_write"):
            for mat in (self._material_3d, self._material_2d, self._material_outline):
                setattr(mat, name, val)
        elif name == "depth_compare":
            # The 2D section keeps SECTION_DEPTH_COMPARE.
            self._material_3d.depth_compare = val
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
            self._render_order = val
            for nodes in self._levels.values():
                nodes.set_render_order(val)

    def on_section_changed(self, event: MeshSectionChangedEvent) -> None:
        """Adopt a change to the model's ``section`` config.

        ``mode``, ``outline`` and ``fill`` are read by the next plan (the
        controller reslices after this); ``outline_width`` is the outline
        material's thickness and applies at once.
        """
        name, value = event.field_name, event.new_value
        if name == "mode":
            self._section_mode = value
        elif name == "outline":
            self._section_outline = bool(value)
        elif name == "fill":
            self._section_fill = bool(value)
        elif name == "outline_width":
            self._material_outline.thickness = float(value)

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """The user's visibility: on the nodes, not on the level children."""
        for node in (self.node_2d, self.node_3d):
            if node is not None:
                node.visible = event.visible

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        self._material_3d.pick_write = event.pick_write
        self._material_2d.pick_write = event.pick_write
        self._material_outline.pick_write = event.pick_write

    # ------------------------------------------------------------------
    # Bounds (M5)
    # ------------------------------------------------------------------

    def set_axis_extents(
        self, axis_extents: Sequence[tuple[float, float]] | None
    ) -> None:
        """Adopt the store's extent after it changed."""
        self._axis_extents = None if axis_extents is None else tuple(axis_extents)
        self._refresh_aabb()

    def _refresh_aabb(self) -> None:
        """Size the bounding boxes and seat the placeholders from the extent.

        From the **store's level-0 extent**, not from the committed faces:
        the box does not blink with the mesh, and a camera fit frames the
        same box whether the mesh is hidden, empty or shown.
        """
        box = self._local_extent()
        corner = np.zeros(3) if box is None else box[0]
        for nodes in self._levels.values():
            nodes.move_placeholder(corner)
        self._aabb_has_bounds = box is not None
        # Only the line of the current dimensionality: the extent is in that
        # node's frame, and the other node is not in the scene.
        line = self._aabb_line
        if line is None:
            return
        if box is not None:
            line.geometry = gfx.Geometry(positions=box_wireframe_positions(*box))
        line.visible = bool(self._aabb_enabled and box is not None)

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Apply AABB param changes to the line nodes."""
        lines = [
            line
            for line in (self._aabb_line_2d, self._aabb_line_3d)
            if line is not None
        ]
        if event.field_name == "enabled":
            self._aabb_enabled = event.new_value
            self._refresh_aabb()
        elif event.field_name == "color":
            self._aabb_color = event.new_value
            for line in lines:
                line.material.color = event.new_value
        elif event.field_name == "line_width":
            self._aabb_line_width = event.new_value
            for line in lines:
                line.material.thickness = event.new_value

    def tick(self) -> None:
        """Called once per rendered frame. No per-frame state to advance."""
