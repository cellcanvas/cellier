# src/cellier/v2/render/visuals/_image_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.image._image_requests import ChunkRequest
from cellier.render._clipping import ClippingPlanesMixin
from cellier.render._level_residency import MEMORY_RASTER_POLICY, LevelResidency
from cellier.render._node_transform import set_node_matrix
from cellier.render._render_planes import RenderPlanesMixin
from cellier.render._spaces import RenderSpaces, node_follows_slice, node_matrix
from cellier.render.scheduling import PlanMode
from cellier.render.shaders._image_volume import IMAGE_VOLUME_MATERIALS
from cellier.render.visuals._brick_route_host import BrickRouteHost
from cellier.render.visuals._pick import memory_image_data_coordinate
from cellier.render.visuals._slicing import (
    axis_selections_from_box,
    image_plane_selection,
)
from cellier.transform import effective_transform
from cellier.visuals._image_memory import effective_transparency_mode
from cellier.visuals._render_plane import PLANE_RENDER_MODE

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        ChannelAppearanceChangedEvent,
        ImageCompositeChangedEvent,
        PickWriteChangedEvent,
        SingleAppearanceChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.scheduling import DesiredSet
    from cellier.transform import (
        AffineTransform,
        BaseTransform,
        RegionSelection,
        WorldCoordinateSystem,
    )
    from cellier.visuals._image_memory import ImageVisual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_colormap(color_map) -> gfx.TextureMap:
    """Convert a cmap Colormap (or name string) to a pygfx TextureMap.

    Appearance mutations driven from the GUI can arrive as a colormap *name*
    string (e.g. the anywidget panel's Unicode trait), and psygnal emits the
    raw assigned value to bus subscribers, so the value reaching this consumer
    may be a ``str`` rather than a ``cmap.Colormap``.  Coerce it here.
    """
    from cmap import Colormap

    if not isinstance(color_map, Colormap):
        color_map = Colormap(color_map)
    return color_map.to_pygfx(N=256)


# Map InMemoryImageAppearance.render_mode -> volume material class.
#
# These are cellier's own subclasses, not pygfx's: the shader behind them
# writes the ``normal`` render target the ambient occlusion pass prefers,
# and fixes the iso branch's depth, which upstream computes as if the
# volume sat at the origin.  See
# ``cellier.render.shaders._image_volume``.
_VOLUME_MATERIALS: dict[str, type] = IMAGE_VOLUME_MATERIALS


def _volume_material_class(render_mode: str) -> type:
    """The material class that draws *render_mode*."""
    return _VOLUME_MATERIALS[render_mode]


def _make_volume_material(appearance, colormap, pick_write: bool):
    """Build the pygfx volume material for an in-memory image appearance.

    Selects the material class from ``appearance.render_mode`` and applies the
    appearance settings.  The ISO material additionally receives
    ``iso_threshold``.

    Parameters
    ----------
    appearance : object
        Provides ``render_mode``, ``clim``, ``interpolation`` and
        ``iso_threshold``.
    colormap : gfx.TextureMap
        Pre-built colormap texture.
    pick_write : bool
        Whether the material writes to the pick buffer.

    Returns
    -------
    gfx.VolumeRayMaterial
        A configured volume material.
    """
    material_cls = _volume_material_class(appearance.render_mode)
    material = material_cls(
        clim=appearance.clim,
        map=colormap,
        interpolation=appearance.interpolation,
        pick_write=pick_write,
    )
    if appearance.render_mode == "iso":
        material.threshold = appearance.iso_threshold
    return material


def _pin_bounding_box(node, box_min: list[float], box_max: list[float]) -> None:
    """Override *node*'s ``get_bounding_box`` to report a fixed local box.

    The in-memory image/volume nodes start with a tiny placeholder texture
    (replaced on the first upload), so the standard pygfx bounding
    box -- derived from the texture size -- would be 1- or 2-voxel and break
    camera fitting before any data is loaded.  Pinning the box to the full data
    footprint in local voxel space (``[-0.5, N-0.5]`` per axis) lets
    ``fit_camera`` frame the data correctly from the start, with no large
    upfront GPU allocation.

    The override is applied to the instance (rather than via a subclass) so the
    node is still built from the current ``gfx.Image`` / ``gfx.Volume`` symbol,
    which tests replace with a mock; a subclass would bind to the real pygfx
    class at import time and bypass that mock.
    """
    box = np.array([box_min, box_max], dtype=np.float32)
    node.get_bounding_box = lambda: box


def _pin_footprint(node, sizes, n_axes: int) -> None:
    """Pin *node*'s local bounding box to a texture of *sizes* voxels.

    *sizes* are in fetch order (the retained data axes, ascending); the
    texture's ``(x, y, z)`` is that order reversed.  A size that does not fit
    the node's *n_axes* (a 2-D store's 3D node) leaves the box as it is.

    The texture a 2D node holds follows the view -- a ``(z, x)`` view's slice
    is ``Z x X``, not the store's last two axes -- so callers pin again from
    the view's retained axes whenever it changes, and from each uploaded
    array.
    """
    if node is None or len(sizes) != n_axes:
        return
    upper = [float(size) - 0.5 for size in reversed(sizes)]
    lower = [-0.5] * n_axes
    if n_axes == 2:
        lower.append(0.0)
        upper.append(0.0)
    _pin_bounding_box(node, lower, upper)


def _box_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return (24, 3) float32 positions for a 3D box wireframe (12 edges x 2 pts)."""
    x0, y0, z0 = float(box_min[0]), float(box_min[1]), float(box_min[2])
    x1, y1, z1 = float(box_max[0]), float(box_max[1]), float(box_max[2])
    return np.array(
        [
            # bottom face
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y1, z0],
            [x0, y0, z0],
            # top face
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y1, z1],
            [x0, y0, z1],
            # verticals
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z0],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _rect_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return (8, 3) float32 positions for a 2D rect wireframe (4 edges x 2 pts)."""
    x0, y0 = float(box_min[0]), float(box_min[1])
    x1, y1 = float(box_max[0]), float(box_max[1])
    return np.array(
        [
            [x0, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y1, 0.0],
            [x1, y1, 0.0],
            [x0, y1, 0.0],
            [x0, y1, 0.0],
            [x0, y0, 0.0],
        ],
        dtype=np.float32,
    )


def _make_aabb_line(
    positions: np.ndarray, color: str, line_width: float = 2.0
) -> gfx.Line:
    """Create a gfx.Line for an AABB wireframe (initially invisible)."""
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineSegmentMaterial(color=color, thickness=line_width),
    )
    line.visible = False
    return line


def _fit_aabb_line(line, sizes, n_axes: int) -> None:
    """Size an outline wireframe to a texture of *sizes* voxels.

    *sizes* are in fetch order, as for :func:`_pin_footprint`.  A hidden line
    still counts toward its group's bounding box, so the wireframe follows
    the view like the pinned box does; the geometry is replaced only when
    the box changes.
    """
    if line is None or len(sizes) != n_axes:
        return
    upper = np.array([float(size) - 0.5 for size in reversed(sizes)])
    lower = np.full(n_axes, -0.5)
    if n_axes == 3:
        positions = _box_wireframe_positions(lower, upper)
    else:
        positions = _rect_wireframe_positions(lower, upper)
    current = np.asarray(line.geometry.positions.data)
    if current.shape == positions.shape and np.array_equal(current, positions):
        return
    line.geometry = gfx.Geometry(positions=positions)


def _plan_from_region(
    selection: RegionSelection,
    transform: AffineTransform,
    world: WorldCoordinateSystem,
    store_shape: tuple[int, ...],
) -> tuple[tuple[int | tuple[int, int], ...], dict[int, float]]:
    """Pull a world-space selection into voxel space and assemble the request.

    Design 3.7 steps 3 and 4.  ``imap_region`` is ``A^T`` on the normals and
    ``d - n . t`` on the offsets: no matrix inverse, no ``select_axes``, and no
    zero-filled probe point.  That last one is the reason to prefer it -- the
    old path invented coordinates for the displayed axes, wrote the slice
    positions into a zero vector and inverted, which is right for a diagonal
    transform and an arbitrary unstated choice for anything else.

    Parameters
    ----------
    selection : RegionSelection
        The region this canvas is showing, in world coordinates.
    transform : BaseTransform
        The visual's ``data -> world`` transform.
    world : WorldCoordinateSystem
        The transform's output system.  Needed to resolve its
        ``broadcast_axes``, which are stored as ids while the arithmetic
        wants indices.
    store_shape : tuple[int, ...]
        The store's shape, one entry per data axis.

    Returns
    -------
    tuple
        ``(axis_selections, collapsed_indices)`` -- the request's per-axis
        selection, and the voxel index of each axis that collapsed, which the
        node matrix needs (design 3.9).
    """
    data_region = transform.imap_region(selection.region, world).simplify()
    box = data_region.bounding_box()
    axis_selections = axis_selections_from_box(box, store_shape)
    collapsed = {
        axis: float(value)
        for axis, value in enumerate(axis_selections)
        if not isinstance(value, tuple)
    }
    return axis_selections, collapsed


# ---------------------------------------------------------------------------
# Whole-slice raster caches (scheduler-only loading design v2, 4.5)
# ---------------------------------------------------------------------------

# Positions in a raster request key.
_SELECTIONS = 0
_COLLAPSED = 1
_COMPOSITE = 2


def raster_request_key(
    axis_selections, collapsed: dict[int, float], composite: bool
) -> tuple:
    """``(axis selections, collapsed indices, composite)`` of one whole slice.

    Everything the read depends on except the store's contents, plus what
    the upload is placed by (D15): the collapsed index of every dropped data
    axis, as a sorted tuple.  The channel index is in the selections.
    """
    return (
        tuple(axis_selections),
        tuple(sorted((int(axis), float(value)) for axis, value in collapsed.items())),
        bool(composite),
    )


def _retained_positions(selections) -> tuple[int, ...]:
    """The data axes a selection keeps (a ``(start, stop)`` range)."""
    return tuple(i for i, value in enumerate(selections) if isinstance(value, tuple))


def raster_keeps_previous(channel_axis: int | None):
    """The ``keeps_previous`` rule of a whole-slice raster cache (D1, D19).

    The previous slice stays drawn while the planned one loads when both
    keep the same data axes (the texture has the same layout).  In single
    mode a channel scrub keeps it (slot 0 follows the slider's channel); in
    composite mode only the same channel does, so a slot given another
    channel never shows the old one (D16).
    """

    def keeps_previous(held: Hashable, planned: Hashable) -> bool:
        held_sel, planned_sel = held[_SELECTIONS], planned[_SELECTIONS]
        if _retained_positions(held_sel) != _retained_positions(planned_sel):
            return False
        if not planned[_COMPOSITE] or channel_axis is None:
            return True
        return held_sel[channel_axis] == planned_sel[channel_axis]

    return keeps_previous


def whole_slice_request(axis_selections) -> ChunkRequest:
    """The store request that reads one whole slice or volume."""
    return ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=tuple(axis_selections),
    )


def placed_collapsed(visual: Any, displayed_axes: tuple[int, ...]) -> dict:
    """The collapsed indices the nodes of *displayed_axes* are placed with.

    D15: in a view whose node follows the slice (``SLICE_SHIFT``) the node
    sits where the shown slice belongs: the indices of the last upload for
    that mode, from its request key.  Until something is uploaded, and in
    every other view, the newest plan's.
    """
    mode = "3d" if len(displayed_axes) == 3 else "2d"
    placed = visual._placed_collapsed.get(mode)
    if placed is None or not node_follows_slice(visual._spaces):
        return visual._collapsed_indices
    return placed


def record_placement(visual: Any, mode: str, request_key: tuple) -> None:
    """Remember where an upload belongs, and move the node there (D15)."""
    visual._placed_collapsed[mode] = dict(request_key[_COLLAPSED])
    displayed = visual._last_displayed_axes
    if (
        node_follows_slice(visual._spaces)
        and displayed is not None
        and (len(displayed) == 3) == (mode == "3d")
    ):
        visual._update_node_matrix(displayed)


# ---------------------------------------------------------------------------
# GFXImageMemoryVisual
# ---------------------------------------------------------------------------


def _release_nothing(level: int) -> None:
    """A slot cache let go of its slice: the visibility rule hides it (D16)."""


class _ImageMemorySlot:
    """One channel's pygfx nodes on an in-memory image visual.

    A slot holds a 2D ``gfx.Image`` and a 3D ``gfx.Volume`` (whichever render
    modes the visual builds), the channel index it currently carries, and
    one whole-slice cache per mode (``residency``, set by the visual), so a
    plan that asks for the same plane again reads nothing, and switching
    2D/3D and back reads nothing either (scheduler-only loading design v2,
    D5).

    Parameters
    ----------
    render_modes : set[str]
        Which nodes to build.
    shape : tuple[int, ...]
        The store's shape.  The trailing two (2D) or three (3D) entries pin
        each node's bounding box to the full data footprint before any data
        arrives, so ``fit_camera`` frames the data from the start; the
        visual pins again from the view's retained axes once it has them.
    pick_write : bool
        Whether the materials write to the pick buffer.
    """

    def __init__(
        self, render_modes: set[str], shape: tuple[int, ...], pick_write: bool
    ) -> None:
        self.key: int | None = None
        self.last_drawn: int = 0
        #: ``mode -> LevelResidency``: the slot's whole-slice caches.
        self.residency: dict[str, LevelResidency] = {}
        self._color_map_source: object = None
        self._colormap: gfx.TextureMap | None = None

        self.node_2d: gfx.Image | None = None
        self.node_3d: gfx.Volume | None = None
        if "2d" in render_modes:
            placeholder = np.zeros((1, 1, 1), dtype=np.float32)
            self.node_2d = gfx.Image(
                gfx.Geometry(grid=gfx.Texture(placeholder, dim=2, format="1xf4")),
                gfx.ImageBasicMaterial(clim=(0.0, 1.0), pick_write=pick_write),
            )
            _pin_footprint(self.node_2d, shape[-2:], 2)
            self.node_2d.visible = False
        if "3d" in render_modes:
            placeholder = np.zeros((2, 2, 2), dtype=np.float32)
            self.node_3d = gfx.Volume(
                gfx.Geometry(grid=gfx.Texture(placeholder, dim=3, format="1xf4")),
                _VOLUME_MATERIALS["mip"](clim=(0.0, 1.0), pick_write=pick_write),
            )
            _pin_footprint(self.node_3d, shape[-3:], 3)
            self.node_3d.visible = False

    def nodes(self) -> tuple:
        """The slot's built nodes."""
        return tuple(n for n in (self.node_2d, self.node_3d) if n is not None)

    def _colormap_for(self, color_map) -> gfx.TextureMap:
        if color_map is not self._color_map_source or self._colormap is None:
            self._colormap = _make_colormap(color_map)
            self._color_map_source = color_map
        return self._colormap

    def apply(
        self,
        shared,
        mode_appearance,
        *,
        pick_write: bool,
        alpha_mode: str,
        planes_overlap: bool,
        render_planes_buffer: gfx.Buffer | None = None,
    ) -> None:
        """Draw with *shared* plus the mode's own appearance (design 3.3).

        Parameters
        ----------
        shared : BaseImageAppearance
            The visual's shared appearance.
        mode_appearance : single or channel appearance
            ``single`` in single mode, ``channels[k]`` in composite mode.
        pick_write : bool
            Whether the materials write to the pick buffer.
        alpha_mode : str
            The effective transparency mode.
        planes_overlap : bool
            True when more than one channel is drawn: every 2D plane sits at
            the same depth, so depth testing and writing are turned off on the
            2D materials or the first plane would hide the rest.  The 3D
            volumes stop writing depth for the same reason.
        render_planes_buffer : gfx.Buffer or None
            The visual's render planes, for a material built for the
            ``"plane"`` render mode.
        """
        colormap = self._colormap_for(mode_appearance.color_map)
        for node in self.nodes():
            node.render_order = shared.render_order
        if self.node_2d is not None:
            material = self.node_2d.material
            material.clim = mode_appearance.clim
            material.map = colormap
            material.interpolation = shared.interpolation
            material.opacity = mode_appearance.opacity
            material.alpha_mode = alpha_mode
            material.pick_write = pick_write
            material.depth_compare = shared.depth_compare
            material.depth_test = shared.depth_test and not planes_overlap
            material.depth_write = shared.depth_write and not planes_overlap
        if self.node_3d is not None:
            material_cls = _volume_material_class(mode_appearance.render_mode)
            material = self.node_3d.material
            if not isinstance(material, material_cls) or type(material) is not (
                material_cls
            ):
                extra = {}
                if mode_appearance.render_mode == PLANE_RENDER_MODE:
                    extra["render_planes_buffer"] = render_planes_buffer
                material = material_cls(
                    clim=mode_appearance.clim,
                    map=colormap,
                    interpolation=shared.interpolation,
                    pick_write=pick_write,
                    **extra,
                )
                self.node_3d.material = material
            material.clim = mode_appearance.clim
            material.map = colormap
            material.interpolation = shared.interpolation
            material.opacity = mode_appearance.opacity
            material.alpha_mode = alpha_mode
            material.pick_write = pick_write
            material.depth_test = shared.depth_test
            # Overlapping channel volumes: the first to draw would write its
            # hit depth and clip the rest into speckle.  Testing stays on so
            # opaque objects still occlude the image.
            material.depth_write = shared.depth_write and not planes_overlap
            material.depth_compare = shared.depth_compare
            if mode_appearance.render_mode == "iso":
                material.threshold = mode_appearance.iso_threshold


class _GFXImageMemoryCore(RenderPlanesMixin, ClippingPlanesMixin):
    """:class:`GFXImageMemoryVisual` without the brick route: whole-slice textures.

    Render-layer visual for one ``ImageVisual`` backed by ``ImageMemoryStore``.

    Draws the image single-channel or composited from a pool of
    :class:`_ImageMemorySlot` (unified image design 3.8).  The pool holds one
    slot when the visual has no ``channel_axis`` and ``max_channels`` slots
    when it does, keyed by channel index: single mode on index ``k`` reuses
    ``k``'s slot, and a full pool reassigns the least recently drawn slot the
    current mode does not need.

    Loads through the chunk scheduler (scheduler-only loading design v2,
    4.5): every slot has one whole-slice cache per mode, and each plan wants
    one full slice or volume per drawn channel.  A key a cache already
    holds reads nothing.  In single mode slot 0 always draws the slider's
    channel (D19), so a channel scrub keeps the previous slice drawn while
    the next one loads.  The whole visual shares one node matrix, one AABB
    wireframe per mode and one visibility switch.

    Parameters
    ----------
    visual_model : ImageVisual
        The model.  Held for the life of the visual: which channels are drawn,
        and with what appearance, is read off it at plan time.
    data_store : ImageMemoryStore
        The backing data store.
    render_modes : set[str]
        Which nodes to build: ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    transform : BaseTransform or None
        The ``data -> world`` transform.
    """

    #: Loads through the chunk scheduler (``is_chunked_visual``).
    chunked: bool = True
    #: Applies the image slicing rule (design 3.2) itself, so the scene
    #: manager's data-coverage pre-check does not skip it.
    decides_empty_slices: bool = True

    def __init__(
        self,
        visual_model: ImageVisual,
        data_store: ImageMemoryStore,
        render_modes: set[str],
        transform: BaseTransform | None = None,
    ) -> None:
        invalid = render_modes - {"2d", "3d"}
        if invalid or not render_modes:
            raise ValueError(
                f"render_modes must be a non-empty subset of {{'2d', '3d'}}, "
                f"got {render_modes!r}"
            )

        self.visual_model_id: UUID = visual_model.id
        self.render_modes: set[str] = render_modes
        self._data_store = data_store
        self._visual_model = visual_model
        self._channel_axis: int | None = visual_model.channel_axis

        # The data -> world transform and the systems the geometry is placed
        # with; the controller supplies both.
        self._transform: BaseTransform | None = effective_transform(transform)
        self._spaces: RenderSpaces | None = None
        self._last_displayed_axes: tuple[int, ...] | None = None
        # The collapsed voxel index per dropped data axis from the last plan
        # (design 3.9).
        self._collapsed_indices: dict[int, float] = {}
        # True while a sliced axis selects no sample (design 3.2).
        self._slice_empty: bool = False
        # mode -> the collapsed indices of the last upload (D15).
        self._placed_collapsed: dict[str, dict[int, float]] = {}

        self._pick_write: bool = visual_model.pick_write
        self._aabb_enabled: bool = visual_model.aabb.enabled
        self._aabb_color: str = visual_model.aabb.color
        self._aabb_line_width: float = visual_model.aabb.line_width
        self._data_ready_2d: bool = False
        self._data_ready_3d: bool = False

        shape = tuple(data_store.shape)
        n_slots = 1 if self._channel_axis is None else visual_model.max_channels
        self._slots = [
            _ImageMemorySlot(render_modes, shape, self._pick_write)
            for _ in range(n_slots)
        ]
        self._slot_for_key: dict[int, int] = {}
        # Channel index -> slot index for the channels the current plan draws.
        self._drawn: dict[int, int] = {}
        self._clock = 0
        keeps_previous = raster_keeps_previous(self._channel_axis)
        for index, slot in enumerate(self._slots):
            slot.residency = {
                mode: LevelResidency(
                    1,
                    upload=self._slot_uploader(index, mode),
                    release=_release_nothing,
                    on_change=self._apply_slot_visibility,
                    keeps_previous=keeps_previous,
                    policy=MEMORY_RASTER_POLICY,
                    keeps_through_invalidation=True,
                )
                for mode in sorted(render_modes)
            }

        self.node_2d: gfx.Group | None = None
        self.node_3d: gfx.Group | None = None
        self._aabb_line_2d: gfx.Line | None = None
        self._aabb_line_3d: gfx.Line | None = None
        if "2d" in render_modes:
            self.node_2d = gfx.Group()
            for slot in self._slots:
                self.node_2d.add(slot.node_2d)
            self._aabb_line_2d = _make_aabb_line(
                _rect_wireframe_positions(np.zeros(2), np.ones(2)),
                self._aabb_color,
                self._aabb_line_width,
            )
            self.node_2d.add(self._aabb_line_2d)
        if "3d" in render_modes:
            self.node_3d = gfx.Group()
            for slot in self._slots:
                self.node_3d.add(slot.node_3d)
            self._aabb_line_3d = _make_aabb_line(
                _box_wireframe_positions(np.zeros(3), np.ones(3)),
                self._aabb_color,
                self._aabb_line_width,
            )
            self.node_3d.add(self._aabb_line_3d)

        for node in (self.node_2d, self.node_3d):
            if node is not None:
                node.render_order = visual_model.appearance.render_order
                node.visible = visual_model.appearance.visible

        # Until the first plan, draw what the model says will be drawn, so
        # a scene fitted before any reslice has something to frame.
        self._drawn = dict(
            zip(
                self._initial_keys(),
                self._assign_slots(self._initial_keys()),
                strict=False,
            )
        )
        self._apply_materials()
        self._apply_slot_visibility()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 -- single-resolution in-memory store."""
        return 1

    @property
    def slots(self) -> tuple[_ImageMemorySlot, ...]:
        """The slot pool."""
        return tuple(self._slots)

    @property
    def _inner_node_2d(self) -> gfx.Image | None:
        """The first slot's 2D node: the one a plain image draws with."""
        return self._slots[0].node_2d

    @property
    def _inner_node_3d(self) -> gfx.Volume | None:
        """The first slot's 3D node: the one a plain image draws with."""
        return self._slots[0].node_3d

    def close(self) -> None:
        """Release the slots and nodes.  The visual is unusable afterwards."""
        for group in (self.node_2d, self.node_3d):
            if group is not None:
                group.clear()
        self._slots = []
        self._slot_for_key = {}
        self._drawn = {}
        self._visual_model = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _channel_size(self) -> int | None:
        if self._channel_axis is None:
            return None
        return int(self._data_store.shape[self._channel_axis])

    def _composite(self) -> bool:
        return bool(self._visual_model.composite) and self._channel_axis is not None

    def _initial_keys(self) -> list[int]:
        if self._composite():
            return list(self._visual_model.drawn_channels(self._channel_size()))
        return [0]

    def _assign_slots(self, keys: list[int]) -> list[int]:
        """Give every key in *keys* a slot, reusing a key's own slot first."""
        self._clock += 1
        wanted = set(keys)
        taken: set[int] = set()
        assigned: list[int] = []
        for key in keys:
            index = self._slot_for_key.get(key)
            if index is None or index in taken:
                free = [
                    i
                    for i, slot in enumerate(self._slots)
                    if slot.key is None and i not in taken
                ]
                if free:
                    index = free[0]
                else:
                    index = min(
                        (
                            i
                            for i, slot in enumerate(self._slots)
                            if i not in taken and slot.key not in wanted
                        ),
                        key=lambda i: self._slots[i].last_drawn,
                    )
                    self._slot_for_key.pop(self._slots[index].key, None)
                slot = self._slots[index]
                slot.key = key
                self._slot_for_key[key] = index
            taken.add(index)
            self._slots[index].last_drawn = self._clock
            assigned.append(index)
        if self._clipping_planes:
            # A slot given another channel is cut at that channel's place.
            self._apply_clipping_planes()
        return assigned

    def _mode_appearance(self, key: int):
        model = self._visual_model
        if self._composite():
            return model.channels.get(key)
        return model.single

    def _apply_materials(self) -> None:
        """Re-apply every drawn slot's appearance from the model."""
        if self._visual_model is None:
            return
        model = self._visual_model
        alpha_mode = effective_transparency_mode(model)
        overlap = len(self._drawn) > 1
        for key, index in self._drawn.items():
            mode_appearance = self._mode_appearance(key)
            if mode_appearance is None:
                continue
            self._slots[index].apply(
                model.appearance,
                mode_appearance,
                pick_write=self._pick_write,
                alpha_mode=alpha_mode,
                planes_overlap=overlap,
                render_planes_buffer=self.render_planes_buffer,
            )
        # A render-mode change gives a slot a new volume material, which
        # starts with no planes (clipping planes design 4.6).
        self._apply_clipping_planes()

    def _apply_slot_visibility(self) -> None:
        """Show a slot's node for a mode only when it has a slice to draw (D16).

        The slot is drawn by the newest plan, the slice is not empty, and
        the slot's cache for that mode holds the planned slice or keeps an
        earlier one on screen while it loads.  Each cache calls this when it
        changes.
        """
        drawn = set(self._drawn.values())
        for index, slot in enumerate(self._slots):
            base = index in drawn and not self._slice_empty
            for mode, node in (("2d", slot.node_2d), ("3d", slot.node_3d)):
                if node is None:
                    continue
                residency = slot.residency.get(mode)
                node.visible = (
                    base and residency is not None and residency.is_drawable(0)
                )

    def _prune_undrawn(self) -> None:
        """Stop drawing channels the model no longer draws, without a plan."""
        if not self._composite():
            return
        drawn = set(self._visual_model.drawn_channels(self._channel_size()))
        self._drawn = {k: i for k, i in self._drawn.items() if k in drawn}

    # ------------------------------------------------------------------
    # Node matrix update (lazy, displayed-axes-aware)
    # ------------------------------------------------------------------

    def set_render_spaces(self, spaces: RenderSpaces | None) -> None:
        """Receive the coordinate systems this visual is placed with."""
        self._spaces = spaces
        if spaces is not None and self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)
        # Which planes are drawn follows the displayed axes.
        self._apply_render_planes()

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Place both groups with the composition of design 3.9.

        One matrix for every slot: the slots differ in which channel they
        carry, not in where they sit.  In a ``SLICE_SHIFT`` view the node
        follows the shown slice, not the newest plan (D15).
        """
        self._last_displayed_axes = displayed_axes
        if self._spaces is None or self._transform is None:
            return
        # The view decides which data axes the textures hold.
        retained = [self._data_store.shape[axis] for axis in self._spaces.retained_axes]
        for slot in self._slots:
            _pin_footprint(slot.node_2d, retained, 2)
            _pin_footprint(slot.node_3d, retained, 3)
        _fit_aabb_line(self._aabb_line_2d, retained, 2)
        _fit_aabb_line(self._aabb_line_3d, retained, 3)
        collapsed = placed_collapsed(self, displayed_axes)
        m = node_matrix(
            self._spaces,
            self._transform,
            {
                axis: float(collapsed.get(axis, 0.0))
                for axis in self._spaces.collapsed_axes
            },
        )
        if self.node_3d is not None:
            set_node_matrix(self.node_3d, m)
        if self.node_2d is not None:
            set_node_matrix(self.node_2d, m)
        # The clip line stays at the newest plan (D15).
        self._apply_clipping_planes()

    def _outline_shape(self):
        """The store's shape: the box a plane's outline is cut by."""
        return tuple(self._data_store.shape)

    def _clip_targets(self):
        """One group per channel slot (clipping planes design 4.1, 4.6).

        A slot draws its channel at that channel's index, so the channel
        axis is pinned per slot: a plane with a channel component cuts each
        channel at its own place.
        """
        if self._spaces is None:
            base: dict[int, float] = {}
        else:
            base = {
                axis: float(self._collapsed_indices.get(axis, 0.0))
                for axis in self._spaces.collapsed_axes
            }
        for slot in self._slots:
            constants = base
            if self._channel_axis in base and slot.key is not None:
                constants = {**base, self._channel_axis: float(slot.key)}
            yield (
                (
                    getattr(slot.node_2d, "material", None),
                    getattr(slot.node_3d, "material", None),
                ),
                constants,
            )

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Return the group for *displayed_axes*, with its matrix up to date."""
        node = self.node_3d if len(displayed_axes) == 3 else self.node_2d
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return node

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        return self.node_3d is not None if mode == "3d" else self.node_2d is not None

    def get_node(self, mode: str) -> gfx.Group | None:
        return self.node_3d if mode == "3d" else self.node_2d

    def build_node(
        self, mode, visual_model, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    def rebuild_node_geometry(
        self, mode, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    # ------------------------------------------------------------------
    # Planning -- build ChunkRequests (synchronous, < 1 ms)
    # ------------------------------------------------------------------

    def _plan(
        self, selection: RegionSelection | None
    ) -> list[tuple[int, tuple[int | tuple[int, int], ...]]]:
        """``(channel index, per-axis selection)`` for every channel to draw.

        The image slicing rule runs first (design 3.2); in composite mode the
        channel axis is exempt, and each drawn channel's selection substitutes
        its own index on that axis (design 3.3).  Empty when a sliced axis
        selects no sample, or a composite has no drawn channel.
        """
        shape = tuple(self._data_store.shape)
        if selection is None or self._spaces is None:
            raise RuntimeError(
                "This visual has no region to plan from: either it has not "
                "been placed in a world or the reslicing request carried no "
                "selection."
            )
        composite = self._composite()
        planned = image_plane_selection(
            selection,
            self._transform,
            self._spaces,
            shape,
            exempt_data_axes=(self._channel_axis,) if composite else (),
        )
        self._slice_empty = planned is None
        if planned is None:
            return []
        base, collapsed = _plan_from_region(
            planned, self._transform, self._spaces.world, shape
        )
        self._collapsed_indices = collapsed
        # The clip line follows the slice (clipping planes design 4.1).
        self._apply_clipping_planes()
        if not composite:
            key = 0 if self._channel_axis is None else int(base[self._channel_axis])
            return [(key, base)]
        channel_axis = self._channel_axis
        return [
            (
                key,
                tuple(
                    key if axis == channel_axis else value
                    for axis, value in enumerate(base)
                ),
            )
            for key in self._visual_model.drawn_channels(self._channel_size())
        ]

    def _single_slot(self, key: int) -> list[int]:
        """D19: single mode draws from slot 0; its key follows the channel.

        The request key carries the channel, so a channel scrub is a key
        change on one cache (kept on screen while it loads, D1), not a move
        to another slot that has to load before anything is drawn.
        """
        self._clock += 1
        for other in self._slots[1:]:
            other.key = None
        slot = self._slots[0]
        slot.key = key
        slot.last_drawn = self._clock
        self._slot_for_key = {key: 0}
        if self._clipping_planes:
            self._apply_clipping_planes()
        return [0]

    def request_key(self, request: ChunkRequest) -> tuple:
        """The cache key of *request*, as the newest plan would build it."""
        return raster_request_key(
            request.axis_selections, self._collapsed_indices, self._composite()
        )

    def plan(
        self,
        request: ReslicingRequest,
        config: VisualRenderConfig | None = None,
        mode: PlanMode = PlanMode.FULL,
    ) -> list[DesiredSet]:
        """One desired set per drawn channel: its slot's cache for the view.

        A whole slice has no backstop, so every plan mode plans it (design
        v2 4.3).  Empty when the slice misses the data or a composite draws
        no channel: the coordinator then retires the caches.
        """
        displayed = request.dims_state.selection.displayed_axes
        view_mode = "3d" if len(displayed) == 3 else "2d"
        wanted = self._plan(request.selection)
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        if not wanted:
            self._apply_slot_visibility()
            return []
        keys = [key for key, _ in wanted]
        composite = self._composite()
        indices = self._assign_slots(keys) if composite else self._single_slot(keys[0])
        self._drawn = dict(zip(keys, indices, strict=True))
        self._apply_materials()
        desired: list[DesiredSet] = []
        for (_key, selections), index in zip(wanted, indices, strict=True):
            residency = self._slots[index].residency.get(view_mode)
            if residency is None:
                continue
            key = raster_request_key(selections, self._collapsed_indices, composite)
            desired.append(
                residency.desired({0: key}, {0}, {0: whole_slice_request(selections)})
            )
        self._apply_slot_visibility()
        return desired

    def residencies(self) -> dict[int, LevelResidency]:
        """``cache_id -> adapter``: every slot's caches, both modes."""
        return {
            residency.cache_id: residency
            for slot in self._slots
            for residency in slot.residency.values()
        }

    @property
    def awaits_data(self) -> bool:
        """Whether a drawn slot shows nothing only because its read is pending.

        True before the first slice of the view's mode lands, and after a
        change no kept slice covers (a new layout, a composite slot given
        another channel).
        """
        displayed = self._last_displayed_axes
        if displayed is None or self._slice_empty:
            return False
        mode = "3d" if len(displayed) == 3 else "2d"
        group = self.node_3d if mode == "3d" else self.node_2d
        if group is None or not group.visible:
            return False
        drawn = set(self._drawn.values())
        return any(
            slot.residency[mode].awaiting
            for index, slot in enumerate(self._slots)
            if index in drawn and mode in slot.residency
        )

    # ------------------------------------------------------------------
    # Upload -- the slot caches' callbacks
    # ------------------------------------------------------------------

    def _slot_uploader(self, index: int, mode: str):
        def upload(level: int, request_key: tuple, data: np.ndarray) -> None:
            self.upload_slice(index, mode, request_key, data)

        return upload

    def upload_slice(
        self, index: int, mode: str, request_key: tuple, data: np.ndarray
    ) -> None:
        """Put one slice (2D) or volume (3D) on slot *index*'s node.

        Called by the slot's cache from ``rebuild_draw``; a test may call it
        directly with ``request_key(request)``.  The node is placed by the
        key's collapsed indices (D15).
        """
        if index >= len(self._slots):
            return
        slot = self._slots[index]
        record_placement(self, mode, request_key)
        if mode == "3d":
            if slot.node_3d is None:
                return
            # No transpose: pygfx maps a numpy texture's *last* axis to
            # texture-x, so (D, H, W) = (z, y, x) already lands as local
            # (x=W, y=H, z=D).
            texture = gfx.Texture(np.ascontiguousarray(data), dim=3, format="1xf4")
            slot.node_3d.geometry = gfx.Geometry(grid=texture)
            _pin_footprint(slot.node_3d, data.shape, 3)
            _fit_aabb_line(self._aabb_line_3d, data.shape, 3)
            if not self._data_ready_3d and self._aabb_line_3d is not None:
                self._data_ready_3d = True
                self._aabb_line_3d.visible = self._aabb_enabled
        else:
            if slot.node_2d is None:
                return
            texture = gfx.Texture(
                np.ascontiguousarray(data[:, :, np.newaxis]), dim=2, format="1xf4"
            )
            slot.node_2d.geometry = gfx.Geometry(grid=texture)
            _pin_footprint(slot.node_2d, data.shape, 2)
            _fit_aabb_line(self._aabb_line_2d, data.shape, 2)
            if not self._data_ready_2d and self._aabb_line_2d is not None:
                self._data_ready_2d = True
                self._aabb_line_2d.visible = self._aabb_enabled

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update the stored transform and the node matrix."""
        self._transform = effective_transform(event.transform)
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """A shared field changed: restyle every drawn slot."""
        if event.field_name == "render_order":
            for group in (self.node_2d, self.node_3d):
                if group is not None:
                    group.render_order = event.new_value
        self._apply_materials()

    def on_single_appearance_changed(self, event: SingleAppearanceChangedEvent) -> None:
        """A single-mode field changed: restyle when single mode is drawn."""
        if not self._composite():
            self._apply_materials()

    def on_channel_appearance_changed(
        self, event: ChannelAppearanceChangedEvent
    ) -> None:
        """A channel field changed: restyle, and hide a channel switched off."""
        if not self._composite():
            return
        if event.field_name == "visible" and not event.new_value:
            self._prune_undrawn()
            self._apply_slot_visibility()
        self._apply_materials()

    def on_image_composite_changed(self, event: ImageCompositeChangedEvent) -> None:
        """The mode switched: restyle now; the controller's reslice follows."""
        self._prune_undrawn()
        self._apply_materials()
        self._apply_slot_visibility()

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle the whole visual."""
        for group in (self.node_2d, self.node_3d):
            if group is not None:
                group.visible = event.visible

    def pick_collapsed_indices(self) -> dict[int, int] | None:
        """The level-0 planes this visual last drew, per collapsed data axis.

        In composite mode the channel axis is left out: every drawn channel
        sits at its own index, so the plan names none of them.

        Returns
        -------
        dict[int, int] or None
            Data axis to voxel index, for collapsed axes only.
        """
        if not self._collapsed_indices:
            return None
        skip = self._channel_axis if self._composite() else None
        return {
            axis: int(value)
            for axis, value in self._collapsed_indices.items()
            if axis != skip
        }

    def pick_channel_index(self, hit_object) -> int | None:
        """The channel index of the slot whose node *hit_object* is, if any."""
        for slot in self._slots:
            if hit_object in slot.nodes():
                return slot.key
        return None

    def drawn_channel_indices(self) -> tuple[int, ...]:
        """The channel indices the last plan draws, ascending."""
        return tuple(sorted(self._drawn))

    def pick_data_coordinate(
        self, hit_object, pick_info: dict
    ) -> tuple[float, ...] | None:
        """Level-0 data coordinate of a pick on this visual (displayed axes).

        Every slot's texture holds a whole slice or volume, so pygfx's
        ``index`` is already the data index; see
        :func:`cellier.render.visuals._pick.memory_image_data_coordinate`.
        """
        ndim = 3 if isinstance(hit_object, gfx.Volume) else 2
        return memory_image_data_coordinate(pick_info, ndim)

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        """Update pick_write on every slot material."""
        self._pick_write = event.pick_write
        for slot in self._slots:
            for node in slot.nodes():
                node.material.pick_write = event.pick_write

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Apply an AABB parameter change to the visual's one wireframe."""
        if event.field_name == "enabled":
            self._aabb_enabled = event.new_value
            if self._aabb_line_2d is not None:
                self._aabb_line_2d.visible = event.new_value and self._data_ready_2d
            if self._aabb_line_3d is not None:
                self._aabb_line_3d.visible = event.new_value and self._data_ready_3d
        elif event.field_name == "color":
            self._aabb_color = event.new_value
            for line in (self._aabb_line_2d, self._aabb_line_3d):
                if line is not None:
                    line.material.color = event.new_value
        elif event.field_name == "line_width":
            self._aabb_line_width = event.new_value
            for line in (self._aabb_line_2d, self._aabb_line_3d):
                if line is not None:
                    line.material.thickness = event.new_value

    def tick(self) -> None:
        """No per-frame state to advance."""


class GFXImageMemoryVisual(BrickRouteHost, _GFXImageMemoryCore):
    """Render-layer visual for one ``ImageVisual`` backed by ``ImageMemoryStore``.

    The planar path and 3D views (:class:`_GFXImageMemoryCore`), and the brick
    route for a tilted 2D view (oblique 2D design v3, 5.4,
    :class:`~cellier.render.visuals._brick_route_host.BrickRouteHost`).
    """

    def _build_inner(self, model, axes):
        from cellier.render.visuals._image import GFXMultiscaleImageVisual

        return GFXMultiscaleImageVisual(
            model, list(self._brick_source.level_shapes), {"3d"}, tuple(axes)
        )

    def on_single_appearance_changed(self, event) -> None:
        super().on_single_appearance_changed(event)
        self._forward("on_single_appearance_changed", event)

    def on_channel_appearance_changed(self, event) -> None:
        super().on_channel_appearance_changed(event)
        self._forward("on_channel_appearance_changed", event)

    def on_image_composite_changed(self, event) -> None:
        super().on_image_composite_changed(event)
        self._forward("on_image_composite_changed", event)
