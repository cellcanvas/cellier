"""CellierController is the coordinator between the models and the rendered views."""

from __future__ import annotations

import asyncio
import contextvars
import difflib
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, NamedTuple
from uuid import UUID, uuid4

import numpy as np

from cellier.data._axes import (
    data_axes_from_world,
    default_data_to_world,
    install_level_transforms,
    level_coordinate_systems,
    scale_and_translation_transform,
    store_level_transforms,
)
from cellier.events import (
    AABBChangedEvent,
    AABBUpdateEvent,
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    BackgroundChangedEvent,
    BackgroundUpdateEvent,
    CameraChangedEvent,
    CanvasAddedEvent,
    CanvasConnectedEvent,
    CanvasSizeChangedEvent,
    ChannelAppearanceChangedEvent,
    ChannelAppearanceUpdateEvent,
    DataStoreContentsChangedEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    DimsUpdateEvent,
    EventBus,
    FrameRenderedEvent,
    ImageCompositeChangedEvent,
    ImageCompositeUpdateEvent,
    OverlayChangedEvent,
    OverlayUpdateEvent,
    PickWriteChangedEvent,
    RenderConfigChangedEvent,
    RenderConfigUpdateEvent,
    ResliceCompletedEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    SingleAppearanceChangedEvent,
    SingleAppearanceUpdateEvent,
    SliderAxesChangedEvent,
    SliderOverrideUpdateEvent,
    SubscriptionHandle,
    SubscriptionSpec,
    TrailChangedEvent,
    TrailUpdateEvent,
    TransformChangedEvent,
    VisualAddedEvent,
    VisualRemovedEvent,
    VisualRenderChangedEvent,
    VisualRenderUpdateEvent,
    VisualVisibilityChangedEvent,
)
from cellier.events._events import (
    PICK_EVENT_TYPES,
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    GraphEdgePickInfo,
    GraphNodePickInfo,
    GraphPickEvent,
    ImagePickEvent,
    ImagePickInfo,
    LabelsPickEvent,
    LabelsPickInfo,
    LinesPickEvent,
    LinesPickInfo,
    MeshPickEvent,
    MeshPickInfo,
    PointsPickEvent,
    PointsPickInfo,
    _CanvasRawPointerEvent,
)
from cellier.logging import _CAMERA_LOGGER, _SOURCE_ID_LOGGER
from cellier.render._capture import capture_scene
from cellier.render._config import RenderManagerConfig
from cellier.render._scene_config import VisualRenderConfig
from cellier.render._spaces import (
    RenderSpaces,
    axis_correspondence,
    build_render_spaces,
)
from cellier.render._visual_lut import (
    KIND_LABEL,
    KIND_LABEL_ALL,
    KIND_WHOLE_OBJECT,
)
from cellier.render.render_manager import RenderManager
from cellier.render.visuals._canvas_overlay import GFXCenteredAxes2D
from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
from cellier.render.visuals._image import GFXMultiscaleImageVisual
from cellier.render.visuals._image_memory import GFXImageMemoryVisual
from cellier.render.visuals._label_memory import GFXLabelMemoryVisual
from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual
from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
from cellier.render.visuals._points_memory import GFXPointsMemoryVisual
from cellier.render.visuals._scene_overlay import GFXSceneBoundingBox
from cellier.scene._background import BackgroundAppearance
from cellier.scene._bounds import scene_world_bounds
from cellier.scene.cameras import (
    CameraType,
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)
from cellier.scene.canvas import Canvas
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    WorldAxesLike,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene
from cellier.transform import (
    AffineTransform,
    CoordinateSystemType,
    NonInvertibleTransformError,
    RegionSelection,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
)
from cellier.viewer_model import DataManager, ViewerModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.visuals._base_visual import BaseVisual, VisualOutline
    from cellier.visuals._label_memory import OutlineMode
from cellier.visuals._canvas_overlay import CanvasOverlay, CenteredAxes2D
from cellier.visuals._graph_memory import (
    GraphAppearance,
    GraphVisual,
    TrailConfig,
)
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)
from cellier.visuals._image_memory import (
    BaseImageVisual,
    ImageVisual,
    InMemoryImageAppearance,
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
)
from cellier.visuals._label_memory import (
    BaseLabelsAppearance,
    BaseLabelsVisual,
    LabelMemoryVisual,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    MultiscaleLabelVisual,
)
from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
from cellier.visuals._mesh_memory import (
    MeshAppearance,
    MeshPhongAppearance,
    MeshVisual,
)
from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual
from cellier.visuals._scene_overlay import SceneBoundingBox, SceneOverlay

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

    from psygnal import EmissionInfo
    from PySide6.QtWidgets import QWidget

    from cellier._state import CameraState, DimsState
    from cellier.data._base_data_store import BaseDataStore
    from cellier.data._changes import StoreChange
    from cellier.data.graph._graph_memory_store import GraphMemoryStore
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.data.label._label_memory_store import LabelMemoryStore
    from cellier.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.data.points._points_memory_store import PointsMemoryStore
    from cellier.gui._protocol import WidgetView
    from cellier.render._config import RenderManagerConfig
    from cellier.render.canvas_view import CanvasView
    from cellier.render.visuals._canvas_overlay import GFXCanvasOverlay
    from cellier.render.visuals._scene_overlay import GFXSceneOverlay
    from cellier.visuals._types import VisualType


# Appearance fields that require a reslice (not just a GPU material update).
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

#: Visuals that load nothing while they draw nothing; showing one reslices it.
_SKIP_WHEN_HIDDEN = (BaseImageVisual,)


#: The pick event for each render-layer pick detail that is complete as
#: decoded.  Image and labels details are promoted first; see
#: ``CellierController._emit_pick_event``.
_PICK_EVENT_FOR_INFO: dict[type, type] = {
    PointsPickInfo: PointsPickEvent,
    LinesPickInfo: LinesPickEvent,
    MeshPickInfo: MeshPickEvent,
    GraphNodePickInfo: GraphPickEvent,
    GraphEdgePickInfo: GraphPickEvent,
}


def _pick_event_type(raw_pick: Any) -> type | None:
    """The public pick event type for a render-layer pick detail, if any."""
    from cellier.render.render_manager import (
        _ImageDisplayedDataCoord,
        _LabelsDisplayedDataCoord,
    )

    if isinstance(raw_pick, _ImageDisplayedDataCoord):
        return ImagePickEvent
    if isinstance(raw_pick, _LabelsDisplayedDataCoord):
        return LabelsPickEvent
    return _PICK_EVENT_FOR_INFO.get(type(raw_pick))


def _voxel_index(
    coordinate: tuple[float, ...], shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    """``floor`` of every component, or ``None`` outside an array of *shape*."""
    if len(coordinate) != len(shape):
        return None
    index = tuple(int(np.floor(value)) for value in coordinate)
    if any(not 0 <= i < size for i, size in zip(index, shape, strict=True)):
        return None
    return index


def _image_pick_positions(
    visual: Any,
    coordinate: tuple[float, ...],
    raw_pick: Any,
    camera_type: str,
) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]]:
    """The reported coordinate and where to read each channel's value.

    Implements the channel rows of design 3.7: no channel axis reads ``{0}``;
    single mode reads the drawn plane the coordinate already names; composite
    mode moves the channel component to the pick-buffer winner and reads
    every drawn channel on a 2D canvas, or only the winner on a 3D one.

    Returns
    -------
    coordinate : tuple[float, ...]
        The data coordinate to report.
    positions : dict[int, tuple[float, ...]]
        Channel index -> the data coordinate to read.
    """
    axis = getattr(visual, "channel_axis", None)
    if axis is None:
        return coordinate, {0: coordinate}
    if not visual.composite:
        return coordinate, {int(np.floor(coordinate[axis])): coordinate}
    drawn = tuple(raw_pick.drawn_channels)
    winner = raw_pick.channel_index
    if winner is None:
        if not drawn:
            return coordinate, {}
        winner = drawn[0]

    def at(channel: int) -> tuple[float, ...]:
        return tuple(
            channel + 0.5 if data_axis == axis else value
            for data_axis, value in enumerate(coordinate)
        )

    channels = (winner,) if camera_type == "3d" else drawn
    return at(winner), {channel: at(channel) for channel in channels}


def _visual_render_config(visual: BaseVisual) -> VisualRenderConfig:
    """Build the render settings one reslice passes for *visual*.

    Parameters
    ----------
    visual : BaseVisual
        The visual model.

    Returns
    -------
    VisualRenderConfig
        LOD settings from a multiscale visual's appearance, and
        ``slicing_enabled=False`` for an image visual that draws nothing:
        hidden, or composite with no drawn channel (unified image design 3.3).
    """
    slicing_enabled = not (
        isinstance(visual, _SKIP_WHEN_HIDDEN) and visual.draws_nothing()
    )
    if isinstance(visual, (MultiscaleImageVisual, MultiscaleLabelVisual)):
        return VisualRenderConfig(
            lod_bias=visual.appearance.lod_bias,
            force_level=visual.appearance.force_level,
            frustum_cull=visual.appearance.frustum_cull,
            slicing_enabled=slicing_enabled,
        )
    return VisualRenderConfig(slicing_enabled=slicing_enabled)


# Context variable used by update_slice_indices / update_appearance_field to
# thread a caller-supplied source_id through the synchronous psygnal bridge.
# Default None means the bridge falls back to the controller's own ID.
_source_id_override: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "_source_id_override", default=None
)

# Parallel context variable for update_aabb_field / _make_aabb_handler.
_aabb_source_id_override: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "_aabb_source_id_override", default=None
)

# Parallel context variable for update_background_field /
# _make_background_handler.  Background writes are unrelated to appearance
# writes, so they get their own variable rather than sharing one that an
# in-flight appearance write may already have set.
_background_source_id_override: contextvars.ContextVar[UUID | None] = (
    contextvars.ContextVar("_background_source_id_override", default=None)
)

# Parallel context variable for update_overlay_field / the overlay bridges.
# Overlays are neither visuals nor scenes, so they get their own variable.
_overlay_source_id_override: contextvars.ContextVar[UUID | None] = (
    contextvars.ContextVar("_overlay_source_id_override", default=None)
)


@dataclass
class _OverlayEntry:
    """The controller's record of one registered overlay.

    Attributes
    ----------
    model : CanvasOverlay or SceneOverlay
        The model-layer overlay, held in ``Canvas.overlays`` or
        ``Scene.overlays``.
    gfx : GFXCanvasOverlay or GFXSceneOverlay
        Its render-layer counterpart.
    kind : "canvas" or "scene"
        Which category it belongs to.
    owner_id : UUID
        The canvas (``kind="canvas"``) or scene (``kind="scene"``) holding it.
    scene_id : UUID
        The scene it is drawn in -- the owner itself for a scene overlay,
        the canvas's scene for a canvas overlay.  What a redraw is asked of.
    handlers : list[tuple]
        ``(signal, handler)`` psygnal connections, for teardown.
    appearance : object or None
        The appearance model the appearance bridge is attached to, so a
        wholesale replacement can move it.
    appearance_handler : Callable or None
        That bridge's handler.
    extent_key : tuple or None
        Scene overlays only: the ``(displayed_axes, bounds)`` last pushed to
        the render layer, so an unchanged scene rebuilds nothing.
    """

    model: Any
    gfx: Any
    kind: Literal["canvas", "scene"]
    owner_id: UUID
    scene_id: UUID
    handlers: list[tuple] = field(default_factory=list)
    appearance: Any = None
    appearance_handler: Callable | None = None
    extent_key: tuple | None = None


# Parallel context variable for the per-visual render settings (outline slot
# and placement, the occlusion tri-state, the labels selection).
_visual_render_source_id_override: contextvars.ContextVar[UUID | None] = (
    contextvars.ContextVar("_visual_render_source_id_override", default=None)
)

#: The message both halves of the pick_write conflict carry.  Outlines are a
#: screen-space post-process and the pick buffer is the only per-pixel
#: identity channel they have, so a visual that does not write pick resolves
#: to LUT entry 0 -- permanently inert -- and is silently not outlined.
_PICK_WRITE_REQUIRED = "outlines require pick_write=True."

#: Same mechanism for an ambient occlusion *exclusion*, which also has to
#: identify the visual per pixel.  An occlusion *inclusion* does not, which is
#: why the normal target is deliberately not gated on pick.
_AO_PICK_WRITE_REQUIRED = "ambient occlusion exclusions require pick_write=True."


class _RenderConfigRoute(NamedTuple):
    """How one render-config field reaches the GPU.

    Attributes
    ----------
    apply : Callable
        Called with ``(controller, new_value)`` after the model has been
        written.  Either forwards to a live setter on ``RenderManager`` or
        re-applies the whole section.
    recompiles : bool
        Whether changing this field recompiles a shader.  Not used to decide
        anything -- the apply route already handles it -- but it is the fact
        a GUI wants in a tooltip, and recording it beside the route is what
        keeps the two from drifting.
    """

    apply: Callable[[Any, Any], None]
    recompiles: bool = False


def _manager_setter(name: str) -> Callable[[Any, Any], None]:
    """Route a field to a live ``RenderManager`` property of *name*."""

    def _apply(controller, value) -> None:
        setattr(controller._render_manager, name, value)

    return _apply


def _reapply_outline(controller, _value) -> None:
    """Route a field to a whole-section re-apply of the outline config."""
    controller._render_manager.apply_outline_config()


def _apply_palette(controller, value) -> None:
    """Re-apply the outline config, warning about slots the palette lost.

    A visual outlined in a slot the palette no longer reaches draws a
    transparent band -- the same failure as asking for a slot past the end,
    reached from the other direction.  Checked here rather than in a widget
    so it fires for ``render_config.outline.palette = [...]`` in a notebook
    just as it does for a button.
    """
    orphaned = controller.visuals_outlined_beyond(len(value))
    if orphaned:
        named = ", ".join(sorted(f"{name!r} (slot {slot})" for name, slot in orphaned))
        warnings.warn(
            f"the palette now holds {len(value)} entries, so these visuals "
            f"draw a transparent outline: {named}",
            RuntimeWarning,
            stacklevel=2,
        )
    controller._render_manager.apply_outline_config()


#: Every settable render-config field, and how it reaches the GPU.
#:
#: Keys are ``(section, dotted field)``.  A field absent from this table is
#: not settable through :meth:`CellierController.update_render_config_field`,
#: which is deliberate for two of them: ``OutlineLayerConfig.color`` exists on
#: the shared layer model but does nothing on the *selection* layer, whose
#: colour comes from the palette slot carried in the LUT.  Recording that here
#: once is what lets every GUI simply not draw a control for it.
_RENDER_CONFIG_ROUTES: dict[tuple[str, str], _RenderConfigRoute] = {
    # -- Outlines.  Thicknesses are shader template vars; the rest are
    # uniforms.  Both go through apply_outline_config, which knows which.
    ("outline", "enabled"): _RenderConfigRoute(_manager_setter("outline_enabled")),
    ("outline", "boundaries.enabled"): _RenderConfigRoute(
        _manager_setter("outline_boundaries_enabled")
    ),
    ("outline", "selection.enabled"): _RenderConfigRoute(
        _manager_setter("outline_selection_enabled")
    ),
    ("outline", "boundaries.inward_thickness"): _RenderConfigRoute(
        _reapply_outline, recompiles=True
    ),
    ("outline", "boundaries.outward_thickness"): _RenderConfigRoute(
        _reapply_outline, recompiles=True
    ),
    ("outline", "selection.inward_thickness"): _RenderConfigRoute(
        _reapply_outline, recompiles=True
    ),
    ("outline", "selection.outward_thickness"): _RenderConfigRoute(
        _reapply_outline, recompiles=True
    ),
    ("outline", "inner_thickness"): _RenderConfigRoute(
        _reapply_outline, recompiles=True
    ),
    ("outline", "boundaries.color"): _RenderConfigRoute(_reapply_outline),
    ("outline", "inner_color"): _RenderConfigRoute(_reapply_outline),
    ("outline", "palette"): _RenderConfigRoute(_apply_palette),
    # -- Ambient occlusion.
    ("ambient_occlusion", "enabled"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_enabled")
    ),
    ("ambient_occlusion", "n_samples"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_n_samples"), recompiles=True
    ),
    ("ambient_occlusion", "blur_radius"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_blur_radius"), recompiles=True
    ),
    ("ambient_occlusion", "radius"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_radius")
    ),
    ("ambient_occlusion", "auto_radius_fraction"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_auto_radius_fraction")
    ),
    ("ambient_occlusion", "bias"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_bias")
    ),
    ("ambient_occlusion", "strength"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_strength")
    ),
    ("ambient_occlusion", "power"): _RenderConfigRoute(
        _manager_setter("ambient_occlusion_power")
    ),
    # -- Temporal accumulation.
    ("temporal", "enabled"): _RenderConfigRoute(_manager_setter("temporal_enabled")),
    ("temporal", "blend_weight"): _RenderConfigRoute(
        _manager_setter("temporal_blend_weight")
    ),
}

#: The sections ``update_render_config_field`` accepts.
RENDER_CONFIG_SECTIONS: tuple[str, ...] = ("outline", "ambient_occlusion", "temporal")

#: Every field ``update_render_config_field``'s per-visual twin accepts.
#:
#: ``outline_selected_labels`` is only meaningful on a labels visual -- every
#: other visual type is outlined as one silhouette -- so it is listed here but
#: rejected per visual at the call.
VISUAL_RENDER_FIELDS: tuple[str, ...] = (
    "outline.slot",
    "outline.placement",
    "ambient_occlusion",
    "outline_selected_labels",
    "outline_mode",
    "pick_write",
)
"""``pick_write`` is here because both features depend on it.

It already has an outgoing event of its own (``PickWriteChangedEvent``), so a
widget driving it subscribes to that and writes through here -- which is why
setting it does *not* also emit ``VisualRenderChangedEvent``: one field, one
outgoing event.
"""


def _resolve_render_config_route(section: str, field: str) -> _RenderConfigRoute:
    """Return the route for one field, or raise with a suggestion.

    The annotation cannot enforce a closed vocabulary here, so the lookup
    does -- at the call, rather than as render-time silence.
    """
    route = _RENDER_CONFIG_ROUTES.get((section, field))
    if route is not None:
        return route
    if section not in RENDER_CONFIG_SECTIONS:
        close = difflib.get_close_matches(section, RENDER_CONFIG_SECTIONS, n=1)
        suggestion = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"{section!r} is not a render config section.{suggestion} "
            f"Valid sections: {list(RENDER_CONFIG_SECTIONS)}."
        )
    valid = sorted(f for s, f in _RENDER_CONFIG_ROUTES if s == section)
    close = difflib.get_close_matches(field, valid, n=1)
    suggestion = f" Did you mean {close[0]!r}?" if close else ""
    raise ValueError(
        f"{field!r} is not a settable field of the {section!r} render config."
        f"{suggestion} Valid fields: {valid}."
    )


#: ``outline_mode`` -> the LUT ``kind`` that implements it.  Spelled out
#: rather than tested against one mode, so a mode added to the model without
#: a kind here fails loudly instead of silently outlining as a silhouette.
_LABELS_OUTLINE_KINDS: dict[str, int] = {
    "per_label": KIND_LABEL,
    "whole_object": KIND_WHOLE_OBJECT,
    "all_boundaries": KIND_LABEL_ALL,
}


def _render_mode_for(displayed_axes: Sequence[int]) -> str:
    """The render mode a displayed-axes tuple selects: 3 axes is 3D, 2 is 2D."""
    return "3d" if len(displayed_axes) == 3 else "2d"


def _outline_kind(visual) -> int:
    """Return the LUT ``kind`` the outline pass should use for *visual*.

    ``kind`` is the shader's mode selector, and it decides two things at
    once: what the outline key is, and where the colour comes from.
    ``KIND_WHOLE_OBJECT`` keys on the pick id, so a region is one object.
    ``KIND_LABEL`` keys on the per-pixel label, so a region is one label and
    touching labels keep a band between them, coloured per label.
    ``KIND_LABEL_ALL`` keys on the label too but colours every one of them
    from the visual's own slot.

    Only a labels visual has a choice, and it makes it through
    ``outline_mode``.  Everything else is one object by construction.
    """
    from cellier.visuals._label_memory import BaseLabelsVisual

    if isinstance(visual, BaseLabelsVisual):
        return _LABELS_OUTLINE_KINDS[visual.outline_mode]
    return KIND_WHOLE_OBJECT


def _default_placement(visual) -> str:
    """Return the default outline placement for one visual.

    ``"outward"`` for anything whose on-screen footprint is a few pixels
    wide by default -- lines, points, and graphs, whose nodes and edges are
    both ``"screen"``-spaced -- because an inward band twice the thickness
    of the thing it outlines consumes it entirely.  ``"inward"`` for
    everything else, so the region never appears to grow.
    """
    from cellier.visuals._graph_memory import GraphVisual
    from cellier.visuals._lines_memory import LinesVisual
    from cellier.visuals._points_memory import PointsVisual

    thin = (LinesVisual, PointsVisual, GraphVisual)
    return "outward" if isinstance(visual, thin) else "inward"


def _apply_render_settings(
    visual_model,
    *,
    outline=None,
    ambient_occlusion: bool | None = None,
    outline_selected_labels: dict[int, int] | None = None,
    pick_write: bool | None = None,
    outline_mode: str | None = None,
):
    """Apply the screen-space render settings an ``add_*`` call carried.

    Written onto the model *before* ``add_visual`` registers it, so the
    values are already in place when ``_seed_visual_render`` pushes them to
    the render layer.  That ordering is what makes a visual added with an
    outline outlined on its first frame, and it is also what makes the
    warnings fire once, from the seed, rather than twice.
    """
    if outline is not None:
        visual_model.outline = outline
    if ambient_occlusion is not None:
        visual_model.ambient_occlusion = ambient_occlusion
    if outline_selected_labels is not None:
        visual_model.outline_selected_labels = dict(outline_selected_labels)
    if pick_write is not None:
        visual_model.pick_write = pick_write
    if outline_mode is not None:
        from typing import get_args

        from cellier.visuals._label_memory import OutlineMode

        # Checked here because the labels model does not validate on
        # assignment: an unknown mode would otherwise be stored silently.
        valid = get_args(OutlineMode)
        if outline_mode not in valid:
            raise ValueError(
                f"outline_mode must be one of {list(valid)}; got {outline_mode!r}."
            )
        visual_model.outline_mode = outline_mode
    return visual_model


class CellierController:
    """The main class for constructing and controlling a cellier visualization.

    Wraps a ViewerModel (model layer) and a RenderManager (render layer) and
    performs the synchronization between both.
    """

    def __init__(
        self,
        widget_parent: object | None = None,
        render_config: RenderManagerConfig | None = None,
        gui: Literal["qt", "anywidget", "offscreen"] = "qt",
    ) -> None:
        # gui selects the render canvas toolkit ("qt", "anywidget" or
        # "offscreen"); threaded
        # to CanvasView via add_canvas.  widget_parent is Qt-only and ignored
        # for the anywidget gui (notebook canvases have no Qt parent).
        self._gui = gui
        self._widget_parent = widget_parent
        self._model = ViewerModel(data=DataManager())
        self._render_manager = RenderManager(config=render_config)
        # Keep model.render_config pointing at the same object so runtime
        # mutations (e.g. camera_settle_threshold_s) are captured on serialize.
        self._model.render_config = self._render_manager._config
        # Reverse map: visual_model_id → scene_id (model layer mirror of render layer)
        self._visual_to_scene: dict[UUID, UUID] = {}
        # Reverse map: canvas_id → scene_id
        self._canvas_to_scene: dict[UUID, UUID] = {}
        # Forward map: scene_id → list[canvas_id]
        self._scene_to_canvases: dict[UUID, list[UUID]] = {}
        # Event buses
        self._id: UUID = uuid4()
        self._outgoing_events: EventBus = EventBus()
        self._incoming_events: EventBus = EventBus()
        # Cache of last-known displayed_axes per scene for change detection
        self._dims_cache: dict[UUID, tuple[int, ...]] = {}
        # Cache of last-known slice positions and thicknesses per scene.  A
        # move here rebuilds only the rendered -> world embedding, whose
        # constant column carries the slice position; the rendered system
        # itself is unchanged, which is what keeps axis ids stable across a
        # slider drag.
        self._slice_cache: dict[UUID, tuple] = {}
        # The last ``Scene.slider_axes`` emitted per scene.  Used only to
        # decide whether a model change moved the derived set and so needs a
        # ``SliderAxesChangedEvent``; nothing reads it as state (design 3.5).
        self._slider_axes_cache: dict[UUID, tuple[int, ...]] = {}
        # id -> coordinate system, for every system this session knows about:
        # the stored ones (world, per-level data) and the runtime ones
        # (rendered, visual).  Three v2 methods -- map_bounding_box, then and
        # validate_against -- take system *objects* while a transform stores
        # only ids, so a lookup is required.  Deliberately a plain dict: no
        # edges, no path finding, no automatic composition (D15).
        self._coordinate_systems: dict[UUID, CoordinateSystemType] = {}
        # canvas_id -> (rendered system, rendered -> world embedding).
        # Rebuilt when displayed_axes changes, including a pure reorder; only
        # the embedding is rebuilt when slice_indices or thickness moves.
        self._rendered: dict[
            UUID, tuple[RenderedCoordinateSystem, AffineTransform]
        ] = {}
        # (visual_id, render_mode) -> the space that visual's GPU geometry is
        # indexed in (D45).  One per mode: a multiscale visual's 3D node is in
        # normalized proxy-box space and its 2D node in level-0 pixels.
        self._visual_spaces: dict[tuple[UUID, str], VisualCoordinateSystem] = {}
        # Canvases whose camera could not be fitted when their displayed
        # axes changed, because the scene was momentarily empty.  Drained
        # by ``_request_draw_for_scene`` once a reslice commits.
        self._canvases_awaiting_fit: set[UUID] = set()
        # render_modes registered per scene (determines which nodes visuals build)
        self._scene_render_modes: dict[UUID, set[Literal["2d", "3d"]]] = {}
        # Camera settle
        self._settle_tasks: dict[UUID, asyncio.Task] = {}
        # Per-canvas count of active pick-event subscribers (``on_pick``).
        # Drives RenderManager pick-detail gating (Decision 4).  Keyed by
        # canvas_id.
        self._pick_subscriber_counts: dict[UUID, int] = {}
        # The same count per (canvas, pick event type): a pick event is built,
        # and image and labels values are read, only while its type has one.
        self._pick_event_counts: dict[tuple[UUID, type], int] = {}
        # A pick SubscriptionHandle's id -> (canvas, event type), so that
        # unsubscribe_pick can decrement the right counters.
        self._pick_handles: dict[int, tuple[UUID, type]] = {}
        # In-flight asynchronous pick value reads (design 3.7): the ``move``
        # read per canvas, which a newer pointer event cancels, and every read
        # per visual, which ``remove_visual`` cancels.
        self._move_pick_reads: dict[UUID, UUID] = {}
        self._pick_reads_by_visual: dict[UUID, set[UUID]] = {}
        # Stored psygnal bridge handlers keyed by visual_id.
        # Each entry is a list of (signal, handler) pairs.  psygnal
        # disconnect() requires the exact handler object; closures are not
        # equality-comparable by value, so references must be retained.
        # Storing the signal alongside the handler avoids branching on visual
        # type during teardown.
        self._visual_psygnal_handlers: dict[UUID, list[tuple]] = {}
        # Per image visual: the channel appearance handlers, rewired whenever
        # ``channels`` is replaced, and the model + handler of the ``single``
        # bridge, moved whenever ``single`` is replaced.
        self._channel_psygnal_handlers: dict[UUID, list[tuple]] = {}
        self._single_bridges: dict[UUID, tuple] = {}
        # Same bookkeeping for the scene-level dims bridge, so it can be
        # disconnected on teardown.  Without a record the handler -- which
        # closes over ``self`` -- keeps the whole controller reachable from
        # the scene's psygnal signal for the lifetime of the process.
        self._scene_psygnal_handlers: dict[UUID, list[tuple]] = {}
        # The BackgroundAppearance object each scene's background bridge is
        # currently attached to, with its handler.  Needed to move the bridge
        # when the whole model is replaced (scene.background = ...), which
        # would otherwise leave the bridge listening to an orphaned object.
        self._scene_background_bridges: dict[UUID, tuple] = {}
        # Every registered overlay, canvas and scene alike, keyed by the
        # overlay model's id.
        self._overlays: dict[UUID, _OverlayEntry] = {}
        # Each registered store's ``data_changed`` connection, as
        # ``(signal, handler)``, keyed by store id -- for teardown.
        self._store_psygnal_handlers: dict[UUID, tuple] = {}
        # When True, transform-change handlers skip reslice_scene.  Managed
        # by the suppress_reslice context manager.  This is a flat boolean, so
        # nested suppress_reslice calls or concurrent async transform mutations
        # will interfere — use a depth counter if that ever becomes necessary.
        self._suppress_reslice: bool = False
        self._outgoing_events.subscribe(
            CameraChangedEvent,
            self._on_camera_changed,
            owner_id=self._id,
        )
        self._outgoing_events.subscribe(
            CanvasSizeChangedEvent,
            self._on_canvas_size_changed,
            owner_id=self._id,
        )
        # Must be called before subscribing _on_dims_changed_bus below so the
        # SliceCoordinator invalidates stale 2D caches before the controller
        # submits new slice requests.
        self._render_manager.connect_event_bus(self._outgoing_events)
        # Controller's own dims handler: reslice the affected scene.
        self._outgoing_events.subscribe(
            DimsChangedEvent,
            self._on_dims_changed_bus,
            owner_id=self._id,
        )
        # New data on the GPU is a content change like any other, and unlike
        # an appearance write nothing on this path asks for a frame.
        self._outgoing_events.subscribe(
            ResliceCompletedEvent,
            self._on_reslice_completed_redraw,
            owner_id=self._id,
        )
        # Subscribe to internal raw pointer events emitted by RenderManager.
        self._outgoing_events.subscribe(
            _CanvasRawPointerEvent,
            self._on_raw_pointer_event,
            owner_id=self._id,
        )
        # Incoming bus: GUI update events dispatched to update_* methods.
        self._incoming_events.subscribe(
            AppearanceUpdateEvent,
            self._on_appearance_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            DimsUpdateEvent,
            self._on_dims_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            SliderOverrideUpdateEvent,
            self._on_slider_override_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            AABBUpdateEvent,
            self._on_aabb_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            ChannelAppearanceUpdateEvent,
            self._on_channel_appearance_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            SingleAppearanceUpdateEvent,
            self._on_single_appearance_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            ImageCompositeUpdateEvent,
            self._on_image_composite_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            BackgroundUpdateEvent,
            self._on_background_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            OverlayUpdateEvent,
            self._on_overlay_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            RenderConfigUpdateEvent,
            self._on_render_config_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            VisualRenderUpdateEvent,
            self._on_visual_render_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            TrailUpdateEvent,
            self._on_trail_update,
            owner_id=self._id,
        )

    @property
    def incoming_events(self) -> EventBus:
        """Incoming event bus for GUI-driven model mutations.

        Emit ``AppearanceUpdateEvent``, ``DimsUpdateEvent``,
        ``AABBUpdateEvent`` or ``BackgroundUpdateEvent`` onto this bus to
        request model changes.
        The controller dispatches each event to the corresponding
        ``update_*`` method, preserving ``source_id`` end-to-end.
        """
        return self._incoming_events

    def set_widget_parent(self, parent: object) -> None:
        """Set the Qt parent for subsequently created canvas widgets.

        Only meaningful when ``self._gui == "qt"``; the anywidget gui ignores
        the parent (notebook canvases are not laid out by a Qt parent).
        """
        self._widget_parent = parent

    # ------------------------------------------------------------------
    # Construction class methods (stubs)
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: ViewerModel,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> CellierController:
        """Construct a controller from a serialized ViewerModel.

        Iteratively adds all data stores, scenes, visuals, and canvases
        through the public API. The order is:

        1. Data stores  — registered before visuals reference them.
        2. Scenes       — registered with render_modes and lighting from model.
        3. Visuals      — added per scene; data stores must already be present.
        4. Canvases     — restored with camera state from model.

        Parameters
        ----------
        model : ViewerModel
            A ViewerModel loaded from disk or constructed programmatically.
        widget_parent : QWidget or None
            Qt parent for canvas widgets. Defaults to None.
        render_config : RenderManagerConfig or None
            Render pipeline configuration. Defaults to None (uses defaults).

        Returns
        -------
        CellierController
        """
        controller = cls(
            widget_parent=widget_parent,
            render_config=render_config
            if render_config is not None
            else model.render_config,
        )

        # 1. Register all data stores.
        for store in model.data.stores.values():
            controller.add_data_store(store)

        # 2. Register all scenes (render_modes and lighting come from the model).
        for scene in model.scenes.values():
            controller.add_scene_model(scene)

        # 3. Add visuals per scene. Data stores are already registered in step 1.
        # The deserialized model's scene.visuals already contains the models.
        # Clear them first so _add_* helpers can re-append with proper wiring.
        for scene in model.scenes.values():
            visual_models = list(scene.visuals)
            scene.visuals.clear()
            for visual_model in visual_models:
                controller.add_visual(scene.id, visual_model)

        # 4. Restore canvases with camera state from the model.
        for scene in model.scenes.values():
            for canvas_model in scene.canvases.values():
                controller.add_canvas_model(scene.id, canvas_model)

        return controller

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> CellierController:
        """Deserialize a ViewerModel from disk and construct a controller.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file previously written by ``to_file``.
        widget_parent : QWidget or None
            Qt parent for canvas widgets.
        render_config : RenderManagerConfig or None
            Render pipeline configuration.

        Returns
        -------
        CellierController
        """
        model = ViewerModel.from_file(path)
        return cls.from_model(
            model,
            widget_parent=widget_parent,
            render_config=render_config,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | pathlib.Path) -> None:
        """Serialize the current model state to a JSON file."""
        self._model.to_file(path)

    def to_model(self) -> ViewerModel:
        """Return a copy of the current model state."""
        return self._model.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_scene_model(self, scene: Scene) -> Scene:
        """Register a pre-built Scene model with the controller.

        Used by ``from_model`` to restore scenes from a serialized
        ViewerModel, and called internally by ``add_scene``.

        Parameters
        ----------
        scene : Scene
            Pre-built scene model.

        Returns
        -------
        Scene
            The same object passed in.
        """
        self._model.scenes[scene.id] = scene
        self._scene_render_modes[scene.id] = scene.render_modes
        self._render_manager.add_scene(
            scene.id, lighting=scene.lighting, background=scene.background
        )
        self._scene_to_canvases[scene.id] = []
        self._wire_dims_model(scene)
        self._wire_scene_background(scene)
        # A scene restored from a serialized model arrives with its overlays
        # already in ``scene.overlays``; wire them without re-appending.
        for overlay in scene.overlays:
            self._check_new_overlay(overlay)
            self._register_scene_overlay(scene.id, overlay)
        self._outgoing_events.emit(
            SceneAddedEvent(source_id=self._id, scene_id=scene.id)
        )
        return scene

    def add_scene(
        self,
        *,
        name: str = "scene",
        dim: Literal["2d", "3d"] = "3d",
        coordinate_system: WorldAxesLike | None = None,
        render_modes: set[Literal["2d", "3d"]] | None = None,
        lighting: Literal["none", "default"] = "none",
        background: BackgroundAppearance | None = None,
    ) -> Scene:
        """Create a Scene from keyword arguments and register it.

        Parameters
        ----------
        name : str
            Human-readable scene name.
        dim : "2d" or "3d"
            Initial display dimensionality.  ``"3d"`` sets
            ``displayed_axes`` to the last three axes of the coordinate
            system; ``"2d"`` sets it to the last two.
        coordinate_system : WorldAxesLike or None
            The scene's world axes: a ``WorldCoordinateSystem``, or a sequence
            of ``Axis`` objects and/or ``(name, axis_type)`` pairs.  Axis
            types are stated, never inferred -- ``spatial_axes("z", "y", "x")``
            is the shorthand for an all-spatial world.  Defaults to a 3-axis
            spatial ``("z", "y", "x")`` world when ``None``.
        render_modes : set or None
            Which rendering modes visuals should support.  Defaults to
            ``{"2d", "3d"}``.
        lighting : "none" or "default"
            Pass ``"default"`` to add ambient/directional lights (required for
            ``MeshPhongAppearance``).
        background : BackgroundAppearance or None
            Background appearance for the scene.  ``None`` uses the model
            defaults (the cellier vertical gray gradient).

        Returns
        -------
        Scene
            The newly created and registered Scene.
        """
        world = (
            world_coordinate_system(spatial_axes("z", "y", "x"))
            if coordinate_system is None
            else world_coordinate_system(coordinate_system)
        )
        ndim = world.ndim
        n_displayed = 3 if dim == "3d" else 2
        if ndim < n_displayed:
            raise ValueError(
                f"coordinate_system has {ndim} axes but dim={dim!r} requires "
                f"at least {n_displayed}."
            )
        displayed_axes = tuple(range(ndim - n_displayed, ndim))
        # Every axis gets a position, displayed ones included (D36).
        slice_indices = dict.fromkeys(range(ndim), 0.0)
        dims = DimsManager(
            world_coordinate_system=world,
            selection=AxisAlignedSelection(
                displayed_axes=displayed_axes,
                slice_indices=slice_indices,
            ),
        )
        scene = Scene(
            name=name,
            dims=dims,
            render_modes=render_modes if render_modes is not None else {"2d", "3d"},
            lighting=lighting,
            background=background if background is not None else BackgroundAppearance(),
        )
        return self.add_scene_model(scene)

    # ------------------------------------------------------------------
    # Data store management
    # ------------------------------------------------------------------

    def add_data_store(self, data_store: BaseDataStore) -> BaseDataStore:
        """Register a data store and return it.

        Parameters
        ----------
        data_store : BaseDataStore
            The store to register.

        Returns
        -------
        BaseDataStore
            The same object passed in.
        """
        self._model.data.stores[data_store.id] = data_store
        self._register_coordinate_systems(*data_store.data_coordinate_systems)
        self._wire_data_store(data_store)
        return data_store

    def _wire_data_store(self, data_store: BaseDataStore) -> None:
        """Relay *data_store*'s change announcements (``_on_store_changed``).

        Idempotent for the same store object; a different object registered
        under the same id replaces the old connection.
        """
        existing = self._store_psygnal_handlers.get(data_store.id)
        if existing is not None:
            signal, handler = existing
            if signal is data_store.data_changed:
                return
            signal.disconnect(handler)
        store_id = data_store.id

        def _on_data_changed(change: StoreChange) -> None:
            self._on_store_changed(store_id, change)

        data_store.data_changed.connect(_on_data_changed)
        self._store_psygnal_handlers[store_id] = (
            data_store.data_changed,
            _on_data_changed,
        )

    def _on_store_changed(self, store_id: UUID, change: StoreChange) -> None:
        """React to a store announcing that its data changed.

        For an ``"extent"`` change, first refresh what is derived from the
        store's extent: the render layer's per-visual extents (the
        out-of-domain slice check) and the scene overlays of every scene
        showing the store.  Then announce the change on the bus and reslice
        every visual reading the store -- for both kinds, so a caller that
        changes a store no longer has to reslice by hand
        (``plans/store_change_events.md``).
        """
        readers = [
            (scene_id, visual.id)
            for scene_id, scene in self._model.scenes.items()
            for visual in scene.visuals
            if UUID(str(visual.data_store_id)) == store_id
            and visual.id in self._visual_to_scene
        ]
        if change.kind == "extent":
            for _scene_id, visual_id in readers:
                self._render_manager.refresh_visual_axis_extents(visual_id)
            for scene_id in dict.fromkeys(scene_id for scene_id, _ in readers):
                self._refresh_scene_overlays(scene_id)
            event: Any = DataStoreMetadataChangedEvent(
                source_id=self._id, data_store_id=store_id
            )
        else:
            event = DataStoreContentsChangedEvent(
                source_id=self._id, data_store_id=store_id, regions=change.regions
            )
        self._outgoing_events.emit(event)
        for _scene_id, visual_id in readers:
            self.reslice_visual(visual_id)

    # ------------------------------------------------------------------
    # Visual management — public API
    # ------------------------------------------------------------------

    def add_visual(
        self,
        scene_id: UUID,
        visual_model: VisualType,
        data_store: BaseDataStore | None = None,
    ) -> VisualType:
        """Register a pre-built visual model with a scene.

        This is the canonical construction path used by ``from_model``.
        All typed convenience methods (``add_image``, ``add_mesh``, etc.)
        delegate to this method internally.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : VisualType
            Pre-built visual model. Its ``data_store_id`` must already be
            registered via ``add_data_store``, or ``data_store`` must be
            passed explicitly.
        data_store : BaseDataStore or None
            If provided, register the store first (no-op if already present),
            then use it. If ``None``, the store is looked up by
            ``visual_model.data_store_id``; a ``KeyError`` is raised if not
            found.

        Returns
        -------
        VisualType
            The same visual_model passed in.

        Raises
        ------
        KeyError
            If ``data_store`` is None and ``visual_model.data_store_id`` is not
            registered.
        TypeError
            If the visual type is not recognized.
        """
        if data_store is not None:
            if data_store.id not in self._model.data.stores:
                self._model.data.stores[data_store.id] = data_store
                self._wire_data_store(data_store)
        else:
            data_store = self._model.data.stores[UUID(visual_model.data_store_id)]

        # Before any GFX object is built: the store must be able to say what
        # its axes are, because everything downstream -- the data -> world
        # transform, the visual space, the region pull-back -- is addressed
        # by axis id.
        self._ensure_data_coordinate_systems(scene_id, data_store, visual_model)
        world = self._model.scenes[scene_id].dims.world_coordinate_system
        supplied = getattr(visual_model, "transform", None)
        if supplied is None:
            visual_model.transform = default_data_to_world(
                data_store.data_coordinate_system, world
            )

        if isinstance(visual_model, MultiscaleImageVisual):
            return self._add_multiscale_image_visual(scene_id, visual_model)
        elif isinstance(visual_model, ImageVisual):
            return self._add_image_visual(scene_id, visual_model)
        elif isinstance(visual_model, MultiscaleLabelVisual):
            return self._add_multiscale_label_visual(scene_id, visual_model)
        elif isinstance(visual_model, LabelMemoryVisual):
            return self._add_label_memory_visual(scene_id, visual_model)
        elif isinstance(visual_model, PointsVisual):
            return self._add_points_visual(scene_id, visual_model)
        elif isinstance(visual_model, LinesVisual):
            return self._add_lines_visual(scene_id, visual_model)
        elif isinstance(visual_model, MeshVisual):
            return self._add_mesh_visual(scene_id, visual_model)
        elif isinstance(visual_model, GraphVisual):
            return self._add_graph_visual(scene_id, visual_model)
        else:
            raise TypeError(
                f"Unrecognized visual type {type(visual_model)!r}. "
                "Register a handler in add_visual."
            )

    def add_image(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        appearance: InMemoryImageAppearance | None = None,
        name: str = "image",
        *,
        single: InMemoryImageSingleAppearance | None = None,
        channel_axis: int | None = None,
        composite: bool = False,
        channels: dict[int, InMemoryImageChannelAppearance] | None = None,
        max_channels: int = 4,
        transform: AffineTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> ImageVisual:
        """Add an in-memory image visual to a scene.

        One visual draws the image single-channel or composited
        (unified image design 3.1): ``composite`` picks the mode, ``single``
        is the appearance single mode draws with, and ``channels`` the
        per-channel appearances composite mode draws with.

        Parameters
        ----------
        data : ImageMemoryStore
            The backing data store.
        scene_id : UUID
            ID of an existing scene.
        appearance : InMemoryImageAppearance or None
            Shared by both modes.  ``None`` uses the defaults.
        name : str
            Human-readable label. Default ``"image"``.
        single : InMemoryImageSingleAppearance or None
            Single mode's appearance.  ``None`` uses the defaults.
        channel_axis : int or None
            The data axis a composite draws channels along.  It must map to a
            world axis.  ``None`` (default) gives an image with no channels.
        composite : bool
            Start in composite mode.  Requires *channel_axis*.
        channels : dict[int, InMemoryImageChannelAppearance] or None
            Composite mode's per-channel appearances.  ``None`` is none.
        max_channels : int
            The most channels the visual may hold.  Default 4.
        transform : AffineTransform or None
            The ``data -> world`` transform.  ``None`` (default) is the
            identity between the store's level-0 system and the world.
        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic.
        pick_write : bool
            Whether the visual writes to the pick buffer.  Default ``True``.

        Returns
        -------
        ImageVisual

        Raises
        ------
        ValueError
            If *composite* is set without *channel_axis*, if *channel_axis*
            maps to no world axis, if the composited axis is displayed, or if
            *channels* has more than *max_channels* entries.
        """
        visual_model = ImageVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance
            if appearance is not None
            else InMemoryImageAppearance(),
            single=single if single is not None else InMemoryImageSingleAppearance(),
            channel_axis=channel_axis,
            composite=composite,
            channels=dict(channels or {}),
            max_channels=max_channels,
            transform=self._prepare_transform(scene_id, data, transform),
        )
        self._check_image_axes(scene_id, visual_model)
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_labels(
        self,
        data: LabelMemoryStore,
        scene_id: UUID,
        appearance: BaseLabelsAppearance | None = None,
        name: str = "labels",
        transform: AffineTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
        outline_selected_labels: dict[int, int] | None = None,
        outline_mode: OutlineMode = "per_label",
    ) -> LabelMemoryVisual:
        """Add an in-memory label visual to a scene.

        Parameters
        ----------
        data : LabelMemoryStore
            Backing int32 label store.
        scene_id : UUID
            ID of an existing scene.
        appearance : BaseLabelsAppearance or None
            Appearance parameters. Defaults to InMemoryLabelsAppearance().
        name : str
            Human-readable label. Default ``"labels"``.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled;
            see :attr:`outline_enabled`.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.
        outline_selected_labels : dict[int, int] or None
            Maps a label value to the palette slot the selection layer
            draws it in.  ``None`` (default) selects no label, so an
            outlined labels visual shows boundaries only.
        outline_mode : {"per_label", "whole_object", "all_boundaries"}
            How the labels are outlined.  ``"per_label"`` (default) outlines
            the label values in ``outline_selected_labels``, each in its own
            slot's colour.  ``"whole_object"`` outlines the volume as one
            silhouette and ``"all_boundaries"`` every label's boundary, both
            in the colour of the ``outline`` slot.

        Returns
        -------
        LabelMemoryVisual
        """
        if appearance is None:
            from cellier.visuals._label_memory import InMemoryLabelsAppearance

            appearance = InMemoryLabelsAppearance()

        resolved_transform = self._prepare_transform(scene_id, data, transform)
        visual_model = LabelMemoryVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
            outline_selected_labels=outline_selected_labels,
            outline_mode=outline_mode,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_mesh(
        self,
        data: MeshMemoryStore,
        scene_id: UUID,
        appearance: MeshAppearance,
        name: str = "mesh",
        transform: AffineTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> MeshVisual:
        """Add a mesh visual to a scene.

        Parameters
        ----------
        data : MeshMemoryStore
            In-memory mesh.  Normals are auto-computed if not supplied;
            indices are coerced to int32.
        scene_id : UUID
            ID of an existing scene.
        appearance : MeshFlatAppearance | MeshPhongAppearance
            Appearance.  Use MeshPhongAppearance with ``lighting="default"``
            on the scene for shaded rendering.
        name : str
            Human-readable label.  Default ``"mesh"``.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled;
            see :attr:`outline_enabled`.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        MeshVisual
        """
        resolved_transform = self._prepare_transform(scene_id, data, transform)
        visual_model = MeshVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_points(
        self,
        data: PointsMemoryStore,
        scene_id: UUID,
        appearance: PointsMarkerAppearance | None = None,
        name: str = "points",
        transform: AffineTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> PointsVisual:
        """Add a points visual backed by a PointsMemoryStore.

        Parameters
        ----------
        data : PointsMemoryStore
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : PointsMarkerAppearance or None
            Appearance model.  Defaults to PointsMarkerAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled;
            see :attr:`outline_enabled`.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        PointsVisual
        """
        if appearance is None:
            appearance = PointsMarkerAppearance()

        resolved_transform = self._prepare_transform(scene_id, data, transform)
        visual_model = PointsVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_lines(
        self,
        data: LinesMemoryStore,
        scene_id: UUID,
        appearance: LinesMemoryAppearance | None = None,
        name: str = "lines",
        transform: AffineTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> LinesVisual:
        """Add a lines visual backed by a LinesMemoryStore.

        Parameters
        ----------
        data : LinesMemoryStore
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : LinesMemoryAppearance or None
            Appearance model.  Defaults to LinesMemoryAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled;
            see :attr:`outline_enabled`.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        LinesVisual
        """
        if appearance is None:
            appearance = LinesMemoryAppearance()

        resolved_transform = self._prepare_transform(scene_id, data, transform)
        visual_model = LinesVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_graph(
        self,
        data: GraphMemoryStore,
        scene_id: UUID,
        appearance: GraphAppearance | None = None,
        name: str = "graph",
        transform: AffineTransform | None = None,
        trail: dict[int, TrailConfig] | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> GraphVisual:
        """Add a spatial-graph visual backed by a GraphMemoryStore.

        Parameters
        ----------
        data : GraphMemoryStore
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : GraphAppearance or None
            Appearance model.  Defaults to ``GraphAppearance()`` if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual.  When ``None``, the
            store's own transform is used if it has one -- a geff file's
            per-axis ``scale`` / ``offset`` (D23) -- and identity otherwise.
            An explicit argument always wins, as it does for every other
            visual: D23 constrains construction, not composition.
        trail : dict[int, TrailConfig] or None
            Axis index -> window configuration.  Keys are validated against
            the store's ``ndim``; an out-of-range axis raises ``ValueError``
            (D21).

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled;
            see :attr:`outline_enabled`.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.

        Returns
        -------
        GraphVisual

        Raises
        ------
        ValueError
            If any ``trail`` key is not a valid axis index for ``data``.
        """
        if appearance is None:
            appearance = GraphAppearance()

        # A geff file states its own per-axis scale and offset (D23), which
        # stands in for an explicit transform.  They are raw numbers -- the
        # store cannot name the scene's world -- so they become a transform
        # here, between the two systems this method knows.
        resolved_transform = transform
        if resolved_transform is None and data.axis_scales is not None:
            self._ensure_data_coordinate_systems(scene_id, data)
            resolved_transform = scale_and_translation_transform(
                data.data_coordinate_system,
                self._model.scenes[scene_id].dims.world_coordinate_system,
                data.axis_scales,
                data.axis_offsets,
            )
        resolved_transform = self._prepare_transform(scene_id, data, resolved_transform)

        visual_model = GraphVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
            trail=dict(trail or {}),
        )
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: MultiscaleImageAppearance | None = None,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
        *,
        single: MultiscaleImageSingleAppearance | None = None,
        channel_axis: int | None = None,
        composite: bool = False,
        channels: dict[int, MultiscaleImageChannelAppearance] | None = None,
        max_channels: int = 4,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual to a scene.

        The multiscale twin of :meth:`add_image`; see it for the two modes.

        Parameters
        ----------
        data : BaseDataStore
            The backing multiscale data store.
        scene_id : UUID
            ID of an existing scene.
        appearance : MultiscaleImageAppearance or None
            Shared by both modes, including the LOD settings.  ``None`` uses
            the defaults.
        name : str
            Human-readable label. Default ``"image"``.
        render_config : MultiscaleImageRenderConfig or None
            GPU cache configuration.  The budget is split evenly between the
            visual's slots: one without a channel axis, ``max_channels`` with.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.
        single : MultiscaleImageSingleAppearance or None
            Single mode's appearance.
        channel_axis : int or None
            The data axis a composite draws channels along.
        composite : bool
            Start in composite mode.  Requires *channel_axis*.
        channels : dict[int, MultiscaleImageChannelAppearance] or None
            Composite mode's per-channel appearances.
        max_channels : int
            The most channels the visual may hold.  Default 4.
        outline : VisualOutline or None
            Screen-space outline assignment.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.
        pick_write : bool
            Whether the visual writes to the pick buffer.  Default ``True``.

        Returns
        -------
        MultiscaleImageVisual

        Raises
        ------
        ValueError
            As :meth:`add_image`.
        """
        if render_config is None:
            render_config = MultiscaleImageRenderConfig()

        resolved_transform = self._prepare_transform(scene_id, data, transform)
        visual_model = MultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            level_transforms=data.level_transforms,
            appearance=(
                appearance if appearance is not None else MultiscaleImageAppearance()
            ),
            single=single if single is not None else MultiscaleImageSingleAppearance(),
            channel_axis=channel_axis,
            composite=composite,
            channels=dict(channels or {}),
            max_channels=max_channels,
            render_config=render_config,
            transform=resolved_transform,
        )
        self._check_image_axes(scene_id, visual_model)
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_labels_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: MultiscaleLabelsAppearance,
        name: str = "labels",
        render_config: MultiscaleLabelRenderConfig | None = None,
        transform: AffineTransform | None = None,
        outline: VisualOutline | None = None,
        ambient_occlusion: bool | None = None,
        pick_write: bool = True,
        outline_selected_labels: dict[int, int] | None = None,
        outline_mode: OutlineMode = "per_label",
    ) -> MultiscaleLabelVisual:
        """Add a multiscale label visual to a scene.

        Parameters
        ----------
        data : BaseDataStore
            The backing label data store (e.g. ``OMEZarrLabelDataStore``).
        scene_id : UUID
            ID of an existing scene.
        appearance : MultiscaleLabelsAppearance
            Visual appearance parameters.
        name : str
            Human-readable label. Default ``"labels"``.
        render_config : MultiscaleLabelRenderConfig or None
            Render-layer configuration. Defaults to
            ``MultiscaleLabelRenderConfig()`` with all default values if None.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.

        outline : VisualOutline or None
            Screen-space outline assignment.  ``None`` (default) leaves the
            visual unoutlined.  Requires the outline pass to be enabled;
            see :attr:`outline_enabled`.
        ambient_occlusion : bool or None
            Whether this visual receives ambient occlusion.  ``None``
            (default) is automatic: excluded while it renders in a
            MIP-family mode, included otherwise.
        pick_write : bool
            Whether the visual writes to the pick buffer.  ``True`` (default)
            makes it pickable.  ``False`` keeps it out -- say, a volume drawn
            over outlined visuals, whose pick ids would otherwise erase their
            outlines.  Outlines and ambient-occlusion exclusions are both
            derived from the pick buffer, so turning it off stops them on
            this visual; asking for an outline as well turns it back on,
            with a warning.
        outline_selected_labels : dict[int, int] or None
            Maps a label value to the palette slot the selection layer
            draws it in.  ``None`` (default) selects no label, so an
            outlined labels visual shows boundaries only.
        outline_mode : {"per_label", "whole_object", "all_boundaries"}
            How the labels are outlined.  ``"per_label"`` (default) outlines
            the label values in ``outline_selected_labels``, each in its own
            slot's colour.  ``"whole_object"`` outlines the volume as one
            silhouette and ``"all_boundaries"`` every label's boundary, both
            in the colour of the ``outline`` slot.

        Returns
        -------
        MultiscaleLabelVisual
        """
        if render_config is None:
            render_config = MultiscaleLabelRenderConfig()

        resolved_transform = self._prepare_transform(scene_id, data, transform)
        visual_model = MultiscaleLabelVisual(
            name=name,
            data_store_id=str(data.id),
            level_transforms=data.level_transforms,
            appearance=appearance,
            render_config=render_config,
            transform=resolved_transform,
        )
        _apply_render_settings(
            visual_model,
            outline=outline,
            ambient_occlusion=ambient_occlusion,
            pick_write=pick_write,
            outline_selected_labels=outline_selected_labels,
            outline_mode=outline_mode,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def _get_visual_model(self, visual_id: UUID) -> VisualType:
        """Return the visual model for *visual_id*.

        Raises
        ------
        KeyError
            If *visual_id* is not registered.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        for visual in scene.visuals:
            if visual.id == visual_id:
                return visual
        raise KeyError(f"Visual {visual_id} not found.")

    def add_channel(
        self,
        visual_id: UUID,
        channel_index: int,
        appearance: InMemoryImageChannelAppearance | MultiscaleImageChannelAppearance,
    ) -> None:
        """Add a channel to an image visual's composite channels.

        The ``channels`` bridge reslices the visual, so a composite draws the
        new channel straight away.

        Parameters
        ----------
        visual_id : UUID
            ID of an ``ImageVisual`` or ``MultiscaleImageVisual``.
        channel_index : int
            Index along the visual's ``channel_axis``.  Must not already be
            present.
        appearance : channel appearance
            The channel's appearance, of the visual's own family.

        Raises
        ------
        ValueError
            If the visual has no ``channel_axis``, *channel_index* is already
            present, or the visual already holds ``max_channels`` channels.
        """
        visual = self._get_visual_model(visual_id)
        if not isinstance(visual, BaseImageVisual) or visual.channel_axis is None:
            raise ValueError(
                f"Visual {visual_id} has no channel_axis, so it has no channels."
            )
        if channel_index in visual.channels:
            raise ValueError(
                f"channel_index={channel_index} already in visual.channels."
            )
        if len(visual.channels) >= visual.max_channels:
            raise ValueError(
                f"The visual already holds max_channels={visual.max_channels} "
                "channels.  Remove one first, or build it with a larger "
                "max_channels."
            )
        new_channels = dict(visual.channels)
        new_channels[channel_index] = appearance
        visual.channels = new_channels

    def remove_channel(self, visual_id: UUID, channel_index: int) -> None:
        """Remove a channel from an image visual's composite channels.

        Any channel may be removed, including the last (D35); an empty
        composite draws nothing.  The ``channels`` bridge reslices.

        Parameters
        ----------
        visual_id : UUID
            ID of an ``ImageVisual`` or ``MultiscaleImageVisual``.
        channel_index : int
            Index of the channel to remove.

        Raises
        ------
        KeyError
            If channel_index is not in visual.channels.
        """
        visual = self._get_visual_model(visual_id)
        if channel_index not in visual.channels:
            raise KeyError(f"channel_index={channel_index} not in visual.channels.")
        visual.channels = {
            k: v for k, v in visual.channels.items() if k != channel_index
        }

    def _composited_world_axis(self, visual: Any) -> int | None:
        """The world axis an image in composite mode composites, else ``None``."""
        if (
            not isinstance(visual, BaseImageVisual)
            or not visual.composite
            or visual.channel_axis is None
            or visual.transform is None
        ):
            return None
        return visual.transform.axis_correspondence().get(visual.channel_axis)

    def _check_image_axes(self, scene_id: UUID, visual: BaseImageVisual) -> None:
        """Refuse an image whose channel axis is unusable where it is going.

        The channel axis must map to a world axis (design 3.1), and a
        composited axis cannot be displayed (design 3.4).

        Raises
        ------
        ValueError
            On either.
        """
        if visual.channel_axis is None or visual.transform is None:
            return
        correspondence = visual.transform.axis_correspondence()
        if visual.channel_axis not in correspondence:
            raise ValueError(
                f"channel_axis={visual.channel_axis} maps to no world axis "
                f"through this visual's transform.  A channel axis with no "
                f"world counterpart is not supported."
            )
        world_axis = correspondence[visual.channel_axis]
        displayed = self._model.scenes[scene_id].dims.selection.displayed_axes
        if visual.composite and world_axis in displayed:
            raise ValueError(
                f"Cannot composite channel_axis={visual.channel_axis}: it maps "
                f"to world axis {world_axis}, which the scene displays "
                f"{tuple(displayed)}."
            )

    # ------------------------------------------------------------------
    # Visual management — private dispatch methods
    # ------------------------------------------------------------------

    def _register_visual(
        self,
        scene_id: UUID,
        visual_model: BaseVisual,
        gfx_visual: Any,
        data_store: Any,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Register a pre-built (visual_model, gfx_visual) pair in one scene.

        Single point of truth for all post-construction wiring:
        - Appends the visual to the scene's visual list.
        - Registers the GFX visual with the RenderManager.
        - Records the visual→scene mapping.
        - Wires psygnal bridges (appearance, channels, aabb, transform,
          pick_write).
        - Subscribes the GFX visual to all EventBus events it handles.
        - Emits VisualAddedEvent.
        """
        scene = self._model.scenes[scene_id]
        scene.visuals.append(visual_model)
        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
        self._visual_to_scene[visual_model.id] = scene_id
        self._rebuild_visual_space(
            visual_model.id, _render_mode_for(displayed_axes), data_store
        )
        setter = getattr(gfx_visual, "set_render_spaces", None)
        if setter is not None:
            setter(self.render_spaces(visual_model.id))

        # psygnal bridges
        if hasattr(visual_model, "appearance"):
            self._wire_appearance(visual_model)
        if isinstance(visual_model, BaseImageVisual):
            self._check_image_axes(scene_id, visual_model)
            self._wire_image(visual_model)
        if isinstance(visual_model, GraphVisual):
            self._wire_trail(visual_model)
        self._wire_aabb(visual_model)
        self._wire_transform(visual_model, scene_id)
        self._wire_pick_write(visual_model)
        self._wire_visual_render(visual_model)
        # Seed the render layer from the model, so a visual constructed with
        # an outline already set is outlined on its first frame rather than
        # needing a post-hoc call.
        self._seed_visual_render(visual_model)
        self._check_slider_axes(scene_id)
        self._refresh_scene_overlays(scene_id)

        # EventBus subscriptions — only subscribe when the GFX visual implements
        # the handler so new visual types get wired automatically.
        for event_type, handler_name in (
            (AppearanceChangedEvent, "on_appearance_changed"),
            (ChannelAppearanceChangedEvent, "on_channel_appearance_changed"),
            (SingleAppearanceChangedEvent, "on_single_appearance_changed"),
            (ImageCompositeChangedEvent, "on_image_composite_changed"),
            (AABBChangedEvent, "on_aabb_changed"),
            (VisualVisibilityChangedEvent, "on_visibility_changed"),
            (TrailChangedEvent, "on_trail_changed"),
            (TransformChangedEvent, "on_transform_changed"),
            (PickWriteChangedEvent, "on_pick_write_changed"),
        ):
            handler = getattr(gfx_visual, handler_name, None)
            if handler is not None:
                self._outgoing_events.subscribe(
                    event_type,
                    handler,
                    entity_id=visual_model.id,
                    owner_id=visual_model.id,
                )

        self._outgoing_events.emit(
            VisualAddedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_model.id,
            )
        )
        self._request_draw_for_scene(scene_id)

    def _add_multiscale_image_visual(
        self,
        scene_id: UUID,
        visual_model: MultiscaleImageVisual,
    ) -> MultiscaleImageVisual:
        """Wire and register a pre-built MultiscaleImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=list(data_store.level_shapes),
            render_modes=render_modes,
            displayed_axes=displayed_axes,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_multiscale_label_visual(
        self,
        scene_id: UUID,
        visual_model: MultiscaleLabelVisual,
    ) -> MultiscaleLabelVisual:
        """Wire and register a pre-built MultiscaleLabelVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMultiscaleLabelVisual.from_cellier_model(
            model=visual_model,
            level_shapes=list(data_store.level_shapes),
            render_modes=render_modes,
            displayed_axes=displayed_axes,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_image_visual(
        self,
        scene_id: UUID,
        visual_model: ImageVisual,
    ) -> ImageVisual:
        """Wire and register a pre-built ImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXImageMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_points_visual(
        self,
        scene_id: UUID,
        visual_model: PointsVisual,
    ) -> PointsVisual:
        """Wire and register a pre-built PointsVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXPointsMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_lines_visual(
        self,
        scene_id: UUID,
        visual_model: LinesVisual,
    ) -> LinesVisual:
        """Wire and register a pre-built LinesVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXLinesMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_graph_visual(
        self,
        scene_id: UUID,
        visual_model: GraphVisual,
    ) -> GraphVisual:
        """Wire and register a pre-built GraphVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        self._validate_trail_axes(visual_model.trail, data_store)

        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXGraphMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    @staticmethod
    def _validate_trail_axes(trail: dict, data_store: Any) -> None:
        """Raise if any trail key is out of range for the store (D21).

        This lives in the controller rather than on the visual because the
        visual holds only ``data_store_id`` while ``ndim`` is on the store.
        An out-of-range axis is always a bug -- there is no view in which it
        becomes meaningful -- so it raises rather than warning.  A *valid*
        axis that merely happens to be displayed is a different situation
        and warns once per (visual, axis); see
        ``GFXGraphMemoryVisual._build_request``.
        """
        ndim = data_store.ndim
        for axis in trail:
            if not isinstance(axis, (int, np.integer)) or not (0 <= axis < ndim):
                raise ValueError(
                    f"Trail axis {axis!r} is out of range for a {ndim}-axis "
                    f"graph; valid axes are 0 to {ndim - 1}."
                )

    def _wire_trail(self, visual: GraphVisual) -> None:
        """Subscribe to trail changes on a graph visual.

        Two subscriptions, and both are needed for the same reason
        ``GraphAppearance`` is flat: psygnal does not propagate nested
        ``EventedModel`` field changes to the parent's event group.

        1. One handler per ``TrailConfig``, so editing ``before`` on an
           existing config reaches the render layer.  Modelled on
           ``_wire_channels``.
        2. A direct connect on ``visual.events.trail``, so whole-dict
           replacement (``visual.trail = {...}``) rewires the per-config
           handlers and reslices.

        Note that an out-of-range axis assigned through route 2 surfaces as
        psygnal's ``EmitLoopError`` wrapping the ``ValueError``, not as a
        bare ``ValueError``: the check needs the store's ``ndim``, so it can
        only run inside the callback, and psygnal wraps whatever a callback
        raises.  The message and ``__cause__`` are preserved.  The
        ``add_graph`` path raises the ``ValueError`` directly.
        """
        for axis, config in visual.trail.items():
            self._connect_trail_config(visual, axis, config)

        handler = self._make_trail_dict_handler(visual)
        visual.events.trail.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.events.trail, handler)
        )

    def _connect_trail_config(
        self, visual: GraphVisual, axis: int, config: TrailConfig
    ) -> None:
        """Connect one ``TrailConfig``'s field events."""
        handler = self._make_trail_config_handler(visual, axis)
        config.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (config.events, handler)
        )

    def _make_trail_config_handler(self, visual: GraphVisual, axis: int) -> Callable:
        """Return a psygnal catch-all handler for one ``TrailConfig``."""

        def _on_trail_config_psygnal(info: EmissionInfo) -> None:
            self._emit_trail_changed(visual, field_name=info.signal.name, axis=axis)

        return _on_trail_config_psygnal

    def _make_trail_dict_handler(self, visual: GraphVisual) -> Callable:
        """Return a handler for whole-dict replacement of ``visual.trail``.

        Revalidates the new keys, rewires the per-config handlers onto the
        new ``TrailConfig`` objects, and reslices.
        """

        def _on_trail_replaced(new_trail: dict) -> None:
            data_store = self._model.data.stores[UUID(visual.data_store_id)]
            self._validate_trail_axes(new_trail, data_store)
            for axis, config in new_trail.items():
                self._connect_trail_config(visual, axis, config)
            self._emit_trail_changed(visual)

        return _on_trail_replaced

    def _emit_trail_changed(
        self,
        visual: GraphVisual,
        field_name: str | None = None,
        axis: int | None = None,
    ) -> None:
        """Emit a TrailChangedEvent and reslice.

        The trail selects *which data is fetched*, so every change to it
        must trigger a reslice -- an appearance push alone would leave the
        old geometry on screen.
        """
        self._outgoing_events.emit(
            TrailChangedEvent(
                source_id=_source_id_override.get() or self._id,
                visual_id=visual.id,
                trail=dict(visual.trail),
                field_name=field_name,
                axis=axis,
            )
        )
        if visual.id in self._visual_to_scene:
            self.reslice_visual(visual.id)

    def _add_mesh_visual(
        self,
        scene_id: UUID,
        visual_model: MeshVisual,
    ) -> MeshVisual:
        """Wire and register a pre-built MeshVisual."""
        import warnings

        if isinstance(visual_model.appearance, MeshPhongAppearance):
            if not self._render_manager.scene_has_lighting(scene_id):
                warnings.warn(
                    "MeshPhongAppearance requires lights in the scene. "
                    "Pass lighting='default' to the Scene model, otherwise "
                    "the mesh will render black.",
                    stacklevel=3,
                )
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXMeshMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    def _add_label_memory_visual(
        self,
        scene_id: UUID,
        visual_model: LabelMemoryVisual,
    ) -> LabelMemoryVisual:
        """Wire and register a pre-built LabelMemoryVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        gfx_visual = GFXLabelMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
            transform=visual_model.transform,
        )
        self._register_visual(
            scene_id, visual_model, gfx_visual, data_store, displayed_axes
        )
        return visual_model

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------
    #
    # Two categories share one registry, one bridge and one event pair:
    #
    # * canvas overlays (``Canvas.overlays``) draw in screen space as a
    #   post-pass on one canvas, with their own camera;
    # * scene overlays (``Scene.overlays``) draw in the scene's world, in the
    #   main pass, by the scene camera.  Their geometry depends on the scene's
    #   contents, so ``_refresh_scene_overlays`` rebuilds it whenever those
    #   change (see ``plans/scene_overlay_implementation.md``).
    #
    # The controller constructs the GFX objects; the render manager is a
    # passive registrar, as for visuals.

    def _build_gfx_canvas_overlay(
        self,
        canvas_id: UUID,
        overlay_model: CanvasOverlay,
    ) -> GFXCanvasOverlay:
        """Construct the render-layer overlay for a canvas overlay model.

        Raises
        ------
        TypeError
            If *overlay_model* has an unrecognised type.
        """
        canvas_view = self._render_manager._canvases[canvas_id]
        if isinstance(overlay_model, CenteredAxes2D):
            return GFXCenteredAxes2D(
                model=overlay_model,
                camera=canvas_view.camera,
            )
        raise TypeError(
            f"Unrecognised canvas overlay type {type(overlay_model)!r}. "
            "Register a handler in _build_gfx_canvas_overlay."
        )

    def _build_gfx_scene_overlay(self, overlay_model: SceneOverlay) -> GFXSceneOverlay:
        """Construct the render-layer overlay for a scene overlay model.

        Raises
        ------
        TypeError
            If *overlay_model* has an unrecognised type.
        """
        if isinstance(overlay_model, SceneBoundingBox):
            return GFXSceneBoundingBox(overlay_model)
        raise TypeError(
            f"Unrecognised scene overlay type {type(overlay_model)!r}. "
            "Register a handler in _build_gfx_scene_overlay."
        )

    def add_canvas_overlay(
        self,
        canvas_id: UUID,
        overlay: CanvasOverlay,
    ) -> CanvasOverlay:
        """Attach a screen-space overlay to a specific canvas.

        The overlay is rendered as a post-pass on top of the main scene each
        frame.  It does not participate in reslicing, has no world-space
        transform, and is not added to ``scene.visuals``.  It is stored in the
        ``Canvas.overlays`` list of *canvas_id*, making it part of the
        serializable model, and its fields are live: assign to them directly
        or through :meth:`update_overlay_field`.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that should display the overlay.  Use
            :meth:`get_canvas_ids` to look up canvas IDs for a scene.
        overlay : CanvasOverlay
            Model-layer overlay description, e.g. a
            :class:`~cellier.visuals.CenteredAxes2D`.

        Returns
        -------
        CanvasOverlay
            The same overlay object passed in (for ID access or chaining).

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        ValueError
            If an overlay with the same id is already registered.
        """
        scene_id = self._canvas_to_scene[canvas_id]
        self._check_new_overlay(overlay)
        canvas_model = self._model.scenes[scene_id].canvases[canvas_id]
        canvas_model.overlays.append(overlay)
        self._register_canvas_overlay(canvas_id, overlay)
        self._request_draw_for_scene(scene_id)
        return overlay

    def add_scene_overlay(
        self,
        scene_id: UUID,
        overlay: SceneOverlay,
    ) -> SceneOverlay:
        """Attach a world-space overlay to a scene.

        The overlay is drawn in the scene's world by the scene camera, in the
        main pass, on every canvas showing the scene.  Its geometry follows
        the scene: it is rebuilt when visuals are added or removed, a
        transform is replaced, the displayed axes change, or a store changes
        (signalled by :meth:`reslice_visual`).  It is stored in
        ``Scene.overlays``, and its fields are live.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene that should hold the overlay.
        overlay : SceneOverlay
            Model-layer overlay description, e.g. a
            :class:`~cellier.visuals.SceneBoundingBox`.

        Returns
        -------
        SceneOverlay
            The same overlay object passed in.

        Raises
        ------
        KeyError
            If *scene_id* is not registered.
        ValueError
            If an overlay with the same id is already registered.
        NotImplementedError
            If the scene's selection is not axis aligned.
        """
        scene = self._model.scenes[scene_id]
        self._check_new_overlay(overlay)
        scene.overlays.append(overlay)
        self._register_scene_overlay(scene_id, overlay)
        return overlay

    def _check_new_overlay(self, overlay: CanvasOverlay | SceneOverlay) -> None:
        """Reject an overlay that is already registered.

        One model drives one render-layer object; registering it twice would
        leave two bridges writing to two nodes and ``remove_overlay`` able to
        find only one of them.
        """
        if overlay.id in self._overlays:
            raise ValueError(
                f"Overlay {overlay.name!r} (id={overlay.id}) is already "
                "registered.  Create a new overlay model instead."
            )

    def _register_canvas_overlay(self, canvas_id: UUID, overlay: CanvasOverlay) -> None:
        """Build, attach and bridge a canvas overlay already in its canvas model."""
        gfx_overlay = self._build_gfx_canvas_overlay(canvas_id, overlay)
        self._render_manager.add_canvas_overlay(canvas_id, gfx_overlay)
        entry = _OverlayEntry(
            model=overlay,
            gfx=gfx_overlay,
            kind="canvas",
            owner_id=canvas_id,
            scene_id=self._canvas_to_scene[canvas_id],
        )
        self._overlays[overlay.id] = entry
        self._wire_overlay(entry)

    def _register_scene_overlay(self, scene_id: UUID, overlay: SceneOverlay) -> None:
        """Build, attach, bridge and size a scene overlay already in its scene."""
        gfx_overlay = self._build_gfx_scene_overlay(overlay)
        self._render_manager.add_scene_overlay(scene_id, overlay.id, gfx_overlay)
        entry = _OverlayEntry(
            model=overlay,
            gfx=gfx_overlay,
            kind="scene",
            owner_id=scene_id,
            scene_id=scene_id,
        )
        self._overlays[overlay.id] = entry
        self._wire_overlay(entry)
        self._refresh_scene_overlays(scene_id, overlay_ids={overlay.id})
        self._request_draw_for_scene(scene_id)

    def _wire_overlay(self, entry: _OverlayEntry) -> None:
        """Bridge an overlay model's field changes to the render layer and bus.

        Two connections, as for the scene background: ``overlay.events`` for
        top-level fields (``visible``, an axis label, a wholesale
        ``appearance`` replacement), and ``overlay.appearance.events`` for
        appearance fields, since psygnal does not propagate a nested model's
        changes to its parent.  The second is moved when the appearance model
        is replaced.
        """
        handler = self._make_overlay_handler(entry.model.id)
        entry.model.events.connect(handler)
        entry.handlers.append((entry.model.events, handler))
        appearance = getattr(entry.model, "appearance", None)
        if appearance is not None:
            self._connect_overlay_appearance(entry, appearance)

    def _connect_overlay_appearance(
        self, entry: _OverlayEntry, appearance: Any
    ) -> None:
        """Attach the appearance bridge to *appearance*, detaching the old one."""
        if entry.appearance is not None and entry.appearance_handler is not None:
            entry.appearance.events.disconnect(entry.appearance_handler)
        handler = self._make_overlay_appearance_handler(entry.model.id)
        appearance.events.connect(handler)
        entry.appearance = appearance
        entry.appearance_handler = handler

    def _make_overlay_handler(self, overlay_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for an overlay's own fields."""

        def _on_overlay_psygnal(info: EmissionInfo) -> None:
            entry = self._overlays.get(overlay_id)
            if entry is None:
                return
            name = info.signal.name
            value = info.args[0]
            if name == "appearance":
                if value is entry.appearance:
                    return
                self._connect_overlay_appearance(entry, value)
            self._push_overlay_change(entry, name, value)

        return _on_overlay_psygnal

    def _make_overlay_appearance_handler(self, overlay_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for an overlay's appearance."""

        def _on_overlay_appearance_psygnal(info: EmissionInfo) -> None:
            entry = self._overlays.get(overlay_id)
            if entry is None:
                return
            self._push_overlay_change(
                entry, f"appearance.{info.signal.name}", info.args[0]
            )

        return _on_overlay_appearance_psygnal

    def _push_overlay_change(
        self, entry: _OverlayEntry, field_name: str, value: Any
    ) -> None:
        """Apply one overlay field change to the render layer and announce it."""
        entry.gfx.apply(field_name, value)
        if entry.kind == "scene" and field_name == "visible" and value:
            # A hidden scene overlay skips rebuilds; catch it up now.
            self._refresh_scene_overlays(entry.owner_id, overlay_ids={entry.model.id})
        resolved_source_id = _overlay_source_id_override.get() or self._id
        _SOURCE_ID_LOGGER.debug(
            "bridge  handler=_on_overlay_psygnal  overlay=%s  field=%s"
            "  resolved_source=%s  override_active=%s",
            entry.model.id,
            field_name,
            resolved_source_id,
            _overlay_source_id_override.get() is not None,
        )
        self._outgoing_events.emit(
            OverlayChangedEvent(
                source_id=resolved_source_id,
                overlay_id=entry.model.id,
                field_name=field_name,
                new_value=value,
            )
        )
        # Overlays are not visuals and never reslice, so nothing else on this
        # path asks for a frame.
        self._request_draw_for_scene(entry.scene_id)

    def _refresh_scene_overlays(
        self, scene_id: UUID, *, overlay_ids: set[UUID] | None = None
    ) -> None:
        """Rebuild the scene overlays of *scene_id* for its current contents.

        Computes the scene's world bounds once -- every visual, hidden ones
        included -- and hands them with the displayed axes to each visible
        scene overlay whose last input differs.  A hidden overlay is skipped
        and caught up when it is shown.

        Parameters
        ----------
        scene_id : UUID
            The scene whose overlays to rebuild.  An unregistered scene is
            ignored.
        overlay_ids : set[UUID] or None
            Restrict the rebuild to these overlays.  ``None`` means all.

        Raises
        ------
        NotImplementedError
            If the scene's selection is not axis aligned: the world ->
            rendered projection is then not a selection of axes.
        """
        scene = self._model.scenes.get(scene_id)
        if scene is None:
            return
        entries = [
            entry
            for entry in self._overlays.values()
            if entry.kind == "scene"
            and entry.owner_id == scene_id
            and entry.model.visible
            and (overlay_ids is None or entry.model.id in overlay_ids)
        ]
        if not entries:
            return
        selection = scene.dims.selection
        if not isinstance(selection, AxisAlignedSelection):
            raise NotImplementedError(
                f"Scene overlays need an axis-aligned selection; scene "
                f"{scene.name!r} uses {type(selection).__name__}."
            )
        displayed_axes = tuple(selection.displayed_axes)
        bounds = scene_world_bounds(scene, self.get_data_store)
        key = (
            displayed_axes,
            None if bounds is None else (bounds[0].tobytes(), bounds[1].tobytes()),
        )
        changed = False
        for entry in entries:
            if entry.extent_key == key:
                continue
            entry.gfx.update_scene_extent(bounds, displayed_axes)
            entry.extent_key = key
            changed = True
        if changed:
            self._request_draw_for_scene(scene_id)

    def get_overlay(self, overlay_id: UUID) -> CanvasOverlay | SceneOverlay:
        """Return the overlay model registered under *overlay_id*.

        Raises
        ------
        KeyError
            If no overlay with that id is registered.
        """
        return self._overlay_entry(overlay_id).model

    def _overlay_entry(self, overlay_id: UUID) -> _OverlayEntry:
        entry = self._overlays.get(overlay_id)
        if entry is None:
            raise KeyError(
                f"No overlay with id={overlay_id!r} found.  Add it with "
                "add_canvas_overlay or add_scene_overlay first."
            )
        return entry

    def remove_overlay(self, overlay_id: UUID) -> None:
        """Remove an overlay of either category.

        Disconnects its bridge, detaches it from the render layer, and removes
        it from ``Canvas.overlays`` / ``Scene.overlays``.

        Parameters
        ----------
        overlay_id : UUID
            ID of the overlay to remove.

        Raises
        ------
        KeyError
            If no overlay with that id is registered.
        """
        entry = self._overlay_entry(overlay_id)
        self._forget_overlay(overlay_id)
        if entry.kind == "scene":
            overlays = self._model.scenes[entry.owner_id].overlays
        else:
            overlays = (
                self._model.scenes[entry.scene_id].canvases[entry.owner_id].overlays
            )
        # By identity: model equality is not safe to rely on (a field
        # ``__eq__`` that raises degrades a model class to identity).
        overlays[:] = [overlay for overlay in overlays if overlay is not entry.model]
        self._request_draw_for_scene(entry.scene_id)

    def _forget_overlay(self, overlay_id: UUID) -> None:
        """Disconnect and detach one overlay, leaving its model list alone.

        Used by :meth:`remove_overlay` and by scene and canvas teardown, where
        the model list goes away with its owner.
        """
        entry = self._overlays.pop(overlay_id, None)
        if entry is None:
            return
        for signal, handler in entry.handlers:
            signal.disconnect(handler)
        if entry.appearance is not None and entry.appearance_handler is not None:
            entry.appearance.events.disconnect(entry.appearance_handler)
        if entry.kind == "scene":
            self._render_manager.remove_scene_overlay(entry.owner_id, overlay_id)
        else:
            self._render_manager.remove_canvas_overlay(entry.owner_id, entry.gfx)

    def update_overlay_field(
        self,
        overlay_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on an overlay model.

        Tags the emitted ``OverlayChangedEvent`` with *source_id*.  GUI
        widgets should pass ``source_id=self._id`` so their own subscription
        can ignore the echo.

        Parameters
        ----------
        overlay_id : UUID
            Target overlay, of either category.
        field : str
            Dotted path on the overlay model: ``"visible"``, or
            ``"appearance.color"`` for an appearance field.
        value : Any
            New value for the field.
        source_id : UUID or None
            UUID to stamp on the emitted event.  Defaults to the controller's
            own ID.

        Raises
        ------
        KeyError
            If no overlay with that id is registered.
        """
        target = self._overlay_entry(overlay_id).model
        *parents, name = field.split(".")
        for parent in parents:
            target = getattr(target, parent)
        token = _overlay_source_id_override.set(source_id)
        try:
            setattr(target, name, value)
        finally:
            _overlay_source_id_override.reset(token)

    def set_overlay_visible(
        self,
        overlay_id: UUID,
        visible: bool,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Show or hide an overlay of either category.

        Equivalent to ``update_overlay_field(overlay_id, "visible", visible)``.

        Parameters
        ----------
        overlay_id : UUID
            ID of the overlay.
        visible : bool
            ``True`` to show the overlay, ``False`` to hide it.
        source_id : UUID or None
            UUID to stamp on the emitted ``OverlayChangedEvent``.

        Raises
        ------
        KeyError
            If no overlay with ``overlay_id`` is registered.
        """
        self.update_overlay_field(
            overlay_id, "visible", bool(visible), source_id=source_id
        )

    def _on_overlay_update(self, event: OverlayUpdateEvent) -> None:
        self.update_overlay_field(
            event.overlay_id, event.field, event.value, source_id=event.source_id
        )

    # ------------------------------------------------------------------
    # Canvas management
    # ------------------------------------------------------------------

    def add_canvas(
        self,
        scene_id: UUID,
        render_modes: set[str] | None = None,
        initial_dim: str | None = None,
        fov: float = 70.0,
        depth_range_3d: tuple[float, float] = (1.0, 8000.0),
        depth_range_2d: tuple[float, float] = (-500.0, 500.0),
        canvas_size: tuple[int, int] | None = None,
    ) -> QWidget:
        """Create a canvas attached to a scene and return its embeddable widget.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        render_modes : set[str] or None
            Which camera modes to prepare on the canvas.  Each entry must be
            ``"2d"`` or ``"3d"``.  When ``None``, defaults to the scene's own
            ``render_modes``.  Pass ``{"2d", "3d"}`` for a canvas that can
            switch between views.
        initial_dim : str or None
            Which mode is active when the canvas first appears.  Must be a
            member of ``render_modes``.  When ``None``, inferred from the
            scene's current ``displayed_axes`` length (3 axes -> ``"3d"``,
            otherwise ``"2d"``).
        fov : float
            Vertical field of view in degrees for the 3D perspective camera.
            Ignored when ``"3d"`` is not in ``render_modes``.  Default ``70.0``.
        depth_range_3d : tuple[float, float]
            ``(near, far)`` clip distances for the 3D perspective camera.
            Default ``(1.0, 8000.0)``.
        depth_range_2d : tuple[float, float]
            ``(near, far)`` clip distances for the 2D orthographic camera.
            Default ``(-500.0, 500.0)``.
        canvas_size : tuple[int, int] or None
            Initial CSS pixel size for the anywidget canvas.  Ignored for the
            Qt gui (which is sized by its parent layout).  Defaults to
            ``(600, 600)`` when ``None`` for the anywidget gui.

        Returns
        -------
        QWidget
            The render widget.  Embed with ``layout.addWidget(widget)``.

        Raises
        ------
        ValueError
            If ``initial_dim`` is supplied but is not a member of
            ``render_modes``.
        """
        scene = self._model.scenes[scene_id]

        if render_modes is None:
            render_modes = scene.render_modes

        if initial_dim is None:
            initial_dim = (
                "3d" if len(scene.dims.selection.displayed_axes) == 3 else "2d"
            )

        if initial_dim not in render_modes:
            raise ValueError(
                f"initial_dim={initial_dim!r} is not in render_modes={render_modes!r}."
            )

        cameras: dict[str, CameraType] = {}
        if "3d" in render_modes:
            cameras["3d"] = PerspectiveCamera(
                fov=fov,
                near_clipping_plane=depth_range_3d[0],
                far_clipping_plane=depth_range_3d[1],
                controller=OrbitCameraController(enabled=True),
            )
        if "2d" in render_modes:
            cameras["2d"] = OrthographicCamera(
                near_clipping_plane=depth_range_2d[0],
                far_clipping_plane=depth_range_2d[1],
                controller=PanZoomCameraController(enabled=True),
            )

        canvas_model = Canvas(cameras=cameras)
        return self.add_canvas_model(
            scene_id, canvas_model, initial_dim=initial_dim, canvas_size=canvas_size
        )

    def add_canvas_model(
        self,
        scene_id: UUID,
        canvas_model: Canvas,
        initial_dim: str | None = None,
        canvas_size: tuple[int, int] | None = None,
    ) -> QWidget:
        """Register a pre-built Canvas model with a scene.

        Used by ``from_model`` to restore canvases from a serialized
        ViewerModel, and called internally by ``add_canvas``.  Camera state
        (position, rotation, fov, depth range) is read from the camera models
        stored in ``canvas_model.cameras``.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        canvas_model : Canvas
            Pre-built canvas model.  Must have at least one entry in
            ``canvas_model.cameras``.
        initial_dim : str or None
            Which dim to activate first.  When ``None``, the first key in
            ``canvas_model.cameras`` is used (insertion order is preserved by
            Python dicts, so this is deterministic for serialized models).
        canvas_size : tuple[int, int] or None
            Initial CSS pixel size for the anywidget canvas.  Ignored for the
            Qt gui.  Defaults to ``(600, 600)`` for the anywidget gui.

        Returns
        -------
        QWidget
            The render widget.

        Raises
        ------
        ValueError
            If ``canvas_model.cameras`` is empty, or if ``initial_dim`` is not
            a key in ``canvas_model.cameras``.
        """
        if not canvas_model.cameras:
            raise ValueError("canvas_model.cameras must not be empty.")

        if initial_dim is None:
            initial_dim = next(iter(canvas_model.cameras))

        if initial_dim not in canvas_model.cameras:
            raise ValueError(
                f"initial_dim={initial_dim!r} is not a key in "
                f"canvas_model.cameras ({list(canvas_model.cameras)!r})."
            )

        active_camera = canvas_model.cameras[initial_dim]
        depth_range = (
            active_camera.near_clipping_plane,
            active_camera.far_clipping_plane,
        )

        # Store the model using its existing id.
        self._model.scenes[scene_id].canvases[canvas_model.id] = canvas_model

        # Use canvas_model.id as the single canonical ID for both layers.
        if isinstance(active_camera, PerspectiveCamera):
            canvas_view = self._render_manager.add_canvas(
                canvas_model.id,
                scene_id,
                parent=self._widget_parent,
                dim=initial_dim,
                fov=active_camera.fov,
                depth_range=depth_range,
                gui=self._gui,
                size=canvas_size,
            )
        else:
            # OrthographicCamera: fov is not meaningful; omit it so CanvasView
            # uses its own internal default for the 3D camera it keeps in reserve.
            canvas_view = self._render_manager.add_canvas(
                canvas_model.id,
                scene_id,
                parent=self._widget_parent,
                dim=initial_dim,
                depth_range=depth_range,
                gui=self._gui,
                size=canvas_size,
            )

        canvas_view.set_event_bus(self._outgoing_events)

        # Fix the reserve (non-active) camera's depth range.  CanvasView builds
        # both the 2D and 3D cameras up front but applies only the active
        # camera's range to both; the reserve camera (e.g. the 3D perspective
        # camera when starting in 2D) is then left with the active camera's
        # range, which may have an invalid near plane and render nothing on the
        # first toggle.  Only the reserve camera is touched here: re-setting the
        # active camera's range would change its captured state and spuriously
        # emit a CameraChangedEvent (scheduling an unwanted settle reslice).
        for dim_key, camera_model in canvas_model.cameras.items():
            if dim_key == initial_dim:
                continue
            canvas_view.set_depth_range_for_dim(
                dim_key,
                (
                    camera_model.near_clipping_plane,
                    camera_model.far_clipping_plane,
                ),
            )

        self._canvas_to_scene[canvas_model.id] = scene_id
        self._scene_to_canvases[scene_id].append(canvas_model.id)

        # Wire any overlays already stored on this canvas model to the render
        # layer.  For canvases created via add_canvas() this loop is a no-op
        # (overlays=[]).  For canvases restored from a serialized ViewerModel
        # the overlays list is already populated, so this call is sufficient —
        # from_model needs no additional overlay-restoration step.
        # _register_canvas_overlay is called directly rather than
        # add_canvas_overlay to avoid re-appending models that are already in
        # canvas_model.overlays.
        for overlay_model in canvas_model.overlays:
            self._register_canvas_overlay(canvas_model.id, overlay_model)
        # The rendered system is derived from (world, displayed_axes,
        # canvas_id) rather than stored on the Canvas: storing it would be a
        # second source of truth for displayed_axes (design 3.1, D9).
        rendered, embedding = self._build_rendered(scene_id, canvas_model.id)
        self._rendered[canvas_model.id] = (rendered, embedding)
        self._register_coordinate_systems(rendered)
        # The first canvas is what makes a rendered system exist, so visuals
        # added before it could not be placed.  They can be now.
        self._push_render_spaces(scene_id)
        self._outgoing_events.emit(
            CanvasAddedEvent(
                source_id=self._id, scene_id=scene_id, canvas_id=canvas_model.id
            )
        )

        return canvas_view.widget

    # ------------------------------------------------------------------
    # Scene and visual lookup
    # ------------------------------------------------------------------

    def get_scene(self, scene_id: UUID) -> Scene:
        """Return the live Scene model for scene_id."""
        return self._model.scenes[scene_id]

    def get_data_store(self, store_id: UUID) -> BaseDataStore:
        """Return the registered data store for store_id.

        Parameters
        ----------
        store_id : UUID
            ID of a previously registered data store.

        Returns
        -------
        BaseDataStore

        Raises
        ------
        KeyError
            If no store with store_id has been registered.
        """
        return self._model.data.stores[store_id]

    def fit_camera(self, scene_id: UUID, canvas_id: UUID | None = None) -> None:
        """Fit the camera to the current scene bounding box.

        Safe to call immediately after ``add_image`` / ``add_image_multiscale``
        and transform assignment — the node matrix is set at construction time
        so no chunk data needs to be loaded first.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose camera should be fitted.
        canvas_id : UUID or None
            If provided, fit only that canvas.  When ``None`` (default),
            all canvases attached to *scene_id* are fitted.
        """
        gfx_scene = self._render_manager.get_scene(scene_id)
        targets = (
            [canvas_id]
            if canvas_id is not None
            else list(self._scene_to_canvases.get(scene_id, []))
        )
        for cid in targets:
            canvas = self._render_manager._canvases.get(cid)
            if canvas is not None:
                canvas.show_object(gfx_scene)
                state = canvas.capture_camera_state()
                self._update_camera_model(scene_id, cid, state)
        # A fit already walked the scene graph, so this is the cheapest
        # moment to re-derive the ambient occlusion radius from the scene
        # bounding box.
        self._render_manager.update_ssao_radius(scene_id)

    def get_scene_by_name(self, name: str) -> Scene:
        """Return the live Scene model for the given name.

        Raises KeyError if no scene with that name exists.
        """
        for scene in self._model.scenes.values():
            if scene.name == name:
                return scene
        raise KeyError(f"No scene named {name!r}")

    def get_canvas_ids(self, scene_id: UUID) -> list[UUID]:
        """Return the IDs of all canvases registered for *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.

        Returns
        -------
        list[UUID]
            Canvas IDs in registration order.  Empty if no canvases have
            been added yet.
        """
        return list(self._scene_to_canvases.get(scene_id, []))

    @property
    def canvas_ids(self) -> tuple[UUID, ...]:
        """IDs of every canvas registered with this controller, across scenes.

        Scene-scoped callers want :meth:`get_canvas_ids`; this is for code that
        must find canvases without knowing which scene they belong to -- the Qt
        window composite in
        :func:`~cellier.convenience.screenshot_window` walks this to discover
        which canvases live inside a given window.
        """
        return tuple(self._render_manager._canvases)

    def get_canvas_view(self, canvas_id: UUID) -> CanvasView:
        """Return the render-layer ``CanvasView`` for *canvas_id*.

        Provides access to the rendering backend (camera, widget, overlays)
        for a canvas registered via :meth:`add_canvas` or
        :meth:`add_canvas_model`.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.

        Returns
        -------
        CanvasView

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        return self._render_manager._canvases[canvas_id]

    def get_camera_state(self, canvas_id: UUID) -> CameraState:
        """Return a snapshot of the current camera state for *canvas_id*.

        Useful for seeding downstream widgets (e.g. orientation overlays)
        with the post-fit camera state without constructing a synthetic
        ``CameraChangedEvent``.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.

        Returns
        -------
        CameraState

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        return self.get_canvas_view(canvas_id).capture_camera_state()

    def screenshot(
        self,
        canvas_id: UUID,
        size: tuple[int, int] | None = None,
        scale: float = 1.0,
        *,
        frames: int | Literal["converged"] = 1,
        **capture_kwargs,
    ) -> np.ndarray:
        """Capture a reproducible screenshot as an RGBA uint8 array.

        The frame is rendered on a **dedicated offscreen canvas** built on the
        same scene, not read back from the canvas on screen.  Two captures of
        the same viewer state therefore produce byte-identical arrays (on the
        same machine and GPU driver), at exactly the size asked for, whether
        or not anything is on screen and whichever GUI toolkit is in use.

        *canvas_id* selects a **viewpoint, not a surface**: the capture copies
        that canvas's camera, dimensionality and depth range, then renders its
        own frame.  Use :meth:`screenshot_scene` to capture a scene that has
        no canvas at all.

        What the capture shows is the data **currently resident on the GPU**.
        It does not reslice, so a multiscale scene is captured at the level of
        detail already loaded; a higher *scale* enlarges that level rather
        than fetching a finer one.  Call :meth:`on_scene_ready` (or use the
        convenience launchers' ``on_ready``) before capturing if a load may
        still be in flight.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas, whose viewpoint the capture copies.
        size : tuple[int, int] or None
            Target ``(width, height)`` in pixels before *scale*.  Defaults to
            the canvas's physical size, so an unqualified call reproduces the
            on-screen framing.
        scale : float
            Multiplier applied to *size*.  ``scale=2`` doubles the output
            resolution.
        frames : int or "converged"
            ``1`` (default) disables temporal accumulation and draws a single
            frame.  ``"converged"`` enables it and draws the number of frames
            the accumulator needs to settle (44 at the default blend weight)
            -- what you want whenever ambient occlusion is enabled, since a
            single-sample AO frame is visibly noisy.  ``N`` draws exactly N
            accumulated frames.
        **capture_kwargs
            Forwarded to the capture helper (``max_frames``, ``residual``).

        Returns
        -------
        np.ndarray
            RGBA uint8 array of shape ``(height, width, 4)``.

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        RuntimeError
            If ``frames="converged"`` would need more than ``max_frames``
            frames to settle.
        """
        return capture_scene(
            self._render_manager,
            self._canvas_to_scene[canvas_id],
            seed_canvas_id=canvas_id,
            size=size,
            scale=scale,
            frames=frames,
            **capture_kwargs,
        )

    def screenshot_scene(
        self,
        scene_id: UUID,
        size: tuple[int, int] | None = None,
        scale: float = 1.0,
        *,
        frames: int | Literal["converged"] = 1,
        dim: str | None = None,
        **capture_kwargs,
    ) -> np.ndarray:
        """Capture *scene_id* without needing a canvas to exist.

        The same offscreen capture as :meth:`screenshot`, but with the camera
        fitted to the scene rather than copied from a canvas -- so a scene can
        be captured with no window, no widget and no event loop.

        **A scene with no canvas has no data.**  Slice requests are planned
        per canvas, from its camera, size and frustum, so ``reslice_all`` on a
        scene with no canvas requests nothing and this returns a correct
        picture of an empty scene.  Add a canvas (``add_canvas``) and let the
        reslice complete before capturing; ``cellier.convenience.capture`` does exactly
        that.  This method's own fit is for the case where a canvas exists but
        its viewpoint is not the one you want.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to render.
        size : tuple[int, int] or None
            Target ``(width, height)`` before *scale*.  Defaults to
            ``(600, 600)``.
        scale : float
            Multiplier applied to *size*.
        frames : int or "converged"
            See :meth:`screenshot`.
        dim : str or None
            ``"2d"`` or ``"3d"``.  Inferred from the scene's displayed axes
            when ``None``.
        **capture_kwargs
            Forwarded to the capture helper.

        Returns
        -------
        np.ndarray
            RGBA uint8 array of shape ``(height, width, 4)``.
        """
        if dim is None:
            scene = self._model.scenes[scene_id]
            dim = "3d" if len(scene.dims.selection.displayed_axes) == 3 else "2d"
        return capture_scene(
            self._render_manager,
            scene_id,
            seed_canvas_id=None,
            size=size,
            scale=scale,
            frames=frames,
            dim=dim,
            **capture_kwargs,
        )

    def get_visual_model(self, visual_id: UUID) -> MultiscaleImageVisual:
        """Return the live visual model for visual_id.

        Searches all scenes.  Raises KeyError if not found.
        """
        for scene in self._model.scenes.values():
            for visual in scene.visuals:
                if visual.id == visual_id:
                    return visual
        raise KeyError(f"No visual with id {visual_id}")

    # ------------------------------------------------------------------
    # Reslicing
    # ------------------------------------------------------------------

    def _build_visual_configs_for_scene(
        self, scene_id: UUID
    ) -> dict[UUID, VisualRenderConfig]:
        """Build a VisualRenderConfig dict from all visuals in a scene."""
        scene = self._model.scenes[scene_id]
        configs: dict[UUID, VisualRenderConfig] = {}
        for visual in scene.visuals:
            configs[visual.id] = _visual_render_config(visual)
        return configs

    # ------------------------------------------------------------------
    # Coordinate systems: the registry, and the runtime systems
    # ------------------------------------------------------------------

    def coordinate_system(self, system_id: UUID) -> CoordinateSystemType:
        """Return the coordinate system with *system_id*.

        Three ``transform`` methods -- ``map_bounding_box``, ``then`` and
        ``validate_against`` -- take coordinate system **objects** while a
        transform stores only their ids, so composing anything needs this
        lookup.  It is a plain dict: no edges, no path finding and no
        automatic composition (D15).

        Parameters
        ----------
        system_id : UUID
            The system's id.

        Returns
        -------
        CoordinateSystemType
            The registered system.

        Raises
        ------
        KeyError
            If no system with that id is registered.  A stored transform
            naming an unregistered system usually means it outlived the scene
            or store that owned its endpoint.
        """
        try:
            return self._coordinate_systems[system_id]
        except KeyError:
            raise KeyError(
                f"No coordinate system {system_id} is registered.  It belongs "
                f"to a scene, data store, canvas or visual that is not in "
                f"this viewer."
            ) from None

    def _register_coordinate_systems(self, *systems: CoordinateSystemType) -> None:
        """Add systems to the registry, keyed by id."""
        for system in systems:
            self._coordinate_systems[system.id] = system

    def _forget_coordinate_systems(self, *systems: CoordinateSystemType) -> None:
        """Drop systems from the registry."""
        for system in systems:
            self._coordinate_systems.pop(system.id, None)

    def _ensure_data_coordinate_systems(
        self,
        scene_id: UUID,
        data_store: Any,
        visual_model: Any = None,
        channel_axis: int | None = None,
    ) -> None:
        """Give *data_store* coordinate systems if it does not have its own.

        A store that can say what its axes are -- an OME-Zarr reader, or any
        store constructed with ``data_coordinate_systems=`` -- already carries
        them, and this is a no-op.  A bare ``ImageMemoryStore(data=arr)``
        cannot say, so it takes the trailing axes of the scene's world: their
        names, types and units, with fresh ids.

        That is not the silent default D3 forbids.  The world was declared
        explicitly by the caller, axis types included; inheriting from it is
        what makes the ``data -> world`` transform typecheck by construction
        rather than by luck.
        """
        if data_store.data_coordinate_systems:
            install_level_transforms(data_store)
            self._register_coordinate_systems(*data_store.data_coordinate_systems)
            return
        ndim = getattr(data_store, "ndim", None)
        if ndim is None:
            return
        world = self._model.scenes[scene_id].dims.world_coordinate_system
        # A store wider than its world has axes the world does not model at
        # all, and for a multichannel visual the widest one is the channel
        # axis -- which the visual is the only object to know about.  When the
        # world does have room for every data axis, it says what they are and
        # nothing is declared: a world that already carries a channel axis
        # must be the one the store's channel axis maps to.
        declared: dict[int, tuple[str, str]] = {}
        if channel_axis is None:
            channel_axis = getattr(visual_model, "channel_axis", None)
        if channel_axis is not None and int(ndim) > world.ndim:
            declared[int(channel_axis)] = ("c", "channel")
        systems = level_coordinate_systems(
            data_store.id,
            data_axes_from_world(world, int(ndim), declared),
            int(getattr(data_store, "n_levels", 1)),
            data_store.name,
        )
        data_store.data_coordinate_systems = systems
        # The pyramid's numbers are the store's; the systems are what makes
        # them transforms, and they only exist now.
        install_level_transforms(data_store)
        self._register_coordinate_systems(*systems)

    def _prepare_transform(
        self,
        scene_id: UUID,
        data_store: Any,
        transform: Any,
        channel_axis: int | None = None,
    ) -> AffineTransform | None:
        """Resolve what a visual's ``data -> world`` transform should be.

        Called by every ``add_*`` before the visual model is built: it
        installs the store's coordinate systems, supplies the default when no
        transform was given, and checks that a supplied one names the store's
        and the scene's own systems rather than a look-alike pair.

        Parameters
        ----------
        scene_id : UUID
            The scene the visual is going into.
        data_store : Any
            The store it reads from.
        transform : Any
            A transform (passed through after its endpoints are checked), or
            ``None`` (the identity between the two systems).
        channel_axis : int or None
            For a multichannel visual, the data axis it composites.

        Returns
        -------
        AffineTransform or None
            ``None`` only when the store cannot say what its axes are, in
            which case nothing downstream can place it either.
        """
        self._ensure_data_coordinate_systems(
            scene_id, data_store, channel_axis=channel_axis
        )
        if not data_store.data_coordinate_systems:
            return transform
        world = self._model.scenes[scene_id].dims.world_coordinate_system
        level_zero = data_store.data_coordinate_systems[0]
        if transform is None:
            return default_data_to_world(level_zero, world)
        if transform.input_coordinate_system != level_zero.id:
            raise ValueError(
                f"This transform maps out of coordinate system "
                f"{transform.input_coordinate_system}, but the store's level-0 "
                f"system is {level_zero.id} ('{level_zero.name}').  Build it "
                f"against the store's own system -- a v2 transform names its "
                f"endpoints, so one built elsewhere describes a different space."
            )
        if transform.output_coordinate_system != world.id:
            raise ValueError(
                f"This transform maps into coordinate system "
                f"{transform.output_coordinate_system}, but the scene's world "
                f"is {world.id} ('{world.name}').  Build it against the "
                f"scene's own world."
            )
        return transform

    def _build_rendered(
        self, scene_id: UUID, canvas_id: UUID
    ) -> tuple[RenderedCoordinateSystem, AffineTransform]:
        """Build one canvas's rendered system and its embedding into the world.

        The rendered system is built in **cellier displayed order** (Part 5,
        D1): for a ``TZYX`` world displayed as ``ZYX`` its axes are
        ``("Z", "Y", "X")``, so ``displayed_axes``, the ``slice_indices``
        keys, the GUI sliders and ``axis_names()`` all agree.  The
        ``(z, y, x) -> (x, y, z)`` reversal is not carried here; it stays at
        the pygfx boundary.

        The embedding's linear block is a selection matrix, and every world
        axis the canvas does not display is a ``constant_output_axes`` entry
        -- never a broadcast one.  A sliced axis sits at its slice position.
        """
        scene = self._model.scenes[scene_id]
        world = scene.dims.world_coordinate_system
        selection = scene.dims.selection
        displayed_axes = tuple(selection.displayed_axes)
        rendered = RenderedCoordinateSystem.from_world(
            world,
            [world.axes[axis].id for axis in displayed_axes],
            canvas_id,
        )
        constant: dict[Any, float] = {}
        for axis in range(world.ndim):
            if axis in displayed_axes:
                continue
            constant[world.axes[axis].id] = float(
                getattr(selection, "slice_indices", {}).get(axis, 0.0)
            )
        embedding = AffineTransform.from_axis_map(
            rendered,
            world,
            axis_map={
                rendered.axes[index].id: world.axes[axis].id
                for index, axis in enumerate(displayed_axes)
            },
            constant_output_axes=constant,
            name="rendered_to_world",
        )
        return rendered, embedding

    def _rebuild_rendered(self, scene_id: UUID) -> None:
        """Rebuild the rendered system and embedding for every canvas on a scene.

        The trigger is a ``displayed_axes`` change, a **pure reorder
        included**: a transpose is a different rendered system with a
        different axis order, even though it fetches identical data.
        """
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            previous = self._rendered.get(canvas_id)
            if previous is not None:
                self._forget_coordinate_systems(previous[0])
            rendered, embedding = self._build_rendered(scene_id, canvas_id)
            self._rendered[canvas_id] = (rendered, embedding)
            self._register_coordinate_systems(rendered)

    def _rebuild_rendered_embedding(self, scene_id: UUID) -> None:
        """Refresh only the ``rendered -> world`` half after a slice move.

        The rendered system itself is unchanged -- same axes, same ids -- so
        it is reused rather than rebuilt, which is what keeps axis ids stable
        across a slider drag.
        """
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            entry = self._rendered.get(canvas_id)
            if entry is None:
                continue
            _, embedding = self._build_rendered(scene_id, canvas_id)
            self._rendered[canvas_id] = (entry[0], embedding)

    def _forget_rendered(self, canvas_id: UUID) -> None:
        """Drop a canvas's rendered system."""
        entry = self._rendered.pop(canvas_id, None)
        if entry is not None:
            self._forget_coordinate_systems(entry[0])

    def _retained_data_axes(
        self, visual_id: UUID, level_zero: Any, scene: Any
    ) -> list[int]:
        """The data axes a visual's geometry keeps, ascending (design 3.14).

        Which data axis a displayed **world** axis names is a question only the
        visual's ``data -> world`` transform can answer.  Subtracting a
        trailing-alignment offset gets the same answer whenever the transform
        preserves axis order, which is nearly always -- and gets a wrong one,
        or an out-of-range one, when it does not: a ``zyx`` store broadcast
        into a ``czyx`` world would be asked for data axis 3, and a store whose
        transform permutes its axes would hand over the wrong columns without
        complaint.  So the correspondence is read off the matrix, and the
        offset is only the fallback for a visual with no transform yet.

        Ascending data-axis order, never ``displayed_axes`` order:
        ``axis_selections`` is assembled per data axis ascending and numpy
        returns an array whose axes are ascending, so a display permutation
        lives in the transform and never in the data.
        """
        displayed = getattr(scene.dims.selection, "displayed_axes", ())
        visual = self._model_visual_or_none(visual_id)
        transform = getattr(visual, "transform", None)
        if transform is not None:
            data_axis_of_world = {
                world_axis: data_axis
                for data_axis, world_axis in axis_correspondence(transform).items()
            }
            return sorted(
                data_axis_of_world[axis]
                for axis in displayed
                if axis in data_axis_of_world
            )
        offset = scene.dims.world_coordinate_system.ndim - level_zero.ndim
        return sorted(
            axis - offset for axis in displayed if 0 <= axis - offset < level_zero.ndim
        )

    def _rebuild_visual_space(
        self, visual_id: UUID, render_mode: str, data_store: Any
    ) -> None:
        """Build the space one visual's geometry is uploaded in, for one mode.

        See :meth:`_retained_data_axes` for why the axis correspondence is read
        off the transform rather than assumed positional.
        """
        if not data_store.data_coordinate_systems:
            return
        level_zero = data_store.data_coordinate_systems[0]
        scene_id = self._visual_to_scene.get(visual_id)
        if scene_id is None:
            return
        retained = self._retained_data_axes(
            visual_id, level_zero, self._model.scenes[scene_id]
        )
        if not retained:
            return
        key = (visual_id, render_mode)
        previous = self._visual_spaces.get(key)
        if previous is not None:
            self._forget_coordinate_systems(previous)
        space = VisualCoordinateSystem.from_data(
            level_zero,
            [level_zero.axes[axis].id for axis in retained],
            visual_id,
            name=f"visual_{render_mode}",
        )
        self._visual_spaces[key] = space
        self._register_coordinate_systems(space)

    def render_spaces(self, visual_id: UUID) -> RenderSpaces | None:
        """The systems a visual's render-layer counterpart places geometry with.

        ``None`` when the visual is not placeable yet -- its store has no
        coordinate systems, or the scene has no canvas and so no rendered
        system.
        """
        scene_id = self._visual_to_scene.get(visual_id)
        if scene_id is None:
            return None
        visual_model = self._get_visual_model(visual_id)
        store = self._model.data.stores.get(UUID(visual_model.data_store_id))
        if store is None or not store.data_coordinate_systems:
            return None
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        space = self._visual_spaces.get((visual_id, _render_mode_for(displayed_axes)))
        if space is None:
            return None
        rendered = self._scene_rendered(scene_id)
        if rendered is None:
            return None
        rendered_cs, rendered_to_world = rendered
        level_zero = store.data_coordinate_systems[0]
        retained = self._retained_data_axes(visual_id, level_zero, scene)
        return build_render_spaces(
            level_zero,
            space,
            scene.dims.world_coordinate_system,
            rendered_cs,
            rendered_to_world,
            visual_model.transform,
            retained,
            data_levels=store.data_coordinate_systems,
            level_transforms=store_level_transforms(store),
        )

    def _scene_rendered(
        self, scene_id: UUID
    ) -> tuple[RenderedCoordinateSystem, AffineTransform] | None:
        """The rendered system node matrices on this scene are expressed in.

        A node matrix lives on a pygfx node, and there is one pygfx scene per
        cellier scene shared by every canvas showing it.  Those canvases all
        display the same axes, so their rendered systems differ only by id and
        any one of them yields the same matrix; the first is used.

        ``None`` when the scene has no canvas yet, which is also when nothing
        needs placing.
        """
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            entry = self._rendered.get(canvas_id)
            if entry is not None:
                return entry
        return None

    def _push_render_spaces(self, scene_id: UUID) -> None:
        """Hand every visual on a scene its rebuilt systems."""
        scene_manager = self._render_manager._scenes.get(scene_id)
        if scene_manager is None:
            return
        for visual_model in self._model.scenes[scene_id].visuals:
            gfx_visual = scene_manager.get_visual(visual_model.id)
            setter = getattr(gfx_visual, "set_render_spaces", None)
            if setter is None:
                continue
            setter(self.render_spaces(visual_model.id))

    def _rebuild_visual_spaces(self, scene_id: UUID) -> None:
        """Rebuild every visual space on a scene after a displayed_axes change.

        Only the mode the scene is actually in.  There is one
        ``displayed_axes`` per scene, so the *other* mode's axes are not known
        here -- the 2D and 3D nodes of one visual are genuinely different
        spaces (D45), and inventing the idle one would put a wrong answer in
        the cache rather than no answer.  Its entry is built when the scene
        switches into it, which is this same trigger.
        """
        displayed = self._model.scenes[scene_id].dims.selection.displayed_axes
        mode = _render_mode_for(displayed)
        for visual_model in self._model.scenes[scene_id].visuals:
            store = self._model.data.stores.get(UUID(visual_model.data_store_id))
            if store is None:
                continue
            self._rebuild_visual_space(visual_model.id, mode, store)

    def _forget_visual_spaces(self, visual_id: UUID) -> None:
        """Drop every render mode's visual space for one visual."""
        for key in [key for key in self._visual_spaces if key[0] == visual_id]:
            self._forget_coordinate_systems(self._visual_spaces.pop(key))

    @staticmethod
    def _slice_signature(selection: Any) -> tuple:
        """A comparable snapshot of where the slice sits and how thick it is.

        Compared rather than the whole selection because ``displayed_axes``
        has its own, coarser, invalidation: this one rebuilds only the
        embedding.  Only the **sliced** axes count: a displayed axis keeps a
        stored position (D36) that nothing is sliced at.
        """
        displayed = set(getattr(selection, "displayed_axes", ()))
        return (
            tuple(
                sorted(
                    (axis, position)
                    for axis, position in getattr(
                        selection, "slice_indices", {}
                    ).items()
                    if axis not in displayed
                )
            ),
            tuple(
                sorted(
                    (axis, half)
                    for axis, half in getattr(selection, "thickness", {}).items()
                    if axis not in displayed
                )
            ),
        )

    def _dims_state_for_scene(self, scene_id: UUID) -> DimsState:
        """Derive a DimsState from the scene's DimsManager."""
        return self._model.scenes[scene_id].dims.to_state()

    # ------------------------------------------------------------------
    # psygnal bridges
    # ------------------------------------------------------------------

    def _wire_dims_model(self, scene: Scene) -> None:
        """Subscribe to all field changes on a scene's DimsManager."""
        self._dims_cache[scene.id] = scene.dims.selection.displayed_axes
        self._slice_cache[scene.id] = self._slice_signature(scene.dims.selection)
        self._register_coordinate_systems(scene.dims.world_coordinate_system)
        handler = self._make_dims_handler(scene.id)
        scene.dims.events.connect(handler)
        self._scene_psygnal_handlers.setdefault(scene.id, []).append(
            (scene.dims.events, handler)
        )
        # ``Scene.slider_axes`` is derived, so its change detection hangs off
        # every model signal that can move it.  A whole-list reassignment of
        # ``scene.visuals`` arrives here; the controller's own add and remove
        # mutate the list in place, which emits nothing, so they call
        # ``_check_slider_axes`` themselves.
        self._slider_axes_cache[scene.id] = scene.slider_axes
        visuals_handler = self._make_slider_axes_check(scene.id)
        scene.events.visuals.connect(visuals_handler)
        self._scene_psygnal_handlers[scene.id].append(
            (scene.events.visuals, visuals_handler)
        )

    def _wire_scene_background(self, scene: Scene) -> None:
        """Bridge a scene's background model to the render layer and the bus.

        Two connections, for the same reason ``_wire_trail`` needs two:

        1. ``scene.background.events`` for per-field changes.  psygnal does
           not propagate a nested ``EventedModel``'s field changes to the
           parent's event group, which is also why ``_wire_dims_model``
           connects to ``scene.dims.events``.
        2. ``scene.events.background`` for wholesale replacement
           (``scene.background = BackgroundAppearance(...)``), which has to
           move connection 1 onto the new object.
        """
        self._connect_background_bridge(scene.id, scene.background)
        assigned_handler = self._make_background_assigned_handler(scene.id)
        scene.events.background.connect(assigned_handler)
        self._scene_psygnal_handlers.setdefault(scene.id, []).append(
            (scene.events.background, assigned_handler)
        )

    def _connect_background_bridge(
        self, scene_id: UUID, background: BackgroundAppearance
    ) -> None:
        """Attach the per-field background bridge for *scene_id* to *background*."""
        handler = self._make_background_handler(scene_id)
        background.events.connect(handler)
        self._scene_background_bridges[scene_id] = (background, handler)
        self._scene_psygnal_handlers.setdefault(scene_id, []).append(
            (background.events, handler)
        )

    def _make_background_assigned_handler(self, scene_id: UUID) -> Callable:
        """Return a handler that moves the bridge onto a replaced background model.

        ``scene.events.background`` also fires for nested field changes (the
        ``Scene`` relay re-emits it), so the identity check against the object
        the bridge is attached to is what distinguishes an actual replacement
        -- and what keeps this from duplicating the per-field path.
        """

        def _on_background_assigned(background: BackgroundAppearance) -> None:
            wired, handler = self._scene_background_bridges[scene_id]
            if background is wired:
                return
            wired.events.disconnect(handler)
            self._scene_psygnal_handlers[scene_id].remove((wired.events, handler))
            self._connect_background_bridge(scene_id, background)
            self._push_background(scene_id, background, field_name=None, new_value=None)

        return _on_background_assigned

    def _make_background_handler(self, scene_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a scene's background model."""

        def _on_background_psygnal(info: EmissionInfo) -> None:
            self._push_background(
                scene_id,
                self._model.scenes[scene_id].background,
                field_name=info.signal.name,
                new_value=info.args[0],
            )

        return _on_background_psygnal

    def _push_background(
        self,
        scene_id: UUID,
        background: BackgroundAppearance,
        *,
        field_name: str | None,
        new_value: Any,
    ) -> None:
        """Apply *background* to the render layer and announce the change.

        The whole model goes to the render layer -- the background modes take
        different numbers of colors, so a per-field push would have to
        reconstruct the rest anyway -- while the bus event also carries the
        delta so widgets can echo-filter and update one control.
        """
        self._render_manager.set_scene_background(scene_id, background)
        resolved_source_id = _background_source_id_override.get() or self._id
        _SOURCE_ID_LOGGER.debug(
            "bridge  handler=_on_background_psygnal  scene=%s  field=%s"
            "  resolved_source=%s  override_active=%s",
            scene_id,
            field_name,
            resolved_source_id,
            _background_source_id_override.get() is not None,
        )
        self._outgoing_events.emit(
            BackgroundChangedEvent(
                source_id=resolved_source_id,
                scene_id=scene_id,
                background=background,
                field_name=field_name,
                new_value=new_value,
            )
        )
        # The background is not a visual and does not reslice, so nothing else
        # on this path asks for a frame.
        self._request_draw_for_scene(scene_id)

    def _make_slider_axes_check(self, scene_id: UUID) -> Callable:
        """Return a psygnal handler that re-checks a scene's slider axes."""

        def _on_slider_axes_input(*_args: Any) -> None:
            self._check_slider_axes(scene_id)
            # The same signal carries a wholesale ``scene.visuals``
            # reassignment, which changes the scene's extent.
            self._refresh_scene_overlays(scene_id)

        return _on_slider_axes_input

    def _check_slider_axes(self, scene_id: UUID) -> None:
        """Emit ``SliderAxesChangedEvent`` if ``Scene.slider_axes`` moved.

        Called after every model change that can alter the derived set: a
        visual added or removed, a transform replaced, an image's composite
        flag, or a slider override (design 3.5, D38).  Recomputes, compares
        with the last value emitted for the scene, and emits only on change.

        Parameters
        ----------
        scene_id : UUID
            The scene to check.  A scene that is no longer registered is
            ignored.
        """
        scene = self._model.scenes.get(scene_id)
        if scene is None or scene_id not in self._slider_axes_cache:
            return
        slider_axes = scene.slider_axes
        if slider_axes == self._slider_axes_cache[scene_id]:
            return
        self._slider_axes_cache[scene_id] = slider_axes
        self._outgoing_events.emit(
            SliderAxesChangedEvent(
                source_id=_source_id_override.get() or self._id,
                scene_id=scene_id,
                slider_axes=slider_axes,
            )
        )

    def _make_dims_handler(self, scene_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a scene's DimsManager."""

        def _on_dims_psygnal(info: EmissionInfo) -> None:
            if info.signal.name == "slider_overrides":
                # Overrides change which sliders are shown, not what is
                # sliced: no reslice and no DimsChangedEvent.
                self._check_slider_axes(scene_id)
                return
            selection = self._model.scenes[scene_id].dims.selection
            new_state = self._model.scenes[scene_id].dims.to_state()
            prev_axes = self._dims_cache[scene_id]
            new_axes = new_state.selection.displayed_axes
            displayed_axes_changed = prev_axes != new_axes
            self._dims_cache[scene_id] = new_axes
            prev_slice = self._slice_cache.get(scene_id)
            new_slice = self._slice_signature(selection)
            self._slice_cache[scene_id] = new_slice
            if displayed_axes_changed:
                # The rendered system and every visual space are rebuilt for
                # a reorder as well as a set change (design 3.14): a transpose
                # is a different rendered system even though the fetch is
                # identical.  The geometry rebuild and camera switch below
                # keep their existing trigger; narrowing them to a set change
                # is the separate optimisation 3.14 describes.
                self._rebuild_rendered(scene_id)
                if set(prev_axes) != set(new_axes):
                    self._rebuild_visual_spaces(scene_id)
                self._push_render_spaces(scene_id)
                self._rebuild_visuals_geometry(
                    scene_id, new_state.selection.displayed_axes
                )
                self._switch_canvas_cameras(
                    scene_id, new_state.selection.displayed_axes
                )
                self._refresh_scene_overlays(scene_id)
            elif prev_slice != new_slice:
                self._rebuild_rendered_embedding(scene_id)
            region_changed = displayed_axes_changed or prev_slice != new_slice
            resolved_source_id = _source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_dims_psygnal  scene=%s"
                "  resolved_source=%s  override_active=%s",
                scene_id,
                resolved_source_id,
                _source_id_override.get() is not None,
            )
            self._outgoing_events.emit(
                DimsChangedEvent(
                    source_id=resolved_source_id,
                    scene_id=scene_id,
                    dims_state=new_state,
                    displayed_axes_changed=displayed_axes_changed,
                    slice_indices=dict(selection.slice_indices),
                    region_changed=region_changed,
                )
            )

        return _on_dims_psygnal

    def _level_shapes_for(self, visual_model) -> list[tuple[int, ...]]:
        """Return the level shapes for *visual_model* from its data store."""
        data_store_id = getattr(visual_model, "data_store_id", None)
        if data_store_id:
            store = self._model.data.stores.get(UUID(data_store_id))
            if store is not None:
                # Single-node memory stores (points/lines/mesh) have no levels;
                # only image/label stores expose ``level_shapes``.
                return list(getattr(store, "level_shapes", []))
        return []

    def _rebuild_visuals_geometry(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Swap each visual's active node after a displayed_axes change.

        Uses the GFXVisual protocol: calls ``build_node`` on the first visit
        to a mode (passing the model so the renderer reads appearance at call
        time without caching it), or ``rebuild_node_geometry`` when the node
        already exists.  Single-node visuals (mesh, lines) handle both as
        no-ops and return the same node, so ``swap_node`` is a no-op too.
        """
        mode = "3d" if len(displayed_axes) == 3 else "2d"
        scene = self._model.scenes[scene_id]
        scene_manager = self._render_manager._scenes[scene_id]

        for visual_model in scene.visuals:
            gfx_visual = scene_manager.get_visual(visual_model.id)
            level_shapes = self._level_shapes_for(visual_model)
            level_transforms = list(getattr(visual_model, "level_transforms", []))
            if not gfx_visual.has_node(mode):
                new_node = gfx_visual.build_node(
                    mode, visual_model, displayed_axes, level_shapes, level_transforms
                )
            else:
                new_node = gfx_visual.rebuild_node_geometry(
                    mode, displayed_axes, level_shapes, level_transforms
                )
            scene_manager.swap_node(visual_model.id, new_node)

    def _switch_canvas_cameras(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Switch cameras on all canvases attached to *scene_id*.

        Calls ``show_object`` on canvases that are being activated for the
        first time in the new dimensionality.
        """
        new_dim = "3d" if len(displayed_axes) == 3 else "2d"
        gfx_scene = self._render_manager.get_scene(scene_id)
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            canvas_view = self._render_manager._canvases[canvas_id]
            first_visit = canvas_view.switch_dim(new_dim)
            if not first_visit:
                continue
            if not canvas_view.show_object(gfx_scene):
                # The visuals' geometry was rebuilt for the new axes a moment
                # ago and the reslice that fills it has not committed yet, so
                # there is nothing to fit to.  Take the fit when the data
                # lands instead of raising here.
                self._canvases_awaiting_fit.add(canvas_id)

    def _wire_transform(
        self,
        visual: BaseVisual,
        scene_id: UUID,
    ) -> None:
        """Subscribe to transform field changes on a visual model."""
        handler = self._make_transform_handler(visual.id, scene_id)
        visual.events.transform.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.events.transform, handler)
        )

    def _make_transform_handler(self, visual_id: UUID, scene_id: UUID) -> Callable:
        """Return a handler that emits TransformChangedEvent and triggers reslice.

        The reslice is skipped when ``_suppress_reslice`` is True, which is
        managed by the :meth:`suppress_reslice` context manager.
        """

        def _on_transform(new_transform: AffineTransform) -> None:
            # A replaced transform can carry a different axis correspondence,
            # which the render spaces read back off the matrix -- and which
            # decides the world axes this visual wants sliders for.
            self._push_render_spaces(scene_id)
            self._check_slider_axes(scene_id)
            self._refresh_scene_overlays(scene_id)
            self._outgoing_events.emit(
                TransformChangedEvent(
                    source_id=self._id,
                    scene_id=scene_id,
                    visual_id=visual_id,
                    transform=new_transform,
                )
            )
            if not self._suppress_reslice:
                self.reslice_scene(scene_id)
            # A transform moves the visual whether or not it reslices, and the
            # reslice path would only redraw once its data commits.
            self._request_draw_for_scene(scene_id)

        return _on_transform

    def _wire_appearance(
        self,
        visual: BaseVisual,
    ) -> None:
        """Subscribe to all field changes on a visual's appearance model."""
        handler = self._make_appearance_handler(visual.id)
        visual.appearance.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.appearance.events, handler)
        )

    def _wire_image(self, visual: BaseImageVisual) -> None:
        """Bridge an image visual's mode, single and channel models (design 3.3).

        Four connections:

        1. ``events.composite`` -- emits ``ImageCompositeChangedEvent``,
           re-checks the scene's slider axes and reslices.  On the model event,
           so a direct ``visual.composite = ...`` behaves like
           :meth:`set_image_composite` (D38).
        2. ``single.events`` -- per-field ``SingleAppearanceChangedEvent``.
        3. ``events.single`` -- a replaced ``single`` model: moves connection 2
           and emits one event with ``field_name=None``.
        4. ``events.channels`` -- a replaced ``channels`` dict: rewires the
           per-channel handlers and reslices when the key set changed.
        """
        visual_id = visual.id
        handlers = self._visual_psygnal_handlers.setdefault(visual_id, [])

        def _on_composite(new_value: bool) -> None:
            resolved_source_id = _source_id_override.get() or self._id
            self._outgoing_events.emit(
                ImageCompositeChangedEvent(
                    source_id=resolved_source_id,
                    visual_id=visual_id,
                    composite=bool(new_value),
                )
            )
            scene_id = self._visual_to_scene.get(visual_id)
            if scene_id is None:
                return
            self._check_slider_axes(scene_id)
            self.reslice_visual(visual_id)
            self._request_draw_for_visual(visual_id)

        visual.events.composite.connect(_on_composite)
        handlers.append((visual.events.composite, _on_composite))

        self._connect_single_bridge(visual)

        def _on_single_replaced(new_single: Any) -> None:
            wired = self._single_bridges.get(visual_id)
            if wired is not None and wired[0] is new_single:
                return
            self._connect_single_bridge(visual)
            self._outgoing_events.emit(
                SingleAppearanceChangedEvent(
                    source_id=_source_id_override.get() or self._id,
                    visual_id=visual_id,
                    field_name=None,
                    new_value=new_single,
                )
            )
            self._request_draw_for_visual(visual_id)

        visual.events.single.connect(_on_single_replaced)
        handlers.append((visual.events.single, _on_single_replaced))

        self._wire_channels(visual)
        known_keys = {"keys": set(visual.channels)}

        def _on_channels_replaced(new_channels: dict) -> None:
            self._wire_channels(visual)
            keys = set(new_channels)
            changed = keys != known_keys["keys"]
            known_keys["keys"] = keys
            if changed and visual_id in self._visual_to_scene:
                self.reslice_visual(visual_id)
                self._request_draw_for_visual(visual_id)

        visual.events.channels.connect(_on_channels_replaced)
        handlers.append((visual.events.channels, _on_channels_replaced))

    def _connect_single_bridge(self, visual: BaseImageVisual) -> None:
        """Attach the per-field bridge to *visual*'s current ``single`` model."""
        previous = self._single_bridges.pop(visual.id, None)
        if previous is not None:
            model, handler = previous
            model.events.disconnect(handler)
        visual_id = visual.id

        def _on_single_field(info: EmissionInfo) -> None:
            self._outgoing_events.emit(
                SingleAppearanceChangedEvent(
                    source_id=_source_id_override.get() or self._id,
                    visual_id=visual_id,
                    field_name=info.signal.name,
                    new_value=info.args[0],
                )
            )
            self._request_draw_for_visual(visual_id)

        visual.single.events.connect(_on_single_field)
        self._single_bridges[visual_id] = (visual.single, _on_single_field)

    def _wire_channels(self, visual: BaseImageVisual) -> None:
        """(Re)subscribe to field changes on every channel appearance of *visual*.

        One psygnal handler per channel.  Called at registration and again
        whenever ``channels`` is replaced, so a channel added later is heard
        from too; the previous handlers are disconnected first.
        """
        for signal, handler in self._channel_psygnal_handlers.pop(visual.id, []):
            signal.disconnect(handler)
        wired = []
        for channel_index, appearance in visual.channels.items():
            handler = self._make_channel_appearance_handler(visual.id, channel_index)
            appearance.events.connect(handler)
            wired.append((appearance.events, handler))
        self._channel_psygnal_handlers[visual.id] = wired

    def _make_channel_appearance_handler(
        self, visual_id: UUID, channel_index: int
    ) -> Callable:
        """Return a psygnal catch-all handler for one channel appearance."""

        def _on_channel_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _source_id_override.get() or self._id
            self._outgoing_events.emit(
                ChannelAppearanceChangedEvent(
                    source_id=resolved_source_id,
                    visual_id=visual_id,
                    channel_index=channel_index,
                    field_name=field_name,
                    new_value=new_value,
                )
            )
            # A hidden channel is left out of every slice request, so showing
            # it needs a load.
            if (
                field_name == "visible"
                and new_value
                and visual_id in self._visual_to_scene
            ):
                self.reslice_visual(visual_id)
            self._request_draw_for_visual(visual_id)

        return _on_channel_appearance_psygnal

    def _wire_aabb(self, visual: BaseVisual) -> None:
        """Subscribe to all field changes on a visual's aabb model."""
        handler = self._make_aabb_handler(visual.id)
        visual.aabb.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.aabb.events, handler)
        )

    def _make_aabb_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a visual's AABBParams."""

        def _on_aabb_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _aabb_source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_aabb_psygnal  visual=%s  field=%s"
                "  resolved_source=%s  override_active=%s",
                visual_id,
                field_name,
                resolved_source_id,
                _aabb_source_id_override.get() is not None,
            )
            self._outgoing_events.emit(
                AABBChangedEvent(
                    source_id=resolved_source_id,
                    visual_id=visual_id,
                    field_name=field_name,
                    new_value=new_value,
                )
            )
            # See _make_appearance_handler: toggling the box changes only
            # scene-graph flags, so nothing else asks for a frame.
            self._request_draw_for_visual(visual_id)

        return _on_aabb_psygnal

    def _wire_visual_render(self, visual: BaseVisual) -> None:
        """Bridge a visual's screen-space render settings to bus and renderer.

        Three sources feed one event: the ``outline`` sub-model, the
        ``ambient_occlusion`` field, and -- on labels visuals only -- the
        ``outline_selected_labels`` map.  The sub-model reports which of its
        fields changed, so it gets the catch-all handler; the two plain
        fields each know their own name.
        """
        handlers = self._visual_psygnal_handlers.setdefault(visual.id, [])

        outline_handler = self._make_visual_outline_handler(visual.id)
        visual.outline.events.connect(outline_handler)
        handlers.append((visual.outline.events, outline_handler))

        ao_handler = self._make_visual_render_handler(visual.id, "ambient_occlusion")
        visual.events.ambient_occlusion.connect(ao_handler)
        handlers.append((visual.events.ambient_occlusion, ao_handler))

        if isinstance(visual, BaseLabelsVisual):
            label_handler = self._make_visual_render_handler(
                visual.id, "outline_selected_labels"
            )
            visual.events.outline_selected_labels.connect(label_handler)
            handlers.append((visual.events.outline_selected_labels, label_handler))

            mode_handler = self._make_visual_render_handler(visual.id, "outline_mode")
            visual.events.outline_mode.connect(mode_handler)
            handlers.append((visual.events.outline_mode, mode_handler))

    def _make_visual_outline_handler(self, visual_id: UUID) -> Callable:
        """Return a catch-all handler for a visual's ``VisualOutline``."""

        def _on_outline(info: EmissionInfo) -> None:
            self._push_visual_render_change(
                visual_id, f"outline.{info.signal.name}", info.args[0]
            )

        return _on_outline

    def _make_visual_render_handler(self, visual_id: UUID, field_name: str) -> Callable:
        """Return a handler for one named render field on a visual."""

        def _on_change(new_value: Any) -> None:
            self._push_visual_render_change(visual_id, field_name, new_value)

        return _on_change

    def _push_visual_render_change(
        self, visual_id: UUID, field_name: str, new_value: Any
    ) -> None:
        """Apply one changed render field, then announce it."""
        visual = self._model_visual_or_none(visual_id)
        if visual is None:
            return
        self._apply_visual_render_field(visual, field_name, new_value)
        resolved_source_id = _visual_render_source_id_override.get() or self._id
        _SOURCE_ID_LOGGER.debug(
            "bridge  handler=_visual_render  visual=%s  field=%s  source=%s",
            visual_id,
            field_name,
            resolved_source_id,
        )
        self._outgoing_events.emit(
            VisualRenderChangedEvent(
                source_id=resolved_source_id,
                visual_id=visual_id,
                field_name=field_name,
                new_value=new_value,
            )
        )
        self._request_draw_for_visual(visual_id)

    def visuals_outlined_beyond(self, n_slots: int) -> list[tuple[str, int]]:
        """Return ``(name, slot)`` for visuals outlined past *n_slots*.

        The slots a palette of *n_slots* entries cannot colour.  Useful to a
        GUI before it shrinks the palette, and to the palette route after.
        """
        return [
            (visual.name, visual.outline.slot)
            for scene in self._model.scenes.values()
            for visual in scene.visuals
            if visual.outline.slot > n_slots
        ]

    def slot_usage(self) -> dict[int, int]:
        """Return ``{slot: how many visuals use it}``, for slots 1 and up.

        What lets a palette editor show that slot 2 is three visuals rather
        than leaving the user to hold it in their head.
        """
        usage: dict[int, int] = {}
        for scene in self._model.scenes.values():
            for visual in scene.visuals:
                slot = visual.outline.slot
                if slot >= 1:
                    usage[slot] = usage.get(slot, 0) + 1
        return usage

    def _seed_visual_render(self, visual: BaseVisual) -> None:
        """Push a newly added visual's render settings to the render layer.

        The render layer's flag map is a *cache* derived from the models, so
        it has to be primed when a visual joins -- otherwise a visual
        constructed with ``outline=VisualOutline(slot=1)`` would draw
        unoutlined until something happened to touch the field.
        """
        if visual.outline.slot >= 1:
            self._push_visual_outline(visual)
        if visual.ambient_occlusion is not None:
            self._render_manager.set_visual_ambient_occlusion(
                visual.id, visual.ambient_occlusion
            )
        if isinstance(visual, BaseLabelsVisual) and visual.outline_selected_labels:
            self._push_label_selection(visual)

    def update_visual_render_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one screen-space render field on a visual.

        The seam a GUI drives, and the twin of
        :meth:`update_render_config_field` for the per-visual half.  Writes
        the model; the psygnal bridge does the rest.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            ``"outline.slot"``, ``"outline.placement"``,
            ``"ambient_occlusion"`` or ``"outline_selected_labels"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``VisualRenderChangedEvent``.  GUI
            widgets should pass ``source_id=self._id`` so their own
            subscription can ignore the echo.

        Raises
        ------
        ValueError
            If *field* is not a settable per-visual render field.
        """
        if field not in VISUAL_RENDER_FIELDS:
            close = difflib.get_close_matches(field, VISUAL_RENDER_FIELDS, n=1)
            suggestion = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(
                f"{field!r} is not a settable per-visual render field."
                f"{suggestion} Valid fields: {list(VISUAL_RENDER_FIELDS)}."
            )
        visual = self._get_visual_model(visual_id)
        if field in {"outline_selected_labels", "outline_mode"} and not isinstance(
            visual, BaseLabelsVisual
        ):
            raise ValueError(
                f"{field} is only available on labels visuals; "
                f"a {type(visual).__name__} is outlined as one silhouette."
            )
        target = visual.outline if field.startswith("outline.") else visual
        leaf = field.split(".")[-1]
        token = _visual_render_source_id_override.set(source_id)
        try:
            setattr(target, leaf, value)
        finally:
            _visual_render_source_id_override.reset(token)

    def _on_visual_render_update(self, event: VisualRenderUpdateEvent) -> None:
        self.update_visual_render_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def update_visual_trail(
        self,
        visual_id: UUID,
        axis: int,
        config: TrailConfig | None,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set or clear the trail window on one axis of a graph visual.

        The seam a GUI drives.  An axis that already has a window is edited
        in place, one field event per field that differs, so nudging one spin
        box reslices once rather than rebuilding the whole trail.  Adding or
        removing an axis replaces ``visual.trail``, which is what rewires the
        per-config handlers (see :meth:`_wire_trail`).

        Parameters
        ----------
        visual_id :
            Target graph visual.
        axis :
            The data-axis index the window is keyed by.
        config :
            The complete window for *axis*, or ``None`` to remove it.  When
            the axis has no window yet the object itself is adopted, so pass
            one no other visual holds.
        source_id :
            UUID to stamp on the emitted ``TrailChangedEvent``.  GUI widgets
            should pass ``source_id=self._id`` so their own subscription can
            ignore the echo.

        Raises
        ------
        TypeError
            If the visual is not a graph visual.
        ValueError
            If *axis* is out of range for the graph's store.
        """
        visual = self._get_visual_model(visual_id)
        if not isinstance(visual, GraphVisual):
            raise TypeError(
                f"Only graph visuals have a trail; got a {type(visual).__name__}."
            )
        existing = visual.trail.get(axis)
        token = _source_id_override.set(source_id)
        try:
            if config is None:
                if existing is not None:
                    visual.trail = {
                        key: value for key, value in visual.trail.items() if key != axis
                    }
            elif existing is None:
                data_store = self._model.data.stores[UUID(visual.data_store_id)]
                # Checked here so a bad axis raises a plain ValueError rather
                # than psygnal's EmitLoopError from inside the dict handler.
                self._validate_trail_axes({axis: config}, data_store)
                visual.trail = {**visual.trail, axis: config}
            else:
                differing = {
                    name: getattr(config, name)
                    for name in type(existing).model_fields
                    if getattr(config, name) != getattr(existing, name)
                }
                if differing:
                    existing.update(differing)
        finally:
            _source_id_override.reset(token)

    def _on_trail_update(self, event: TrailUpdateEvent) -> None:
        self.update_visual_trail(
            event.visual_id, event.axis, event.config, source_id=event.source_id
        )

    def _model_visual_or_none(self, visual_id: UUID) -> BaseVisual | None:
        """Return the visual model, or ``None`` if it has been removed."""
        try:
            return self._get_visual_model(visual_id)
        except KeyError:
            return None

    def _apply_visual_render_field(
        self, visual: BaseVisual, field_name: str, new_value: Any
    ) -> None:
        """Push one changed render field onto the render layer, with warnings."""
        if field_name in ("outline.slot", "outline.placement"):
            self._push_visual_outline(visual)
        elif field_name == "ambient_occlusion":
            if new_value is False and not visual.pick_write:
                warnings.warn(
                    f"{_AO_PICK_WRITE_REQUIRED} pick_write set to True",
                    RuntimeWarning,
                    stacklevel=2,
                )
                visual.pick_write = True
            self._render_manager.set_visual_ambient_occlusion(visual.id, new_value)
        elif field_name == "outline_selected_labels":
            self._push_label_selection(visual)
        elif field_name == "outline_mode":
            # The mode is carried in the LUT entry's ``kind``, so re-pushing
            # the outline covers the shader side -- no material rebuild.  The
            # selection goes with it because ``all_boundaries`` suppresses it
            # on the GPU: entering the mode has to clear it and leaving has to
            # put it back.
            self._push_visual_outline(visual)
            self._push_label_selection(visual)

    def _push_visual_outline(self, visual: BaseVisual) -> None:
        """Send a visual's outline assignment to the render layer.

        Warns rather than silently doing nothing for the three ways an
        outline can be configured and still not appear.
        """
        slot = int(visual.outline.slot)
        if slot >= 1:
            if not visual.pick_write:
                warnings.warn(
                    f"{_PICK_WRITE_REQUIRED} pick_write set to True",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # Set it on the model, not the material: the
                # PickWriteChangedEvent wiring propagates it for us.
                visual.pick_write = True
            if not self.render_config.outline.enabled:
                warnings.warn(
                    "the outline pass is off, so this outline will not draw; "
                    "set controller.outline_enabled = True",
                    RuntimeWarning,
                    stacklevel=2,
                )
            palette = self.render_config.outline.palette
            if slot > len(palette):
                warnings.warn(
                    f"outline slot {slot} has no palette entry (the palette "
                    f"holds {len(palette)}), so the outline draws transparent",
                    RuntimeWarning,
                    stacklevel=2,
                )
        placement = visual.outline.placement or _default_placement(visual)
        kind = _outline_kind(visual)
        self._render_manager.set_visual_outline(
            visual.id, slot=slot, placement=placement, kind=kind
        )

    def _push_label_selection(self, visual: BaseVisual) -> None:
        """Send a labels visual's per-label selection to the render layer.

        In ``all_boundaries`` mode the selection is suppressed on the GPU
        rather than pushed.  It is not only a colour there: a selected
        label's outline key *is* its slot number, so two touching labels
        sharing a slot would share a key and lose the boundary between them
        -- in the one mode whose whole purpose is showing every boundary.
        The model field is left untouched, so switching back to
        ``per_label`` restores the selection exactly.
        """
        selection = (
            {}
            if visual.outline_mode == "all_boundaries"
            else dict(visual.outline_selected_labels)
        )
        self._render_manager.set_label_selection(visual.id, selection)

    def _wire_pick_write(self, visual: BaseVisual) -> None:
        """Subscribe to pick_write field changes on a visual model."""
        handler = self._make_pick_write_handler(visual.id)
        visual.events.pick_write.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.events.pick_write, handler)
        )

    def _make_pick_write_handler(self, visual_id: UUID) -> Callable:
        """Return a handler that emits PickWriteChangedEvent on pick_write changes."""

        def _on_pick_write(new_value: bool) -> None:
            visual = self._model_visual_or_none(visual_id)
            if visual is not None and not new_value:
                # The other half of the conflict.  The user's decision about
                # picking stands -- the most recent explicit action wins --
                # but the feature it silently disables says so.
                if visual.outline.slot >= 1:
                    warnings.warn(_PICK_WRITE_REQUIRED, RuntimeWarning, stacklevel=2)
                if visual.ambient_occlusion is False:
                    warnings.warn(_AO_PICK_WRITE_REQUIRED, RuntimeWarning, stacklevel=2)
            self._outgoing_events.emit(
                PickWriteChangedEvent(
                    source_id=self._id,
                    visual_id=visual_id,
                    pick_write=new_value,
                )
            )

        return _on_pick_write

    def _make_appearance_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for any visual's appearance model.

        Routes field changes to one of two bus events: ``visible`` field changes
        become ``VisualVisibilityChangedEvent``; all other fields become
        ``AppearanceChangedEvent`` with ``requires_reslice=True`` for fields in
        ``_RESLICE_FIELDS`` (``lod_bias``, ``force_level``, ``frustum_cull``).
        """

        def _on_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_appearance_psygnal  visual=%s  field=%s"
                "  resolved_source=%s  override_active=%s",
                visual_id,
                field_name,
                resolved_source_id,
                _source_id_override.get() is not None,
            )
            if field_name == "visible":
                self._outgoing_events.emit(
                    VisualVisibilityChangedEvent(
                        source_id=resolved_source_id,
                        visual_id=visual_id,
                        visible=new_value,
                    )
                )
                # A hidden image skipped every reslice while hidden, so what
                # it holds is from its last visible slice.
                if (
                    new_value
                    and visual_id in self._visual_to_scene
                    and isinstance(self.get_visual_model(visual_id), _SKIP_WHEN_HIDDEN)
                ):
                    self.reslice_visual(visual_id)
            else:
                self._outgoing_events.emit(
                    AppearanceChangedEvent(
                        source_id=resolved_source_id,
                        visual_id=visual_id,
                        field_name=field_name,
                        new_value=new_value,
                        requires_reslice=(field_name in _RESLICE_FIELDS),
                    )
                )
                if field_name in _RESLICE_FIELDS:
                    self.reslice_visual(visual_id)

            # An appearance change repaints the same data, so it triggers no
            # reslice (only the three _RESLICE_FIELDS do) and nothing else in
            # the pipeline asks for a frame.  Measured in a headless harness,
            # every appearance write requested zero draws.
            #
            # This also discards the canvas's accumulation history, without
            # which the frame would be an average with the pre-change picture
            # -- see CanvasView.invalidate_accumulation.
            self._request_draw_for_visual(visual_id)

        return _on_appearance_psygnal

    def reslice_all(self) -> None:
        """Trigger a data load for all visuals across all scenes."""
        for scene_id in self._model.scenes:
            self.reslice_scene(scene_id)

    def _selections_for_scene(self, scene_id: UUID) -> dict[UUID, RegionSelection]:
        """The region each of a scene's canvases is showing (design 3.1).

        ``DimsManager`` is the editor and emits the artifact (D43), but it
        needs the canvas's rendered system, which is per canvas and is not
        model state.  Both halves meet here: the controller owns the rendered
        systems and hands one to the model, and the model layer never reaches
        into the render layer.
        """
        scene = self._model.scenes.get(scene_id)
        if scene is None:
            return {}
        selections: dict[UUID, RegionSelection] = {}
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            entry = self._rendered.get(canvas_id)
            if entry is None:
                continue
            rendered, embedding = entry
            selections[canvas_id] = scene.dims.to_selection(rendered, embedding)
        return selections

    def reslice_scene(
        self,
        scene_id: UUID,
        *,
        on_ready: Callable[[], None] | None = None,
        owner_id: UUID | None = None,
    ) -> None:
        """Trigger a data load for all visuals in one scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice.
        on_ready : Callable[[], None] or None
            If provided, a zero-argument callback fired exactly once after
            *all* visuals loaded by this reslice have committed to the GPU
            (across every canvas attached to the scene).  Visuals with no data
            in the current view (culled, empty, or hidden) do not delay it.
            Works uniformly for in-memory, multiscale, multichannel, and
            geometry visuals.  See :meth:`on_scene_ready`.

            The callback tracks *this* reslice generation.  If a superseding
            reslice (e.g. a camera-settle reload) cancels these in-flight reads
            before they commit, the cancelled visual never reports completion
            and the callback may not fire.  Callers that need a guaranteed
            startup signal should suppress camera-driven reslicing during the
            load (the convenience launchers do this automatically).
        owner_id : UUID or None
            Owner under which the temporary ``on_ready`` subscriptions are
            registered (for teardown).  Defaults to the controller's own id.
        """
        dims_state = self._dims_state_for_scene(scene_id)
        visual_configs = self._build_visual_configs_for_scene(scene_id)

        selections = self._selections_for_scene(scene_id)

        if on_ready is None:
            self._render_manager.reslice_scene(
                scene_id, dims_state, visual_configs, selections=selections
            )
            return

        self._notify_when_resliced(
            scene_id,
            on_ready,
            owner_id or self._id,
            lambda: self._render_manager.reslice_scene(
                scene_id, dims_state, visual_configs, selections=selections
            ),
        )

    def _notify_when_resliced(
        self,
        scene_id: UUID,
        callback: Callable[[], None],
        owner_id: UUID,
        trigger: Callable[[], None],
    ) -> None:
        """Arm a one-shot quiescence tracker, then run *trigger* to reslice.

        The tracker counts visuals announced by ``ResliceStartedEvent`` and
        decrements on each ``ResliceCompletedEvent`` for *scene_id*; *callback*
        fires once the count drains to zero.  Because planning and the
        synchronous ``ResliceStartedEvent`` emissions all run inside *trigger*
        on a synchronous bus, the pending count is fully known the instant
        *trigger* returns — so the "armed" flag suppresses any premature fire
        from synchronous empty-batch completions until the full count is in.
        """
        state = {"pending": 0, "armed": True, "fired": False}
        handles: list[SubscriptionHandle] = []

        def _finish() -> None:
            if state["fired"]:
                return
            state["fired"] = True
            for handle in handles:
                self._outgoing_events.unsubscribe(handle)
            callback()

        def _maybe_fire() -> None:
            if not state["armed"] and not state["fired"] and state["pending"] <= 0:
                _finish()

        def _on_started(event: ResliceStartedEvent) -> None:
            state["pending"] += len(event.visual_ids)

        def _on_completed(event: ResliceCompletedEvent) -> None:
            if event.scene_id != scene_id:
                return
            state["pending"] -= 1
            _maybe_fire()

        handles.append(
            self._outgoing_events.subscribe(
                ResliceStartedEvent,
                _on_started,
                entity_id=scene_id,
                owner_id=owner_id,
            )
        )
        # ResliceCompletedEvent routes by visual_id, so subscribe as a
        # catch-all and filter on scene_id inside the handler.
        handles.append(
            self._outgoing_events.subscribe(
                ResliceCompletedEvent,
                _on_completed,
                owner_id=owner_id,
            )
        )

        trigger()

        # All ResliceStartedEvents for this generation have now been emitted
        # synchronously, so the pending count is complete.  Close the arming
        # window and fire immediately if there was nothing to load.
        state["armed"] = False
        _maybe_fire()

    def reslice_visual(self, visual_id: UUID) -> None:
        """Trigger a data load for one visual.

        Not needed after changing a store: stores announce their own changes
        (reassigning a data field, or ``store.notify_changed``) and the
        controller reslices every visual reading them.
        """
        scene_id = self._visual_to_scene[visual_id]
        dims_state = self._dims_state_for_scene(scene_id)
        cfg = _visual_render_config(self.get_visual_model(visual_id))
        self._render_manager.reslice_visual(
            visual_id, dims_state, cfg, selections=self._selections_for_scene(scene_id)
        )

    @contextmanager
    def suppress_reslice(self) -> Generator[None, None, None]:
        """Context manager that blocks reslice_scene inside transform handlers.

        Use this when updating a visual's transform without needing to reload
        its underlying data — for example, repositioning a static-geometry mesh
        by translation only.

        .. warning::
            ``_suppress_reslice`` is a flat boolean.  Nested calls or concurrent
            async tasks that mutate transforms inside overlapping
            ``suppress_reslice`` blocks will interfere.  Replace with a depth
            counter if that becomes necessary.
        """
        self._suppress_reslice = True
        try:
            yield
        finally:
            self._suppress_reslice = False

    def data_to_world(
        self,
        scene_id: UUID,
        data_store: Any,
        scale: Sequence[float] | None = None,
        translation: Sequence[float] | None = None,
    ) -> AffineTransform:
        """Build a ``data -> world`` transform from a per-axis scale and offset.

        A transform names the two coordinate systems it maps between, and only
        the viewer knows both: the store's level-0 system and the scene's
        world.  This is the short way to say "this dataset is 4 um in z and
        sits 10 um along it" without assembling an axis map by hand.

        Before Phase 8 the same thing was said with a bare
        v1 ``AffineTransform``, which stated the numbers and
        named nothing; ``add_*`` accepted one and attached the endpoints
        itself.  That went with v1, and this replaces it.

        Parameters
        ----------
        scene_id : UUID
            The scene whose world the transform maps into.
        data_store : Any
            The store whose voxel space it maps from.  Given its coordinate
            systems if it does not have them.
        scale : Sequence[float] or None
            Per-data-axis scale.  ``None`` is all ones.
        translation : Sequence[float] or None
            Per-data-axis offset, in world units.  ``None`` is all zeros.

        Returns
        -------
        AffineTransform
            Ready to hand to any ``add_*`` or to
            :meth:`set_visual_transform`.

        Raises
        ------
        ValueError
            If the store and the world disagree about how many axes they
            have: the correspondence here is positional, so there is nothing
            to infer.  Use ``AffineTransform.from_axis_map`` to state it.
        """
        self._ensure_data_coordinate_systems(scene_id, data_store)
        data = data_store.data_coordinate_system
        world = self._model.scenes[scene_id].dims.world_coordinate_system
        factors = tuple(scale) if scale is not None else (1.0,) * data.ndim
        return scale_and_translation_transform(data, world, factors, translation)

    def set_visual_transform(
        self,
        visual_id: UUID,
        transform: AffineTransform,
        *,
        reslice: bool = True,
    ) -> None:
        """Update the data-to-world transform of a visual.

        Assigns *transform* to the live visual model, which fires its psygnal
        field event and propagates the change to the render layer via the
        event bus.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to update.
        transform : AffineTransform
            New data-to-world transform.
        reslice : bool
            If True (default), a full reslice is triggered after the transform
            is applied — required for image visuals where the transform changes
            which data falls in the current slab.  Pass False for
            static-geometry visuals (mesh, points, lines) where the transform
            only repositions the node and the underlying data is unchanged.
        """
        visual_model = self.get_visual_model(visual_id)
        if reslice:
            visual_model.transform = transform
        else:
            with self.suppress_reslice():
                visual_model.transform = transform

    def _on_dims_changed_bus(self, event: DimsChangedEvent) -> None:
        """Bus handler -- reslice the scene when what it shows changed."""
        if not event.region_changed:
            return
        self.reslice_scene(event.scene_id)

    # ------------------------------------------------------------------
    # Model mutation with source-ID threading
    # ------------------------------------------------------------------

    def update_slice_indices(
        self,
        scene_id: UUID,
        slice_indices: Mapping[int, float],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Move the slice position of one or more world axes on a scene.

        **Merges** into the scene's positions: axes absent from
        *slice_indices* keep theirs.  Every world axis has a position whether
        or not it is displayed (D36), so this never adds or removes one.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        slice_indices :
            Mapping of world axis index -> world slice position.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.

        Raises
        ------
        ValueError
            If a key is not an axis of the scene's world.  Nothing is changed.
        """
        dims = self._model.scenes[scene_id].dims
        unknown = sorted(set(slice_indices) - set(range(dims.ndim)))
        if unknown:
            raise ValueError(
                f"slice_indices names axes {unknown} outside the scene's "
                f"world {dims.axis_labels} (ndim={dims.ndim})."
            )
        merged = dict(dims.selection.slice_indices)
        merged.update(
            {int(axis): float(value) for axis, value in slice_indices.items()}
        )
        if merged == dims.selection.slice_indices:
            return
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  scene=%s  source=%s",
            scene_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            dims.selection.slice_indices = merged
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  scene=%s", scene_id)

    def update_thickness(
        self,
        scene_id: UUID,
        thickness: Mapping[int, float],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Replace a scene's per-axis half-thicknesses.

        Parameters
        ----------
        scene_id :
            Target scene.
        thickness :
            World axis index -> half-thickness in world units.  An axis
            absent from the mapping slices a plane.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.

        Raises
        ------
        ValueError
            If a key is not an axis of the scene's world.
        """
        dims = self._model.scenes[scene_id].dims
        unknown = sorted(set(thickness) - set(range(dims.ndim)))
        if unknown:
            raise ValueError(
                f"thickness names axes {unknown} outside the scene's "
                f"world {dims.axis_labels} (ndim={dims.ndim})."
            )
        new = {int(axis): float(value) for axis, value in thickness.items()}
        if new == dims.selection.thickness:
            return
        token = _source_id_override.set(source_id)
        try:
            dims.selection.thickness = new
        finally:
            _source_id_override.reset(token)

    def set_slider_override(
        self,
        scene_id: UUID,
        axis: int,
        value: bool | None,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Force a world axis's slider shown or hidden, or return it to automatic.

        Parameters
        ----------
        scene_id :
            Target scene.
        axis :
            World axis index.
        value :
            ``True`` force-shows the slider, ``False`` force-hides it, and
            ``None`` removes the override so the visuals decide.
        source_id :
            UUID to stamp on the ``SliderAxesChangedEvent`` this may emit.

        Raises
        ------
        ValueError
            If *axis* is not an axis of the scene's world.
        """
        dims = self._model.scenes[scene_id].dims
        if not 0 <= int(axis) < dims.ndim:
            raise ValueError(
                f"axis {axis} is outside the scene's world {dims.axis_labels}."
            )
        overrides = dict(dims.slider_overrides)
        if value is None:
            overrides.pop(int(axis), None)
        else:
            overrides[int(axis)] = bool(value)
        if overrides == dims.slider_overrides:
            return
        token = _source_id_override.set(source_id)
        try:
            dims.slider_overrides = overrides
        finally:
            _source_id_override.reset(token)

    def update_appearance_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a visual's appearance model.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``AppearanceChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            Attribute name on the appearance model, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``AppearanceChangedEvent``.  Defaults
            to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  field=%s  visual=%s  source=%s",
            field,
            visual_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.appearance, field, value)
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  field=%s  visual=%s", field, visual_id)

    def update_background_field(
        self,
        scene_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a scene's background appearance model.

        Tags the emitted bus event with *source_id*.  GUI widgets should pass
        ``source_id=self._id`` so their own ``BackgroundChangedEvent``
        subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        field :
            Attribute name on the background model, e.g. ``"top_color"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``BackgroundChangedEvent``.  Defaults
            to the controller's own ID.
        """
        background = self._model.scenes[scene_id].background
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  background_field=%s  scene=%s  source=%s",
            field,
            scene_id,
            resolved_source_id,
        )
        token = _background_source_id_override.set(source_id)
        try:
            setattr(background, field, value)
        finally:
            _background_source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug(
                "reset  background_field=%s  scene=%s", field, scene_id
            )

    def set_background(
        self,
        scene_id: UUID,
        background: BackgroundAppearance,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Replace a scene's background appearance model wholesale.

        Emits a single ``BackgroundChangedEvent`` with ``field_name=None``.
        Assigning ``scene.background`` directly does the same thing; this
        method exists to stamp a *source_id* on the resulting event.

        Parameters
        ----------
        scene_id :
            Target scene.
        background :
            The background appearance to apply.
        source_id :
            UUID to stamp on the emitted ``BackgroundChangedEvent``.
        """
        token = _background_source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].background = background
        finally:
            _background_source_id_override.reset(token)

    def update_appearance_group_field(
        self,
        visual_ids: list[UUID],
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one appearance field across a group of visuals in lock-step.

        Fan-out over :meth:`update_appearance_field` so every visual in the
        group -- the per-panel visuals of an ``OrthoViewer``, say -- receives
        the same change.  This is the programmatic write-side companion to the
        widget subscribe-to-all read side, matching
        :meth:`update_channel_group_field`.

        Parameters
        ----------
        visual_ids :
            Target visuals, kept equal.
        field :
            Attribute name on each appearance model, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on each emitted event.  Defaults to the
            controller's own ID.
        """
        for visual_id in visual_ids:
            self.update_appearance_field(visual_id, field, value, source_id=source_id)

    def update_aabb_group_field(
        self,
        visual_ids: list[UUID],
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one AABB field across a group of visuals in lock-step.

        The AABB is **not** an appearance field: it lives on ``visual.aabb``
        and travels on ``AABBChangedEvent``, so it needs its own group helper
        rather than riding on :meth:`update_appearance_group_field`.

        Parameters
        ----------
        visual_ids :
            Target visuals, kept equal.
        field :
            Attribute name on each AABB model, e.g. ``"enabled"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on each emitted ``AABBChangedEvent``.  Defaults to
            the controller's own ID.
        """
        for visual_id in visual_ids:
            self.update_aabb_field(visual_id, field, value, source_id=source_id)

    def update_channel_appearance_field(
        self,
        visual_id: UUID,
        channel_index: int,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on one channel of an image visual.

        Tags the emitted bus event with *source_id*.  GUI widgets should pass
        ``source_id=self._id`` so their own ``ChannelAppearanceChangedEvent``
        subscription can ignore the echo.

        This mutates a single visual only.  When a channel is shared in
        lock-step across several panels (e.g. an ``OrthoViewer``), calling this
        on one panel's visual leaves the sibling panels unequal until they are
        written too; use ``update_channel_group_field`` to keep the group in
        lock-step.

        A ``pydantic.ValidationError`` from a malformed *value* is allowed to
        propagate (matching ``update_appearance_field``).

        Parameters
        ----------
        visual_id :
            Target visual.
        channel_index :
            Index into ``visual.channels`` selecting the channel appearance.
        field :
            Attribute name on the channel appearance model, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``ChannelAppearanceChangedEvent``.
            Defaults to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  channel=%d  field=%s  visual=%s  source=%s",
            channel_index,
            field,
            visual_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.channels[channel_index], field, value)
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug(
                "reset  channel=%d  field=%s  visual=%s",
                channel_index,
                field,
                visual_id,
            )

    def update_channel_group_field(
        self,
        visual_ids: list[UUID],
        channel_index: int,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one channel field across a group of visuals in lock-step.

        Fan-out over ``update_channel_appearance_field`` so every visual in the
        group (e.g. the per-panel visuals of an ``OrthoViewer``) receives the
        same channel change.  This is the programmatic write-side companion to
        the widget subscribe-to-all read side.

        Parameters
        ----------
        visual_ids :
            Target visuals sharing the channel set.
        channel_index :
            Index into each visual's ``channels`` mapping.
        field :
            Attribute name on the channel appearance model.
        value :
            New value for the field.
        source_id :
            UUID to stamp on each emitted ``ChannelAppearanceChangedEvent``.
            Defaults to the controller's own ID.
        """
        for visual_id in visual_ids:
            self.update_channel_appearance_field(
                visual_id, channel_index, field, value, source_id=source_id
            )

    def update_single_appearance_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on an image visual's single-mode appearance.

        A ``pydantic.ValidationError`` from a malformed *value* propagates.

        Parameters
        ----------
        visual_id :
            Target image visual.
        field :
            Attribute name on ``visual.single``, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``SingleAppearanceChangedEvent``.
        """
        visual = self.get_visual_model(visual_id)
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.single, field, value)
        finally:
            _source_id_override.reset(token)

    def update_single_group_field(
        self,
        visual_ids: list[UUID],
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one single-mode field across a group of image visuals in lock-step.

        Parameters
        ----------
        visual_ids :
            Target visuals, kept equal -- an ``OrthoViewer``'s panel siblings.
        field :
            Attribute name on each ``single`` model.
        value :
            New value for the field.
        source_id :
            UUID to stamp on each emitted event.
        """
        for visual_id in visual_ids:
            self.update_single_appearance_field(
                visual_id, field, value, source_id=source_id
            )

    def set_image_composite(
        self,
        visual_id: UUID,
        composite: bool,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Switch an image visual between single and composite mode.

        Validates, then assigns.  The reslice, ``ImageCompositeChangedEvent``
        and any ``SliderAxesChangedEvent`` come from the ``composite`` model
        bridge, so a direct ``visual.composite = ...`` gets them too -- but a
        direct assignment skips the checks below (design 3.4).

        Parameters
        ----------
        visual_id :
            Target image visual.
        composite :
            ``True`` for composite mode.
        source_id :
            UUID to stamp on the emitted ``ImageCompositeChangedEvent``.

        Raises
        ------
        ValueError
            If *composite* is ``True`` and the visual has no ``channel_axis``,
            or its channel axis is displayed.  The model is unchanged.
        """
        visual = self.get_visual_model(visual_id)
        if not isinstance(visual, BaseImageVisual):
            raise TypeError(f"Visual {visual_id} is not an image visual.")
        composite = bool(composite)
        if composite == visual.composite:
            return
        if composite:
            if visual.channel_axis is None:
                raise ValueError(
                    f"Visual {visual_id} has no channel_axis, so it cannot composite."
                )
            scene_id = self._visual_to_scene[visual_id]
            world_axis = visual.transform.axis_correspondence().get(visual.channel_axis)
            displayed = self._model.scenes[scene_id].dims.selection.displayed_axes
            if world_axis in displayed:
                raise ValueError(
                    f"Cannot composite channel_axis={visual.channel_axis}: it "
                    f"maps to world axis {world_axis}, which the scene "
                    f"displays {tuple(displayed)}."
                )
        token = _source_id_override.set(source_id)
        try:
            visual.composite = composite
        finally:
            _source_id_override.reset(token)

    def set_image_composite_group(
        self,
        visual_ids: list[UUID],
        composite: bool,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Switch a group of image visuals' mode in lock-step.

        Every visual is validated before any is changed, so the group is
        never left split between the two modes.

        Parameters
        ----------
        visual_ids :
            Target visuals -- an ``OrthoViewer``'s panel siblings.
        composite :
            ``True`` for composite mode.
        source_id :
            UUID to stamp on each emitted event.

        Raises
        ------
        ValueError
            As :meth:`set_image_composite`, for any visual in the group.
        """
        if composite:
            for visual_id in visual_ids:
                visual = self.get_visual_model(visual_id)
                if visual.channel_axis is None:
                    raise ValueError(
                        f"Visual {visual_id} has no channel_axis, so it cannot "
                        "composite."
                    )
                scene_id = self._visual_to_scene[visual_id]
                world_axis = visual.transform.axis_correspondence().get(
                    visual.channel_axis
                )
                displayed = self._model.scenes[scene_id].dims.selection.displayed_axes
                if world_axis in displayed:
                    raise ValueError(
                        f"Cannot composite visual {visual_id}: world axis "
                        f"{world_axis} is displayed {tuple(displayed)}."
                    )
        for visual_id in visual_ids:
            self.set_image_composite(visual_id, composite, source_id=source_id)

    def update_aabb_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a visual's AABB params model.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``AABBChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            Attribute name on the AABB model, e.g. ``"enabled"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``AABBChangedEvent``.  Defaults
            to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  field=%s  visual=%s  source=%s",
            field,
            visual_id,
            resolved_source_id,
        )
        token = _aabb_source_id_override.set(source_id)
        try:
            setattr(visual.aabb, field, value)
        finally:
            _aabb_source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  field=%s  visual=%s", field, visual_id)

    def update_displayed_axes(
        self,
        scene_id: UUID,
        displayed_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set ``displayed_axes`` on a scene's dims.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        displayed_axes :
            Tuple of axis indices to display; length 2 for 2D, 3 for 3D.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.
        """
        scene = self._model.scenes[scene_id]
        for visual in scene.visuals:
            world_axis = self._composited_world_axis(visual)
            if world_axis is not None and world_axis in displayed_axes:
                raise ValueError(
                    f"Cannot display world axis {world_axis}: image visual "
                    f"'{visual.name}' composites it.  Switch it to single mode "
                    f"first."
                )
        token = _source_id_override.set(source_id)
        try:
            scene.dims.selection.displayed_axes = displayed_axes
        finally:
            _source_id_override.reset(token)

    def set_displayed_axes(
        self,
        scene_id: UUID,
        displayed_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set displayed axes on a scene's dims (preferred public API).

        Equivalent to :meth:`update_displayed_axes`.  The controller's
        psygnal bridge fires ``_rebuild_visuals_geometry`` and
        ``_switch_canvas_cameras`` automatically when the model field changes.

        Parameters
        ----------
        scene_id :
            Target scene.
        displayed_axes :
            Tuple of axis indices to display; length 2 for 2D, 3 for 3D.
        source_id :
            UUID stamped on the emitted ``DimsChangedEvent``.
        """
        self.update_displayed_axes(scene_id, displayed_axes, source_id=source_id)

    # ------------------------------------------------------------------
    # IncomingEventBus handlers
    # ------------------------------------------------------------------

    def _on_appearance_update(self, event: AppearanceUpdateEvent) -> None:
        self.update_appearance_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def _on_dims_update(self, event: DimsUpdateEvent) -> None:
        # No ordering between the two: every axis keeps a position whether or
        # not it is displayed (D36), so neither write can leave one uncovered.
        if event.slice_indices is not None:
            self.update_slice_indices(
                event.scene_id, event.slice_indices, source_id=event.source_id
            )
        if event.displayed_axes is not None:
            self.update_displayed_axes(
                event.scene_id, event.displayed_axes, source_id=event.source_id
            )

    def _on_slider_override_update(self, event: SliderOverrideUpdateEvent) -> None:
        self.set_slider_override(
            event.scene_id, event.axis, event.value, source_id=event.source_id
        )

    def _on_aabb_update(self, event: AABBUpdateEvent) -> None:
        self.update_aabb_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def _on_background_update(self, event: BackgroundUpdateEvent) -> None:
        self.update_background_field(
            event.scene_id, event.field, event.value, source_id=event.source_id
        )

    def _on_single_appearance_update(self, event: SingleAppearanceUpdateEvent) -> None:
        self.update_single_appearance_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def _on_image_composite_update(self, event: ImageCompositeUpdateEvent) -> None:
        self.set_image_composite(
            event.visual_id, event.composite, source_id=event.source_id
        )

    def _on_channel_appearance_update(
        self, event: ChannelAppearanceUpdateEvent
    ) -> None:
        self.update_channel_appearance_field(
            event.visual_id,
            event.channel_index,
            event.field,
            event.value,
            source_id=event.source_id,
        )

    # ------------------------------------------------------------------
    # Camera settle
    # ------------------------------------------------------------------

    @property
    def camera_reslice_enabled(self) -> bool:
        """Whether camera movement triggers automatic reslicing."""
        return self._render_manager.config.camera.reslice_enabled

    @camera_reslice_enabled.setter
    def camera_reslice_enabled(self, value: bool) -> None:
        self._render_manager.config.camera.reslice_enabled = value
        if not value:
            self._cancel_settle_tasks()

    def _cancel_settle_tasks(self) -> None:
        """Cancel and forget every pending camera-settle task.

        Shared by the ``camera_reslice_enabled`` setter and :meth:`close`.
        ``remove_scene`` and ``remove_canvas`` cancel only their own subset,
        because they leave the rest of the controller running.
        """
        for task in self._settle_tasks.values():
            if not task.done():
                task.cancel()
        self._settle_tasks.clear()

    @property
    def render_config(self) -> RenderManagerConfig:
        """Live rendering configuration.

        Mutating a field here changes the model but not the GPU state, and
        notifies no widget.  :meth:`update_render_config_field` and the
        dedicated properties (``ambient_occlusion_power`` and friends) do all three in
        one step, and are what a GUI should drive.
        """
        return self._render_manager.config

    @property
    def render_manager(self) -> RenderManager:
        """The render manager owning the canvases and the GPU-side state."""
        return self._render_manager

    def update_render_config_field(
        self,
        section: str,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field of the render configuration and apply it.

        The single seam every render-config write goes through: it updates
        the model, pushes the change to the GPU by whichever route that
        field needs, and emits a ``RenderConfigChangedEvent`` so subscribed
        widgets follow along.  Which fields recompile a shader and which are
        plain uniforms is a property of the field, recorded once in
        :data:`_RENDER_CONFIG_ROUTES`, so no caller has to know.

        Parameters
        ----------
        section :
            ``"outline"``, ``"ambient_occlusion"`` or ``"temporal"``.
        field :
            Dotted attribute path within the section, e.g. ``"power"`` or
            ``"selection.inward_thickness"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``RenderConfigChangedEvent``.  GUI
            widgets should pass ``source_id=self._id`` so their own
            subscription can ignore the echo.  Defaults to the controller's
            own ID.

        Raises
        ------
        ValueError
            If *section* or *field* is not a settable render-config field.
        """
        route = _resolve_render_config_route(section, field)
        config_section = getattr(self.render_config, section)

        # Write through the section model first so pydantic validates the
        # value before any of it reaches the GPU.
        target = config_section
        *parents, leaf = field.split(".")
        for name in parents:
            target = getattr(target, name)
        setattr(target, leaf, value)

        route.apply(self, getattr(target, leaf))

        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  render_config=%s.%s  source=%s", section, field, resolved_source_id
        )
        self._outgoing_events.emit(
            RenderConfigChangedEvent(
                source_id=resolved_source_id,
                section=section,
                config=config_section,
                field_name=field,
                new_value=getattr(target, leaf),
            )
        )

    def _on_render_config_update(self, event: RenderConfigUpdateEvent) -> None:
        self.update_render_config_field(
            event.section, event.field, event.value, source_id=event.source_id
        )

    # -- Outlines ------------------------------------------------------

    @property
    def outline_enabled(self) -> bool:
        """Whether the screen-space outline pass is active."""
        return self._render_manager.outline_enabled

    @outline_enabled.setter
    def outline_enabled(self, value: bool) -> None:
        self.update_render_config_field("outline", "enabled", value)

    @property
    def outline_boundaries_enabled(self) -> bool:
        """Whether the boundaries layer (every outlined region) draws."""
        return self._render_manager.outline_boundaries_enabled

    @outline_boundaries_enabled.setter
    def outline_boundaries_enabled(self, value: bool) -> None:
        self.update_render_config_field("outline", "boundaries.enabled", value)

    @property
    def outline_selection_enabled(self) -> bool:
        """Whether the selection layer (regions with a palette slot) draws."""
        return self._render_manager.outline_selection_enabled

    @outline_selection_enabled.setter
    def outline_selection_enabled(self, value: bool) -> None:
        self.update_render_config_field("outline", "selection.enabled", value)

    # -- Ambient occlusion ---------------------------------------------

    @property
    def ambient_occlusion_enabled(self) -> bool:
        """Whether the screen-space ambient occlusion pass is active.

        Ambient occlusion darkens creases by sampling the depth buffer, and
        is the cheapest shape cue available for cellier's default unlit
        isosurfaces.  It runs in 3D only.
        """
        return self._render_manager.ambient_occlusion_enabled

    @ambient_occlusion_enabled.setter
    def ambient_occlusion_enabled(self, value: bool) -> None:
        self.update_render_config_field("ambient_occlusion", "enabled", value)

    @property
    def ambient_occlusion_radius(self) -> float | None:
        """Occlusion hemisphere radius in scene units, or ``None`` for auto.

        ``None`` derives the radius from the scene bounding box diagonal
        (:attr:`ambient_occlusion_auto_radius_fraction`, 2 percent by default), which is
        the only default that means anything across cellier's coordinate
        systems.  :attr:`ambient_occlusion_effective_radius` reports what that came to.
        """
        return self._render_manager.ambient_occlusion_radius

    @ambient_occlusion_radius.setter
    def ambient_occlusion_radius(self, value: float | None) -> None:
        self.update_render_config_field("ambient_occlusion", "radius", value)

    @property
    def ambient_occlusion_auto_radius_fraction(self) -> float:
        """Fraction of the scene bounding box diagonal used when radius is auto."""
        return self._render_manager.ambient_occlusion_auto_radius_fraction

    @ambient_occlusion_auto_radius_fraction.setter
    def ambient_occlusion_auto_radius_fraction(self, value: float) -> None:
        self.update_render_config_field(
            "ambient_occlusion", "auto_radius_fraction", value
        )

    @property
    def ambient_occlusion_effective_radius(self) -> float | None:
        """The occlusion radius actually in use, in scene units.

        The explicit :attr:`ambient_occlusion_radius` when one is set, otherwise the
        auto-derived value.  Read-only, and the number worth showing next to
        the radius control: a radius means nothing until it can be compared
        with the scale of the thing being rendered.  ``None`` when there is
        no canvas to ask.
        """
        return self._render_manager.ambient_occlusion_effective_radius

    @property
    def ambient_occlusion_strength(self) -> float:
        """How far the occlusion is applied, 0 (off) to 1 (full)."""
        return self._render_manager.ambient_occlusion_strength

    @ambient_occlusion_strength.setter
    def ambient_occlusion_strength(self, value: float) -> None:
        self.update_render_config_field("ambient_occlusion", "strength", value)

    @property
    def ambient_occlusion_power(self) -> float:
        """Contrast exponent applied to the occlusion before the multiply."""
        return self._render_manager.ambient_occlusion_power

    @ambient_occlusion_power.setter
    def ambient_occlusion_power(self, value: float) -> None:
        self.update_render_config_field("ambient_occlusion", "power", value)

    @property
    def ambient_occlusion_bias(self) -> float:
        """Depth-comparison bias, as a fraction of the effective radius.

        Dimensionless on purpose: an absolute bias tuned for one coordinate
        system self-occludes a flat plane in another.
        """
        return self._render_manager.ambient_occlusion_bias

    @ambient_occlusion_bias.setter
    def ambient_occlusion_bias(self, value: float) -> None:
        self.update_render_config_field("ambient_occlusion", "bias", value)

    @property
    def ambient_occlusion_n_samples(self) -> int:
        """Hemisphere samples per pixel.  Changing this recompiles the shader."""
        return self._render_manager.ambient_occlusion_n_samples

    @ambient_occlusion_n_samples.setter
    def ambient_occlusion_n_samples(self, value: int) -> None:
        self.update_render_config_field("ambient_occlusion", "n_samples", value)

    @property
    def ambient_occlusion_blur_radius(self) -> int:
        """Occlusion box-blur half-width in internal pixels.  Recompiles."""
        return self._render_manager.ambient_occlusion_blur_radius

    @ambient_occlusion_blur_radius.setter
    def ambient_occlusion_blur_radius(self, value: int) -> None:
        self.update_render_config_field("ambient_occlusion", "blur_radius", value)

    # -- Temporal accumulation -----------------------------------------

    @property
    def temporal_enabled(self) -> bool:
        """Whether the temporal accumulation pass is active.

        The pass averages successive jittered frames, which is what lets the
        volume raymarcher and the occlusion kernel use few samples per frame
        and still settle to a clean image when the camera stops.  It is off
        in 2D whatever this says.
        """
        return self._render_manager.temporal_enabled

    @temporal_enabled.setter
    def temporal_enabled(self, value: bool) -> None:
        self.update_render_config_field("temporal", "enabled", value)

    @property
    def temporal_blend_weight(self) -> float:
        """Minimum EMA blend weight for the current frame, in ``(0, 1]``.

        Lower values give a smoother settled image and take longer to get
        there after a camera move.
        """
        return self._render_manager.temporal_blend_weight

    @temporal_blend_weight.setter
    def temporal_blend_weight(self, value: float) -> None:
        self.update_render_config_field("temporal", "blend_weight", value)

    def reset_temporal_accumulation(self) -> None:
        """Discard the accumulated history on every canvas.

        The next frame is shown raw and accumulation restarts from it.
        Cellier already does this on every camera and content change; this
        is for a caller who has changed something cellier cannot see.
        """
        self._render_manager.reset_temporal_accumulation()

    def apply_ambient_occlusion_config(self) -> None:
        """Push ``render_config.ambient_occlusion`` onto every canvas's occlusion pass.

        Needed after mutating the config model in place.  ``n_samples`` and
        ``blur_radius`` are shader template vars, so changing them
        recompiles; the rest are uniforms and do not.
        """
        self._render_manager.apply_ambient_occlusion_config()

    def apply_outline_config(self) -> None:
        """Push ``render_config.outline`` onto every canvas's outline pass.

        Call this after mutating thicknesses or colours in place.  Changing
        a thickness recompiles the outline shader; enables, colours and the
        palette do not.
        """
        self._render_manager.apply_outline_config()

    @property
    def camera_settle_threshold_s(self) -> float:
        """Debounce delay before reslice after camera movement."""
        return self._render_manager.config.camera.settle_threshold_s

    @camera_settle_threshold_s.setter
    def camera_settle_threshold_s(self, value: float) -> None:
        self._render_manager.config.camera.settle_threshold_s = value

    def _on_canvas_size_changed(self, event: CanvasSizeChangedEvent) -> None:
        """Update Canvas.size in the model when the backend reports a resize."""
        scene_id = self._canvas_to_scene.get(event.canvas_id)
        if scene_id is None:
            return
        canvas_model = self._model.scenes[scene_id].canvases.get(event.canvas_id)
        if canvas_model is None:
            return
        canvas_model.size = (event.width, event.height)

    def _on_camera_changed(self, event: CameraChangedEvent) -> None:
        """Synchronous bus handler: updates camera model and schedules settle task."""
        self._update_camera_model(event.scene_id, event.source_id, event.camera_state)

        if not self._render_manager.config.camera.reslice_enabled:
            return

        canvas_id = event.source_id
        existing = self._settle_tasks.get(canvas_id)
        if existing is not None and not existing.done():
            _CAMERA_LOGGER.debug(
                "settle_cancel  canvas=%s  scene=%s",
                canvas_id,
                event.scene_id,
            )
            existing.cancel()

        _CAMERA_LOGGER.debug(
            "settle_schedule  canvas=%s  scene=%s  threshold=%.3fs",
            canvas_id,
            event.scene_id,
            self._render_manager.config.camera.settle_threshold_s,
        )
        self._settle_tasks[canvas_id] = asyncio.create_task(
            self._settle_after(canvas_id, event.scene_id)
        )

    def _update_camera_model(
        self, scene_id: UUID, canvas_id: UUID, camera_state: CameraState
    ) -> None:
        """Write a CameraState snapshot back into the model-layer camera.

        Branches on the actual model type: ``PerspectiveCamera`` writes
        ``up_direction`` and ``fov``; ``OrthographicCamera`` writes ``width``
        and ``height`` (``extent``).

        Parameters
        ----------
        scene_id : UUID
            Scene that owns the canvas.
        canvas_id : UUID
            The specific canvas whose camera moved (``event.source_id``).
        camera_state : CameraState
            Snapshot to write back.
        """
        scene = self._model.scenes[scene_id]
        canvas_model = scene.canvases.get(canvas_id)
        if canvas_model is None:
            return
        canvas_view = self._render_manager._canvases.get(canvas_id)
        if canvas_view is not None:
            active_dim = canvas_view._dim
        else:
            # No render-layer canvas (e.g. headless tests): infer from type.
            active_dim = "2d" if camera_state.camera_type == "orthographic" else "3d"
        camera_model = canvas_model.cameras.get(active_dim)
        if camera_model is None:
            return

        # Common fields present on both camera model types.
        camera_model.position = np.array(camera_state.position, dtype=np.float32)
        camera_model.rotation = np.array(camera_state.rotation, dtype=np.float32)
        camera_model.zoom = camera_state.zoom
        camera_model.near_clipping_plane = camera_state.depth_range[0]
        camera_model.far_clipping_plane = camera_state.depth_range[1]

        # Type-specific fields.
        if isinstance(camera_model, PerspectiveCamera):
            camera_model.up_direction = np.array(camera_state.up, dtype=np.float32)
            camera_model.fov = camera_state.fov
        elif isinstance(camera_model, OrthographicCamera):
            camera_model.width = camera_state.extent[0]
            camera_model.height = camera_state.extent[1]

    async def _settle_after(self, canvas_id: UUID, scene_id: UUID) -> None:
        """Wait for the settle threshold, then reslice camera-sensitive visuals."""
        try:
            await asyncio.sleep(self._render_manager.config.camera.settle_threshold_s)
        except asyncio.CancelledError:
            raise

        scene = self._model.scenes[scene_id]
        target_ids = frozenset(v.id for v in scene.visuals if v.requires_camera_reslice)

        if not target_ids:
            return

        dims_state = self._dims_state_for_scene(scene_id)
        visual_configs = self._build_visual_configs_for_scene(scene_id)

        _CAMERA_LOGGER.info(
            "settle_reslice  scene=%s  visuals=%d",
            scene_id,
            len(target_ids),
        )

        self._render_manager.reslice_scene(
            scene_id=scene_id,
            dims_state=dims_state,
            visual_configs=visual_configs,
            target_visual_ids=target_ids,
            selections=self._selections_for_scene(scene_id),
        )

    # ------------------------------------------------------------------
    # Camera operations
    # ------------------------------------------------------------------

    def look_at_visual(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        view_direction: tuple[float, float, float] = (-1, -1, -1),
        up: tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """Fit the camera to a visual's bounding box.

        Parameters
        ----------
        visual_id : UUID
            ID of the target visual.
        canvas_id : UUID
            ID of the canvas whose camera should be fitted.
        view_direction : tuple[float, float, float]
            Camera look direction vector (need not be normalized).
        up : tuple[float, float, float]
            Camera up vector.
        """
        self._render_manager.look_at_visual(visual_id, canvas_id, view_direction, up)
        scene_id = self._visual_to_scene[visual_id]
        state = self.get_canvas_view(canvas_id).capture_camera_state()
        self._update_camera_model(scene_id, canvas_id, state)

    def set_camera_depth_range(
        self,
        canvas_id: UUID,
        depth_range: tuple[float, float],
    ) -> None:
        """Set the near/far clip distances for a canvas camera.

        Parameters
        ----------
        canvas_id : UUID
            ID of the target canvas.
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        self._render_manager.set_camera_depth_range(canvas_id, depth_range)

    # ------------------------------------------------------------------
    # Stubs for future features
    # ------------------------------------------------------------------

    def add_paint_controller(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        brush_value: int = 1,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
        autosave_interval_s: float | None = None,
    ):
        """Create and wire a paint controller for a labels visual.

        Only labels can be painted.  The controller is chosen by the visual's
        type -- a ``LabelMemoryVisual`` gets a ``SyncPaintController`` and a
        ``MultiscaleLabelVisual`` a ``MultiscalePaintController`` -- rather
        than by its store, because a multiscale labels visual may be backed
        by a generic ``MultiscaleZarrDataStore``.  Image visuals raise.

        Parameters
        ----------
        visual_id : UUID
            Visual to paint on.
        canvas_id : UUID
            Canvas to bind to.  Its camera controller is disabled for
            the session.  Pass ``controller.get_canvas_ids(scene_id)[0]``
            for the common single-canvas case.
        brush_value : int
            Integer label ID written to every painted voxel.
        brush_radius_voxels : float
            Brush radius in level-0 voxel units.
        history_depth : int
            Maximum undoable strokes.
        autosave_interval_s : float | None
            Seconds between automatic flushes for ``MultiscalePaintController``.
            Each autosave rebuilds the pyramid and resets GPU paint textures.
            ``None`` disables autosave.  Ignored for ``SyncPaintController``.

        Returns
        -------
        AbstractPaintController
            Fully wired; caller owns the object.

        Raises
        ------
        TypeError
            If the visual type has no registered paint controller.
        """
        from cellier.visuals._label_memory import LabelMemoryVisual
        from cellier.visuals._labels import MultiscaleLabelVisual

        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        visual_model = next(v for v in scene.visuals if v.id == visual_id)
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]

        if isinstance(visual_model, LabelMemoryVisual):
            from cellier.paint import SyncPaintController

            displayed_axes = scene.dims.selection.displayed_axes
            if len(displayed_axes) != 2:
                raise NotImplementedError(
                    "SyncPaintController currently only supports 2-D "
                    f"displayed-axis configurations; scene {scene_id} has "
                    f"displayed_axes={displayed_axes!r}."
                )
            return SyncPaintController(
                cellier_controller=self,
                visual_id=visual_id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                data_store=data_store,
                displayed_axes=displayed_axes,
                brush_value=brush_value,
                brush_radius_voxels=brush_radius_voxels,
                history_depth=history_depth,
            )

        if isinstance(visual_model, MultiscaleLabelVisual):
            from cellier.paint import MultiscalePaintController

            displayed_axes = scene.dims.selection.displayed_axes
            if len(displayed_axes) != 2:
                raise NotImplementedError(
                    "MultiscalePaintController currently only supports 2-D "
                    f"displayed-axis configurations; scene {scene_id} has "
                    f"displayed_axes={displayed_axes!r}.  3-D paint "
                    "feedback is a Phase 3 follow-up."
                )

            block_size = visual_model.render_config.block_size
            return MultiscalePaintController(
                cellier_controller=self,
                visual_id=visual_id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                data_store=data_store,
                visual_block_size=block_size,
                displayed_axes=displayed_axes,
                brush_value=brush_value,
                brush_radius_voxels=brush_radius_voxels,
                history_depth=history_depth,
                autosave_interval_s=autosave_interval_s,
            )

        raise TypeError(
            f"No PaintController implementation for visual type "
            f"{type(visual_model).__name__!r}.  "
            f"Supported: LabelMemoryVisual, MultiscaleLabelVisual."
        )

    def remove_scene(self, scene_id: UUID) -> None:
        """Remove a scene and all its visuals and canvases.

        Teardown order mirrors ``remove_visual`` for each child visual, then
        cleans up the scene-level maps and render layer.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to remove.

        Raises
        ------
        KeyError
            If ``scene_id`` is not registered.
        """
        scene = self._model.scenes[scene_id]

        # 1. Tear down all child visuals (psygnal + bus + render layer each).
        for visual_model in list(scene.visuals):
            self.remove_visual(visual_model.id)

        # 2. Cancel any pending camera-settle tasks for this scene's canvases.
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            task = self._settle_tasks.pop(canvas_id, None)
            if task is not None and not task.done():
                task.cancel()

        # 3. Overlays drawn in this scene -- its own and its canvases' --
        #    lose their bridges and render objects with it.
        for overlay_id in [
            overlay_id
            for overlay_id, entry in self._overlays.items()
            if entry.scene_id == scene_id
        ]:
            self._forget_overlay(overlay_id)

        # 3b. Bus cleanup for canvases and the scene itself.
        for canvas_id in self._scene_to_canvases.pop(scene_id, []):
            self._outgoing_events.unsubscribe_all(canvas_id)
            self._canvas_to_scene.pop(canvas_id, None)
            self._forget_rendered(canvas_id)
        self._outgoing_events.unsubscribe_all(scene_id)

        # 4. Disconnect the scene-level psygnal bridge, as remove_visual does
        #    for its own.
        for signal, handler in self._scene_psygnal_handlers.pop(scene_id, []):
            signal.disconnect(handler)

        # 5. Clean up controller-side scene maps.
        self._forget_coordinate_systems(scene.dims.world_coordinate_system)
        self._dims_cache.pop(scene_id, None)
        self._slice_cache.pop(scene_id, None)
        self._slider_axes_cache.pop(scene_id, None)
        self._scene_render_modes.pop(scene_id, None)
        self._scene_background_bridges.pop(scene_id, None)

        # 6. Remove from model layer.
        self._model.scenes.pop(scene_id)

        # 7. Render-layer teardown (drops gfx.Scene, canvas widgets, GPU refs).
        self._render_manager.remove_scene(scene_id)

        # 8. Notify external observers.
        self._outgoing_events.emit(
            SceneRemovedEvent(source_id=self._id, scene_id=scene_id)
        )

    def remove_canvas(self, canvas_id: UUID) -> None:
        """Remove a canvas from its scene, disconnecting all wiring.

        Teardown order mirrors ``remove_scene`` for the canvas-level steps:
        1. Cancel any pending camera-settle task.
        2. Remove bus subscriptions owned by this canvas.
        3. Update controller lookup maps.
        4. Remove from the model layer.
        5. Render-layer teardown (drops widget and GPU references).

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas to remove.

        Raises
        ------
        KeyError
            If ``canvas_id`` is not registered.
        """
        scene_id = self._canvas_to_scene[canvas_id]

        # 1. Cancel any pending camera-settle task.
        task = self._settle_tasks.pop(canvas_id, None)
        if task is not None and not task.done():
            task.cancel()

        # 2. Remove bus subscriptions owned by this canvas.
        self._outgoing_events.unsubscribe_all(canvas_id)
        self._pick_subscriber_counts.pop(canvas_id, None)
        for key in [key for key in self._pick_event_counts if key[0] == canvas_id]:
            del self._pick_event_counts[key]
        self._cancel_move_pick_read(canvas_id)

        # 3. Update controller lookup maps, dropping the canvas's overlays.
        for overlay_id in [
            overlay_id
            for overlay_id, entry in self._overlays.items()
            if entry.kind == "canvas" and entry.owner_id == canvas_id
        ]:
            self._forget_overlay(overlay_id)
        self._canvas_to_scene.pop(canvas_id)
        self._scene_to_canvases[scene_id].remove(canvas_id)
        self._forget_rendered(canvas_id)

        # 4. Remove from the model layer.
        self._model.scenes[scene_id].canvases.pop(canvas_id)

        # 5. Render-layer teardown.
        self._render_manager.remove_canvas(canvas_id)

    def remove_visual(self, visual_id: UUID) -> None:
        """Remove a visual from its scene, disconnecting all wiring.

        Teardown order:
        1. Psygnal bridge handlers disconnected first — prevents the
           bridge closures from firing during any subsequent model access.
        2. Bus subscriptions removed — prevents dangling GFX-layer handlers
           from receiving events after the node is gone from the scene graph.
        3. Render-layer removal — drops scene-graph node and GPU references.
        4. ``VisualRemovedEvent`` emitted for external observers.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.

        Raises
        ------
        KeyError
            If ``visual_id`` is not registered.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        visual_model = self.get_visual_model(visual_id)

        # 1. Remove from model layer.
        scene.visuals.remove(visual_model)

        # 2. Disconnect psygnal bridge handlers.
        for signal, handler in self._visual_psygnal_handlers.pop(visual_id, []):
            signal.disconnect(handler)
        for signal, handler in self._channel_psygnal_handlers.pop(visual_id, []):
            signal.disconnect(handler)
        single_bridge = self._single_bridges.pop(visual_id, None)
        if single_bridge is not None:
            single_bridge[0].events.disconnect(single_bridge[1])

        # 3. Remove bus subscriptions for this visual's GFX handlers, and
        #    cancel its pick value reads, press and release included.
        self._outgoing_events.unsubscribe_all(visual_id)
        self._cancel_visual_pick_reads(visual_id)

        # 4. Remove from controller lookup maps.
        self._visual_to_scene.pop(visual_id)
        self._forget_visual_spaces(visual_id)

        # 5. Render-layer teardown.
        self._render_manager.remove_visual(visual_id)
        self._check_slider_axes(scene_id)
        self._refresh_scene_overlays(scene_id)

        # 6. Notify external observers.
        self._outgoing_events.emit(
            VisualRemovedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_id,
            )
        )
        # Removal happens after the visual left _visual_to_scene, so this has
        # to go through the scene rather than the (now unmapped) visual.
        self._request_draw_for_scene(scene_id)

    def remove_data_store(self, data_store_id: UUID) -> None:
        """Remove a data store from the model.

        Raises ``ValueError`` if any live visual still references the store.
        Call ``remove_visual`` for each referencing visual first.

        Parameters
        ----------
        data_store_id : UUID
            ID of the data store to remove.

        Raises
        ------
        ValueError
            If one or more visuals still reference the store.  The error
            message names each visual so the caller can identify them.
        KeyError
            If ``data_store_id`` is not registered.
        """
        referencing = [
            v
            for scene in self._model.scenes.values()
            for v in scene.visuals
            if UUID(v.data_store_id) == data_store_id
        ]
        if referencing:
            names = ", ".join(f"{v.name!r} ({v.id})" for v in referencing)
            raise ValueError(
                f"Cannot remove data store {data_store_id}: "
                f"still referenced by visuals: {names}"
            )
        store = self._model.data.stores.pop(data_store_id)
        self._forget_coordinate_systems(*store.data_coordinate_systems)
        connection = self._store_psygnal_handlers.pop(data_store_id, None)
        if connection is not None:
            connection[0].disconnect(connection[1])

    # ------------------------------------------------------------------
    # External event subscriptions
    # ------------------------------------------------------------------

    def on_dims_changed(
        self,
        scene_id: UUID,
        callback: Callable[[DimsChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the dims for *scene_id* change.

        The callback receives the full ``DimsChangedEvent``, which includes
        ``source_id`` for echo-filtering and ``dims_state`` for the new state.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``DimsChangedEvent`` on each dims change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.  Use for
            transient widgets that may be destroyed outside the controller's
            teardown path.  Cannot be used with lambdas.

        Returns
        -------
        SubscriptionHandle
            Pass to ``EventBus.unsubscribe()`` for individual removal.
        """
        return self._outgoing_events.subscribe(
            DimsChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_camera_changed(
        self,
        scene_id: UUID,
        callback: Callable[[CameraChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the camera for *scene_id* changes.

        The callback receives a ``CameraChangedEvent`` carrying the latest
        ``CameraState``, including ``extent`` (width, height) for
        ``OrthographicCamera`` scenes.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``CameraChangedEvent`` on each camera change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            CameraChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def _promote_pick_coordinate(
        self,
        displayed_data_coord: Sequence[float],
        *,
        displayed_axes: Sequence[int],
        slice_indices: Mapping[int, float],
        world_ndim: int,
        hit_visual_id: UUID | None,
        collapsed_data_indices: Sequence[tuple[int, int]] | None = None,
    ) -> tuple[float, ...]:
        """Join a partial pick coordinate into a level-0 data coordinate.

        The render layer decodes only the axes it drew.  The rest are the
        planes the visual collapsed, and it reports them itself in
        *collapsed_data_indices* -- read off the plan it last drew, so the
        answer is the slice on screen rather than the one the current dims
        state implies.  Those two differ while a reslice is in flight, and a
        pick is a question about the screen.

        **One convention throughout.** ``floor`` of every component gives the
        voxel index, displayed and collapsed alike.  The displayed components
        arrive that way already (``_pick`` adds the half-voxel).  A collapsed
        axis has no sub-voxel position -- the visual drew one plane -- so its
        component is the centre of that plane, ``index + 0.5``.  Emitting the
        raw pulled-back position instead would put the two halves of one tuple
        in two conventions, and ``floor`` would be right for one and wrong for
        the other whenever the slider sat past a voxel's midpoint.

        The fallback, for a visual that reported nothing, pulls the world
        slice positions back through the transform and rounds them the way the
        selection assembler does (``round_world_to_voxel``, half-up).  It lands
        on the same plane whenever no reslice is pending, and it keeps a
        headlessly constructed visual answerable.

        The result is in **data**-axis order and has the hit visual's rank: a
        world axis the data broadcasts over contributes nothing, because the
        data has no such axis.  A data axis with no world counterpart -- a
        multichannel store's composited channel axis -- is zero, since nothing
        in the pick says which channel was hit.

        Parameters
        ----------
        displayed_data_coord : Sequence[float]
            Level-0 data position on the displayed axes only, in pygfx
            ``(x, y[, z])`` order.
        displayed_axes : Sequence[int]
            The scene's displayed world axes, in the order the canvas draws
            them.
        slice_indices : Mapping[int, float]
            World slice position per non-displayed world axis.  Used only by
            the fallback.
        world_ndim : int
            The scene world's rank, used for the fallback below.
        hit_visual_id : UUID or None
            The visual that was hit.  When it cannot be resolved to a model
            with a transform -- a background miss, or a removed visual -- there
            is nothing to pull the world positions back through, so the world
            positions are returned on their world axes as they were before this
            pull-back existed.
        collapsed_data_indices : Sequence[tuple[int, int]] or None
            ``(data axis, level-0 voxel index)`` from the visual's last plan.

        Returns
        -------
        tuple[float, ...]
            One coordinate per data axis, ascending.
        """
        # The render layer decodes displayed-axis coordinates in pygfx
        # (x, y[, z]) order, which is the reverse of cellier's ascending
        # ``displayed_axes`` (..., row, col) order.  Reverse so each value lands
        # on its true axis (e.g. the column coordinate on the x axis, not on the
        # first displayed axis) -- otherwise the displayed axes come out
        # transposed.
        decoded = tuple(displayed_data_coord)[::-1]

        visual = (
            self._model_visual_or_none(hit_visual_id)
            if hit_visual_id is not None
            else None
        )
        transform = getattr(visual, "transform", None)
        if transform is None:
            coordinate = np.zeros(world_ndim, dtype=np.float64)
            for slot, world_axis in enumerate(displayed_axes):
                coordinate[world_axis] = decoded[slot]
            for world_axis, world_position in slice_indices.items():
                coordinate[world_axis] = float(world_position)
            return tuple(float(value) for value in coordinate)

        data_axis_of_world = {
            world_axis: data_axis
            for data_axis, world_axis in axis_correspondence(transform).items()
        }
        coordinate = np.zeros(transform.input_ndim, dtype=np.float64)
        for slot, world_axis in enumerate(displayed_axes):
            data_axis = data_axis_of_world.get(world_axis)
            if data_axis is not None:
                coordinate[data_axis] = decoded[slot]

        if collapsed_data_indices is not None:
            for data_axis, index in collapsed_data_indices:
                if 0 <= data_axis < coordinate.size:
                    coordinate[data_axis] = float(index) + 0.5
            return tuple(float(value) for value in coordinate)

        # Fallback: the visual reported no plan, so derive the plane from the
        # dims state the way the selection assembler would have.
        #
        # Pulled back through imap_coordinates rather than by reading
        # .linear / .translation, so a non-affine data -> world transform
        # works here too.  Mapping the whole world point at once is exact
        # because axis_correspondence above already established that the
        # transform is axis-aligned -- each data axis is fed by exactly one
        # world axis, so the axes this loop does not fill cannot perturb the
        # ones it reads.
        world_point = np.zeros(transform.output_ndim, dtype=np.float64)
        for world_axis, world_position in slice_indices.items():
            if 0 <= world_axis < world_point.size:
                world_point[world_axis] = float(world_position)
        try:
            data_point = np.asarray(transform.imap_coordinates(world_point))
        except NonInvertibleTransformError:
            # No left inverse: there is no data coordinate to report, and a
            # pick is not worth raising over.
            return tuple(float(value) for value in coordinate)
        for world_axis in slice_indices:
            data_axis = data_axis_of_world.get(world_axis)
            if data_axis is None:
                # The visual broadcasts over this world axis: it exists at
                # every position along it and has no coordinate of its own.
                continue
            position = float(data_point[data_axis])
            if not np.isfinite(position):
                # Outside this visual's extent on a bounded axis: it has no
                # data coordinate there at all.
                continue
            coordinate[data_axis] = float(np.floor(position + 0.5)) + 0.5
        return tuple(float(value) for value in coordinate)

    def _on_raw_pointer_event(self, event: _CanvasRawPointerEvent) -> None:
        """Emit the public mouse event, then the typed pick event for a hit.

        Reads displayed_axes and slice_indices from the scene model to fill
        the full world coordinate, emits the canvas mouse event, and then --
        for a hit that some ``on_pick`` subscriber wants -- the matching pick
        event (design 3.7).
        """
        from cellier.events._events import CanvasPickInfo

        scene_model = self._model.scenes[event.scene_id]
        dims = scene_model.dims
        displayed_axes = dims.selection.displayed_axes
        # A displayed axis keeps a stored position (D36), but the pointer
        # supplies its coordinate; only the sliced axes come from dims.
        slice_indices = {
            axis: position
            for axis, position in dims.selection.slice_indices.items()
            if axis not in displayed_axes
        }
        axis_labels = dims.axis_labels
        n_dims = len(axis_labels)

        pick_info = CanvasPickInfo(hit_visual_id=event.hit_visual_id)
        world_coord: np.ndarray | None = None

        if event.camera_type == "2d":
            world_coord = np.empty(n_dims, dtype=np.float64)
            # Embed pygfx position_2d into world_coord verbatim:
            #   world_coord[displayed_axes[0]] = position_2d[0]  (pygfx-x)
            #   world_coord[displayed_axes[1]] = position_2d[1]  (pygfx-y)
            # Both image visuals render data[r, c] at pygfx (x=c, y=r), so
            # world_coord[ax_row] == pygfx_x (column) and
            # world_coord[ax_col] == pygfx_y (row) after this assignment.
            # Paint controllers swap ax_row ↔ ax_col to recover (row, col)
            # voxel order before calling imap_coordinates.
            for i, axis in enumerate(displayed_axes):
                world_coord[axis] = event.position_2d[i]  # type: ignore[index]
            for axis, idx in slice_indices.items():
                world_coord[axis] = float(idx)

            _CLS_2D = {
                "press": CanvasMousePress2DEvent,
                "move": CanvasMouseMove2DEvent,
                "release": CanvasMouseRelease2DEvent,
            }
            self._outgoing_events.emit(
                _CLS_2D[event.action](
                    source_id=event.canvas_id,
                    scene_id=event.scene_id,
                    world_coordinate=world_coord,
                    pick_info=pick_info,
                    button=event.button,
                    buttons=event.buttons,
                    modifiers=event.modifiers,
                    gesture_id=event.gesture_id,
                )
            )
        else:
            _CLS_3D = {
                "press": CanvasMousePress3DEvent,
                "move": CanvasMouseMove3DEvent,
                "release": CanvasMouseRelease3DEvent,
            }
            self._outgoing_events.emit(
                _CLS_3D[event.action](
                    source_id=event.canvas_id,
                    scene_id=event.scene_id,
                    ray=event.ray,
                    pick_info=pick_info,
                    button=event.button,
                    buttons=event.buttons,
                    modifiers=event.modifiers,
                    gesture_id=event.gesture_id,
                )
            )

        # A newer pointer event supersedes an in-flight ``move`` value read;
        # ``press`` and ``release`` reads always complete (design 3.7).
        self._cancel_move_pick_read(event.canvas_id)
        if event.hit_visual_id is not None and event.pick_details is not None:
            self._emit_pick_event(
                event,
                world_coordinate=world_coord,
                displayed_axes=displayed_axes,
                slice_indices=slice_indices,
                world_ndim=n_dims,
            )

    def _emit_pick_event(
        self,
        event: _CanvasRawPointerEvent,
        *,
        world_coordinate: np.ndarray | None,
        displayed_axes: tuple[int, ...],
        slice_indices: dict[int, float],
        world_ndim: int,
    ) -> None:
        """Emit the typed pick event for a hit, if its type has a subscriber.

        Points, lines, mesh and graph details are complete as decoded.  Image
        and labels picks are promoted to full-rank data coordinates first, and
        carry the values under them: read now from an in-memory store's array,
        or at level 0 through the slicer for any other store, in which case the
        event is emitted when the read completes.
        """
        from cellier.render.render_manager import (
            _ImageDisplayedDataCoord,
            _LabelsDisplayedDataCoord,
        )

        raw_pick = event.pick_details
        event_type = _pick_event_type(raw_pick)
        if event_type is None or not self._pick_event_counts.get(
            (event.canvas_id, event_type)
        ):
            return
        common = {
            "source_id": event.canvas_id,
            "scene_id": event.scene_id,
            "visual_id": event.hit_visual_id,
            "action": event.action,
            "camera_type": event.camera_type,
            "world_coordinate": world_coordinate,
            "ray": event.ray,
            "button": event.button,
            "buttons": event.buttons,
            "modifiers": event.modifiers,
            "gesture_id": event.gesture_id,
        }
        if not isinstance(
            raw_pick, (_ImageDisplayedDataCoord, _LabelsDisplayedDataCoord)
        ):
            self._outgoing_events.emit(event_type(**common, pick_info=raw_pick))
            return

        visual = self._model_visual_or_none(event.hit_visual_id)
        store = (
            None
            if visual is None
            else self._model.data.stores.get(UUID(visual.data_store_id))
        )
        if store is None:
            return
        # The render layer decodes the displayed axes into level-0 data
        # coordinates; the other axes come from the plan the visual last drew,
        # or failing that from the dims state pulled back through the
        # visual's transform.
        coordinate = self._promote_pick_coordinate(
            raw_pick.displayed_data_coord,
            displayed_axes=displayed_axes,
            slice_indices=slice_indices,
            world_ndim=world_ndim,
            hit_visual_id=event.hit_visual_id,
            collapsed_data_indices=raw_pick.collapsed_data_indices,
        )
        shape = tuple(int(size) for size in store.level_shapes[0])

        if isinstance(raw_pick, _LabelsDisplayedDataCoord):
            index = _voxel_index(coordinate, shape)
            reads = {} if index is None else {0: index}
            labels_coordinate = coordinate

            def build(values: dict) -> LabelsPickEvent:
                return LabelsPickEvent(
                    **common,
                    pick_info=LabelsPickInfo(
                        data_coordinate=labels_coordinate,
                        value=int(values.get(0, 0)),
                    ),
                )

        else:
            image_coordinate, positions = _image_pick_positions(
                visual, coordinate, raw_pick, event.camera_type
            )
            reads = {}
            for channel, position in positions.items():
                index = _voxel_index(position, shape)
                if index is not None:
                    reads[channel] = index

            def build(values: dict) -> ImagePickEvent:
                return ImagePickEvent(
                    **common,
                    pick_info=ImagePickInfo(
                        data_coordinate=image_coordinate,
                        channel_values={
                            channel: float(values[channel])
                            for channel in sorted(values)
                        },
                    ),
                )

        self._read_pick_values(event, store, reads, build)

    def _read_pick_values(
        self,
        event: _CanvasRawPointerEvent,
        store: Any,
        reads: dict[int, tuple[int, ...]],
        build: Callable[[dict], Any],
    ) -> None:
        """Read the voxels in *reads* from *store* and emit ``build(values)``.

        An in-memory store holds a plain array (its ``get_data`` is a
        coroutine only for the slicer's sake), so the read and the emit happen
        here.  Any other store is read at level 0 through the render layer's
        cancellable slicer, and the event is emitted when every value has
        arrived -- never, if the read is cancelled.
        """
        array = getattr(store, "data", None)
        if not reads or isinstance(array, np.ndarray):
            values = (
                {key: array[index] for key, index in reads.items()} if reads else {}
            )
            self._outgoing_events.emit(build(values))
            return

        from cellier.data.image._image_requests import ChunkRequest

        read_id = uuid4()
        keys: dict[UUID, int] = {}
        requests = []
        for key, index in reads.items():
            request = ChunkRequest(
                chunk_request_id=uuid4(),
                slice_request_id=read_id,
                scale_index=0,
                axis_selections=index,
            )
            keys[request.chunk_request_id] = key
            requests.append(request)
        values: dict[int, Any] = {}
        canvas_id = event.canvas_id
        visual_id = event.hit_visual_id

        def on_batch(batch: list) -> None:
            for request, data in batch:
                values[keys[request.chunk_request_id]] = np.asarray(data).reshape(-1)[0]

        def on_complete() -> None:
            self._forget_pick_read(canvas_id, visual_id, read_id)
            self._outgoing_events.emit(build(values))

        self._render_manager.submit_pick_read(
            requests, store.get_data, on_batch, on_complete
        )
        self._pick_reads_by_visual.setdefault(visual_id, set()).add(read_id)
        if event.action == "move":
            self._move_pick_reads[canvas_id] = read_id

    def _forget_pick_read(
        self, canvas_id: UUID, visual_id: UUID, read_id: UUID
    ) -> None:
        """Drop a finished pick read from the bookkeeping."""
        reads = self._pick_reads_by_visual.get(visual_id)
        if reads is not None:
            reads.discard(read_id)
            if not reads:
                del self._pick_reads_by_visual[visual_id]
        if self._move_pick_reads.get(canvas_id) == read_id:
            del self._move_pick_reads[canvas_id]

    def _cancel_move_pick_read(self, canvas_id: UUID) -> None:
        """Cancel the canvas's in-flight ``move`` value read, if any."""
        read_id = self._move_pick_reads.pop(canvas_id, None)
        if read_id is None:
            return
        self._render_manager.cancel_pick_read(read_id)
        for visual_id, reads in list(self._pick_reads_by_visual.items()):
            reads.discard(read_id)
            if not reads:
                del self._pick_reads_by_visual[visual_id]

    def _cancel_visual_pick_reads(self, visual_id: UUID) -> None:
        """Cancel every in-flight value read for *visual_id*."""
        reads = self._pick_reads_by_visual.pop(visual_id, set())
        for read_id in reads:
            self._render_manager.cancel_pick_read(read_id)
        for canvas_id, read_id in list(self._move_pick_reads.items()):
            if read_id in reads:
                del self._move_pick_reads[canvas_id]

    def _register_pick_subscriber(
        self, canvas_id: UUID, handle: SubscriptionHandle
    ) -> SubscriptionHandle:
        """Record a pick subscription and enable pick details for its canvas.

        Increments the per-canvas picking-subscriber count; the first
        subscriber flips on element-detail extraction in the render layer.

        Parameters
        ----------
        canvas_id : UUID
            The canvas the subscription watches.
        handle : SubscriptionHandle
            The handle returned by the bus subscription.

        Returns
        -------
        SubscriptionHandle
            The same handle, for convenient return from the callers.
        """
        count = self._pick_subscriber_counts.get(canvas_id, 0)
        self._pick_subscriber_counts[canvas_id] = count + 1
        if count == 0:
            self._render_manager.set_pick_details_enabled(canvas_id, True)
        return handle

    def unsubscribe_mouse(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription created by an ``on_mouse_*`` method.

        Mouse subscriptions do not gate pick details; this is
        ``EventBus.unsubscribe``, kept so a handle from ``on_pick`` passed
        here by mistake still updates the pick counters.

        Parameters
        ----------
        handle : SubscriptionHandle
            A handle returned by one of the ``on_mouse_*`` methods.
        """
        self.unsubscribe_pick(handle)

    def on_pick(
        self,
        canvas_id: UUID,
        event_type: type,
        callback: Callable[[Any], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a visual of one kind is picked.

        Mouse events say *that* something was hit; pick events say *what*
        (unified image design 3.7).  A pick event follows the mouse event for
        the same pointer event, carries the same ``gesture_id``, ``action``,
        button, buttons and modifiers, and is emitted only for hits.  Image
        and labels events carry the values under the pointer; for multiscale
        visuals those are read asynchronously, so match events by
        ``gesture_id`` and ``action`` rather than by arrival order.

        Pick details are extracted while the canvas has any pick subscriber,
        and image and labels values are read only while that event type has
        one.

        Parameters
        ----------
        canvas_id : UUID
            The canvas to watch.
        event_type : type
            One of :data:`cellier.events.PICK_EVENT_TYPES`: ``ImagePickEvent``,
            ``LabelsPickEvent``, ``PointsPickEvent``, ``LinesPickEvent``,
            ``MeshPickEvent`` or ``GraphPickEvent``.
        callback : Callable
            Called with each pick event.
        owner_id : UUID
            UUID under which this subscription is registered for bulk removal
            via ``unsubscribe_all(owner_id)``.
        weak : bool
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
            Pass it to :meth:`unsubscribe_pick`.

        Raises
        ------
        TypeError
            If *event_type* is not a pick event type.
        """
        if event_type not in PICK_EVENT_TYPES:
            names = ", ".join(t.__name__ for t in PICK_EVENT_TYPES)
            raise TypeError(
                f"on_pick takes a pick event type ({names}), got {event_type!r}."
            )
        handle = self._outgoing_events.subscribe(
            event_type,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        key = (canvas_id, event_type)
        self._pick_event_counts[key] = self._pick_event_counts.get(key, 0) + 1
        self._pick_handles[id(handle)] = key
        return self._register_pick_subscriber(canvas_id, handle)

    def unsubscribe_pick(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription created by :meth:`on_pick`.

        Keeps the per-canvas and per-type subscriber counts accurate: the last
        subscriber for a type stops its value reads, and the last for a canvas
        disables pick-detail extraction there.

        Parameters
        ----------
        handle : SubscriptionHandle
            A handle returned by :meth:`on_pick`.
        """
        key = self._pick_handles.pop(id(handle), None)
        if key is not None:
            canvas_id, _event_type = key
            remaining = self._pick_event_counts.get(key, 0) - 1
            if remaining <= 0:
                self._pick_event_counts.pop(key, None)
            else:
                self._pick_event_counts[key] = remaining
            count = self._pick_subscriber_counts.get(canvas_id, 0) - 1
            if count <= 0:
                self._pick_subscriber_counts.pop(canvas_id, None)
                self._render_manager.set_pick_details_enabled(canvas_id, False)
            else:
                self._pick_subscriber_counts[canvas_id] = count
        self._outgoing_events.unsubscribe(handle)

    def on_mouse_press_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMousePress2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-down event on a 2D canvas.

        Parameters
        ----------
        canvas_id :
            The canvas to watch.
        callback :
            Called with the CanvasMousePress2DEvent on each press.
        owner_id :
            UUID under which this subscription is registered for bulk
            removal via unsubscribe_all(owner_id).
        weak :
            If True, hold only a weak reference to *callback*.
        """
        handle = self._outgoing_events.subscribe(
            CanvasMousePress2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return handle

    def on_mouse_move_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseMove2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-move event on a 2D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseMove2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return handle

    def on_mouse_release_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseRelease2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-up event on a 2D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseRelease2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return handle

    def on_mouse_press_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMousePress3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-down event on a 3D canvas.

        Parameters
        ----------
        canvas_id :
            The canvas to watch.
        callback :
            Called with the CanvasMousePress3DEvent on each press.
        owner_id :
            UUID under which this subscription is registered for bulk
            removal via unsubscribe_all(owner_id).
        weak :
            If True, hold only a weak reference to *callback*.
        """
        handle = self._outgoing_events.subscribe(
            CanvasMousePress3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return handle

    def on_mouse_move_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseMove3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-move event on a 3D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseMove3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return handle

    def on_mouse_release_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseRelease3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-up event on a 3D canvas."""
        handle = self._outgoing_events.subscribe(
            CanvasMouseRelease3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )
        return handle

    def set_camera_controller_enabled(self, canvas_id: UUID, enabled: bool) -> None:
        """Enable or disable the camera controller for one canvas.

        Parameters
        ----------
        canvas_id : UUID
            The canvas whose controller state should change.
        enabled : bool
            False disables the controller (paint session active).
            True restores normal camera interaction (session ended).
        """
        self._render_manager._canvases[canvas_id].set_controller_enabled(enabled)

    def _invalidate_painted_tiles_2d(
        self, visual_id: UUID, dirty_grid_coords: set[tuple[int, int]]
    ) -> int:
        """Evict the visible 2D-cache tiles for *visual_id* listed in dirty grid coords.

        Used by :class:`MultiscalePaintController` after staging level-0
        writes so the next reslice re-fetches through the open paint
        transaction.  No-op for visuals that don't own a 2D tile cache.

        Parameters
        ----------
        visual_id :
            The painted multiscale visual.
        dirty_grid_coords :
            Set of ``(gy, gx)`` tile-grid coordinates whose finest-level
            tiles should be evicted.

        Returns
        -------
        int
            Number of tiles evicted (0 if the visual has no 2D cache or
            no entries matched).
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "invalidate_painted_tiles_2d"):
            return 0
        return gfx_visual.invalidate_painted_tiles_2d(dirty_grid_coords)

    def _patch_painted_tiles_2d(
        self,
        visual_id: UUID,
        voxel_indices: np.ndarray,
        values: np.ndarray,
        displayed_axes: tuple[int, int],
    ) -> int:
        """Write paint into the visual's GPU paint cache (Phase-2 fast path).

        Used by :class:`MultiscalePaintController._write_values` for
        sub-frame visible feedback in 2-D paint sessions.  Mirrors the
        architectural pattern of :meth:`_invalidate_painted_tiles_2d`.

        Parameters
        ----------
        visual_id :
            The painted multiscale visual.
        voxel_indices :
            Shape ``(N, ndim)`` int64.  Level-0 voxel indices in
            data-array axis order.
        values :
            Shape ``(N,)`` float32.  Brush values.
        displayed_axes :
            ``(row_axis, col_axis)`` for the 2-D display.

        Returns
        -------
        int
            Number of tiles successfully patched.  Returns 0 if the visual
            has no 2-D paint resources or every tile was rejected by the
            slot manager.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "patch_paint_texture"):
            return 0
        return gfx_visual.patch_paint_texture(voxel_indices, values, displayed_axes)

    def _clear_painted_tiles_2d(self, visual_id: UUID) -> None:
        """Clear the GPU paint textures for *visual_id*.

        Called by :class:`MultiscalePaintController.commit` and ``abort`` at
        session end.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "clear_paint_textures"):
            return
        gfx_visual.clear_paint_textures()

    def on_appearance_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AppearanceChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the appearance of *visual_id* changes.

        The callback receives the full ``AppearanceChangedEvent``, which
        includes ``source_id`` for echo-filtering, ``field_name``, and
        ``new_value``.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``AppearanceChangedEvent`` on each appearance change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            AppearanceChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_visual_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AppearanceChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Deprecated. Use ``on_appearance_changed`` instead."""
        import warnings

        warnings.warn(
            "on_visual_changed is deprecated; use on_appearance_changed instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.on_appearance_changed(
            visual_id, callback, owner_id=owner_id, weak=weak
        )

    def on_visibility_changed(
        self,
        visual_id: UUID,
        callback: Callable[[VisualVisibilityChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the visibility of *visual_id* changes.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``VisualVisibilityChangedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            VisualVisibilityChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def set_visual_visible(self, visual_id: UUID, visible: bool) -> None:
        """Show or hide a visual.

        Parameters
        ----------
        visual_id :
            Target visual.
        visible :
            ``True`` to show, ``False`` to hide.
        """
        self.update_appearance_field(visual_id, "visible", visible)

    def set_visual_outline(
        self,
        visual_id: UUID,
        slot: int = 1,
        placement: str | None = None,
    ) -> None:
        """Outline a visual with a screen-space contour.

        Requires ``render_config.outline.enabled`` (or
        ``controller.render_manager.outline_enabled = True``); the outline
        pass is off by default.

        Parameters
        ----------
        visual_id :
            Target visual.
        slot :
            ``0`` removes the outline.  ``1..15`` selects palette entry
            ``slot - 1`` from ``OutlineConfig.palette`` for the selection
            layer; any nonzero slot also makes the visual visible to the
            boundaries layer.
        placement :
            ``"inward"`` puts the band inside the region's own footprint,
            so the region never appears to grow but one thinner than twice
            the thickness is consumed entirely.  ``"outward"`` puts it
            outside, reading as a halo and leaving the region intact.
            Defaults by visual type: ``"outward"`` for lines and points,
            whose screen-space defaults (2 px thickness, 5 px marker size)
            are too thin to survive an inward band at any zoom level;
            ``"inward"`` for everything else.

        Raises
        ------
        ValueError
            If *slot* or *placement* is out of range.

        Notes
        -----
        A nonzero *slot* sets ``pick_write = True`` on the visual, since
        outlines are derived from the pick buffer.  **This also enables
        mouse picking for that visual**, which is a side effect worth
        knowing about if you had deliberately turned picking off.

        On a labels visual what *slot* means depends on
        ``outline_mode``, and the default mode is ``"per_label"``: there
        the visual is outlined per label rather than as one silhouette, and
        **the colour comes from the label, not from** *slot* -- a nonzero
        *slot* only makes the visual eligible for the boundaries layer, and
        :meth:`set_label_selection` is what puts a palette colour on
        individual labels.  In ``"whole_object"`` and ``"all_boundaries"``
        mode *slot* is the colour, exactly as on every other visual.  Per
        label keys need the canvas to have been built with outlines
        enabled; without that the visual falls back to a silhouette.
        """
        visual = self._get_visual_model(visual_id)
        # Write the model and let the bridge do the rest: the warnings, the
        # placement default, the push to the render layer and the event all
        # hang off the field change rather than off this method, so a direct
        # ``visual.outline.slot = 1`` behaves identically.
        if placement is not None:
            visual.outline.placement = placement
        visual.outline.slot = int(slot)

    def set_visual_ambient_occlusion(
        self, visual_id: UUID, enabled: bool | None = None
    ) -> None:
        """Choose whether one visual receives ambient occlusion.

        Requires ``render_config.ambient_occlusion.enabled`` (or
        ``controller.ambient_occlusion_enabled = True``); the occlusion pass is off by
        default, and off in 2D always.

        Parameters
        ----------
        visual_id :
            Target visual.
        enabled :
            ``None`` (the default) restores the automatic rule: excluded
            while the visual renders in a MIP-family mode (``mip``,
            ``attenuated_mip``, ``minip``), included otherwise, re-derived
            whenever the render mode changes.  ``True`` and ``False`` are
            explicit and survive a render-mode change.

        Notes
        -----
        A MIP-family mode writes the depth of the brightest sample along
        the ray rather than of a surface, and that depth jumps between
        neighbouring pixels, so occlusion computed from it shimmers.  That
        is what the automatic rule exists for; an explicit ``True`` is
        available for anyone who wants it anyway.

        Excluding a visual sets ``pick_write = True`` on it, because the
        occlusion pass identifies pixels through the pick buffer -- the
        same side effect :meth:`set_visual_outline` has, and worth knowing
        if you had deliberately turned picking off.  **The automatic rule
        needs picking too**: a MIP visual with ``pick_write = False``
        cannot be identified per pixel, so it receives occlusion despite
        the default.

        This controls whether the visual *receives* occlusion, not whether
        it *casts* it: the occlusion loop reads raw depth, so an excluded
        visual's depth still darkens its neighbours.
        """
        self._get_visual_model(visual_id).ambient_occlusion = enabled

    def get_visual_ambient_occlusion(self, visual_id: UUID) -> bool | None:
        """Return the explicit occlusion setting for *visual_id*.

        Parameters
        ----------
        visual_id :
            Target visual.

        Returns
        -------
        bool or None
            ``None`` when the visual is on the automatic rule, which is
            the default.
        """
        return self._get_visual_model(visual_id).ambient_occlusion

    def get_visual_outline(self, visual_id: UUID) -> tuple[int, str] | None:
        """Return ``(slot, placement)`` for *visual_id*, or ``None``.

        Parameters
        ----------
        visual_id :
            Target visual.

        Returns
        -------
        tuple[int, str] or None
            ``None`` when the visual is not outlined.
        """
        visual = self._get_visual_model(visual_id)
        if visual.outline.slot == 0:
            return None
        placement = visual.outline.placement or _default_placement(visual)
        return visual.outline.slot, placement

    def set_label_selection(self, visual_id: UUID, selection: dict[int, int]) -> None:
        """Choose which label values the selection layer outlines.

        Parameters
        ----------
        visual_id :
            A labels visual, already given an outline with
            :meth:`set_visual_outline`.
        selection :
            ``{label value: palette slot}``.  Slots are clamped into
            ``1..15`` and index ``OutlineConfig.palette`` as ``slot - 1``.
            An empty dict clears the selection, leaving the boundaries layer
            to draw every label boundary.

        Notes
        -----
        Selection is **exact**: the label key is range-partitioned, with
        ``1..15`` reserved for selected labels and everything above for
        unselected ones, so an unselected label can never be mistaken for a
        selected one.

        Requires a canvas built with outlines enabled -- the per-label key
        lives in a render target that is only allocated then.  Without it
        the visual still gets a whole-object silhouette.
        """
        visual = self._get_visual_model(visual_id)
        if not isinstance(visual, BaseLabelsVisual):
            raise ValueError(
                "set_label_selection is only available on labels visuals; "
                f"a {type(visual).__name__} is outlined as one silhouette."
            )
        # The model, not the material: a labels material can be rebuilt
        # underneath a selection written straight to the GPU, and the
        # multiscale visual rebuilds its materials whenever the displayed
        # level shapes change.
        visual.outline_selected_labels = dict(selection)

    def _on_reslice_completed_redraw(self, event: ResliceCompletedEvent) -> None:
        """Redraw once a reslice has committed its data to the GPU.

        Fires per visual per canvas at the end of a reslice round, so a
        progressive multiscale load still averages its intermediate levels
        together -- only the settled result is guaranteed a clean frame.
        """
        self._request_draw_for_visual(event.visual_id)

    def _request_draw_for_visual(self, visual_id: UUID) -> None:
        """Ask every canvas showing *visual_id*'s scene to redraw."""
        scene_id = self._visual_to_scene.get(visual_id)
        if scene_id is None:
            return
        self._request_draw_for_scene(scene_id)

    def _request_draw_for_scene(self, scene_id: UUID) -> None:
        """Ask every canvas showing *scene_id* to redraw.

        For changes that are not attributable to one visual's appearance --
        a visual added or removed, a transform, freshly committed data.
        ``CanvasView.request_draw`` also discards the accumulation history,
        which is what these need: they all change the image, and a frame
        averaged with the previous content would show the change fading in.

        Also the moment a deferred camera fit becomes possible: a canvas that
        changed its displayed axes could not be fitted while the scene was
        empty, and freshly committed data is exactly what it was waiting for.
        """
        gfx_scene = None
        for canvas_id in self.get_canvas_ids(scene_id):
            canvas_view = self._render_manager._canvases.get(canvas_id)
            if canvas_view is None:
                continue
            if canvas_id in self._canvases_awaiting_fit:
                if gfx_scene is None:
                    gfx_scene = self._render_manager.get_scene(scene_id)
                if canvas_view.show_object(gfx_scene):
                    self._canvases_awaiting_fit.discard(canvas_id)
                    self._update_camera_model(
                        scene_id, canvas_id, canvas_view.capture_camera_state()
                    )
            canvas_view.request_draw()

    def on_scene_added(
        self,
        scene_id: UUID,
        callback: Callable[[SceneAddedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *scene_id* is added.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``SceneAddedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            SceneAddedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_scene_removed(
        self,
        scene_id: UUID,
        callback: Callable[[SceneRemovedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *scene_id* is removed.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``SceneRemovedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            SceneRemovedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_visual_added(
        self,
        visual_id: UUID,
        callback: Callable[[VisualAddedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *visual_id* is added.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``VisualAddedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            VisualAddedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_visual_removed(
        self,
        visual_id: UUID,
        callback: Callable[[VisualRemovedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when *visual_id* is removed.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``VisualRemovedEvent``.
        owner_id :
            UUID under which this subscription is registered.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            VisualRemovedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def cancel_pending_slices(self, scene_id: UUID) -> None:
        """Cancel all in-flight slice requests for *scene_id*.

        Parameters
        ----------
        scene_id :
            ID of the scene whose pending slices to cancel.
        """
        self._render_manager._slice_coordinator.cancel_scene(scene_id)

    def close(self) -> None:
        """Cancel in-flight slices and close every canvas this viewer owns.

        Releases the render surfaces and GPU resources held by the controller.
        Closing is explicit because the canvases are owned by the GUI backend,
        not by Python refcounting, so dropping the controller alone leaks them
        (see :meth:`CanvasView.close`).

        Cancels the pending camera-settle tasks and every in-flight slice
        task -- including the ones the slice coordinator no longer tracks --
        so a closed controller holds no live ``asyncio.Task``.

        Also disconnects the psygnal bridges from the model and clears the
        event buses.  Those hold the
        controller's own handlers, and psygnal keeps them **strongly**, so a
        closed-but-connected controller stays reachable from the models it was
        watching -- and keeps reacting to them.  ``remove_visual`` and
        ``remove_scene`` already do this for what they remove; this does it for
        whatever is left.

        Safe to call more than once; the controller must not be used afterwards.
        """
        # Both of these cancel rather than await: close is synchronous, so a
        # task only observes its CancelledError once the loop runs again.  What
        # is guaranteed here is that nothing stays tracked and nothing is left
        # un-cancelled -- not that everything has already stopped.
        self._cancel_settle_tasks()
        # cancel_all rather than a cancel_pending_slices walk per scene: a
        # superseded non-cancellable reslice is no longer named by any scene's
        # bookkeeping, so the per-scene walk cannot reach it.
        self._render_manager._slice_coordinator.cancel_all()
        self._render_manager.close()
        self._scene_to_canvases.clear()
        self._canvases_awaiting_fit.clear()

        for registry in (self._visual_psygnal_handlers, self._scene_psygnal_handlers):
            for handlers in registry.values():
                for signal, handler in handlers:
                    # A signal whose model is already gone, or a handler
                    # disconnected by an earlier remove_*, is not an error here.
                    with suppress(Exception):
                        signal.disconnect(handler)
            registry.clear()
        for overlay_id in list(self._overlays):
            with suppress(Exception):
                self._forget_overlay(overlay_id)
        for signal, handler in self._store_psygnal_handlers.values():
            with suppress(Exception):
                signal.disconnect(handler)
        self._store_psygnal_handlers.clear()

        # The buses hold strong references to every handler subscribed to
        # them -- render visuals, widgets, and the controller's own methods --
        # so dropping the controller without this leaves that whole graph
        # reachable through them.
        self._outgoing_events.clear()
        self._incoming_events.clear()

    def on_aabb_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AABBChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the AABB params of *visual_id* change.

        The callback receives the full ``AABBChangedEvent``, which includes
        ``source_id`` for echo-filtering, ``field_name``, and ``new_value``.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``AABBChangedEvent`` on each AABB change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            AABBChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_reslice_started(
        self,
        scene_id: UUID,
        callback: Callable[[ResliceStartedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a reslice cycle begins for *scene_id*.

        Useful for showing a loading indicator.  Fired once per reslice
        submission, before any async data fetching starts.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``ResliceStartedEvent``.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            ResliceStartedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_reslice_completed(
        self,
        visual_id: UUID,
        callback: Callable[[ResliceCompletedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a reslice cycle completes for *visual_id*.

        Useful for hiding a loading indicator.  Fired once per visual per
        reslice cycle, after all bricks/tiles in the batch are committed.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``ResliceCompletedEvent``.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._outgoing_events.subscribe(
            ResliceCompletedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_scene_ready(
        self,
        scene_id: UUID,
        callback: Callable[[], None],
        *,
        owner_id: UUID | None = None,
    ) -> None:
        """Reslice *scene_id* and fire *callback* once all its data is on the GPU.

        This triggers a reslice of every visual in the scene and invokes
        *callback* exactly once, after all visuals loaded by that reslice have
        committed to the GPU across every attached canvas.  Visuals with no
        data in the current view (frustum-culled, empty slab, or hidden) do not
        delay the callback.

        Unlike :meth:`on_reslice_completed` (which fires per-visual on every
        cycle), this is a scene-level, one-shot readiness signal — the right
        hook for fitting the camera or hiding a startup spinner once a mixed
        scene of multiscale images, in-memory images, and geometry has fully
        loaded.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice and watch.
        callback : Callable[[], None]
            Zero-argument callback fired once the scene's data is resident.
        owner_id : UUID or None
            Owner under which the temporary subscriptions are registered.
            Defaults to the controller's own id.
        """
        self.reslice_scene(scene_id, on_ready=callback, owner_id=owner_id)

    def on_canvas_connected(
        self,
        canvas_id: UUID,
        callback: Callable[[], None],
        *,
        owner_id: UUID | None = None,
    ) -> None:
        """Fire *callback* once *canvas_id*'s front end is live and able to draw.

        The weaker, earlier sibling of :meth:`on_canvas_first_frame`: it says
        the canvas *can* render, not that it *has*.  On Qt the two nearly
        coincide; on the anywidget backend the canvas exists in Python long
        before the browser mounts it, and a frame may never arrive at all --
        so work that only needs a usable canvas should wait on this instead.

        Fires immediately if the canvas has already connected.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas to watch.
        callback : Callable[[], None]
            Zero-argument callback, fired once.
        owner_id : UUID or None
            Owner for the temporary subscription.  Defaults to the
            controller's own id.
        """
        canvas_view = self._render_manager._canvases.get(canvas_id)
        if canvas_view is not None and canvas_view.connected:
            callback()
            return

        state: dict[str, Any] = {"fired": False, "handle": None}

        def _on_connected(event: CanvasConnectedEvent) -> None:
            if state["fired"]:
                return
            state["fired"] = True
            if state["handle"] is not None:
                self._outgoing_events.unsubscribe(state["handle"])
            callback()

        state["handle"] = self._outgoing_events.subscribe(
            CanvasConnectedEvent,
            _on_connected,
            entity_id=canvas_id,
            owner_id=owner_id or self._id,
        )

    def on_canvas_first_frame(
        self,
        canvas_id: UUID,
        callback: Callable[[], None],
        *,
        owner_id: UUID | None = None,
    ) -> None:
        """Fire *callback* once *canvas_id* has rendered its first frame.

        The first rendered frame guarantees the canvas has reached its final
        logical size and its camera matrix has been applied — the precondition
        for fitting the camera and computing view-dependent (multiscale) slice
        requests at the correct level of detail.  This is a timer-free
        replacement for deferring startup work with ``QTimer.singleShot(0)``.

        A draw is requested immediately so the frame is guaranteed to arrive
        whether or not the event loop is already running.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas to watch.
        callback : Callable[[], None]
            Zero-argument callback fired once, on the first frame.
        owner_id : UUID or None
            Owner under which the temporary subscription is registered.
            Defaults to the controller's own id.
        """
        owner = owner_id or self._id
        state: dict[str, Any] = {"fired": False, "handle": None}

        def _on_frame(event: FrameRenderedEvent) -> None:
            if state["fired"]:
                return
            state["fired"] = True
            if state["handle"] is not None:
                self._outgoing_events.unsubscribe(state["handle"])
            callback()

        state["handle"] = self._outgoing_events.subscribe(
            FrameRenderedEvent,
            _on_frame,
            entity_id=canvas_id,
            owner_id=owner,
        )

        # Request a draw so a FrameRenderedEvent is guaranteed to be emitted.
        canvas_view = self._render_manager._canvases.get(canvas_id)
        if canvas_view is not None:
            canvas_view.request_draw()

    def unsubscribe_owner(self, owner_id: UUID) -> None:
        """Remove all event subscriptions registered under *owner_id*.

        GUI widgets should call this from their Qt ``closeEvent`` or
        ``destroyed`` signal handler to deterministically clean up their
        bus subscriptions.

        Parameters
        ----------
        owner_id :
            The UUID used as ``owner_id`` when the subscriptions were
            registered (typically the widget's own ``self._id``).
        """
        self._outgoing_events.unsubscribe_all(owner_id)

    def connect_widget(
        self,
        widget: WidgetView,
        *,
        subscription_specs: list[SubscriptionSpec] | None = None,
    ) -> None:
        """Wire a widget's psygnal signals to the bus and register subscriptions.

        Widgets declare their intent through two psygnal signals and an optional
        list of ``SubscriptionSpec`` objects, so they never import or hold a
        reference to ``CellierController``.

        The caller is responsible for constructing the widget and passing the
        specs — typically obtained from ``widget.subscription_specs()``.

        Parameters
        ----------
        widget :
            Any object exposing:

            * ``widget._id`` — a ``UUID`` identifying the widget.
            * ``widget.changed`` — a psygnal ``Signal`` that emits a
              ``CellierUpdateEventTypes`` instance when the user changes a
              value.  Connected to ``incoming_events.emit``.
            * ``widget.closed`` — a psygnal ``Signal`` (no arguments) emitted
              when the widget is closed.  Triggers
              ``unsubscribe_owner(widget._id)``.

        subscription_specs :
            Optional list of ``SubscriptionSpec`` entries describing which
            outgoing bus events the widget wants to receive.  Pass ``None``
            (or omit the argument) for pure-output widgets that do not need
            model-driven updates.
        """
        widget.changed.connect(self._incoming_events.emit)
        owner_id = widget._id
        widget.closed.connect(lambda: self.unsubscribe_owner(owner_id))
        for spec in subscription_specs or []:
            self._outgoing_events.subscribe(
                spec.event_type,
                spec.handler,
                entity_id=spec.entity_id,
                owner_id=owner_id,
                weak=spec.weak,
            )
