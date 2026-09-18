"""All event NamedTuples for the cellier v2 EventBus."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

#: The empty slice-position mapping, shared because it is immutable.
NO_SLICE_POSITIONS: Mapping[int, float] = MappingProxyType({})

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    import numpy as np

    from cellier._state import CameraState, DimsState
    from cellier.scene._background import BackgroundAppearance
    from cellier.transform import BaseTransform


class DimsChangedEvent(NamedTuple):
    """The dims editor changed.

    ``slice_indices`` -- world axis to world position -- is here rather than
    on ``dims_state`` because the two have different audiences.  The render
    layer takes the ``RegionSelection`` the controller emits alongside and
    never reads a raw position (D5, landed in Phase 8); the dims *editor*
    widgets do need the positions, to resync their sliders when something
    else moves them, and this is the channel they arrive on.

    ``region_changed`` is ``False`` when the change moved only positions or
    thicknesses of **displayed** axes.  Every axis keeps a position while
    displayed (D36), but a displayed axis is not sliced, so what the scene
    shows is unchanged and nothing needs reslicing.
    """

    source_id: UUID
    scene_id: UUID
    dims_state: DimsState
    displayed_axes_changed: bool
    slice_indices: Mapping[int, float] = NO_SLICE_POSITIONS
    region_changed: bool = True


class SliderAxesChangedEvent(NamedTuple):
    """The set of world axes that get a slider changed on a scene.

    ``Scene.slider_axes`` is a derived property with no signal of its own;
    the controller recomputes it after every model change that can move it
    and emits this only when the value differs (design 3.5).
    """

    source_id: UUID
    scene_id: UUID
    slider_axes: tuple[int, ...]


class CameraChangedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    camera_state: CameraState


class AppearanceChangedEvent(NamedTuple):
    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any
    requires_reslice: bool


class ChannelAppearanceChangedEvent(NamedTuple):
    """Emitted when a field on one ``ChannelAppearance`` object changes.

    Routed from the psygnal bridge in ``CellierController._wire_channels``
    to subscribers that hold a ``GFXMultichannel*Visual``.

    Attributes
    ----------
    source_id : UUID
        ID of the event emitter (typically the controller).
    visual_id : UUID
        Model-layer ID of the multichannel visual that owns the channel.
    channel_index : int
        Index of the channel whose appearance changed.
    field_name : str
        Name of the field that changed (e.g. ``"color_map"``, ``"clim"``).
    new_value : Any
        The new field value.
    """

    source_id: UUID
    visual_id: UUID
    channel_index: int
    field_name: str
    new_value: Any


class SingleAppearanceChangedEvent(NamedTuple):
    """A field on an image visual's single-mode appearance changed.

    ``field_name`` is ``None`` when the whole ``single`` model was replaced;
    ``new_value`` is then the new model, and a consumer re-reads every field.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str | None
    new_value: Any


class ImageCompositeChangedEvent(NamedTuple):
    """An image visual switched between single and composite mode."""

    source_id: UUID
    visual_id: UUID
    composite: bool


class PickWriteChangedEvent(NamedTuple):
    """Emitted when ``BaseVisual.pick_write`` changes.

    Attributes
    ----------
    source_id : UUID
        ID of the event emitter (typically the controller).
    visual_id : UUID
        Model-layer ID of the visual whose pick_write changed.
    pick_write : bool
        The new pick_write value.
    """

    source_id: UUID
    visual_id: UUID
    pick_write: bool


class AABBChangedEvent(NamedTuple):
    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any


class VisualVisibilityChangedEvent(NamedTuple):
    source_id: UUID
    visual_id: UUID
    visible: bool


class DataStoreMetadataChangedEvent(NamedTuple):
    """Emitted when a store's data may now occupy a different region.

    The bus form of a store's ``"extent"`` change
    (``plans/store_change_events.md``): new geometry positions, image data of
    a new shape, a store that grew.  Everything derived from the store's
    ``axis_extents`` is stale.  Relayed by the controller from
    ``BaseDataStore.data_changed``, after it has refreshed extent-derived
    state and requested a reslice of every visual reading the store.

    Attributes
    ----------
    source_id : UUID
        ID of the emitter (the controller).
    data_store_id : UUID
        The store that changed.
    """

    source_id: UUID
    data_store_id: UUID


class DataStoreContentsChangedEvent(NamedTuple):
    """Emitted when a store's values changed within the same region.

    The bus form of a store's ``"contents"`` change: a paint stroke, new
    colours, a frame streamed into the existing extent.  Only what was drawn
    from the data is stale.  Relayed by the controller from
    ``BaseDataStore.data_changed``, after it has requested a reslice of every
    visual reading the store.

    Attributes
    ----------
    source_id : UUID
        ID of the emitter (the controller).
    data_store_id : UUID
        The store that changed.
    regions : tuple[DataRegion, ...] or None
        Where the values changed, as per-axis ``(start, stop)`` in level-0
        data coordinates, or ``None`` for anywhere.  In the store's own
        coordinates because the store does not know the render layer's brick
        grid; converting to brick keys is the render layer's job
        (``plans/streaming_acquisition_design.md`` 8b).
    """

    source_id: UUID
    data_store_id: UUID
    regions: tuple[tuple[tuple[float, float], ...], ...] | None = None


class ResliceStartedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    canvas_id: UUID
    visual_ids: frozenset[UUID]


class ResliceCompletedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    canvas_id: UUID
    visual_id: UUID
    brick_count: int


class ResliceCancelledEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class FrameRenderedEvent(NamedTuple):
    source_id: UUID
    canvas_id: UUID
    frame_time_ms: float


class CanvasConnectedEvent(NamedTuple):
    """Emitted once a canvas's front end is live and able to draw.

    Distinct from :class:`FrameRenderedEvent`, which says a frame *has* been
    drawn.  On Qt the two nearly coincide -- the widget is shown and the
    continuous scheduler draws immediately -- but on the anywidget backend
    they can be far apart, or the frame may never arrive at all: the canvas
    exists in Python long before the browser mounts it, and nothing can be
    rendered until it does.

    Anything that must wait for a *usable* canvas rather than a *painted* one
    should watch this.

    Attributes
    ----------
    source_id : UUID
        ID of the ``CanvasView`` reporting the connection.
    canvas_id : UUID
        Model-layer canvas identifier.
    gui : str
        Which front end connected, for diagnostics.
    """

    source_id: UUID
    canvas_id: UUID
    gui: str


class CanvasSizeChangedEvent(NamedTuple):
    """Emitted when the physical pixel size of a canvas changes.

    Fired by ``CanvasView`` whenever the underlying render surface is
    resized -- via the Qt ``resizeEvent`` on the Qt backend or via the
    ``ResizeObserver`` message on the anywidget backend.  The controller
    listens and updates ``Canvas.size`` in the model.

    Attributes
    ----------
    source_id : UUID
        ID of the ``CanvasView`` that detected the resize.
    canvas_id : UUID
        Model-layer canvas identifier.
    width : int
        New canvas width in logical pixels.
    height : int
        New canvas height in logical pixels.
    """

    source_id: UUID
    canvas_id: UUID
    width: int
    height: int


class VisualAddedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class VisualRemovedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class SceneAddedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID


class CanvasAddedEvent(NamedTuple):
    """Emitted once a canvas has been registered on a scene.

    Fired at the end of ``CellierController.add_canvas_model``, after the
    canvas's stored overlays are wired and its rendered system exists.  A
    convenience viewer uses it to attach canvas overlays requested before
    any canvas existed.

    Attributes
    ----------
    source_id : UUID
        ID of the emitter (the controller).
    scene_id : UUID
        Model-layer ID of the scene the canvas renders.
    canvas_id : UUID
        Model-layer ID of the new canvas.
    """

    source_id: UUID
    scene_id: UUID
    canvas_id: UUID


class OverlayChangedEvent(NamedTuple):
    """Emitted when a field of an overlay model changes.

    Covers both overlay categories -- canvas overlays and scene overlays --
    keyed by the overlay's own id.  Routed from the controller's psygnal
    bridges on ``overlay.events`` and ``overlay.appearance.events``.

    Attributes
    ----------
    source_id : UUID
        ID of the emitter (the controller, or the widget that requested the
        change).
    overlay_id : UUID
        Model-layer ID of the overlay.
    field_name : str
        Dotted path of the changed field: a top-level field such as
        ``"visible"``, ``"appearance.<field>"`` for an appearance field, or
        ``"appearance"`` when the whole appearance model was replaced.
    new_value : Any
        The new value (the new appearance model for ``"appearance"``).
    """

    source_id: UUID
    overlay_id: UUID
    field_name: str
    new_value: Any


class BackgroundChangedEvent(NamedTuple):
    """Emitted when a scene's ``BackgroundAppearance`` changes.

    Routed from the psygnal bridge in
    ``CellierController._wire_scene_background`` -- both from a field change
    on the background model and from wholesale replacement
    (``scene.background = BackgroundAppearance(...)``).  The background is a
    property of the scene rather than of any visual, so it travels on its own
    event keyed by ``scene_id`` rather than riding on
    ``AppearanceChangedEvent``.

    Attributes
    ----------
    source_id : UUID
        ID of the event emitter (the controller, or the widget that
        requested the change).
    scene_id : UUID
        Model-layer ID of the scene whose background changed.
    background : BackgroundAppearance
        The scene's complete background model *after* the change.  Carried
        whole, not just the delta, because which fields are meaningful
        depends on ``mode`` -- a subscriber holding a snapshot cannot
        reconstruct the rest from one field.
    field_name : str or None
        Name of the changed field (e.g. ``"top_color"``, ``"mode"``), or
        ``None`` when the whole model was replaced.
    new_value : Any
        The new field value, or ``None`` for whole-model replacement.
    """

    source_id: UUID
    scene_id: UUID
    background: BackgroundAppearance
    field_name: str | None = None
    new_value: Any = None


class VisualRenderChangedEvent(NamedTuple):
    """Emitted when one visual's screen-space render settings change.

    Covers the per-visual half of the outline and ambient occlusion
    features: which palette slot a visual is outlined in, which side of its
    edge the band sits on, whether it receives occlusion, and -- for labels
    visuals -- which label values the selection layer draws.

    One event type rather than three, keyed by ``visual_id`` and carrying a
    dotted ``field_name``, so a per-visual widget needs one subscription
    rather than one per field.  The *global* half of both features travels
    on ``RenderConfigChangedEvent`` instead, because it belongs to the
    renderer rather than to any visual.

    Attributes
    ----------
    source_id : UUID
        ID of the event emitter (the controller, or the widget that
        requested the change).
    visual_id : UUID
        Model-layer ID of the visual whose settings changed.
    field_name : str
        ``"outline.slot"``, ``"outline.placement"``,
        ``"ambient_occlusion"`` or ``"outline_selected_labels"``.
    new_value : Any
        The new field value.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any


class RenderConfigChangedEvent(NamedTuple):
    """Emitted when one section of the render configuration changes.

    Render configuration belongs to the ``RenderManager`` rather than to a
    scene, a visual or a canvas, so this event has no entity id and is
    deliberately absent from ``EventBus._ENTITY_FIELD`` -- every subscriber
    receives every render-config event and filters on ``section`` itself,
    the way the appearance widgets filter on ``field_name``.

    Attributes
    ----------
    source_id : UUID
        ID of the event emitter (the controller, or the widget that
        requested the change).
    section : str
        Which configuration section changed: ``"outline"``, ``"ambient_occlusion"`` or
        ``"temporal"``.
    config : Any
        The section's complete model *after* the change (an
        ``OutlineConfig``, an ``AmbientOcclusionConfig`` or a
        ``TemporalAccumulationConfig``).  Carried whole, not just the
        delta, for the same reason ``BackgroundChangedEvent`` is: a
        subscriber holding a snapshot
        cannot reconstruct the rest of an ``OutlineConfig`` from one field.
    field_name : str or None
        Dotted path of the changed field within the section, e.g.
        ``"power"`` or ``"selection.inward_thickness"``.  ``None`` when the
        whole section was replaced.
    new_value : Any
        The new field value, or ``None`` for whole-section replacement.
    """

    source_id: UUID
    section: str
    config: Any
    field_name: str | None = None
    new_value: Any = None


class TrailChangedEvent(NamedTuple):
    """Emitted when a graph visual's trail configuration changes.

    Routed from the psygnal bridge in ``CellierController._wire_trail`` --
    both from a field change on one ``TrailConfig`` and from whole-dict
    replacement (``visual.trail = {...}``).  Both are needed because psygnal
    does not propagate nested ``EventedModel`` field changes to the parent's
    event group, which is the same fact that flattens ``GraphAppearance``.

    Attributes
    ----------
    source_id : UUID
        ID of the event emitter (typically the controller).
    visual_id : UUID
        Model-layer ID of the graph visual whose trail changed.
    trail : dict
        The visual's complete ``{axis: TrailConfig}`` mapping *after* the
        change.  The whole dict rather than the delta, because the GFX
        visual holds a snapshot and rebuilds it wholesale.
    field_name : str | None
        Name of the changed ``TrailConfig`` field, or ``None`` when the
        whole dict was replaced.
    axis : int | None
        Axis whose config changed, or ``None`` for whole-dict replacement.
    """

    source_id: UUID
    visual_id: UUID
    trail: dict
    field_name: str | None = None
    axis: int | None = None


class TransformChangedEvent(NamedTuple):
    """Fired when a visual's data-to-world transform is replaced."""

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    transform: BaseTransform


class SceneRemovedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID


class PointsPickInfo(NamedTuple):
    """Element-level pick result for a points visual.

    Attributes
    ----------
    point_index : int
        Index of the picked point within the visual's point array.
    """

    point_index: int


class LinesPickInfo(NamedTuple):
    """Element-level pick result for a lines visual.

    Attributes
    ----------
    edge_index : int
        Index of the picked edge (vertex *pair*) within the visual's edge
        array.  See the GFX line visual for the vertex-index-to-edge-index
        mapping.
    """

    edge_index: int


class ImagePickInfo(NamedTuple):
    """Element-level pick result for an image or volume visual.

    Attributes
    ----------
    data_coordinate : tuple[float, ...]
        Position of the picked point in the hit visual's level-0 data-array
        coordinate system, one component per **data** axis, ascending.  That is
        the store's rank, not the scene's: an axis the visual broadcasts over
        has no component, because the data has no such axis.

        ``floor`` of each component yields the integer voxel index — one rule,
        every component.  The coordinate uses the ``[i, i + 1)`` convention
        where voxel ``i`` spans ``[i, i + 1)`` and its center is ``i + 0.5``.

        Displayed axes carry the pick-decoded position (for 3-D canvases this
        is the actual surface hit — MIP maximum or ISO surface — snapped into
        data space).  Non-displayed axes carry the **plane the visual last
        drew**, at its centre: a collapsed axis has no sub-voxel position, and
        reporting the plane rather than the dims state is what keeps the answer
        agreeing with the screen while a reslice is in flight.  A visual with
        nothing planned yet falls back to the dims state, rounded the way the
        selection assembler rounds it.

        The channel component names the channel the value was read from: the
        drawn plane in single mode, and the pick-buffer winner in composite
        mode.
    channel_values : dict[int, float]
        Index along ``channel_axis`` -> the value at the picked voxel.  A
        visual with ``channel_axis=None`` reports ``{0: value}``.  Single mode
        reports its drawn channel.  Composite mode reports every drawn
        (visible) channel on a 2D canvas and only the pick-buffer winner on a
        3D canvas.  Empty when the voxel lies outside the data.
    """

    data_coordinate: tuple[float, ...]
    channel_values: dict[int, float]


class MeshPickInfo(NamedTuple):
    """Element-level pick result for a mesh visual (stub; filled in a later phase)."""

    face_index: int


class LabelsPickInfo(NamedTuple):
    """Element-level pick result for a labels visual.

    Attributes
    ----------
    data_coordinate : tuple[float, ...]
        Same convention as ``ImagePickInfo.data_coordinate``.  ``floor`` of
        every component indexes the label array to recover the label id under
        the cursor.
    value : int
        The label id at ``floor(data_coordinate)``; in 3D, the label at the
        hit voxel.  ``0`` when the voxel lies outside the data.
    """

    data_coordinate: tuple[float, ...]
    value: int


class GraphNodePickInfo(NamedTuple):
    """Element-level pick result for a node of a graph visual.

    Attributes
    ----------
    node_id : Any
        The store's **original** node id, not the render-buffer row (D14).
        Typed ``Any`` rather than ``int`` because a geff file may carry
        uint64 or string ids; this is the one place the graph pick payload
        is more weakly typed than the points one.
    node_row : int
        Row of the node in the store's position array.  Stable under D7's
        immutability and useful for indexing the store's own arrays.
    """

    node_id: Any
    node_row: int


class GraphEdgePickInfo(NamedTuple):
    """Element-level pick result for an edge of a graph visual.

    Attributes
    ----------
    edge_index : int
        Row of the edge in the store's edge array, or ``-1`` when it could
        not be resolved.  Stable only because D7 makes the store immutable.
    source_node_id : Any
        Original node id of the edge's first endpoint.  See
        ``GraphNodePickInfo.node_id`` on the dtype.
    target_node_id : Any
        Original node id of the edge's second endpoint.
    """

    edge_index: int
    source_node_id: Any
    target_node_id: Any


VisualPickDetails = (
    PointsPickInfo
    | LinesPickInfo
    | ImagePickInfo
    | MeshPickInfo
    | LabelsPickInfo
    | GraphNodePickInfo
    | GraphEdgePickInfo
)


class CanvasPickInfo(NamedTuple):
    """Model-layer pick result attached to canvas mouse events.

    Mouse events report *that* something was hit; the typed pick events
    (``ImagePickEvent`` and friends, subscribed with
    ``CellierController.on_pick``) report *what* was hit.

    Attributes
    ----------
    hit_visual_id : UUID or None
        Model-layer ID of the visual whose active scene-graph node was
        hit, or None if the pointer landed on the background.
    """

    hit_visual_id: UUID | None


class ViewRay(NamedTuple):
    """A view ray from the camera through a screen-space point.

    Used by 3D canvas mouse events in place of a single world coordinate.

    Attributes
    ----------
    origin : np.ndarray
        Near-plane world position where the ray starts.  Shape (3,), float64.
    direction : np.ndarray
        Unit vector in world space giving the ray direction.  Shape (3,), float64.
    """

    origin: np.ndarray
    direction: np.ndarray


class CanvasMousePress2DEvent(NamedTuple):
    """Emitted when the primary pointer button is pressed on a 2D canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseMove2DEvent(NamedTuple):
    """Emitted on every pointer-move event on a 2D canvas (button up or down)."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseRelease2DEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a 2D canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMousePress3DEvent(NamedTuple):
    """Emitted when the primary pointer button is pressed on a 3D canvas."""

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseMove3DEvent(NamedTuple):
    """Emitted on every pointer-move event on a 3D canvas (button up or down)."""

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseRelease3DEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a 3D canvas."""

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class ImagePickEvent(NamedTuple):
    """An image visual was hit on a canvas (design 3.7).

    Emitted only for a hit, after the mouse event for the same pointer event:
    at once for an in-memory image, and when the level-0 value read completes
    for a multiscale image.  A newer pointer event on the canvas cancels an
    in-flight ``move`` read; ``press`` and ``release`` reads always complete,
    so match events by ``gesture_id`` and ``action``, not arrival order.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    scene_id : UUID
        The canvas's scene.
    visual_id : UUID
        The picked visual.
    action : str
        ``"press"``, ``"move"`` or ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.
    world_coordinate : np.ndarray or None
        The pointer's world position on a 2D canvas, as on the mouse event.
    ray : ViewRay or None
        The view ray on a 3D canvas, as on the mouse event.
    button : int
        As on the mouse event.
    buttons : tuple
        As on the mouse event.
    modifiers : tuple
        As on the mouse event.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    pick_info : ImagePickInfo
        What was hit.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    action: Literal["press", "move", "release"]
    camera_type: Literal["2d", "3d"]
    world_coordinate: np.ndarray | None
    ray: ViewRay | None
    button: int
    buttons: tuple
    modifiers: tuple
    gesture_id: UUID | None
    pick_info: ImagePickInfo


class LabelsPickEvent(NamedTuple):
    """A labels visual was hit on a canvas (design 3.7).

    Timing as for ``ImagePickEvent``: synchronous for in-memory labels,
    asynchronous for multiscale labels.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    scene_id : UUID
        The canvas's scene.
    visual_id : UUID
        The picked visual.
    action : str
        ``"press"``, ``"move"`` or ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.
    world_coordinate : np.ndarray or None
        The pointer's world position on a 2D canvas, as on the mouse event.
    ray : ViewRay or None
        The view ray on a 3D canvas, as on the mouse event.
    button : int
        As on the mouse event.
    buttons : tuple
        As on the mouse event.
    modifiers : tuple
        As on the mouse event.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    pick_info : LabelsPickInfo
        What was hit.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    action: Literal["press", "move", "release"]
    camera_type: Literal["2d", "3d"]
    world_coordinate: np.ndarray | None
    ray: ViewRay | None
    button: int
    buttons: tuple
    modifiers: tuple
    gesture_id: UUID | None
    pick_info: LabelsPickInfo


class PointsPickEvent(NamedTuple):
    """A points visual was hit on a canvas (design 3.7).

    Emitted synchronously,
    right after the mouse event for the same pointer event, and only for a hit.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    scene_id : UUID
        The canvas's scene.
    visual_id : UUID
        The picked visual.
    action : str
        ``"press"``, ``"move"`` or ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.
    world_coordinate : np.ndarray or None
        The pointer's world position on a 2D canvas, as on the mouse event.
    ray : ViewRay or None
        The view ray on a 3D canvas, as on the mouse event.
    button : int
        As on the mouse event.
    buttons : tuple
        As on the mouse event.
    modifiers : tuple
        As on the mouse event.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    pick_info : PointsPickInfo
        What was hit.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    action: Literal["press", "move", "release"]
    camera_type: Literal["2d", "3d"]
    world_coordinate: np.ndarray | None
    ray: ViewRay | None
    button: int
    buttons: tuple
    modifiers: tuple
    gesture_id: UUID | None
    pick_info: PointsPickInfo


class LinesPickEvent(NamedTuple):
    """A lines visual was hit on a canvas (design 3.7).

    Emitted synchronously,
    right after the mouse event for the same pointer event, and only for a hit.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    scene_id : UUID
        The canvas's scene.
    visual_id : UUID
        The picked visual.
    action : str
        ``"press"``, ``"move"`` or ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.
    world_coordinate : np.ndarray or None
        The pointer's world position on a 2D canvas, as on the mouse event.
    ray : ViewRay or None
        The view ray on a 3D canvas, as on the mouse event.
    button : int
        As on the mouse event.
    buttons : tuple
        As on the mouse event.
    modifiers : tuple
        As on the mouse event.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    pick_info : LinesPickInfo
        What was hit.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    action: Literal["press", "move", "release"]
    camera_type: Literal["2d", "3d"]
    world_coordinate: np.ndarray | None
    ray: ViewRay | None
    button: int
    buttons: tuple
    modifiers: tuple
    gesture_id: UUID | None
    pick_info: LinesPickInfo


class MeshPickEvent(NamedTuple):
    """A mesh visual was hit on a canvas (design 3.7).

    Emitted synchronously,
    right after the mouse event for the same pointer event, and only for a hit.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    scene_id : UUID
        The canvas's scene.
    visual_id : UUID
        The picked visual.
    action : str
        ``"press"``, ``"move"`` or ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.
    world_coordinate : np.ndarray or None
        The pointer's world position on a 2D canvas, as on the mouse event.
    ray : ViewRay or None
        The view ray on a 3D canvas, as on the mouse event.
    button : int
        As on the mouse event.
    buttons : tuple
        As on the mouse event.
    modifiers : tuple
        As on the mouse event.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    pick_info : MeshPickInfo
        What was hit.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    action: Literal["press", "move", "release"]
    camera_type: Literal["2d", "3d"]
    world_coordinate: np.ndarray | None
    ray: ViewRay | None
    button: int
    buttons: tuple
    modifiers: tuple
    gesture_id: UUID | None
    pick_info: MeshPickInfo


class GraphPickEvent(NamedTuple):
    """A graph visual was hit on a canvas (design 3.7).

    Emitted synchronously,
    right after the mouse event for the same pointer event, and only for a hit.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    scene_id : UUID
        The canvas's scene.
    visual_id : UUID
        The picked visual.
    action : str
        ``"press"``, ``"move"`` or ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.
    world_coordinate : np.ndarray or None
        The pointer's world position on a 2D canvas, as on the mouse event.
    ray : ViewRay or None
        The view ray on a 3D canvas, as on the mouse event.
    button : int
        As on the mouse event.
    buttons : tuple
        As on the mouse event.
    modifiers : tuple
        As on the mouse event.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    pick_info : GraphNodePickInfo or GraphEdgePickInfo
        What was hit.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    action: Literal["press", "move", "release"]
    camera_type: Literal["2d", "3d"]
    world_coordinate: np.ndarray | None
    ray: ViewRay | None
    button: int
    buttons: tuple
    modifiers: tuple
    gesture_id: UUID | None
    pick_info: GraphNodePickInfo | GraphEdgePickInfo


#: Every typed pick event, for ``CellierController.on_pick``.
PICK_EVENT_TYPES: tuple[type, ...] = (
    ImagePickEvent,
    LabelsPickEvent,
    PointsPickEvent,
    LinesPickEvent,
    MeshPickEvent,
    GraphPickEvent,
)

PickEvent = (
    ImagePickEvent
    | LabelsPickEvent
    | PointsPickEvent
    | LinesPickEvent
    | MeshPickEvent
    | GraphPickEvent
)


class _CanvasRawPointerEvent(NamedTuple):
    """Internal event emitted by RenderManager after render-layer translation.

    Not part of the public EventBus catalogue.
    Consumed only by CellierController._on_raw_pointer_event.

    Attributes
    ----------
    canvas_id : UUID
    scene_id : UUID
        Filled by RenderManager from _canvas_to_scene.
    action : str
        One of ``"press"``, ``"move"``, ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.  Determines which of ``position_2d`` / ``ray``
        is populated.
    position_2d : np.ndarray or None
        2D world position for the displayed axes, already unprojected through
        the active orthographic camera.  Shape (2,), float64.
        Set iff ``camera_type == "2d"``.
    ray : ViewRay or None
        View ray unprojected through the active perspective camera.
        Set iff ``camera_type == "3d"``.
    hit_visual_id : UUID or None
        Visual ID translated from the pick-buffer world object via
        SceneManager.get_visual_id_for_node.  None on background hits.
    button : int
        Mouse button identifier from the pygfx event.
    modifiers : tuple[str, ...]
        Active keyboard modifiers from the pygfx event.
    buttons : tuple
        Held-button mask from the pygfx event (``PointerEvent.buttons``).
    gesture_id : UUID or None
        Synthesized id correlating one press->move->release bracket.  Set on
        press, carried through moves, cleared on release.
    pick_details : VisualPickDetails or None
        Typed element-level pick identity extracted in the render layer, or
        None on a miss or stubbed visual kind.

        For image and labels visuals the render layer stores the render-layer
        intermediate types ``_ImageDisplayedDataCoord`` or
        ``_LabelsDisplayedDataCoord`` (defined in ``render_manager.py``) here
        rather than the public ``ImagePickInfo`` / ``LabelsPickInfo``.  The
        controller promotes them to full-N-dim public types in
        ``_on_raw_pointer_event`` after filling the non-displayed axes from the
        dims state.  At runtime ``VisualPickDetails`` is not enforced, so the
        wider set of internal types flows through transparently.
    """

    canvas_id: UUID
    scene_id: UUID
    action: str
    camera_type: str
    position_2d: np.ndarray | None
    ray: ViewRay | None
    hit_visual_id: UUID | None
    button: int
    modifiers: tuple
    buttons: tuple = ()
    gesture_id: UUID | None = None
    pick_details: Any = None


CellierEventTypes = (
    DimsChangedEvent
    | SliderAxesChangedEvent
    | CameraChangedEvent
    | CanvasSizeChangedEvent
    | AppearanceChangedEvent
    | ChannelAppearanceChangedEvent
    | SingleAppearanceChangedEvent
    | ImageCompositeChangedEvent
    | PickWriteChangedEvent
    | AABBChangedEvent
    | VisualVisibilityChangedEvent
    | DataStoreMetadataChangedEvent
    | DataStoreContentsChangedEvent
    | ResliceStartedEvent
    | ResliceCompletedEvent
    | ResliceCancelledEvent
    | FrameRenderedEvent
    | VisualAddedEvent
    | VisualRemovedEvent
    | TrailChangedEvent
    | TransformChangedEvent
    | SceneAddedEvent
    | CanvasAddedEvent
    | OverlayChangedEvent
    | BackgroundChangedEvent
    | RenderConfigChangedEvent
    | VisualRenderChangedEvent
    | SceneRemovedEvent
    | CanvasMousePress2DEvent
    | CanvasMouseMove2DEvent
    | CanvasMouseRelease2DEvent
    | CanvasMousePress3DEvent
    | CanvasMouseMove3DEvent
    | CanvasMouseRelease3DEvent
)
