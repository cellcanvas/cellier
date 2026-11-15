"""All event NamedTuples for the EventBus."""

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
    never reads a raw position; the dims *editor* widgets do need the
    positions, to resync their sliders when something else moves them, and
    this is the channel they arrive on.

    ``region_changed`` is ``False`` when the change moved only positions or
    thicknesses of **displayed** axes.  Every axis keeps a position while
    displayed, but a displayed axis is not sliced, so what the scene shows
    is unchanged and nothing needs reslicing.

    ``interactive`` is ``True`` when the change is a tick of a dims scrub:
    it was marked interactive, or it came inside an open scope (see
    ``DimsInteractionEvent``).  A scrub's start and end have their own
    event; this flag is the per-tick "in progress" state.

    Attributes
    ----------
    source_id : UUID
        Who made the change: a widget's id, or the controller's.
    scene_id : UUID
        The scene whose dims changed.  The routing key.
    dims_state : DimsState
        The dims after the change.
    displayed_axes_changed : bool
        Whether the set of displayed axes changed (a 2D/3D switch or a
        different pair of axes).
    slice_indices : Mapping[int, float]
        World axis index to world position, for every axis.
    region_changed : bool
        Whether the sliced region changed (see above).
    interactive : bool
        Whether this is a tick of a dims scrub (see above).
    """

    source_id: UUID
    scene_id: UUID
    dims_state: DimsState
    displayed_axes_changed: bool
    slice_indices: Mapping[int, float] = NO_SLICE_POSITIONS
    region_changed: bool = True
    interactive: bool = False


class DimsInteractionEvent(NamedTuple):
    """A dims scrub started or ended on a scene.

    A scrub is a run of interactive slice-position changes: slider ticks,
    or programmatic moves marked ``interactive=True`` or made inside
    ``CellierController.dims_interaction``.  The start is emitted **before**
    the first tick changes the dims model.  No event is emitted per tick;
    ``DimsChangedEvent.interactive`` carries that.

    Attributes
    ----------
    source_id : UUID
        The source of the tick or scope that caused the transition.
    scene_id : UUID
        The scene being scrubbed.
    phase : {"start", "end"}
        Which transition this is.
    reason : {"release", "settle", "jump", "cancel"} or None
        Why the scrub ended: the last scope closed, the dims were still for
        ``SchedulerConfig.dims_settle_s``, a change that was not interactive
        arrived, or the displayed axes changed.  ``None`` for a start.
    axes : frozenset[int]
        The world axes moved by the scrub: the first tick's for a start,
        every tick's for an end.
    """

    source_id: UUID
    scene_id: UUID
    phase: Literal["start", "end"]
    reason: Literal["release", "settle", "jump", "cancel"] | None = None
    axes: frozenset[int] = frozenset()


class SliderAxesChangedEvent(NamedTuple):
    """The set of world axes that get a slider changed on a scene.

    ``Scene.slider_axes`` is a derived property with no signal of its own;
    the controller recomputes it after every model change that can move it
    and emits this only when the value differs.

    Attributes
    ----------
    source_id : UUID
        Who made the change: a widget's id, or the controller's.
    scene_id : UUID
        The scene.  The routing key.
    slider_axes : tuple[int, ...]
        The world axes that now get a slider, in world order.
    """

    source_id: UUID
    scene_id: UUID
    slider_axes: tuple[int, ...]


class CameraChangedEvent(NamedTuple):
    """A canvas's camera changed.

    ``source_id`` is the canvas's id.  ``interactive`` is ``True`` for a
    change detected between frames (the camera controller, or anything else
    that moved the pygfx camera) and for a programmatic move made with
    ``interactive=True`` or inside ``CellierController.camera_interaction``;
    it is ``False`` for a programmatic jump (``fit_camera``,
    ``look_at_visual``, ``set_camera_state``).

    Attributes
    ----------
    source_id : UUID
        The canvas whose camera changed.
    scene_id : UUID
        The canvas's scene.  The routing key.
    camera_state : CameraState
        The camera after the change.
    interactive : bool
        Whether the change is part of a camera motion (see above).
    """

    source_id: UUID
    scene_id: UUID
    camera_state: CameraState
    interactive: bool = True


class CameraInteractionEvent(NamedTuple):
    """A camera motion started or ended on a canvas.

    The camera counterpart of ``DimsInteractionEvent``, with the same
    fields plus the canvas.  No event is emitted per frame;
    ``CameraChangedEvent.interactive`` carries that.

    Attributes
    ----------
    source_id : UUID
        The source of the tick or scope that caused the transition.  The
        canvas's id for motion driven by its camera controller.
    scene_id : UUID
        The scene the canvas shows.
    canvas_id : UUID
        The canvas whose camera is moving.
    phase : {"start", "end"}
        Which transition this is.
    reason : {"release", "settle", "jump", "cancel"} or None
        Why the motion ended: the camera controller stopped driving the
        camera (or the last scope closed), the camera was still for
        ``CameraConfig.settle_threshold_s``, a programmatic move that was
        not interactive arrived, or the canvas switched between 2D and 3D.
        ``None`` for a start.
    """

    source_id: UUID
    scene_id: UUID
    canvas_id: UUID
    phase: Literal["start", "end"]
    reason: Literal["release", "settle", "jump", "cancel"] | None = None


class AppearanceChangedEvent(NamedTuple):
    """A field on a visual's ``appearance`` changed.

    Fired for every field except ``visible``, which fires
    ``VisualVisibilityChangedEvent`` instead.  Image visuals' single and
    per-channel appearances fire ``SingleAppearanceChangedEvent`` and
    ``ChannelAppearanceChangedEvent``.  Subscribe with
    ``CellierController.on_appearance_changed``.

    Attributes
    ----------
    source_id : UUID
        Who made the change: the id a widget passed with its update event,
        or the controller's id for a change made on the model directly.
        Compare it with your own id to ignore echoes of your own edits.
    visual_id : UUID
        The visual whose appearance changed.  The routing key.
    field_name : str
        Name of the field that changed, e.g. ``"opacity"``.
    new_value : Any
        The field's new value.
    requires_reslice : bool
        ``True`` for the fields that change which data is loaded
        (``settled_lod_bias``, ``force_level``, ``frustum_cull``); the
        controller reslices the visual after emitting the event.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any
    requires_reslice: bool


class ChannelAppearanceChangedEvent(NamedTuple):
    """Emitted when a field on one ``ChannelAppearance`` object changes.

    Emitted by the controller when a field of one of an image visual's
    ``channels`` changes.  Subscribe with ``entity_id`` set to the visual's
    id.

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

    Attributes
    ----------
    source_id : UUID
        Who made the change: a widget's id, or the controller's.
    visual_id : UUID
        The image visual.  The routing key.
    field_name : str or None
        The field that changed, or ``None`` for a replaced model.
    new_value : Any
        The field's new value, or the new ``single`` model.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str | None
    new_value: Any


class ImageCompositeChangedEvent(NamedTuple):
    """An image visual switched between single and composite mode.

    Attributes
    ----------
    source_id : UUID
        Who made the change: a widget's id, or the controller's.
    visual_id : UUID
        The image visual.  The routing key.
    composite : bool
        ``True`` when it now draws its channels as a composite.
    """

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
    """A field on a visual's bounding-box settings (``visual.aabb``) changed.

    Subscribe with ``CellierController.on_aabb_changed``.

    Attributes
    ----------
    source_id : UUID
        Who made the change: the id a widget passed with its update event,
        or the controller's id for a change made on the model directly.
        Compare it with your own id to ignore echoes of your own edits.
    visual_id : UUID
        The visual whose bounding box changed.  The routing key.
    field_name : str
        Name of the ``AABBParams`` field that changed, e.g. ``"visible"``.
    new_value : Any
        The field's new value.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any


class MeshSectionChangedEvent(NamedTuple):
    """A field of a mesh visual's ``section`` config changed.

    Attributes
    ----------
    source_id : UUID
        Who made the change, for echo filtering.
    visual_id : UUID
        The mesh visual.
    field_name : str
        ``"mode"``, ``"outline"``, ``"fill"`` or ``"outline_width"``.
    new_value : Any
        The field's new value.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any


class VisualVisibilityChangedEvent(NamedTuple):
    """A visual was shown or hidden (``appearance.visible`` changed).

    Subscribe with ``CellierController.on_visibility_changed``.

    Attributes
    ----------
    source_id : UUID
        Who made the change: the id a widget passed with its update event,
        or the controller's id for a change made on the model directly.
        Compare it with your own id to ignore echoes of your own edits.
    visual_id : UUID
        The visual shown or hidden.  The routing key.
    visible : bool
        The new value of ``appearance.visible``.
    """

    source_id: UUID
    visual_id: UUID
    visible: bool


class DataStoreMetadataChangedEvent(NamedTuple):
    """Emitted when a store's data may now occupy a different region.

    The bus form of a store's ``"extent"`` change: new geometry positions,
    image data of a new shape, a store that grew.  Everything derived from
    the store's ``axis_extents`` is stale.  Relayed by the controller from
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
        grid; converting to brick keys is the render layer's job.
    """

    source_id: UUID
    data_store_id: UUID
    regions: tuple[tuple[tuple[float, float], ...], ...] | None = None


class ResliceStartedEvent(NamedTuple):
    """A reslice began loading data for some of a scene's visuals.

    Emitted once per reslice request (one per canvas showing the scene),
    before any data for it is read.  Each visual it names later fires one
    ``ResliceCompletedEvent`` for the same canvas.  Not emitted when the
    request plans nothing.  Subscribe with
    ``CellierController.on_reslice_started``.

    Attributes
    ----------
    source_id : UUID
        The render layer's slice coordinator.
    scene_id : UUID
        The scene being resliced.  The routing key.
    canvas_id : UUID
        The canvas whose view the request was planned for.
    visual_ids : frozenset[UUID]
        The visuals that have data to load in this reslice.  Visuals with
        nothing to load (hidden, culled, outside the slice) are not listed.
    """

    source_id: UUID
    scene_id: UUID
    canvas_id: UUID
    visual_ids: frozenset[UUID]


class ResliceCompletedEvent(NamedTuple):
    """One visual's data for a reslice is on the GPU.

    Fires when the visual's latest load pass is resident (a chunk the
    scheduler gave up on counts as done), once for each
    ``ResliceStartedEvent`` that named the visual since its last
    completion.  A newer reslice that starts before an older one finishes
    therefore answers both.  For loading progress in between, see
    ``ResliceProgressEvent`` and ``BackstopCompleteEvent``.
    Subscribe with ``CellierController.on_reslice_completed``.

    Attributes
    ----------
    source_id : UUID
        The render layer's slice coordinator.
    scene_id : UUID
        The scene that owns the visual.
    canvas_id : UUID
        The canvas whose reslice request this completes.  A scene shown on
        several canvases completes once per canvas.
    visual_id : UUID
        The visual whose data is committed.  The routing key.
    brick_count : int
        Number of bricks or tiles the visual's latest pass needed, over
        every level it loaded.
    """

    source_id: UUID
    scene_id: UUID
    canvas_id: UUID
    visual_id: UUID
    brick_count: int


def sum_intermediate(
    a: tuple[tuple[int, int, int], ...], b: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    """Add two progresses' intermediate stages by offset.

    Parameters
    ----------
    a, b : tuple of (offset, needed, resident)
        One triple per stage.

    Returns
    -------
    tuple of (offset, needed, resident)
        Coarsest (largest offset) first.  The same offset adds its counts;
        an offset only one side has is kept.
    """
    if not b:
        return a
    counts: dict[int, tuple[int, int]] = {}
    for offset, needed, resident in (*a, *b):
        had = counts.get(offset, (0, 0))
        counts[offset] = (had[0] + needed, had[1] + resident)
    return tuple((offset, *counts[offset]) for offset in sorted(counts, reverse=True))


def loading_stage(
    needed_backstop: int,
    resident_backstop: int,
    intermediate: tuple[tuple[int, int, int], ...],
    needed_target: int,
    resident_target: int,
) -> str | None:
    """Which stage a progress's counts are still waiting for, coarsest first.

    Returns
    -------
    str or None
        ``"coarse"``, ``"intermediate_1"`` ... ``"intermediate_n"`` (numbered
        from 1 in the order of *intermediate*), ``"target"``, or ``None``
        when every needed chunk is resident.
    """
    if resident_backstop < needed_backstop:
        return "coarse"
    for number, (_offset, needed, resident) in enumerate(intermediate, start=1):
        if resident < needed:
            return f"intermediate_{number}"
    if resident_target < needed_target:
        return "target"
    return None


class LoadingProgress(NamedTuple):
    """How far a multiscale visual's latest plan has loaded.

    Summed over every atlas of the visual (a composite image has one per
    drawn channel).  Counts are chunks: bricks in 3D, tiles in 2D.

    Attributes
    ----------
    needed_backstop, resident_backstop : int
        Coarse backstop chunks the plan wants, and how many are on the GPU.
    needed_target, resident_target : int
        The same for the target: the level of detail the view asked for.
    in_flight : int
        Reads outstanding, including reads for chunks no longer wanted.
    failed : int
        Wanted chunks given up after the retry limit.
    truncated_target, truncated_backstop : int
        Chunks the plan dropped to fit the GPU cache, or the backstop cap.
    backstop_complete : bool
        Every wanted backstop chunk is resident or given up.  The view then
        shows the current slice everywhere, at least blurry.
    complete : bool
        Every wanted chunk is resident or given up.
    target_deferred : bool
        The latest plan was the backstop only: a tick of a dims scrub with
        ``coarsest_while_moving`` on for the view.  The target is planned once the
        scrub ends, so ``complete`` here does not mean full detail.
    intermediate : tuple of (offset, needed, resident)
        One triple per intermediate stage
        (``ProgressiveLoadingConfig.intermediate_offsets``), coarsest first:
        the chunks the stage wants, and how many are on the GPU.  ``()``
        with no stage.  A stage that is empty for the view counts 0 of 0.
    truncated_intermediate : int
        Stage chunks the plan dropped under the stages' slot cap.

    Notes
    -----
    The first eight fields are counts, and are summed by position when
    progress is combined over several visuals (an ``OrthoViewer`` panel
    group); a new field goes after ``target_deferred``.
    """

    needed_backstop: int = 0
    resident_backstop: int = 0
    needed_target: int = 0
    resident_target: int = 0
    in_flight: int = 0
    failed: int = 0
    truncated_target: int = 0
    truncated_backstop: int = 0
    backstop_complete: bool = True
    complete: bool = True
    target_deferred: bool = False
    intermediate: tuple[tuple[int, int, int], ...] = ()
    truncated_intermediate: int = 0

    @property
    def stage(self) -> str | None:
        """What is loading: the most important class not yet all resident.

        ``"coarse"``, ``"intermediate_1"`` ... ``"intermediate_n"`` (numbered
        from 1 in the order of ``intermediate_offsets``), ``"target"``, or
        ``None`` when nothing is loading.  The target is not held behind a
        stage, so this can name a stage while target chunks arrive: it is
        what the picture still waits for, coarsest first.
        """
        if self.complete:
            return None
        return loading_stage(
            self.needed_backstop,
            self.resident_backstop,
            self.intermediate,
            self.needed_target,
            self.resident_target,
        )

    @property
    def fraction(self) -> float:
        """Target chunks resident over needed, in ``[0, 1]``; 1 when none."""
        if self.needed_target == 0:
            return 1.0
        return self.resident_target / self.needed_target


class ResliceProgressEvent(NamedTuple):
    """A multiscale visual's loading progress changed.

    Emitted at most once per event-loop iteration per visual: after a plan,
    after a commit round that wrote its data, after a read is given up, and
    after an invalidation.  Not emitted per read, so ``in_flight`` is the
    count at the last of those moments.

    Attributes
    ----------
    source_id : UUID
        ID of the emitter (the render layer).
    scene_id : UUID
        Scene that owns the visual.
    visual_id : UUID
        The visual.  The routing key.
    progress : LoadingProgress
        The visual's counts.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    progress: LoadingProgress


class LoadingConfigChangedEvent(NamedTuple):
    """A multiscale visual's ``render_config.loading`` changed.

    Emitted for every change, whether it came from
    ``CellierController.set_loading_config`` (or a
    ``LoadingConfigUpdateEvent``) or from assigning ``render_config``
    directly, so a control showing the settings stays in step.

    Attributes
    ----------
    source_id : UUID
        Who asked for the change: the widget's id for a GUI edit, otherwise
        the controller's.
    visual_id : UUID
        The visual.  The routing key.
    loading : ProgressiveLoadingConfig
        The complete config after the change.
    """

    source_id: UUID
    visual_id: UUID
    loading: Any


class PlaneGizmoMovedEvent(NamedTuple):
    """A plane gizmo in a canvas was grabbed, moved or released.

    Emitted by the canvas that draws the gizmo.  A drag is one ``"start"``,
    then at most one ``"move"`` per frame the canvas draws, then one
    ``"end"``.  The last pose of a drag is always reported as a ``"move"``
    before the ``"end"``.

    The pose is in the scene's rendered space, in ``(x, y, z)`` order.  It
    holds no render-layer types.

    Attributes
    ----------
    source_id : UUID
        The canvas.
    canvas_id : UUID
        The canvas the gizmo is drawn in.
    gizmo_id : UUID
        The gizmo.  The routing key.
    point : tuple[float, float, float]
        The gizmo's position: a point on the plane.
    normal : tuple[float, float, float]
        The plane's unit normal there.  It points into the kept half-space.
    phase : str
        ``"start"``, ``"move"`` or ``"end"``.
    handle_kind : str
        ``"translate"``, ``"rotate"`` or ``"scale"``: the handle being
        dragged.  Only a gizmo on a render plane shows scale handles.
    handle_axis : int or tuple[int, int]
        The axis of the handle in the gizmo's own frame, where axis 0 is
        the normal and axes 1 and 2 are the in-plane axes: the axis a
        one-axis handle moves along, turns about or scales, or the pair of
        axes a two-axis translate handle moves in.
    in_plane_axis_0, in_plane_axis_1 : tuple[float, float, float]
        The gizmo's two in-plane axes, unit vectors.  With ``normal`` they
        are the gizmo's whole frame: ``normal`` is their cross product.
    scale : tuple[float, float]
        The gizmo's scale along the two in-plane axes: 1.0 unless a scale
        handle is held, when it is the factor of the drag so far.
    """

    source_id: UUID
    canvas_id: UUID
    gizmo_id: UUID
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    phase: str
    handle_kind: str
    handle_axis: Any
    in_plane_axis_0: tuple[float, float, float] = (0.0, 1.0, 0.0)
    in_plane_axis_1: tuple[float, float, float] = (0.0, 0.0, 1.0)
    scale: tuple[float, float] = (1.0, 1.0)


class ClippingPlanesChangedEvent(NamedTuple):
    """A visual's ``clipping_planes`` changed.

    Emitted for every change, whether it came from
    ``CellierController.set_clipping_planes`` (or a
    ``ClippingPlanesUpdateEvent``) or from assigning
    ``visual.clipping_planes`` directly.

    Attributes
    ----------
    source_id : UUID
        Who asked for the change: the widget's id for a GUI edit, otherwise
        the controller's.
    visual_id : UUID
        The visual.  The routing key.
    clipping_planes : tuple[ClippingPlane, ...]
        The complete tuple after the change.
    """

    source_id: UUID
    visual_id: UUID
    clipping_planes: Any


class RenderPlanesChangedEvent(NamedTuple):
    """A visual's ``render_planes`` changed.

    Emitted for every change, whether it came from
    ``CellierController.set_render_planes`` (or a
    ``RenderPlanesUpdateEvent``) or from assigning ``visual.render_planes``
    directly, and whether or not the visual is in ``"plane"`` render mode.

    Attributes
    ----------
    source_id : UUID
        Who asked for the change: the widget's id for a GUI edit, otherwise
        the controller's.
    visual_id : UUID
        The visual.  The routing key.
    render_planes : tuple[RenderPlane, ...]
        The complete tuple after the change.
    """

    source_id: UUID
    visual_id: UUID
    render_planes: Any


class PlaneInteractionEvent(NamedTuple):
    """A drag of a visual's planes started or ended.

    One tracker per visual serves its clipping planes and its render planes.
    A drag is a run of ``clipping_planes`` or ``render_planes`` changes made
    inside ``CellierController.plane_interaction`` (a gizmo opens one for
    the length of a drag).  The start is emitted with the first change,
    ahead of its ``ClippingPlanesChangedEvent`` or
    ``RenderPlanesChangedEvent``.  No event is emitted per change.

    A multiscale image or labels visual in a 3D view with
    ``appearance.coarsest_while_moving_3d`` on (the default) plans nothing
    between the start and the end, and its target is planned at the end.
    Every other visual plans a plane change the same way inside a drag as
    outside one.

    Attributes
    ----------
    source_id : UUID
        The source of the change or scope that caused the transition.
    visual_id : UUID
        The visual whose planes are dragged.  The routing key.
    phase : {"start", "end"}
        Which transition this is.
    reason : {"release", "settle", "jump", "cancel"} or None
        Why the drag ended: the last scope closed, the planes were still
        for ``SchedulerConfig.dims_settle_s``, or a change arrived after
        the scopes had closed.  ``None`` for a start.
    """

    source_id: UUID
    visual_id: UUID
    phase: Literal["start", "end"]
    reason: Literal["release", "settle", "jump", "cancel"] | None = None


class TransformInteractionEvent(NamedTuple):
    """A drag of a visual's transform started or ended.

    One tracker per visual.  A drag is a run of ``transform`` changes made
    inside ``CellierController.transform_interaction`` (a rotation slider
    opens one for the length of a drag).  The start is emitted with the
    first change, ahead of its ``TransformChangedEvent``.  No event is
    emitted per change.

    In a 2D view, a visual with ``appearance.coarsest_while_moving_2d`` on
    plans only its backstop between the start and the end.  In a 3D view,
    a multiscale image or labels visual with
    ``appearance.coarsest_while_moving_3d`` on keeps its last plan, whose
    bricks move with the transform.  The target is planned at the end.

    Attributes
    ----------
    source_id : UUID
        The source of the change or scope that caused the transition.
    visual_id : UUID
        The visual whose transform is dragged.  The routing key.
    phase : {"start", "end"}
        Which transition this is.
    reason : {"release", "settle", "jump", "cancel"} or None
        Why the drag ended: the last scope closed, the transform was still
        for ``SchedulerConfig.dims_settle_s``, or a change arrived after
        the scopes had closed.  ``None`` for a start.
    """

    source_id: UUID
    visual_id: UUID
    phase: Literal["start", "end"]
    reason: Literal["release", "settle", "jump", "cancel"] | None = None


class PlaneGizmoChangedEvent(NamedTuple):
    """The plane a canvas's plane gizmo edits changed.

    A canvas has at most one plane gizmo, on a clipping plane or on a
    render plane.  Emitted when one is added, when it is replaced by one on
    another plane, and when it is closed, by its owner or by itself (its
    plane or visual was removed, the canvas left 3D, a render plane's
    visual left plane mode).

    Attributes
    ----------
    source_id : UUID
        Who asked: the widget's id for a GUI toggle, otherwise the
        controller's.
    canvas_id : UUID
        The canvas.  The routing key.
    visual_id : UUID or None
        The visual whose plane the gizmo edits; ``None`` when the canvas
        has no gizmo.
    kind : {"clipping", "render"} or None
        Which of the visual's tuples the plane is in: ``clipping_planes``
        or ``render_planes``.  ``None`` when the canvas has no gizmo.
    plane_id : UUID or None
        That plane's id; ``None`` when the canvas has no gizmo.
    """

    source_id: UUID
    canvas_id: UUID
    visual_id: UUID | None = None
    kind: Literal["clipping", "render"] | None = None
    plane_id: UUID | None = None


class LodConfigChangedEvent(NamedTuple):
    """A multiscale mesh's ``lod`` config changed.

    Emitted for every change, whether it came from
    ``CellierController.set_lod_config`` (or a ``LodConfigUpdateEvent``) or
    from assigning ``visual.lod`` directly, so a control showing the
    settings stays in step.

    Attributes
    ----------
    source_id : UUID
        Who asked for the change: the widget's id for a GUI edit, otherwise
        the controller's.
    visual_id : UUID
        The visual.  The routing key.
    lod : GeometryLodConfig
        The complete config after the change.
    """

    source_id: UUID
    visual_id: UUID
    lod: Any


class BackstopCompleteEvent(NamedTuple):
    """Every backstop chunk of a multiscale visual's latest plan is done.

    From here on the view shows the current slice everywhere, possibly at
    the backstop's coarse level.  Emitted once per plan whose backstop
    completes, and only when the plan has a backstop.

    Attributes
    ----------
    source_id : UUID
        ID of the emitter (the render layer).
    scene_id : UUID
        Scene that owns the visual.
    visual_id : UUID
        The visual.  The routing key.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class FrameRenderedEvent(NamedTuple):
    """A canvas finished rendering a frame.

    Emitted at the end of every draw, so it is frequent.  To wait for a
    canvas's first frame, use ``CellierController.on_canvas_first_frame``,
    which subscribes to this event once.

    Attributes
    ----------
    source_id : UUID
        The canvas that rendered (equal to ``canvas_id``).
    canvas_id : UUID
        The canvas that rendered.  The routing key.
    frame_time_ms : float
        Wall-clock time the frame took to render, in milliseconds.
    """

    source_id: UUID
    canvas_id: UUID
    frame_time_ms: float


class CanvasConnectedEvent(NamedTuple):
    """Emitted once a canvas's front end is live and able to draw.

    Distinct from ``FrameRenderedEvent``, which says a frame *has* been
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
    """Emitted when the size of a canvas changes, in logical pixels.

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
    """A visual was added to a scene and is ready to draw.

    Emitted after the visual's render objects and subscriptions exist.
    Subscribe with ``CellierController.on_visual_added``.

    Attributes
    ----------
    source_id : UUID
        The controller.
    scene_id : UUID
        The scene the visual was added to.  The routing key.
    visual_id : UUID
        The visual added.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class VisualRemovedEvent(NamedTuple):
    """A visual was removed from a scene.

    Emitted last, after the visual's render objects and subscriptions are
    gone, so ``visual_id`` no longer resolves on the controller.  Subscribe
    with ``CellierController.on_visual_removed``.

    Attributes
    ----------
    source_id : UUID
        The controller.
    scene_id : UUID
        The scene the visual was removed from.  The routing key.
    visual_id : UUID
        The visual removed.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class SceneAddedEvent(NamedTuple):
    """A scene was added to the controller.

    Subscribe with ``CellierController.on_scene_added``.

    Attributes
    ----------
    source_id : UUID
        The controller.
    scene_id : UUID
        The scene added.  The routing key.
    """

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

    Emitted both for a field change on the background model and for
    wholesale replacement
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
        One of ``VISUAL_RENDER_FIELDS``, such as ``"outline.slot"``,
        ``"ambient_occlusion"`` or ``"outline_depth_range"``.
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
    scene, a visual or a canvas, so this event has no entity id: it cannot
    be subscribed to with an ``entity_id``.  Every subscriber receives every
    render-config event and filters on ``section`` itself, the way the
    appearance widgets filter on ``field_name``.

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

    Emitted both for a field change on one ``TrailConfig`` and for
    whole-dict replacement (``visual.trail = {...}``).  In the first case
    ``axis`` and ``field_name`` say which field changed; in the second they
    are ``None``.

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
    """Fired when a visual's data-to-world transform is replaced.

    Attributes
    ----------
    source_id : UUID
        The controller.
    scene_id : UUID
        The visual's scene.
    visual_id : UUID
        The visual.  The routing key.
    transform : BaseTransform
        The new data-to-world transform.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    transform: BaseTransform


class DrawStatus(NamedTuple):
    """Whether a scene's current view draws a visual, and why not.

    A visual whose transform the view cannot draw -- a 2D view whose slice
    cuts an image at an angle, say -- draws nothing and reports it here
    rather than drawing something wrong.

    Attributes
    ----------
    visual_id : UUID
        The visual.
    scene_id : UUID
        Its scene.
    name : str
        The visual's name, for messages.
    alignment : str
        The view's ``cellier.transform.SliceAlignment`` value for the
        visual (``"axis_aligned"``, ``"in_plane"``, ``"slice_shift"`` or
        ``"oblique"``).
    drawable : bool
        Whether the view draws the visual.
    reason : str or None
        What the user can do about it, when not drawable.
    """

    visual_id: UUID
    scene_id: UUID
    name: str
    alignment: str
    drawable: bool
    reason: str | None


class DrawStatusChangedEvent(NamedTuple):
    """Fired when a visual's ``DrawStatus`` in its scene's view changes.

    Attributes
    ----------
    source_id : UUID
        The controller.
    scene_id : UUID
        The scene whose view changed.  The routing key.
    visual_id : UUID
        The visual whose status changed.
    status : DrawStatus
        The new status.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    status: DrawStatus


class SceneRemovedEvent(NamedTuple):
    """A scene was removed, with its visuals and canvases.

    Emitted after its visuals and canvases are torn down.  Subscribe with
    ``CellierController.on_scene_removed``.

    Attributes
    ----------
    source_id : UUID
        The controller.
    scene_id : UUID
        The scene removed.  The routing key.
    """

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
        Index along ``channel_axis`` -> the value at the picked voxel.  On a
        multiscale visual that voxel is the one of the pyramid level drawn at
        the hit (per channel), so the value agrees with what is on screen: a
        coarse level's surface can sit a voxel from the level-0 one.  The value
        is level 0's where the level cannot be said (an in-memory visual, or a
        brick table rewritten since the frame was drawn).  A visual with
        ``channel_axis=None`` reports ``{0: value}``.  Single mode
        reports its drawn channel.  Composite mode reports every drawn
        (visible) channel on a 2D canvas and only the pick-buffer winner on a
        3D canvas.  Empty when the voxel lies outside the data.

    Notes
    -----
    **Precision of 3D picks.**  A multiscale visual drawn as a volume (a 3D
    canvas, or a tilted 2D view) stores the hit in 44 bits of the pick buffer,
    shared between the three displayed axes by their lengths: an axis of ``n``
    voxels first gets the ``bit_length(n - 1)`` bits that name each of its
    voxels, and the bits left over split a voxel into finer steps.  When those
    bits add up to at most 44 and no displayed axis is longer than ``2 ** 24``
    voxels, ``floor(data_coordinate)`` is always the voxel hit: a lightsheet
    volume of 2048 x 10000 x 40000 voxels needs 11 + 14 + 16 = 41 bits.  A
    larger volume resolves to steps of 2, 4 or more voxels on its longest axes,
    and a ``UserWarning`` names the step when the visual's volume is built with
    that shape.
    """

    data_coordinate: tuple[float, ...]
    channel_values: dict[int, float]


class MeshPickInfo(NamedTuple):
    """Element-level pick result for a mesh visual: what was drawn there.

    Attributes
    ----------
    face_index : int or None
        The face under the cursor, in the numbering of the level that was
        drawn (the store's faces for a single-level mesh).  ``None`` for
        ``part="fill"``: the area a section's loop encloses is no face.
    part : {"face", "outline", "fill"}
        ``"face"``: a face of the mesh (a 3D view, or a face lying in the
        slice plane of a 2D view).  ``"outline"``: the section's outline;
        ``face_index`` is the face the plane crosses there.  ``"fill"``: the
        section's fill.
    level : int
        The level that was drawn, 0 the finest.
    """

    face_index: int | None
    part: Literal["face", "outline", "fill"] = "face"
    level: int = 0


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

        On a multiscale visual, in 2D and 3D, the hit is on the pyramid level
        that was drawn, and ``value`` is the label that level holds there: the
        label you see.  A coarse voxel covers several level-0 voxels and can
        stand a voxel or more beyond the level-0 surface, so reading level 0 at
        the same point could return the background.  The value is level 0's
        where the level cannot be said (a brick or tile table rewritten since
        the frame was drawn) and for in-memory labels, which have one level.

    Notes
    -----
    **Precision of 3D picks.**  A multiscale visual drawn as a volume (a 3D
    canvas, or a tilted 2D view) stores the hit in 44 bits of the pick buffer,
    shared between the three displayed axes by their lengths: an axis of ``n``
    voxels first gets the ``bit_length(n - 1)`` bits that name each of its
    voxels, and the bits left over split a voxel into finer steps.  When those
    bits add up to at most 44 and no displayed axis is longer than ``2 ** 24``
    voxels, ``floor(data_coordinate)`` is always the voxel hit: a lightsheet
    volume of 2048 x 10000 x 40000 voxels needs 11 + 14 + 16 = 41 bits.  A
    larger volume resolves to steps of 2, 4 or more voxels on its longest axes,
    and a ``UserWarning`` names the step when the visual's volume is built with
    that shape.
    """

    data_coordinate: tuple[float, ...]
    value: int


class MaskedImagePickInfo(NamedTuple):
    """Element-level pick result for a MaskedImage.

    Every value is read at **level 0** and at the **nearest voxel**, as image
    picks are, so a pick can differ slightly from the drawn colour while a
    coarse level is on screen, or under linear interpolation.

    Attributes
    ----------
    data_coordinate : tuple[float, ...]
        The hit in the **mask**'s level-0 data coordinates, the convention of
        ``ImagePickInfo.data_coordinate``: ``floor`` gives the voxel index.
    value_coordinate : tuple[float, ...]
        The same point in the **value** image's level-0 data coordinates, the
        same convention.
    mask_value : float or None
        The mask at the hit; ``None`` outside the mask's data.
    surface_value : float or None
        The value image at the hit; ``None`` outside the value's data.
    projected_value : float or None
        3D: the max, mean or min of the value along the inward normal over
        the placed ``surface_thickness``, as the surface is coloured (on a
        clipping plane's cut face, along the plane's normal into the kept
        side).  2D, and 3D with thickness 0: ``surface_value``.  ``None``
        where no sample lies in the value's data.

    Notes
    -----
    **Precision of 3D picks.**  A MaskedImage drawn as a volume (a 3D canvas,
    or a tilted 2D view) stores the hit in 44 bits of the pick buffer,
    shared between the three displayed axes by their lengths: an axis of ``n``
    voxels first gets the ``bit_length(n - 1)`` bits that name each of its
    voxels, and the bits left over split a voxel into finer steps.  When those
    bits add up to at most 44 and no displayed axis is longer than ``2 ** 24``
    voxels, ``floor(data_coordinate)`` is always the voxel hit: a lightsheet
    volume of 2048 x 10000 x 40000 voxels needs 11 + 14 + 16 = 41 bits.  A
    larger volume resolves to steps of 2, 4 or more voxels on its longest axes,
    and a ``UserWarning`` names the step when the visual's volume is built with
    that shape.  The axis lengths are the mask's; ``value_coordinate`` follows
    ``data_coordinate``.
    """

    data_coordinate: tuple[float, ...]
    value_coordinate: tuple[float, ...]
    mask_value: float | None
    surface_value: float | None
    projected_value: float | None


class GraphNodePickInfo(NamedTuple):
    """Element-level pick result for a node of a graph visual.

    Attributes
    ----------
    node_id : Any
        The store's **original** node id, not the render-buffer row. Typed
        ``Any`` rather than ``int`` because a geff file may carry uint64
        or string ids; this is the one place the graph pick payload is
        more weakly typed than the points one.
    node_row : int
        Row of the node in the store's position array.  Stable because the
        store is immutable, and useful for indexing the store's own arrays.
    """

    node_id: Any
    node_row: int


class GraphEdgePickInfo(NamedTuple):
    """Element-level pick result for an edge of a graph visual.

    Attributes
    ----------
    edge_index : int
        Row of the edge in the store's edge array, or ``-1`` when it could
        not be resolved.  Stable only because the store is immutable.
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
    | MaskedImagePickInfo
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
    """Emitted when the primary pointer button is pressed on a 2D canvas.

    Subscribe with ``CellierController.on_mouse_press_2d``.  For what
    was hit beyond the visual's id, subscribe to the typed pick events with
    ``CellierController.on_pick``.

    Attributes
    ----------
    source_id : UUID
        The canvas.  The routing key.
    scene_id : UUID
        The canvas's scene.
    world_coordinate : np.ndarray
        The pointer's position in world coordinates, one value per world
        axis (float64): the displayed axes from the pointer, the sliced axes
        at their current slice positions.
    pick_info : CanvasPickInfo
        The visual under the pointer, if any.
    button : int
        The button that changed: ``0`` none (a move), ``1`` left, ``2`` right,
        ``3`` middle, ``4`` and ``5`` the extra buttons.
    buttons : tuple
        Every button held, as the same numbers.
    modifiers : tuple
        The modifier keys held, as strings: ``"Shift"``, ``"Control"``,
        ``"Alt"``, ``"Meta"``.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    """

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseMove2DEvent(NamedTuple):
    """Emitted on every pointer-move event on a 2D canvas (button up or down).

    Subscribe with ``CellierController.on_mouse_move_2d``.  For what
    was hit beyond the visual's id, subscribe to the typed pick events with
    ``CellierController.on_pick``.

    Attributes
    ----------
    source_id : UUID
        The canvas.  The routing key.
    scene_id : UUID
        The canvas's scene.
    world_coordinate : np.ndarray
        The pointer's position in world coordinates, one value per world
        axis (float64): the displayed axes from the pointer, the sliced axes
        at their current slice positions.
    pick_info : CanvasPickInfo
        The visual under the pointer, if any.
    button : int
        The button that changed: ``0`` none (a move), ``1`` left, ``2`` right,
        ``3`` middle, ``4`` and ``5`` the extra buttons.
    buttons : tuple
        Every button held, as the same numbers.
    modifiers : tuple
        The modifier keys held, as strings: ``"Shift"``, ``"Control"``,
        ``"Alt"``, ``"Meta"``.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    """

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseRelease2DEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a 2D canvas.

    Subscribe with ``CellierController.on_mouse_release_2d``.  For what
    was hit beyond the visual's id, subscribe to the typed pick events with
    ``CellierController.on_pick``.

    Attributes
    ----------
    source_id : UUID
        The canvas.  The routing key.
    scene_id : UUID
        The canvas's scene.
    world_coordinate : np.ndarray
        The pointer's position in world coordinates, one value per world
        axis (float64): the displayed axes from the pointer, the sliced axes
        at their current slice positions.
    pick_info : CanvasPickInfo
        The visual under the pointer, if any.
    button : int
        The button that changed: ``0`` none (a move), ``1`` left, ``2`` right,
        ``3`` middle, ``4`` and ``5`` the extra buttons.
    buttons : tuple
        Every button held, as the same numbers.
    modifiers : tuple
        The modifier keys held, as strings: ``"Shift"``, ``"Control"``,
        ``"Alt"``, ``"Meta"``.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    """

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMousePress3DEvent(NamedTuple):
    """Emitted when the primary pointer button is pressed on a 3D canvas.

    Subscribe with ``CellierController.on_mouse_press_3d``.  For what
    was hit beyond the visual's id, subscribe to the typed pick events with
    ``CellierController.on_pick``.

    Attributes
    ----------
    source_id : UUID
        The canvas.  The routing key.
    scene_id : UUID
        The canvas's scene.
    ray : ViewRay
        The view ray from the camera through the pointer, in world
        coordinates.
    pick_info : CanvasPickInfo
        The visual under the pointer, if any.
    button : int
        The button that changed: ``0`` none (a move), ``1`` left, ``2`` right,
        ``3`` middle, ``4`` and ``5`` the extra buttons.
    buttons : tuple
        Every button held, as the same numbers.
    modifiers : tuple
        The modifier keys held, as strings: ``"Shift"``, ``"Control"``,
        ``"Alt"``, ``"Meta"``.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    """

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseMove3DEvent(NamedTuple):
    """Emitted on every pointer-move event on a 3D canvas (button up or down).

    Subscribe with ``CellierController.on_mouse_move_3d``.  For what
    was hit beyond the visual's id, subscribe to the typed pick events with
    ``CellierController.on_pick``.

    Attributes
    ----------
    source_id : UUID
        The canvas.  The routing key.
    scene_id : UUID
        The canvas's scene.
    ray : ViewRay
        The view ray from the camera through the pointer, in world
        coordinates.
    pick_info : CanvasPickInfo
        The visual under the pointer, if any.
    button : int
        The button that changed: ``0`` none (a move), ``1`` left, ``2`` right,
        ``3`` middle, ``4`` and ``5`` the extra buttons.
    buttons : tuple
        Every button held, as the same numbers.
    modifiers : tuple
        The modifier keys held, as strings: ``"Shift"``, ``"Control"``,
        ``"Alt"``, ``"Meta"``.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    """

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseRelease3DEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a 3D canvas.

    Subscribe with ``CellierController.on_mouse_release_3d``.  For what
    was hit beyond the visual's id, subscribe to the typed pick events with
    ``CellierController.on_pick``.

    Attributes
    ----------
    source_id : UUID
        The canvas.  The routing key.
    scene_id : UUID
        The canvas's scene.
    ray : ViewRay
        The view ray from the camera through the pointer, in world
        coordinates.
    pick_info : CanvasPickInfo
        The visual under the pointer, if any.
    button : int
        The button that changed: ``0`` none (a move), ``1`` left, ``2`` right,
        ``3`` middle, ``4`` and ``5`` the extra buttons.
    buttons : tuple
        Every button held, as the same numbers.
    modifiers : tuple
        The modifier keys held, as strings: ``"Shift"``, ``"Control"``,
        ``"Alt"``, ``"Meta"``.
    gesture_id : UUID or None
        Shared by one press, its drag moves and its release; ``None`` for a
        hover move.
    """

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class ImagePickEvent(NamedTuple):
    """An image visual was hit on a canvas.

    Emitted only for a hit, after the mouse event for the same pointer event:
    at once for an in-memory image, and when the value read completes for a
    multiscale image (at the level drawn at the hit).  A newer pointer event on
    the canvas cancels an in-flight ``move`` read; ``press`` and ``release``
    reads always complete, so match events by ``gesture_id`` and ``action``, not
    arrival order.

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
    """A labels visual was hit on a canvas.

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


class MaskedImagePickEvent(NamedTuple):
    """A MaskedImage was hit on a canvas.

    Timing as for ``ImagePickEvent``: synchronous when both stores are in
    memory, otherwise emitted when the level-0 reads complete.  The reads run
    in two passes, the mask first (its value and the normal at the hit), then
    the value image (at the hit and along the normal); a newer pointer event
    cancels an in-flight ``move`` read in either pass.

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
    pick_info : MaskedImagePickInfo
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
    pick_info: MaskedImagePickInfo


class PointsPickEvent(NamedTuple):
    """A points visual was hit on a canvas.

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
    """A lines visual was hit on a canvas.

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
    """A mesh visual was hit on a canvas.

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
    """A graph visual was hit on a canvas.

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
    MaskedImagePickEvent,
    LabelsPickEvent,
    PointsPickEvent,
    LinesPickEvent,
    MeshPickEvent,
    GraphPickEvent,
)

PickEvent = (
    ImagePickEvent
    | MaskedImagePickEvent
    | LabelsPickEvent
    | PointsPickEvent
    | LinesPickEvent
    | MeshPickEvent
    | GraphPickEvent
)
"""Any pick event: the union of ``PICK_EVENT_TYPES``.

For annotating a callback that handles several pick event types.
"""


class _CameraControllerEvent(NamedTuple):
    """Internal: a canvas's camera controller started or stopped driving.

    Emitted by ``CanvasView`` from inside a draw, after that frame's camera
    change (if any) was reported.  ``driving`` is ``True`` while the pygfx
    controller has a running action: a drag held (still or not), its damped
    tail, a wheel or key animation.  Not part of the public catalogue;
    consumed only by ``CellierController``, which maps it to a scope on the
    canvas's camera tracker.
    """

    source_id: UUID
    canvas_id: UUID
    scene_id: UUID
    driving: bool


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
        The pointer in the pygfx scene ``(x, y)``, already unprojected through
        the active orthographic camera: ``x`` is the last displayed axis and
        ``y`` the one before it.  Shape (2,), float64.
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
        None on a miss or a visual kind with no pick details.

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
    | DimsInteractionEvent
    | SliderAxesChangedEvent
    | CameraChangedEvent
    | CameraInteractionEvent
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
    | ResliceProgressEvent
    | BackstopCompleteEvent
    | LoadingConfigChangedEvent
    | LodConfigChangedEvent
    | MeshSectionChangedEvent
    | ClippingPlanesChangedEvent
    | RenderPlanesChangedEvent
    | PlaneInteractionEvent
    | TransformInteractionEvent
    | PlaneGizmoChangedEvent
    | PlaneGizmoMovedEvent
    | FrameRenderedEvent
    | CanvasConnectedEvent
    | VisualAddedEvent
    | VisualRemovedEvent
    | TrailChangedEvent
    | TransformChangedEvent
    | DrawStatusChangedEvent
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
