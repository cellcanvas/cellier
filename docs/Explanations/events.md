# Events

## Overview

The EventBus is the sole communication channel between the model layer (psygnal
`EventedModel` instances held by the controller) and the render layer (`RenderManager`,
`CanvasView`, `GFX*Visual`). No model class ever imports a render-layer class. No
render-layer class ever imports a model class. Every cross-layer communication passes
through a typed event on the bus.

The bus is an internal implementation detail. Application developers never hold a
reference to it. External state change notifications are surfaced through the
`controller.on_*` callback registration methods.

This document covers: the shared state NamedTuples, the event catalogue, the bus API,
bounce-back prevention, the ContextVar source-ID pattern, async handler conventions,
how the controller bridges psygnal signals to bus events, and the registration lifecycle.

---

## Module structure

```
src/cellier/v2/
├── _state.py          # DimsState, CameraState — shared by events and render layer
├── events/
│   ├── __init__.py    # exports EventBus, SubscriptionSpec, SubscriberInfo,
│   │                  #   SubscriptionHandle, and all event types
│   ├── _bus.py        # EventBus, SubscriptionHandle, SubscriberInfo, _Subscription
│   ├── _events.py     # all outgoing event NamedTuples
│   └── _update_events.py  # AppearanceUpdateEvent, DimsUpdateEvent,
│                          # AABBUpdateEvent, SubscriptionSpec
└── gui/
    ├── _scene.py           # QtDimsControl, QtCanvasWidget
    └── visuals/
        ├── _contrast_limits.py
        ├── _colormap.py
        ├── _lod_bias.py
        ├── _image.py
        └── _aabb.py
```

**GUI widgets import only from `cellier.v2.events`. No widget imports
`cellier.v2.controller`.**

Nothing outside `src/cellier/v2/` imports from this package. The controller is the only
production caller of `EventBus`.

---

## Widget integration

### The widget contract

Every reusable cellier widget that connects to the event bus must expose:

- `_id: UUID` — a stable UUID generated at construction (one per widget instance).
- `changed: Signal = Signal(object)` — psygnal signal emitted when the widget's value
  changes.  The payload is an `*UpdateEvent` (`AppearanceUpdateEvent`,
  `DimsUpdateEvent`, or `AABBUpdateEvent`).
- `closed: Signal = Signal()` — psygnal signal emitted when the widget closes, so the
  controller can clean up its bus subscriptions automatically.
- `subscription_specs() -> list[SubscriptionSpec]` — declares which bus events the
  widget wants to receive.  Returns an empty list for write-only widgets.

Widgets never import `cellier.v2.controller` and never hold a controller reference.
They are pure UI components that communicate only through signals and
`SubscriptionSpec` declarations.  This makes them distributable as reusable plugins
in separate packages that depend only on `cellier.v2.events`.

### `SubscriptionSpec`

```python
class SubscriptionSpec(NamedTuple):
    """One inbound event subscription declared by a cellier widget."""

    event_type: type
    handler: Callable[..., None]
    entity_id: UUID | None = None
    weak: bool = False
```

`SubscriptionSpec` lives in `src/cellier/v2/events/_update_events.py` and is
re-exported from `cellier.v2.events`.  Placing it in the `events` package means
widgets can import it without creating a circular dependency on any render-layer or
controller module.

### `CellierController.connect_widget`

```python
def connect_widget(
    self,
    widget: Any,
    *,
    subscription_specs: list[SubscriptionSpec] | None = None,
) -> None:
```

`connect_widget` does three things:

1. Connects `widget.changed → self._incoming_events.emit` so that `*UpdateEvent`
   payloads enter the controller's incoming-event pipeline.
2. Connects `widget.closed → self.unsubscribe_owner(widget._id)` so that bus
   subscriptions are cleaned up automatically when the widget closes.
3. Registers each `SubscriptionSpec` on the outgoing event bus with
   `owner_id=widget._id`.

Typical call site after construction:

```python
slider = QtClimRangeSlider(visual_id, clim_range=(0, 255), initial_clim=(0, 200))
controller.connect_widget(slider, subscription_specs=slider.subscription_specs())
```

Write-only widgets (no inbound subscriptions) omit `subscription_specs`:

```python
controller.connect_widget(my_button)
```

### Application code exception

Script-local closures and callbacks that live in the same file as the controller
instantiation may call `controller.incoming_events.emit()` directly rather than
wrapping themselves in the widget contract.  The widget contract exists to enable
reusable components distributed in separate packages.

---

## Design principles

### NamedTuples for all events and state snapshots

All events and shared state objects are `NamedTuple`s — immutable, fast, and consistent
with the project convention. Because Python `NamedTuple` subclasses cannot add fields,
there is no common base class with shared fields. Instead, every event NamedTuple
independently declares a `source_id: UUID` as its first field. The `CellierEventTypes`
Union alias at module level groups them for type annotations on the bus.

### One universal field: `source_id`

Every event carries a `source_id: UUID` identifying the object that triggered the
change. This is the primary mechanism for preventing feedback loops (see
§Bounce-back prevention and the ContextVar pattern). The bus itself is source-agnostic;
filtering on `source_id` is the subscriber's responsibility.

### No model references inside the bus

The bus is constructed before any model or render object exists. It holds no reference
to `ViewerModel`, `DimsManager`, or any psygnal object. The controller owns all the
callbacks that bridge psygnal signals to bus events (see §How the controller bridges
psygnal to the bus).

### Typed dispatch with optional entity filtering

Subscribers declare the event type they want, optionally scoped to a specific entity
(scene, visual, canvas, data store). New event types require no changes to the bus
class itself.

### Synchronous bus, async-safe handlers

All bus dispatch is synchronous. Handlers that need to await work wrap it in
`asyncio.create_task()` and return immediately. This is a documented convention, not
a constraint enforced by the bus.

### Strong references by default; weak references opt-in

Subscriptions hold strong references to callbacks by default. The controller's explicit
`unsubscribe_all(owner_id)` lifecycle is the primary cleanup mechanism and is sufficient
for all render-layer objects whose lifetimes the controller manages directly.

Weak references (`weak=True`) are a safety net for subscribers that may be
garbage-collected outside the controller's teardown path — for example, transient
widgets closed by Qt independently. When a weak subscription's target is collected,
the entry is cleaned up lazily on the next emission of that event type.

### Symmetric registration lifecycle

Every entity that is registered can be fully deregistered. All callback references and
entity filters are cleaned up atomically on deregistration via `unsubscribe_all()`.
Weak subscriptions are additionally cleaned up automatically when their target is
collected, without requiring any explicit call.

---

## Shared state NamedTuples

These are not events — they are the immutable state payloads that events carry. They
live in `src/cellier/v2/_state.py`. This module sits above both the events package and
the render layer so neither needs to import the other to reach shared types.

### `DimsState`

`DimsState` is a two-level structure. The top level carries `axis_labels` alongside a
`selection` object. The selection is typed as `SelectionState`, which is currently
either an `AxisAlignedSelectionState` (the only concrete implementation) or the stub
`PlaneSelectionState`.

```python
SelectionState = AxisAlignedSelectionState | PlaneSelectionState


class AxisAlignedSelectionState(NamedTuple):
    """Immutable snapshot of an axis-aligned slice selection.

    Parameters
    ----------
    displayed_axes :
        Indices of axes currently rendered, e.g. ``(0, 1, 2)`` for 3-D
        or ``(1, 2)`` for a 2-D XY slice.
    slice_indices :
        Mapping of every world axis to its position.  A displayed axis
        keeps a stored position, which the selection ignores; the scene's
        derived ``slider_axes`` say which positions a slider shows.  An
        image in composite mode draws all of its channels whatever the
        position on its channel axis.
    """

    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, float]

    def to_index_selection(self, ndim: int) -> tuple[int | slice, ...]:
        """Return a per-axis numpy indexer in axis order.

        Displayed axes → ``slice(None)``, sliced axes → their int value.
        """
        ...


class DimsState(NamedTuple):
    """Current dimension display state for a scene.

    Parameters
    ----------
    axis_labels :
        Ordered labels for every axis in the scene coordinate system,
        e.g. ``("z", "y", "x")``.
    selection :
        Immutable snapshot of which axes are displayed and the current
        slice position for all non-displayed axes.
    """

    axis_labels: tuple[str, ...]
    selection: SelectionState
```

Access patterns — accessing displayed axes and slice indices from a `DimsState`:

```python
# in a bus event handler
displayed = event.dims_state.selection.displayed_axes  # e.g. (0, 1, 2)
z_index   = event.dims_state.selection.slice_indices.get(0, 0)
```

### `CameraState`

Numpy-free immutable snapshot suitable for event payloads, external callbacks, and
serialization. Distinct from `ReslicingRequest`, which carries numpy arrays for the
hot slicing path.

```python
class CameraState(NamedTuple):
    """Immutable snapshot of the active camera's logical state.

    For perspective cameras: ``fov`` is populated; ``extent`` is
    ``(0.0, 0.0)``.
    For orthographic cameras: ``extent`` is populated; ``fov`` is
    ``0.0``.
    ``zoom`` applies to both types.

    Parameters
    ----------
    camera_type :
        ``"perspective"`` or ``"orthographic"``.
    position :
        World-space camera position ``(x, y, z)``.
    rotation :
        Camera orientation as a unit quaternion ``(x, y, z, w)``,
        matching pygfx's convention.
    up :
        World-space up vector ``(x, y, z)``.
    fov :
        Vertical field of view in degrees.  Perspective cameras only;
        ``0.0`` for orthographic.
    zoom :
        Unitless zoom multiplier.  Applies to both camera types.
    extent :
        ``(width, height)`` in world units.  Orthographic cameras only;
        ``(0.0, 0.0)`` for perspective.
    depth_range :
        ``(near, far)`` clip distances.
    """

    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    zoom: float
    extent: tuple[float, float]
    depth_range: tuple[float, float]
```

`CameraState` and `DimsState` are both defined in `src/cellier/v2/_state.py` and
re-exported from `cellier.v2.events` so subscribers have a single import location.

---

## Event catalogue

All events live in `src/cellier/v2/events/_events.py`. All are `NamedTuple`s. The
first field of every event is `source_id: UUID`.

### Scene / dims events

```python
class DimsChangedEvent(NamedTuple):
    """Fired when a scene's DimsManager state changes.

    Triggers: dims slider move, programmatic mutation of
    ``dims.selection.slice_indices``, ``dims.selection.displayed_axes``.

    Primary consumers:
    - ``SliceCoordinator``: invalidate stale 2D caches
    - Controller: reslice the affected scene
    - Dims slider widget: update slider position without re-emitting

    Parameters
    ----------
    source_id :
        UUID of the object that caused the change.  The dims slider
        widget passes its own ID via ``controller.update_slice_indices``
        so it can echo-filter the resulting event.
    scene_id :
        The scene whose ``DimsManager`` changed.
    dims_state :
        Full snapshot of the new dims state.
    displayed_axes_changed :
        ``True`` when ``displayed_axes`` changed, not just
        ``slice_indices``.  The controller uses this flag to rebuild
        visual geometry and switch canvas cameras.
    """

    source_id: UUID
    scene_id: UUID
    dims_state: DimsState
    displayed_axes_changed: bool
```

```python
class CameraChangedEvent(NamedTuple):
    """Fired when the camera moves in a scene.

    Source: ``CanvasView`` detects camera movement each frame by
    comparing a cached ``CameraState`` snapshot; it emits this event
    with ``source_id = canvas_id`` when the state changes.

    Primary consumers:
    - Controller: update camera model and schedule debounced reslice

    Parameters
    ----------
    source_id :
        ID of the canvas that moved the camera (its ``canvas_id``, not
        the canvas's internal ``_id``).
    scene_id :
        The scene whose active camera moved.
    camera_state :
        Full snapshot of the new camera state.
    """

    source_id: UUID
    scene_id: UUID
    camera_state: CameraState
```

### Visual / appearance events

```python
class AppearanceChangedEvent(NamedTuple):
    """Fired when any field on a visual's appearance model changes.

    Emitted by the controller's psygnal bridge whenever a field event
    fires on a ``BaseAppearance`` subclass.  ``visible`` field changes
    emit ``VisualVisibilityChangedEvent`` instead.

    Primary consumers:
    - ``GFX*Visual``: update material / shader parameters
    - External ``on_visual_changed`` callbacks

    Parameters
    ----------
    source_id :
        UUID of the object that caused the change.  Widget-driven
        changes carry the widget's own ID (injected via the ContextVar
        pattern in ``update_appearance_field``); direct model mutations
        fall back to the controller's ID.
    visual_id :
        The visual whose appearance changed.
    field_name :
        The name of the changed field, e.g. ``"clim"``.
    new_value :
        The new value.  Typed as ``Any``; consumers cast as needed.
    requires_reslice :
        ``True`` for fields that invalidate the current brick set
        (``lod_bias``, ``force_level``, ``frustum_cull``).
        ``False`` for pure GPU-side changes (``color_map``, ``clim``).
    """

    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any
    requires_reslice: bool


class ChannelAppearanceChangedEvent(NamedTuple):
    """Fired when a field on one channel's appearance model changes.

    Emitted by the controller's per-channel psygnal bridge (one handler
    per channel appearance, wired in ``_wire_channels``) for image
    visuals.  Replacing ``visual.channels`` rewires those handlers, and
    reslices when the channel indices change.

    Primary consumers:
    - ``GFXImageMemoryVisual`` / ``GFXMultiscaleImageVisual``: restyle the
      channel's slot when composite mode draws it

    Parameters
    ----------
    source_id :
        UUID of the object that caused the change.  Widget-driven
        changes carry the widget's own ID; direct model mutations fall
        back to the controller's ID.
    visual_id :
        The image visual that owns the channel.
    channel_index :
        Index of the channel whose appearance changed.
    field_name :
        The name of the changed field, e.g. ``"color_map"`` or ``"clim"``.
    new_value :
        The new value.
    """

    source_id: UUID
    visual_id: UUID
    channel_index: int
    field_name: str
    new_value: Any


class PickWriteChangedEvent(NamedTuple):
    """Fired when ``BaseVisual.pick_write`` changes.

    ``source_id`` is always the controller's own ID; ``pick_write`` is
    not driven through the ContextVar widget pattern.

    Primary consumers:
    - ``GFX*Visual``: toggle whether the node writes to the pick buffer

    Parameters
    ----------
    source_id :
        Always the controller's own ID.
    visual_id :
        The visual whose ``pick_write`` changed.
    pick_write :
        The new ``pick_write`` value.
    """

    source_id: UUID
    visual_id: UUID
    pick_write: bool


class VisualVisibilityChangedEvent(NamedTuple):
    """Fired when ``appearance.visible`` changes.

    Separate from ``AppearanceChangedEvent`` so consumers that only
    need to show/hide a node do not have to inspect ``field_name``.

    Parameters
    ----------
    source_id :
        UUID of the object that caused the change.  Widget-driven
        changes carry the widget's own ID; direct model mutations fall
        back to the controller's ID.
    visual_id :
        The visual whose visibility changed.
    visible :
        The new visibility state.
    """

    source_id: UUID
    visual_id: UUID
    visible: bool


class AABBChangedEvent(NamedTuple):
    """Fired when any field on a visual's AABB params model changes.

    Primary consumers:
    - ``GFX*Visual``: update bounding-box display parameters

    Parameters
    ----------
    source_id :
        UUID of the object that caused the change.  Widget-driven
        changes carry the widget's own ID (injected via
        ``update_aabb_field``); direct model mutations fall back to
        the controller's ID.
    visual_id :
        The visual whose AABB params changed.
    field_name :
        The name of the changed field, e.g. ``"enabled"``.
    new_value :
        The new value.
    """

    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any


class TransformChangedEvent(NamedTuple):
    """Fired when a visual's data-to-world transform is replaced.

    Primary consumers:
    - ``GFX*Visual``: update the scene-graph node matrix

    Parameters
    ----------
    source_id :
        Always the controller's own ID; transform changes originate
        from application code via ``set_visual_transform``.
    scene_id :
        The scene that contains the visual.
    visual_id :
        The visual whose transform changed.
    transform :
        The new ``AffineTransform``.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    transform: AffineTransform
```

### Data store events

A store announces its own changes on ``BaseDataStore.data_changed`` -- when a
data field is reassigned, or when ``store.notify_changed(kind, regions)`` is
called after an in-place write.  The controller relays each announcement to
the bus as one of these two events, after it has refreshed extent-derived
state and requested a reslice of every visual reading the store.  See
``plans/store_change_events.md``.

```python
class DataStoreMetadataChangedEvent(NamedTuple):
    """A store's data may now occupy a different region (``"extent"``).

    New geometry positions, image data of a new shape, a store that grew.
    Everything derived from the store's ``axis_extents`` is stale.
    """

    source_id: UUID
    data_store_id: UUID


class DataStoreContentsChangedEvent(NamedTuple):
    """A store's values changed within the same region (``"contents"``).

    A paint stroke, new colours, a frame streamed into the existing extent.
    ``regions`` is per-axis ``(start, stop)`` in level-0 data coordinates, or
    ``None`` for anywhere.
    """

    source_id: UUID
    data_store_id: UUID
    regions: tuple[tuple[tuple[float, float], ...], ...] | None = None
```

### Slicer lifecycle events

```python
class ResliceStartedEvent(NamedTuple):
    """Fired before submitting async reslice tasks for a scene.

    Parameters
    ----------
    source_id :
        Always the ``SliceCoordinator``'s ID.
    scene_id :
        The scene being resliced.
    visual_ids :
        The visuals in this batch.
    """

    source_id: UUID
    scene_id: UUID
    visual_ids: frozenset[UUID]


class ResliceCompletedEvent(NamedTuple):
    """Fired after all bricks in one visual's slice batch are committed.

    Fired once per visual per reslice cycle, not once per brick.

    Parameters
    ----------
    source_id :
        Always the completing visual's ID.
    scene_id :
        The scene that contains the visual.
    visual_id :
        The visual that finished loading.
    brick_count :
        Number of bricks committed in this batch.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    brick_count: int


class ResliceCancelledEvent(NamedTuple):
    """Fired when an in-flight reslice task is cancelled.

    Parameters
    ----------
    source_id :
        Always the ``SliceCoordinator``'s ID.
    scene_id :
        The scene whose task was cancelled.
    visual_id :
        The visual whose task was cancelled.
    """

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
```

### Render events

```python
class FrameRenderedEvent(NamedTuple):
    """Fired by ``CanvasView`` at the end of each rendered frame.

    Parameters
    ----------
    source_id :
        Always the ``CanvasView``'s ``canvas_id``.
    canvas_id :
        The canvas that completed a frame.
    frame_time_ms :
        Wall-clock time for this frame in milliseconds.
    """

    source_id: UUID
    canvas_id: UUID
    frame_time_ms: float
```

### Structural events

```python
class VisualAddedEvent(NamedTuple):
    """Fired after a visual is fully wired into a scene."""

    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class VisualRemovedEvent(NamedTuple):
    """Fired after a visual is fully removed from a scene."""

    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class SceneAddedEvent(NamedTuple):
    """Fired after a scene is registered with the controller."""

    source_id: UUID
    scene_id: UUID


class SceneRemovedEvent(NamedTuple):
    """Fired after a scene and all its canvases and visuals are removed."""

    source_id: UUID
    scene_id: UUID
```

### Canvas mouse events

Pointer interaction is surfaced as a family of typed events, one per
action (press / move / release) and per camera dimensionality (2D / 3D).
The 2D variants carry a `world_coordinate`; the 3D variants carry a
`ViewRay` instead, because a single screen point maps to a ray rather
than one world position under a perspective camera. Both carry a
`CanvasPickInfo` describing what the pointer landed on, translated to a
model-layer `visual_id` by the render layer (no `gfx.*` types leak out).

For these events the `source_id` is the originating canvas's ID and is
also the entity-filter key (there is no separate canvas field).

Each event additionally carries the *gesture context* — `button` (the
button that triggered this event), `buttons` (the held-button mask, the
cross-backend-consistent signal for drag-vs-hover), `modifiers` (active
keyboard modifiers), and `gesture_id`. `gesture_id` is a UUID synthesized
by the render layer on press, carried verbatim through every move, and
cleared on release, so the phases of one click-drag share an id; a
hover-move with no active press carries `None`. Phase stays encoded as the
event type (press / move / release).

#### Pick events

Mouse events report *that* something was hit (`CanvasPickInfo.hit_visual_id`);
typed pick events report *what* was hit.  There is one event per visual
family, and all six share the mouse event's context fields -- `source_id`
(the canvas), `scene_id`, `visual_id`, `action`, `camera_type`,
`world_coordinate` (2D) or `ray` (3D), `button`, `buttons`, `modifiers` and
`gesture_id` -- apart from `pick_info`:

| Event | Visuals | `pick_info` |
|---|---|---|
| `ImagePickEvent` | `ImageVisual`, `MultiscaleImageVisual` | `ImagePickInfo` |
| `LabelsPickEvent` | `LabelMemoryVisual`, `MultiscaleLabelVisual` | `LabelsPickInfo` |
| `PointsPickEvent` | `PointsVisual` | `PointsPickInfo` |
| `LinesPickEvent` | `LinesVisual` | `LinesPickInfo` |
| `MeshPickEvent` | `MeshVisual` | `MeshPickInfo` |
| `GraphPickEvent` | `GraphVisual` | `GraphNodePickInfo` or `GraphEdgePickInfo` |

Subscribe with `controller.on_pick(canvas_id, event_type, callback, *,
owner_id, weak=False)` (mirrored on `Viewer` and `OrthoViewer`) and remove
with `unsubscribe_pick`.  A pick event is emitted only for a hit, after the
mouse event for the same pointer event, and carries its `gesture_id`: a press,
its drag moves and its release share one id, and a hover move has `None`.

Timing:

- Points, lines, mesh and graph events, and in-memory image and labels
  events, are synchronous: they arrive straight after the mouse event.
- Multiscale image and labels values are read at level 0 through the
  cancellable async slicer, so their events arrive when the read completes.
  A newer pointer event on the canvas cancels an in-flight `move` read;
  `press` and `release` reads always complete, and removing the visual
  cancels all of its reads.  Match events by `gesture_id` and `action`, not
  by arrival order.

The element identity is decoded in the render layer from the pygfx pick
payload: points report `vertex_index` directly; the `LineSegmentMaterial`
lays out one explicit vertex pair per edge, so the line visual maps the
picked vertex to `edge_index = vertex_index // 2`; image and labels visuals
decode a level-0 data coordinate, which the controller completes with the
planes the visual last drew.

`ImagePickInfo.channel_values` maps a channel index to the value at the
picked voxel.  A visual without a channel axis reports `{0: value}`; single
mode reports its drawn channel; composite mode reports every drawn (visible)
channel on a 2D canvas and only the pick-buffer winner's channel on a 3D
canvas.  `LabelsPickInfo.value` is the label id at the picked voxel.

Detail extraction is **gated**: it runs only while a canvas has at least one
`on_pick` subscriber (`RenderManager.set_pick_details_enabled`, driven by the
controller's per-canvas subscriber count), and image and labels values are
read only while that event type has a subscriber.  Mouse subscribers, and
consumers that subscribe directly on the bus (e.g. the paint controller), do
not enable it.  `hit_visual_id` is always computed -- it is needed for
visual-level hits and misses and is already paid for upstream by pygfx.  The
GPU->CPU pick-buffer readback itself is performed by pygfx before our handler
runs; this gate only suppresses our own dispatch and reads.

```python
class PointsPickInfo(NamedTuple):
    point_index: int


class LinesPickInfo(NamedTuple):
    edge_index: int


class ImagePickInfo(NamedTuple):
    data_coordinate: tuple[float, ...]
    channel_values: dict[int, float]


class LabelsPickInfo(NamedTuple):
    data_coordinate: tuple[float, ...]
    value: int


class ImagePickEvent(NamedTuple):
    source_id: UUID  # the canvas
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


class CanvasPickInfo(NamedTuple):
    """Model-layer pick result attached to canvas mouse events.

    Parameters
    ----------
    hit_visual_id :
        Model-layer ID of the visual whose active scene-graph node was
        hit, or ``None`` if the pointer landed on the background.
    """

    hit_visual_id: UUID | None


class ViewRay(NamedTuple):
    """A view ray from the camera through a screen-space point.

    Used by 3D canvas mouse events in place of a single world coordinate.

    Parameters
    ----------
    origin :
        Near-plane world position where the ray starts.  Shape ``(3,)``,
        float64.
    direction :
        Unit vector in world space giving the ray direction.  Shape
        ``(3,)``, float64.
    """

    origin: np.ndarray
    direction: np.ndarray


# The 2D events (the 3D events are identical but carry `ray: ViewRay`
# in place of `world_coordinate`).
class CanvasMousePress2DEvent(NamedTuple):
    """Primary pointer button pressed on a 2D canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseMove2DEvent(NamedTuple):
    """Pointer moved on a 2D canvas (button up or down)."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None


class CanvasMouseRelease2DEvent(NamedTuple):
    """Primary pointer button released on a 2D canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo
    button: int = 0
    buttons: tuple = ()
    modifiers: tuple = ()
    gesture_id: UUID | None = None
```

`RenderManager` additionally emits an internal `_CanvasRawPointerEvent`
that carries the un-dispatched pointer data (canvas/scene IDs, action,
camera type, raw 2D position or ray, hit visual, button, modifiers,
`buttons`, `gesture_id`, and the extracted `pick_details`). It is consumed
only by `CellierController._on_raw_pointer_event`, which translates it
into the public 2D/3D events above. It is **not** part of the public event
catalogue and subscribers should not depend on it.

### Union alias

```python
CellierEventTypes = (
    DimsChangedEvent
    | CameraChangedEvent
    | AppearanceChangedEvent
    | ChannelAppearanceChangedEvent
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
    | TransformChangedEvent
    | SceneAddedEvent
    | SceneRemovedEvent
    | CanvasMousePress2DEvent
    | CanvasMouseMove2DEvent
    | CanvasMouseRelease2DEvent
    | CanvasMousePress3DEvent
    | CanvasMouseMove3DEvent
    | CanvasMouseRelease3DEvent
)
```

---

## EventBus API

### Internal subscription record: `_Subscription`

Subscriptions are stored as `_Subscription` instances rather than plain tuples.
Weak reference logic is fully encapsulated in this class — there is no standalone
`make_weak_callback` helper.

```python
class _Subscription:
    __slots__ = (
        "_strong_cb",   # callback (strong ref)
        "_weak_func",   # weakref to unbound function (weak bound methods or weak fns)
        "_weak_obj",    # weakref to bound instance (weak bound methods only)
        "entity_id",
        "handle",
        "is_weak",
        "owner_id",
    )

    def is_alive(self) -> bool:
        """Return True if the callback target is still reachable."""
        ...

    def call(self, event) -> None:
        """Invoke the callback. Assumes is_alive() is True."""
        ...

    def __repr__(self) -> str:
        # Produces e.g.:
        # _Subscription(callback='QtDimsControl._on_dims_changed'
        #   owner_id=3f2a... entity_id=9c1b... strong/alive)
        ...
```

### `SubscriberInfo` — debugging helper

`get_subscribers()` returns `SubscriberInfo` dataclass instances (not raw
`_Subscription` objects) for external inspection. See
[debug_events.md](../How_To/debug_events.md) for usage.

```python
@dataclass
class SubscriberInfo:
    callback_qualname: str   # e.g. "QtDimsControl._on_dims_changed"
    callback_instance: Any   # bound object, or None for plain functions
    owner_id: UUID | None
    entity_id: UUID | None
    is_weak: bool
    is_alive: bool
```

### `EventBus`

```python
class EventBus:
    def __init__(self) -> None:
        # event_type → list[_Subscription], in registration order
        self._subs: dict[type, list[_Subscription]] = defaultdict(list)

        # handle._id (UUID) → (event_type, _Subscription) for O(1) single removal
        self._handle_index: dict[UUID, tuple[type, _Subscription]] = {}
```

**Note on `unsubscribe_all` performance.** There is no forward owner index.
`unsubscribe_all(owner_id)` scans all subscriptions across all event types linearly.
This is O(n) in the total number of subscriptions. In practice the subscription count
is small (tens of entries) so this has no measurable cost.

```python
    def subscribe(
        self,
        event_type: type,
        callback: Callable,
        *,
        entity_id: UUID | None = None,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register callback to be called when event_type is emitted.

        Parameters
        ----------
        event_type :
            The event class to subscribe to, e.g. ``DimsChangedEvent``.
        callback :
            Called synchronously for each matching event.
        entity_id :
            When provided, the callback fires only for events whose
            canonical entity ID matches this value.  ``None`` subscribes
            to all entities.
        owner_id :
            Groups this subscription for bulk removal via
            ``unsubscribe_all(owner_id)``.  Pass the owning object's own
            ``._id``.
        weak :
            Store a weak reference to the callback.  Dead weak
            subscriptions are cleaned up lazily during emission.
            Do not pass lambdas with ``weak=True``.

        Returns
        -------
        SubscriptionHandle
            Opaque token for individual removal via ``unsubscribe()``.

        Raises
        ------
        ValueError
            If ``weak=True`` is combined with a lambda.
        """
        ...

    def unsubscribe(self, handle: SubscriptionHandle) -> None: ...

    def unsubscribe_all(self, owner_id: UUID) -> None: ...

    def emit(self, event) -> None: ...

    def get_subscribers(
        self,
        event_type: type,
        *,
        entity_id: UUID | None = None,
    ) -> list[SubscriberInfo]: ...
```

### Entity ID resolution

The bus resolves the "canonical entity ID" of an event for `entity_id` filtering using a
module-level mapping. Each event type declares which of its fields is the filterable
entity:

```python
_ENTITY_FIELD: dict[type, str] = {
    DimsChangedEvent:                  "scene_id",
    CameraChangedEvent:                "scene_id",
    AppearanceChangedEvent:            "visual_id",
    ChannelAppearanceChangedEvent:     "visual_id",
    PickWriteChangedEvent:             "visual_id",
    AABBChangedEvent:                  "visual_id",
    VisualVisibilityChangedEvent:      "visual_id",
    DataStoreMetadataChangedEvent:     "data_store_id",
    DataStoreContentsChangedEvent:     "data_store_id",
    ResliceStartedEvent:               "scene_id",
    ResliceCompletedEvent:             "visual_id",
    ResliceCancelledEvent:             "visual_id",
    FrameRenderedEvent:                "canvas_id",
    VisualAddedEvent:                  "scene_id",
    VisualRemovedEvent:                "scene_id",
    TransformChangedEvent:             "visual_id",
    SceneAddedEvent:                   "scene_id",
    SceneRemovedEvent:                 "scene_id",
    CanvasMousePress2DEvent:           "source_id",
    CanvasMouseMove2DEvent:            "source_id",
    CanvasMouseRelease2DEvent:         "source_id",
    CanvasMousePress3DEvent:           "source_id",
    CanvasMouseMove3DEvent:            "source_id",
    CanvasMouseRelease3DEvent:         "source_id",
}
```

---

## Bounce-back prevention and the ContextVar pattern

### The problem

Preventing feedback loops requires two things:

1. **Echo filtering** — a widget that drives a model change must be able to identify
   the resulting bus event as its own and ignore it, rather than redundantly
   re-applying the value it just set.

2. **Re-entrancy prevention** — when widget B is programmatically updated in response
   to a change, it must not re-broadcast that update back into the system.

Re-entrancy is handled at the widget level with Qt's `blockSignals(True/False)` around
every programmatic `setValue()` call. This prevents the widget's own `valueChanged`
signal from firing during a model-driven update.

Echo filtering requires that every bus event carry a `source_id` that identifies the
originating widget. Widgets check `if event.source_id == self._id: return` at the top
of their bus handlers.

### The challenge: psygnal does not carry caller metadata

Widgets do not emit bus events directly. Instead they mutate model fields through
controller methods, and a psygnal bridge handler fires synchronously to emit the bus
event. The bridge handler's signature only receives the new field value via
`EmissionInfo` — there is no mechanism within psygnal to pass the calling widget's UUID
down to the handler.

### The solution: ContextVar side-channel

A `contextvars.ContextVar` is set by the controller method before the model mutation
and reset after. The psygnal handler, running synchronously in the same call frame,
reads the var to retrieve the caller-supplied `source_id`.

```python
# Module-level — not on the controller class
_source_id_override: ContextVar[UUID | None] = ContextVar(
    "_source_id_override", default=None
)
_aabb_source_id_override: ContextVar[UUID | None] = ContextVar(
    "_aabb_source_id_override", default=None
)
```

Two separate vars exist — one for dims and appearance mutations, one for AABB mutations
— so that concurrent mutations to different sub-objects do not interfere with each
other's source attribution.

The three controller methods that use this pattern:

```python
def update_slice_indices(
    self, scene_id, slice_indices, *, source_id=None
) -> None:
    token = _source_id_override.set(source_id)
    try:
        self._model.scenes[scene_id].dims.selection.slice_indices = slice_indices
        # ↑ psygnal fires _on_dims_psygnal synchronously here
    finally:
        _source_id_override.reset(token)   # always restored, even on exception


def update_appearance_field(
    self, visual_id, field, value, *, source_id=None
) -> None:
    visual = self.get_visual_model(visual_id)
    token = _source_id_override.set(source_id)
    try:
        setattr(visual.appearance, field, value)
        # ↑ psygnal fires _on_appearance_psygnal synchronously here
    finally:
        _source_id_override.reset(token)


def update_aabb_field(
    self, visual_id, field, value, *, source_id=None
) -> None:
    visual = self.get_visual_model(visual_id)
    token = _aabb_source_id_override.set(source_id)
    try:
        setattr(visual.aabb, field, value)
        # ↑ psygnal fires _on_aabb_psygnal synchronously here
    finally:
        _aabb_source_id_override.reset(token)
```

Each bridge handler reads the var before emitting:

```python
# Inside _on_appearance_psygnal (returned by _make_appearance_handler)
resolved_source_id = _source_id_override.get() or self._id
self._outgoing_events.emit(
    AppearanceChangedEvent(source_id=resolved_source_id, ...)
)
```

If a model field is mutated directly — bypassing the controller methods — the ContextVar
is `None` and `source_id` falls back to the controller's own ID. No widget will
echo-filter such an event, which is the correct behaviour: the change looks external
to all subscribers.

### Why ContextVar rather than a plain instance variable

`ContextVar` is semantically correct for async code: each `asyncio.Task` has its own
copy of the context, so concurrent async mutations cannot stomp each other's
`source_id`. A plain `self._pending_source_id` instance variable would also work in
practice because psygnal fires synchronously (the bridge handler always runs and
completes before `set()` returns, so there is no race in a single-threaded async loop),
but `ContextVar` is the more principled choice and matches the intended use case.

### Full echo-filtering example

```python
class QtDimsControl:
    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(self, scene_id, ...):
        self._id = uuid4()
        self._scene_id = scene_id
        # No controller reference — wired externally via connect_widget.

    def subscription_specs(self) -> list[SubscriptionSpec]:
        return [SubscriptionSpec(
            event_type=DimsChangedEvent,
            handler=self._on_dims_changed,
            entity_id=self._scene_id,
        )]

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own slider move; ignore
        for axis, sld in self._sliders.items():
            value = event.dims_state.selection.slice_indices.get(axis)
            if value is not None:
                self._set_value(axis, value)   # uses blockSignals internally

    def _submit_slider_values(self) -> None:
        updates = {axis: sld.value() for axis, sld in self._sliders.items()
                   if axis not in self._displayed_axes}
        # Emit on changed; connect_widget has wired this to _incoming_events.emit.
        self.changed.emit(DimsUpdateEvent(
            source_id=self._id,
            scene_id=self._scene_id,
            slice_indices=updates,
            displayed_axes=None,
        ))
```

`DimsUpdateEvent` carries `source_id`, `scene_id`, `slice_indices` and
`displayed_axes`. Every field except `source_id` and `scene_id` is optional
(`None` = leave unchanged), so a widget sets only the fields it owns, and
`slice_indices` merges: send only the axes that moved.

Both guards are necessary and serve different roles:

- **`source_id` check** — prevents the originating widget from processing its own
  echo (redundant work, since the model already reflects the new value).
- **`blockSignals`** — prevents a programmatically updated widget from re-broadcasting
  the change into the system. Without it, widget B being updated in response to
  widget A's change would fire widget B's `valueChanged`, triggering another
  `update_slice_indices`, producing another bus event, updating widget A — an
  infinite loop. `blockSignals` is the guard that breaks that cycle.

---

## Async handler convention

The bus dispatches synchronously. Handlers that need to await async work must call
`asyncio.create_task()` and return immediately. Awaiting directly inside a handler will
deadlock the Qt event loop.

This convention is **mandatory** for any handler that touches the slicer pipeline. It
is documented in every such handler's docstring.

```python
class SliceCoordinator:

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        """Synchronous entry point — invalidates stale caches."""
        # Synchronous work only. Any async submission is the controller's
        # responsibility via reslice_scene().
        ...
```

---

## How the controller bridges psygnal to the bus

The controller is the only place where psygnal models and the EventBus coexist. When a
psygnal field event fires on a model — e.g. `visual.appearance.clim = (0.0, 1.0)`
triggers `appearance.events.clim.emit(...)` — the controller converts it into a typed
bus event and calls `bus.emit()`.

This bridging is done by private factory methods on the controller. Each factory
captures an entity ID by value and returns a closure that psygnal calls. The factory
pattern is required because Python loop closures bind by reference — without it, all
handlers wired in a loop would share the last iteration's variable.

### Reading the new value from `EmissionInfo`

The bridge handlers read the new field value from `info.args[0]`, which psygnal
populates with the emitted value. They do not call `Signal.current_emitter()` or
re-read from the model object.

```python
def _make_appearance_handler(self, visual_id: UUID) -> Callable:
    def _on_appearance_psygnal(info: EmissionInfo) -> None:
        field_name: str = info.signal.name
        new_value = info.args[0]  # new value from psygnal args
        resolved_source_id = _source_id_override.get() or self._id
        if field_name == "visible":
            self._outgoing_events.emit(VisualVisibilityChangedEvent(
                source_id=resolved_source_id,
                visual_id=visual_id,
                visible=new_value,
            ))
        else:
            self._outgoing_events.emit(AppearanceChangedEvent(
                source_id=resolved_source_id,
                visual_id=visual_id,
                field_name=field_name,
                new_value=new_value,
                requires_reslice=(field_name in _RESLICE_FIELDS),
            ))
            if field_name in _RESLICE_FIELDS:
                self.reslice_visual(visual_id)

    return _on_appearance_psygnal
```

### Handler storage for teardown

The controller stores every psygnal bridge handler it connects so that they can be
explicitly disconnected during visual teardown.  psygnal's `disconnect()` requires
the exact handler object — closures are not equality-comparable by value, so the
reference must be retained.

```python
# On the controller instance
self._visual_psygnal_handlers: dict[UUID, list[tuple]] = {}
# Each value is a list of (signal, handler) pairs — one per _wire_* call.
```

The `(signal, handler)` pair form is used rather than storing the handler alone so
that teardown can call `signal.disconnect(handler)` without branching on visual type.

Wiring is **per-capability, not per-visual-type**. At `add_visual` time the
controller probes the model and wires whichever signals it exposes:

```python
# In add_visual, after the model is registered
if hasattr(visual_model, "appearance"):
    self._wire_appearance(visual_model)
if hasattr(visual_model, "channels"):
    self._wire_channels(visual_model)
self._wire_aabb(visual_model)        # always
self._wire_transform(visual_model, scene_id)  # always
self._wire_pick_write(visual_model)  # always
```

| Wiring step | When | Emits |
|---|---|---|
| `_wire_appearance` | model has an `appearance` | `AppearanceChangedEvent` / `VisualVisibilityChangedEvent` |
| `_wire_image` | image visuals | `SingleAppearanceChangedEvent`, `ImageCompositeChangedEvent`, and `ChannelAppearanceChangedEvent` (one handler per channel, via `_wire_channels`) |
| `_wire_aabb` | always | `AABBChangedEvent` |
| `_wire_transform` | always | `TransformChangedEvent` |
| `_wire_pick_write` | always | `PickWriteChangedEvent` |

Each `_wire_*` method creates the handler, connects it, and appends the
`(signal, handler)` pair to `_visual_psygnal_handlers[visual.id]`:

```python
def _wire_appearance(self, visual) -> None:
    handler = self._make_appearance_handler(visual.id)
    visual.appearance.events.connect(handler)
    self._visual_psygnal_handlers.setdefault(visual.id, []).append(
        (visual.appearance.events, handler)
    )
```

### DimsManager wiring

The dims bridge reads the full state from the model rather than from `EmissionInfo`,
because `DimsState` is a composite snapshot of the entire `DimsManager` — not a single
field value.

```python
def _wire_dims_model(self, scene: Scene) -> None:
    self._dims_cache[scene.id] = scene.dims.selection.displayed_axes
    scene.dims.events.connect(self._make_dims_handler(scene.id))


def _make_dims_handler(self, scene_id: UUID) -> Callable:
    def _on_dims_psygnal(info: EmissionInfo) -> None:
        new_state = self._model.scenes[scene_id].dims.to_state()
        prev_axes = self._dims_cache[scene_id]
        displayed_axes_changed = prev_axes != new_state.selection.displayed_axes
        self._dims_cache[scene_id] = new_state.selection.displayed_axes
        if displayed_axes_changed:
            self._rebuild_visuals_geometry(scene_id, new_state.selection.displayed_axes)
            self._switch_canvas_cameras(scene_id, new_state.selection.displayed_axes)
        resolved_source_id = _source_id_override.get() or self._id
        self._outgoing_events.emit(DimsChangedEvent(
            source_id=resolved_source_id,
            scene_id=scene_id,
            dims_state=new_state,
            displayed_axes_changed=displayed_axes_changed,
        ))

    return _on_dims_psygnal
```

Note that `_rebuild_visuals_geometry` and `_switch_canvas_cameras` are called
synchronously inside the bridge handler, before the bus event is emitted. Camera
switching and visual geometry rebuilding happen as a direct consequence of the dims
change, not through bus subscriptions.

### AABB wiring

AABB is wired for every visual (`visual: BaseVisual`), not just image visuals.

```python
def _wire_aabb(self, visual: BaseVisual) -> None:
    handler = self._make_aabb_handler(visual.id)
    visual.aabb.events.connect(handler)
    self._visual_psygnal_handlers.setdefault(visual.id, []).append(
        (visual.aabb.events, handler)
    )


def _make_aabb_handler(self, visual_id: UUID) -> Callable:
    def _on_aabb_psygnal(info: EmissionInfo) -> None:
        field_name: str = info.signal.name
        new_value = info.args[0]
        resolved_source_id = _aabb_source_id_override.get() or self._id
        self._outgoing_events.emit(AABBChangedEvent(
            source_id=resolved_source_id,
            visual_id=visual_id,
            field_name=field_name,
            new_value=new_value,
        ))

    return _on_aabb_psygnal
```

### Channel wiring

Image visuals carry `single` (single mode's appearance) and a `channels`
mapping of per-channel appearances (composite mode). `_wire_image` bridges
all three: a field change on `single`, or replacing it, emits
`SingleAppearanceChangedEvent`; a change of `composite` emits
`ImageCompositeChangedEvent`, rechecks the scene's derived slider axes and
reslices the visual (so a direct `visual.composite = True` behaves like
`set_image_composite`, minus that method's composited-axis check); and
`_wire_channels` connects one handler per channel, each emitting a
`ChannelAppearanceChangedEvent` tagged with that channel's index. Replacing
`visual.channels` rewires those handlers, and reslices when the channel
indices change.

```python
def _wire_channels(self, visual) -> None:
    for channel_index, appearance in visual.channels.items():
        handler = self._make_channel_appearance_handler(visual.id, channel_index)
        appearance.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (appearance.events, handler)
        )


def _make_channel_appearance_handler(self, visual_id, channel_index) -> Callable:
    def _on_channel_appearance_psygnal(info: EmissionInfo) -> None:
        resolved_source_id = _source_id_override.get() or self._id
        self._outgoing_events.emit(ChannelAppearanceChangedEvent(
            source_id=resolved_source_id,
            visual_id=visual_id,
            channel_index=channel_index,
            field_name=info.signal.name,
            new_value=info.args[0],
        ))

    return _on_channel_appearance_psygnal
```

### pick_write wiring

`pick_write` is wired for every visual. Like transforms, it does not use the
ContextVar pattern — `source_id` is always the controller's own ID.

```python
def _wire_pick_write(self, visual: BaseVisual) -> None:
    handler = self._make_pick_write_handler(visual.id)
    visual.events.pick_write.connect(handler)
    self._visual_psygnal_handlers.setdefault(visual.id, []).append(
        (visual.events.pick_write, handler)
    )


def _make_pick_write_handler(self, visual_id: UUID) -> Callable:
    def _on_pick_write(new_value: bool) -> None:
        self._outgoing_events.emit(PickWriteChangedEvent(
            source_id=self._id,
            visual_id=visual_id,
            pick_write=new_value,
        ))

    return _on_pick_write
```

### Transform wiring

Transform changes do not use the ContextVar pattern because transforms are not driven
by GUI sliders — they come from application code via `set_visual_transform`. The
`source_id` is always the controller's own ID.

```python
def _make_transform_handler(self, visual_id: UUID, scene_id: UUID) -> Callable:
    def _on_transform(new_transform: AffineTransform) -> None:
        self._outgoing_events.emit(TransformChangedEvent(
            source_id=self._id,
            scene_id=scene_id,
            visual_id=visual_id,
            transform=new_transform,
        ))
        if not self._suppress_reslice:
            self.reslice_scene(scene_id)

    return _on_transform
```

---

## Registration and deregistration lifecycle

Every render-layer object passes its own ID as `owner_id` when subscribing. When the
controller removes an entity, it calls `bus.unsubscribe_all(entity_id)` to clean up
all subscriptions registered by that entity atomically.

```python
# Controller: wiring a new GFX visual
bus.subscribe(
    AppearanceChangedEvent,
    gfx_visual.on_appearance_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)
bus.subscribe(
    AABBChangedEvent,
    gfx_visual.on_aabb_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)
bus.subscribe(
    VisualVisibilityChangedEvent,
    gfx_visual.on_visibility_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)
bus.subscribe(
    TransformChangedEvent,
    gfx_visual.on_transform_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)

# Controller: removing the visual
bus.unsubscribe_all(owner_id=visual_id)  # removes all four in one call
```

### Visual teardown sequence

A visual has two independent sets of subscriptions that must both be cleaned up:

1. **psygnal side** — the bridge closures stored in `_visual_psygnal_handlers`.
   These live on psygnal `EventedModel` signals (`appearance.events`,
   `aabb.events`, `events.transform`) and are disconnected by calling
   `signal.disconnect(handler)` for each stored `(signal, handler)` pair.

2. **bus side** — the GFX-layer subscriptions registered with `owner_id=visual_id`
   (appearance, AABB, visibility, transform handlers on `GFX*Visual`).
   Removed atomically with `bus.unsubscribe_all(visual_id)`.

The teardown order inside `remove_visual` is:

```
1. scene.visuals.remove(visual_model)          # model layer
2. signal.disconnect(handler) for each pair    # psygnal bridge: stop emitting events
3. bus.unsubscribe_all(visual_id)              # bus: stop receiving events in GFX layer
4. _visual_to_scene.pop(visual_id)             # controller map
5. render_manager.remove_visual(visual_id)     # scene graph + GPU ref release
6. bus.emit(VisualRemovedEvent(...))            # notify external observers
```

Psygnal is disconnected **before** the bus so that no new bus events can be emitted
from stale bridge closures during or after step 5.  The bus is unsubscribed before
render-layer removal so that the GFX handlers cannot fire on a node that is already
removed from the scene graph.

`remove_scene` cascades: it calls `remove_visual` for every child visual before
tearing down the scene-level maps, then delegates to
`render_manager.remove_scene(scene_id)` which drops references to the `gfx.Scene`
and all `CanvasView` objects (GC then reclaims GPU resources — pygfx has no
explicit destroy API).

External callbacks registered by the application developer receive the full event
object, not a stripped-down state payload:

```python
# Application code
handle = controller.on_dims_changed(scene_id, my_callback, owner_id=my_id)

# my_callback receives the full DimsChangedEvent:
def my_callback(event: DimsChangedEvent) -> None:
    print(event.source_id, event.dims_state.selection.displayed_axes)
```

---

## Subscription matrix

Canonical specification for what the controller wires at construction time or at
object registration time.

| Subscriber | Event type | `entity_id` filter | Action |
|---|---|---|---|
| `SliceCoordinator` | `DimsChangedEvent` | *(none — all scenes)* | Invalidate stale 2D caches |
| Controller | `DimsChangedEvent` | *(none — all scenes)* | Call `reslice_scene` for the changed scene |
| Controller | `CameraChangedEvent` | *(none — all canvases)* | Update camera model; schedule debounced reslice |
| `GFX*Visual` | `AppearanceChangedEvent` | `visual_id` | Apply material / shader parameter |
| `GFX*Visual` | `AABBChangedEvent` | `visual_id` | Update bounding-box display |
| `GFX*Visual` | `VisualVisibilityChangedEvent` | `visual_id` | Show / hide scene-graph node |
| `GFX*Visual` | `TransformChangedEvent` | `visual_id` | Update scene-graph node matrix |
| External callback | `DimsChangedEvent` | `scene_id` | Fire `on_dims_changed` user callback |
| External callback | `CameraChangedEvent` | `scene_id` | Fire `on_camera_changed` user callback |
| External callback | `AppearanceChangedEvent` | `visual_id` | Fire `on_visual_changed` user callback |
| External callback | `AABBChangedEvent` | `visual_id` | Fire `on_aabb_changed` user callback |
| External callback | `ResliceStartedEvent` | `scene_id` | Fire `on_reslice_started` user callback |
| External callback | `ResliceCompletedEvent` | `visual_id` | Fire `on_reslice_completed` user callback |
| External callback | `CanvasMouse{Press,Move,Release}{2D,3D}Event` | `canvas_id` | Fire `on_mouse_*` user callback |
| External callback | `{Image,Labels,Points,Lines,Mesh,Graph}PickEvent` | `canvas_id` | Fire `on_pick` user callback; enables pick-detail extraction |
| Paint controller | `CanvasMouse{Press,Move,Release}2DEvent` | `canvas_id` | Accumulate brush stroke (direct bus subscription; does not enable pick details) |

`SliceCoordinator` and the controller subscribe without `entity_id` filters because
they manage all scenes centrally. Camera switching and visual geometry rebuilding on
dims changes happen synchronously inside the controller's psygnal bridge handler
(`_make_dims_handler`), not through bus subscriptions.

`CanvasView` is not a bus subscriber. It emits `CameraChangedEvent` (detected by
comparing a cached `CameraState` snapshot each frame) but does not subscribe to any
event type. The controller receives `CameraChangedEvent` and handles model sync and
reslice scheduling.

---

## End-to-end example: appearance change propagation

This example traces a single colormap change through the full system, starting from
a widget that follows the decoupled widget contract.

```
QtClimRangeSlider
      ↓  changed.emit(AppearanceUpdateEvent(source_id=slider._id, field="clim", ...))
controller._incoming_events   ← connect_widget wired slider.changed → this bus
      ↓
_on_appearance_update()       ← incoming bus subscriber; calls update_appearance_field
      ↓
update_appearance_field()     ← sets ContextVar; mutates visual.appearance.clim
      ↓
psygnal EventedModel          ← clim signal fires synchronously
      ↓
Controller bridge             ← reads ContextVar; emits AppearanceChangedEvent
      ↓
EventBus.emit()               ← dispatches to subscribers
      ↙ requires_reslice=False          requires_reslice=True ↘
GFX*Visual                   Controller
on_appearance_changed()       reslice_scene() → RenderManager
      ↓                                ↓
pygfx renderer               AsyncSlicer
deferred GPU upload           concurrent brick reads
```

### The trigger

The slider emits its `changed` signal with an `AppearanceUpdateEvent` payload.
`connect_widget` has already wired `slider.changed → controller._incoming_events.emit`,
so the event enters the controller's incoming pipeline without any direct controller
call in widget code:

```python
# Inside QtClimRangeSlider._on_slider_changed — no controller import required:
self.changed.emit(
    AppearanceUpdateEvent(
        source_id=self._id,
        visual_id=self._visual_id,
        field="clim",
        value=value,
    )
)
```

### Stage 1 — Incoming bus dispatches to `_on_appearance_update`; model is mutated

`_on_appearance_update` receives the `AppearanceUpdateEvent` and forwards it to
`update_appearance_field`, preserving `source_id`:

```python
def _on_appearance_update(self, event: AppearanceUpdateEvent) -> None:
    self.update_appearance_field(
        event.visual_id, event.field, event.value, source_id=event.source_id
    )
```

`update_appearance_field` sets `_source_id_override` to `slider._id`, then sets
`visual.appearance.clim = (0.0, 1.0)`. psygnal fires `_on_appearance_psygnal`
synchronously before `setattr` returns.

### Stage 2 — Bridge handler reads ContextVar and emits bus event

```python
def _on_appearance_psygnal(info: EmissionInfo) -> None:
    field_name = info.signal.name  # "clim"
    new_value = info.args[0]  # (0.0, 1.0)
    resolved_source_id = _source_id_override.get() or self._id  # widget._id
    self._outgoing_events.emit(AppearanceChangedEvent(
        source_id=resolved_source_id,
        visual_id=visual_id,
        field_name="clim",
        new_value=(0.0, 1.0),
        requires_reslice=False,
    ))
```

`_source_id_override` is reset by `update_appearance_field`'s `finally` block after
`setattr` returns.

### Stage 3 — EventBus dispatches to subscribers

`emit()` dispatches `AppearanceChangedEvent` to all matching subscriptions. Two fire:

1. `GFXMultiscaleImageVisual.on_appearance_changed` — registered with
   `entity_id=visual_id`
2. Any external `on_visual_changed` callback registered by the application

The widget that triggered the change also has a subscription to
`AppearanceChangedEvent`. It checks `if event.source_id == self._id: return` and
skips the event. `blockSignals` is not needed here — the widget is only reading
the event, not driving a Qt signal.

### Stage 4 — `GFXMultiscaleImageVisual.on_appearance_changed`

The render-layer handler updates the pygfx material directly. No async work is
initiated — that is the controller's responsibility for `requires_reslice=True` fields.

### Stage 5 — pygfx deferred GPU upload

On the next call to `renderer.render()` in `CanvasView._draw_frame`, pygfx detects the
dirty material and uploads the updated parameters to the GPU.

### The reslice path: `lod_bias`, `force_level`, `frustum_cull`

When a field in `_RESLICE_FIELDS` changes, `AppearanceChangedEvent` carries
`requires_reslice=True`. The `GFX*Visual` handler does nothing for these fields. The
reslice is triggered **directly inside the psygnal bridge** `_on_appearance_psygnal`,
which calls `self.reslice_visual(visual_id)` after emitting the event — it is not
driven by a separate controller bus subscription. The updated field value is read from
the live model at planning time, so the event payload itself does not need to carry
planning parameters.

---

## What does not flow through the bus

- **Data payloads.** Brick data from `AsyncSlicer` is delivered directly to
  `GFX*Visual.on_data_ready()` via the `callback` argument to `slicer.submit()`.
  Only the lifecycle signal (`ResliceCompletedEvent`) goes through the bus.
- **Structural add/remove operations.** `add_visual()`, `remove_visual()`, etc. are
  synchronous and handled atomically by the controller. `VisualAddedEvent` and
  `VisualRemovedEvent` are emitted *after* the operation completes, for external
  observers such as a layer list widget.
- **Frame render loop.** The pygfx render loop runs via rendercanvas. `CanvasView`
  emits `FrameRenderedEvent` from its draw callback, but the loop itself is not
  bus-driven.

---

## Module exports

```python
# src/cellier/v2/events/__init__.py

from cellier._state import CameraState, DimsState
from cellier.events._bus import EventBus, SubscriberInfo, SubscriptionHandle
from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    CameraChangedEvent,
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    CanvasPickInfo,
    CellierEventTypes,
    ChannelAppearanceChangedEvent,
    DataStoreContentsChangedEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    FrameRenderedEvent,
    GraphPickEvent,
    ImagePickEvent,
    ImagePickInfo,
    LabelsPickEvent,
    LabelsPickInfo,
    LinesPickEvent,
    LinesPickInfo,
    MeshPickEvent,
    MeshPickInfo,
    PickWriteChangedEvent,
    PointsPickEvent,
    PointsPickInfo,
    ResliceCancelledEvent,
    ResliceCompletedEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    TransformChangedEvent,
    ViewRay,
    VisualAddedEvent,
    VisualPickDetails,
    VisualRemovedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.events._update_events import (
    AABBUpdateEvent,
    AppearanceUpdateEvent,
    CellierUpdateEventTypes,
    DimsUpdateEvent,
    SubscriptionSpec,
)
```

---

## Pending

- **Higher-level interaction events.** The low-level canvas pointer events
  (press / move / release, 2D and 3D — see §Canvas mouse events) are implemented,
  with synthesized `gesture_id` and typed pick events for element identity. A library
  gesture layer (compound `on_drag`, double-click via `event.clicks`) is
  deliberately deferred; apps assemble gestures from the phase events plus
  `gesture_id`, as the paint controller does. Higher-level semantic events for
  selection and annotation — e.g. a `SelectionChangedEvent` — are still to be
  designed once that interaction model is specified. They will follow the same
  NamedTuple / `source_id` pattern.
- **Upstream pick-readback suppression.** The true Decision-4 perf win — stopping
  pygfx from performing the GPU→CPU readback when no consumer needs it — is pending
  a pygfx pick-configuration investigation. The current gate only skips our own
  cheap dispatch.
