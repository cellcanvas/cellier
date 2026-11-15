# Events

Cellier tells you about changes through callbacks. Each `on_*` method on
[`CellierController`](controller.md) (and the few that `Viewer` and
`OrthoViewer` mirror) registers a callback for one event type, scoped to one
scene, visual or canvas, and returns a
[`SubscriptionHandle`](#cellier.events.SubscriptionHandle).

```python
from uuid import uuid4

from cellier.events import DimsChangedEvent

my_id = uuid4()

def on_dims(event: DimsChangedEvent) -> None:
    if event.source_id == my_id:
        return  # an echo of my own edit
    print(event.slice_indices)

handle = controller.on_dims_changed(scene.id, on_dims, owner_id=my_id)
...
handle.unsubscribe()                 # remove this one subscription
controller.unsubscribe_owner(my_id)  # or everything registered under my_id
```

Events are immutable named tuples, delivered synchronously on the thread that
made the change. Every event has a `source_id`. For a change made through a
widget it is the widget's id, and the event comes back to the widget that made
it, so the widget can recognise its own edit and ignore it. One field of each
event (named on each class as "the routing key") is the id that the `on_*`
method filters on.

| Method | Event |
|--------|-------|
| `on_pick` | one of [`PICK_EVENT_TYPES`](#cellier.events.PICK_EVENT_TYPES) |
| `on_mouse_press_2d`, `on_mouse_move_2d`, `on_mouse_release_2d` | `CanvasMousePress2DEvent`, `CanvasMouseMove2DEvent`, `CanvasMouseRelease2DEvent` |
| `on_mouse_press_3d`, `on_mouse_move_3d`, `on_mouse_release_3d` | `CanvasMousePress3DEvent`, `CanvasMouseMove3DEvent`, `CanvasMouseRelease3DEvent` |
| `on_dims_changed`, `on_dims_interaction` | `DimsChangedEvent`, `DimsInteractionEvent` |
| `on_camera_changed`, `on_camera_interaction` | `CameraChangedEvent`, `CameraInteractionEvent` |
| `on_appearance_changed`, `on_visibility_changed`, `on_aabb_changed` | `AppearanceChangedEvent`, `VisualVisibilityChangedEvent`, `AABBChangedEvent` |
| `on_clipping_planes_changed`, `on_render_planes_changed`, `on_plane_interaction`, `on_plane_gizmo_changed` | `ClippingPlanesChangedEvent`, `RenderPlanesChangedEvent`, `PlaneInteractionEvent`, `PlaneGizmoChangedEvent` |
| `on_section_changed`, `on_lod_config_changed` | `MeshSectionChangedEvent`, `LodConfigChangedEvent` |
| `on_draw_status_changed` | `DrawStatusChangedEvent` |
| `on_scene_added`, `on_scene_removed`, `on_visual_added`, `on_visual_removed` | `SceneAddedEvent`, `SceneRemovedEvent`, `VisualAddedEvent`, `VisualRemovedEvent` |
| `on_reslice_started`, `on_reslice_completed` | `ResliceStartedEvent`, `ResliceCompletedEvent` |
| `on_reslice_progress`, `on_backstop_complete` | `ResliceProgressEvent`, `BackstopCompleteEvent` |
| `on_canvas_connected`, `on_canvas_first_frame`, `on_scene_ready` | one-shot: the callback takes no argument |
| `subscribe(event_type, callback, *, owner_id, entity_id=None)` | any event the controller emits, including the [other changes](#other-changes) below, which have no `on_*` method; not pick events |

## Subscriptions

::: cellier.events.SubscriptionHandle

## Picking

Mouse events say *that* something was hit; pick events say *what* was hit.
See [`CellierController.on_pick`](controller.md#cellier.CellierController.on_pick).

::: cellier.events.PICK_EVENT_TYPES
::: cellier.events.PickEvent
::: cellier.events.ImagePickEvent
::: cellier.events.ImagePickInfo
::: cellier.events.MaskedImagePickEvent
::: cellier.events.MaskedImagePickInfo
::: cellier.events.LabelsPickEvent
::: cellier.events.LabelsPickInfo
::: cellier.events.PointsPickEvent
::: cellier.events.PointsPickInfo
::: cellier.events.LinesPickEvent
::: cellier.events.LinesPickInfo
::: cellier.events.MeshPickEvent
::: cellier.events.MeshPickInfo
::: cellier.events.GraphPickEvent
::: cellier.events.GraphNodePickInfo
::: cellier.events.GraphEdgePickInfo

## Mouse

::: cellier.events.CanvasMousePress2DEvent
::: cellier.events.CanvasMouseMove2DEvent
::: cellier.events.CanvasMouseRelease2DEvent
::: cellier.events.CanvasMousePress3DEvent
::: cellier.events.CanvasMouseMove3DEvent
::: cellier.events.CanvasMouseRelease3DEvent
::: cellier.events.CanvasPickInfo
::: cellier.events.ViewRay

## Dims and camera

::: cellier.events.DimsChangedEvent
::: cellier.events.DimsInteractionEvent
::: cellier.events.DimsState
::: cellier.events.AxisAlignedSelectionState
::: cellier.events.CameraChangedEvent
::: cellier.events.CameraInteractionEvent
::: cellier.events.CameraState
::: cellier.events.OrbitControllerState
::: cellier.events.PanZoomControllerState

## Visuals

::: cellier.events.AppearanceChangedEvent
::: cellier.events.VisualVisibilityChangedEvent
::: cellier.events.AABBChangedEvent
::: cellier.events.ClippingPlanesChangedEvent
::: cellier.events.RenderPlanesChangedEvent
::: cellier.events.PlaneInteractionEvent
::: cellier.events.PlaneGizmoChangedEvent
::: cellier.events.MeshSectionChangedEvent
::: cellier.events.LodConfigChangedEvent
::: cellier.events.DrawStatusChangedEvent
::: cellier.events.DrawStatus

## Scenes and visuals added or removed

::: cellier.events.SceneAddedEvent
::: cellier.events.SceneRemovedEvent
::: cellier.events.VisualAddedEvent
::: cellier.events.VisualRemovedEvent

## Loading

::: cellier.events.ResliceStartedEvent
::: cellier.events.ResliceCompletedEvent
::: cellier.events.ResliceProgressEvent
::: cellier.events.BackstopCompleteEvent
::: cellier.events.LoadingProgress

## Canvases

::: cellier.events.CanvasConnectedEvent
::: cellier.events.FrameRenderedEvent

## Other changes

These events have no `on_*` method. Subscribe with
[`CellierController.subscribe`](controller.md#cellier.CellierController.subscribe),
passing the event's routing key as `entity_id` to receive one entity's events
only:

```python
from cellier.events import TransformChangedEvent

handle = controller.subscribe(
    TransformChangedEvent, on_transform, owner_id=my_id, entity_id=visual.id
)
```

`RenderConfigChangedEvent` has no routing key: every subscriber receives every
section's changes and filters on `section`.

### Visuals

::: cellier.events.SingleAppearanceChangedEvent
::: cellier.events.ChannelAppearanceChangedEvent
::: cellier.events.ImageCompositeChangedEvent
::: cellier.events.PickWriteChangedEvent
::: cellier.events.LoadingConfigChangedEvent
::: cellier.events.VisualRenderChangedEvent
::: cellier.events.TrailChangedEvent
::: cellier.events.TransformChangedEvent
::: cellier.events.TransformInteractionEvent

### Scenes, canvases and overlays

::: cellier.events.SliderAxesChangedEvent
::: cellier.events.BackgroundChangedEvent
::: cellier.events.CanvasAddedEvent
::: cellier.events.CanvasSizeChangedEvent
::: cellier.events.OverlayChangedEvent
::: cellier.events.PlaneGizmoMovedEvent
::: cellier.events.RenderConfigChangedEvent

### Data stores

::: cellier.events.DataStoreContentsChangedEvent
::: cellier.events.DataStoreMetadataChangedEvent

## Debugging

The controller's event bus can list who is subscribed to what; see
[Debug events](../How_To/debug_events.md).

::: cellier.events.EventBus
::: cellier.events.SubscriberInfo
