# Debug events

Cellier uses a synchronous event bus to coordinate changes between the model
layer, GUI widgets, and the render layer. When something isn't updating as
expected (e.g., a widget not responding or a render not triggering), the event system is often the place to look,

This document describes the three debugging tools available and walks through
the most common failure scenarios.

## Concepts

### Events and entities

Every change in the system is announced as a typed event on the `EventBus`.
Each event type is associated with one *entity* — the model object that
changed. The entity field is the routing key the bus uses to deliver events
only to subscribers that care about a specific object:

| Event type | Entity | Entity field |
|---|---|---|
| `DimsChangedEvent`, `CameraChangedEvent`, `ResliceStartedEvent`, `VisualAddedEvent`, `VisualRemovedEvent`, `SceneAddedEvent`, `SceneRemovedEvent` | Scene | `scene_id` |
| `AppearanceChangedEvent`, `ChannelAppearanceChangedEvent`, `PickWriteChangedEvent`, `AABBChangedEvent`, `VisualVisibilityChangedEvent`, `TransformChangedEvent`, `ResliceCompletedEvent` | Visual | `visual_id` |
| `FrameRenderedEvent` | Canvas | `canvas_id` |
| `DataStoreMetadataChangedEvent`, `DataStoreContentsChangedEvent` | Data store | `data_store_id` |
| `CanvasMousePress2DEvent`, `CanvasMouseMove2DEvent`, `CanvasMouseRelease2DEvent`, `CanvasMousePress3DEvent`, `CanvasMouseMove3DEvent`, `CanvasMouseRelease3DEvent` | Canvas (emitter) | `source_id` |

A subscriber registered with `entity_id=some_visual_id` only receives events
for that visual. A subscriber registered with `entity_id=None` receives every
event of that type regardless of which entity changed.

### source_id and echo filtering

Every event carries a `source_id` — the UUID of the object that triggered the
change. GUI widgets use this to prevent a loop where its own update event triggers an update. That is, the
widget checks `if event.source_id == self._id: return` to ignore changes it
caused itself.

`source_id` is injected via a `ContextVar` side-channel in the three controller
mutation methods:

```python
controller.update_slice_indices(scene_id, {0: 5}, source_id=widget._id)
controller.update_appearance_field(visual_id, "clim", (0.0, 1.0), source_id=widget._id)
controller.update_aabb_field(visual_id, "enabled", True, source_id=widget._id)
```

If a model field is mutated directly (bypassing these methods), `source_id`
falls back to the controller's own `._id`.

---

## Tool 1 — `CellierController.get_subscribers()`

The primary inspection tool. Returns a list of `SubscriberInfo` objects
describing every callback registered for a given event type.

```python
from cellier.events import DimsChangedEvent, AppearanceChangedEvent

subscribers = controller.get_subscribers(DimsChangedEvent)
for info in subscribers:
    print(info.callback_qualname, info.owner_id, info.entity_id, info.is_alive)
```

Example output:

```
SliceCoordinator._on_dims_changed     <UUID-coordinator>  None    True
CellierController._on_dims_changed_bus  <UUID-controller>   None    True
QtDimsControl._on_dims_changed        <UUID-slider>       <UUID-scene>  True
```

### Filtering by entity

Pass `entity_id` to scope results to one scene, visual, or canvas. The results
include both subscriptions scoped to that entity **and** unscoped subscriptions
(`entity_id=None`), because both would fire for that entity's events:

```python
# Everything that fires when scene X's dims change.
subscribers = controller.get_subscribers(DimsChangedEvent, entity_id=scene.id)
```

### Inspecting the live subscriber object

`SubscriberInfo.callback_instance` holds the bound object for method callbacks.
Use this in a debugging session to inspect the subscriber directly:

```python
for info in controller.get_subscribers(AppearanceChangedEvent):
    print(type(info.callback_instance).__name__, info.callback_qualname)
    # e.g. GFXMultiscaleImageVisual  on_appearance_changed
```

### `SubscriberInfo` fields

| Field | Type | Description |
|---|---|---|
| `callback_qualname` | `str` | Dotted name of the callback, e.g. `"QtDimsControl._on_dims_changed"`. `"(dead)"` if the weak reference was collected. |
| `callback_instance` | `object` | The bound instance for a method callback; `None` for plain functions or dead weak refs. |
| `owner_id` | `UUID or None` | UUID used to group this subscription for bulk removal via `controller.unsubscribe_owner`. |
| `entity_id` | `UUID or None` | Entity this subscription is scoped to, or `None` for unscoped. |
| `is_weak` | `bool` | `True` when the bus holds a weak reference to the callback. |
| `is_alive` | `bool` | `False` when a weak reference has been garbage-collected. |

---

## Tool 2 — `_Subscription.__repr__`

The internal `_Subscription` objects stored in `EventBus._subs` now have a
human-readable `repr`. This is useful when you break inside `EventBus.emit()`
or inspect `_subs` directly in a debugger:

```python
# In a debugger, evaluate:
controller._outgoing_events._subs[DimsChangedEvent]
```

Each entry now prints as:

```
_Subscription(callback='QtDimsControl._on_dims_changed' owner_id=3f2a... entity_id=9c1b... strong/alive)
_Subscription(callback='CellierController._on_dims_changed_bus' owner_id=1a2b... entity_id=None strong/alive)
_Subscription(callback='(dead)' owner_id=7d4e... entity_id=9c1b... weak/dead)
```

The `strong/alive`, `weak/alive`, and `weak/dead` suffixes immediately show
whether a subscription is still active.

---

## Tool 3 — Source ID logging

Traces `source_id` injection through the `ContextVar` → psygnal → bus bridge.
Enable it to see who triggered each model mutation, which bridge handler
resolved the `ContextVar`, and whether it fell back to the controller's own ID.

```python
from cellier.logging import enable_debug_logging

enable_debug_logging(categories=("source_id",))
```

### What the output looks like

A slider moving its contrast-limits produces three lines:

```
[SOURCE_ID] set    field=clim  visual=9c1b...  source=3f2a...
[SOURCE_ID] bridge handler=_on_appearance_psygnal  visual=9c1b...  field=clim  resolved_source=3f2a...  override_active=True
[SOURCE_ID] reset  field=clim  visual=9c1b...
```

- `set` — `update_appearance_field` was called with `source_id=widget._id`.
- `bridge` — the psygnal handler fired, read the `ContextVar`, and resolved the
  source. `override_active=True` confirms it came from a controller method.
- `reset` — the `ContextVar` was restored after the mutation.

The `[SOURCE_ID]` prefix is added by the Rich handler (the default,
`use_rich=True`). With `enable_debug_logging(..., use_rich=False)` the records
carry no color label — each line is prefixed with the logger name
`cellier.render.source_id` instead. The `set` / `bridge` / `reset` message
bodies are identical either way.

### Spotting a direct model mutation

If `override_active=False` appears in a `bridge` line, the model field was
mutated directly (bypassing `update_appearance_field` or similar), and the
`source_id` fell back to the controller's own ID:

```
[SOURCE_ID] bridge handler=_on_appearance_psygnal  visual=9c1b...  field=clim  resolved_source=<controller-id>  override_active=False
```

This means no widget will echo-filter the event — all subscribers will receive
it as if an external change occurred.

---

## Common failure scenarios

### Subscriber not firing

**Symptom:** a widget or render layer callback is not called when the model
changes.

1. Check that the subscription exists and is alive:
   ```python
   subscribers = controller._outgoing_events.get_subscribers(
       AppearanceChangedEvent, entity_id=visual.id
   )
   for info in subscribers:
       print(info.callback_qualname, info.is_alive, info.entity_id)
   ```
2. If `is_alive=False` — the subscriber was garbage-collected. The subscription
   was registered with `weak=True` and the owning object has been destroyed.
   Ensure the object is kept alive for as long as the subscription is needed, or
   call `controller.unsubscribe_owner(owner_id)` at teardown rather than relying
   on GC.
3. If the subscriber is missing entirely — `subscribe` was never called, or
   `unsubscribe_all` was called prematurely. Check widget teardown paths.
4. If `entity_id` on the subscription does not match the `entity_id` of the
   emitted event — the event will be silently skipped. Confirm that the UUID
   passed to `subscribe` matches the UUID of the model object that changed.

### Widget updating when it shouldn't (echo not filtered)

**Symptom:** a widget re-applies a value it just set, causing flickering or
redundant work.

1. Enable source ID logging and move the widget:
   ```python
   enable_debug_logging(categories=("source_id",))
   ```
2. Check `override_active` in the `bridge` line. If it is `False`, the mutation
   bypassed `update_appearance_field` and `source_id` was set to the
   controller's ID — the widget's `if event.source_id == self._id` check will
   never match.
3. Confirm the widget passes its own `._id` as `source_id`:
   ```python
   controller.update_appearance_field(visual_id, "clim", value, source_id=self._id)
   ```
   Not passing `source_id` defaults to the controller's ID, which no widget
   will match.

### Infinite update loop

**Symptom:** the application hangs or the stack overflows after a widget
interaction.

The safeguard chain is: `source_id` echo filter (skip redundant self-updates)
and `blockSignals` (prevent programmatically-updated Qt widgets from re-firing
their signals). Both must be in place. If `blockSignals` is missing from a
widget's `_set_value` path, a programmatic update to widget B will fire
widget B's `valueChanged`, which will call `update_*`, which will emit an event,
which will update widget A, and so on.

To diagnose:

1. Check `get_subscribers` for the event type — confirm the chain of
   subscriptions matches your expectations.
2. Add a temporary breakpoint in `EventBus.emit` and inspect the call stack
   depth. A loop will show repeated frames.
3. Confirm every widget's programmatic update path calls `blockSignals(True)`
   before setting the value and `blockSignals(False)` after.

### Source ID belongs to the wrong object

**Symptom:** a widget receives an event and does not filter it, even though the
widget itself caused the change.

Enable source ID logging and compare the `resolved_source` in the `bridge` line
against `widget._id`. Common causes:

- The widget passed a different UUID to `source_id` than the one it uses in
  the `if event.source_id == self._id` check.
- Two different widget instances share the same model visual but each have their
  own `._id`. Widget A's change arrives at widget B with `source_id=A._id`,
  which B does not filter — this is correct behaviour. The `blockSignals` guard
  in B's `_set_value` prevents the cascade.
