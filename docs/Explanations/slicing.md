# Slicing

This document describes how slicing works in Cellier. 

## Core concepts & data types
The slicing pipeline is broken up into three main steps:

1. Plan: determine which data needs to be loaded based on the current view.
2. Fetch: request the required data from the data stores. The requests are performed asynchronously.
3. Commit: as the requested data arrives from the data stores, upload it to the appropriate buffer/texture on the GPU.
 
The table below describes the core objects and data types used to orchestrate slicing. 

| **Component** | **Explanation** |
|---|---|
| **`ReslicingRequest`** | Immutable per-canvas snapshot of camera + dims at trigger time. The unit of work. One per canvas so each camera drives its own LOD/culling. Carries `request_id` (one per trigger) — *not* the ID used for fetch cancellation; see `slice_request_id` below. |
| **`DimsState` / `AxisAlignedSelectionState`** | Which axes are *displayed* (→ `slice(None)`) vs *sliced* (→ integer index). The selection that defines the plane/slab. |
| **`ChunkRequest`** | A single padded brick/region read: `scale_index` + `axis_selections` (per-axis `int` for sliced, `(start,stop)` for displayed). Coords may be out-of-bounds; `get_data` clamps + zero-pads. Carries `slice_request_id`, shared by all chunks from one planning event. |
| **`AsyncSlicer`** | Generic cancellable batch-fetch service. One asyncio.Task per `slice_request_id` (the shared `ChunkRequest` ID, *not* `ReslicingRequest.request_id`); data source injected per-submit as `fetch_fn`. `submit` returns the `slice_request_id`, which `SliceCoordinator` stores keyed by `(scene, canvas, visual)` and later passes to `cancel`. |
| **`SliceCoordinator`** | Orchestrator: per-`(scene,canvas,visual)` cancellation, planning dispatch, reslice-start/complete events. Routes multiscale visuals (2D and 3D) to the chunk scheduler. |
| **`ChunkScheduler`** (`cellier.render.scheduling`) | Loads multiscale visuals, 2D and 3D. A persistent registry per atlas that each reslice *re-prioritises* instead of cancelling: one global in-flight budget, nearest bricks first, slots chosen at commit time, and only unwanted (`RECENT`) bricks evicted. |
| **`Residency`** / `ImageResidency3D` / `ImageResidency2D` | The scheduler's adapter for one atlas: writes a brick (tile) into the slot the scheduler chose, turns keys back into store requests, and repaints the LUT. |
| **GPU brick/tile cache** (multiscale) | Fixed-slot texture atlas. `write_brick` / `write_tile` upload into a slot. |
| **`TileManager2D` / `TileManager3D`** | The atlas's slot geometry and a read-only `tilemap` view of what the LUT draws.  Which brick lives in which slot is the scheduler's registry. |
| **LUT indirection texture** | Maps virtual brick-grid coordinates → physical atlas slot + level; the shader walks it to sample resident bricks. Painted coarsest→finest so finer bricks cover coarser fallbacks; bricks from earlier views are painted underneath (oldest first) until the current view is complete. The writer and the shaders share one cell → brick rule; see [Multiscale brick lookup](multiscale_brick_lookup.md). |

## Slicing flow

From the perspective of slicing, there are two main flavors of visuals: in-memory and multiscale. The in-memory visuals load the whole scene at a single scale at one time. The multiscale visuals have to determine the level of detail to load and only load the rendered region of the scene. While they flow through the same components, there are some differences in how they are sliced. See an explanation of how slicing works using the `InMemoryImageVisual` and `MultiscaleImageVisual` as an example.

### In-memory visual

**Plan** (synchronous)

4. **`CellierController.reslice_scene`**: snapshots the dims state and per-visual render config.
5. **`RenderManager.reslice_scene`**: issues one reslicing request per canvas.
6. **`CanvasView.capture_reslicing_request`**: freezes the camera and dims into an immutable `ReslicingRequest`.
7. **`SliceCoordinator.submit`**: cancels any in-flight task for the visual, then plans.
8. **`SceneManager.build_slice_requests`**: dispatches to the 2D or 3D planner by displayed-axis count.
9. **`GFXImageMemoryVisual.build_slice_request`**: maps the world slice position into data space and emits one `ChunkRequest` for the whole slice.

**Fetch** (async, off render thread)

10. **`AsyncSlicer.submit`**: makes the request for a single read on one async task.
11. **`ImageMemoryStore.get_data`**: returns the requested slice / sub-volume array.

**Commit** (main thread)

12. **`GFXImageMemoryVisual.on_data_ready`**: uploads the whole array to the GPU.
13. **`Renderer.render`**: draws the next frame.

### Multiscale visual (the chunk scheduler)

2D and 3D take the same path.  A visual has one atlas per drawn channel and mode (a 3D brick atlas and a 2D tile atlas), each registered with the scheduler under its own cache id.

**Plan** (synchronous)

1. **`CellierController.reslice_scene`** and **`RenderManager.reslice_scene`**: as above, one `ReslicingRequest` per canvas.
2. **`SliceCoordinator.submit`**: never cancels a multiscale visual; registers its atlases with the scheduler.
3. **`SceneManager.plan_chunked`** → **`GFXMultiscaleImageVisual.plan`**: LOD selection, nearest-first sort, frustum (3D) or viewport (2D) cull and truncation, returning one `DesiredSet` of packed keys per drawn channel.  Each set leads with the **backstop**: by default the coarsest level over the whole volume (3D) or slice (2D), nearest first, capped at `backstop_max_slot_fraction` of the atlas (`render_config.loading`, a `ProgressiveLoadingConfig`).  Backstop reads go before every target read, from every visual, so the view is blurry rather than blank (or showing the previous slice) while the target loads.  Planning touches no GPU state.  The atlases of the other mode get an empty pass (they are *retired*): their bricks stay resident as `RECENT`, so switching back costs no reads.
4. **`ChunkScheduler.pass_`**: passes in one loop turn coalesce.  Each atlas's registry is diffed against its desired set: new bricks are queued, bricks still wanted are re-prioritised, bricks no longer wanted become `RECENT` (queued ones are simply deleted; ones in flight still land).

**Fetch** (async)

5. The scheduler issues reads, nearest first, up to `SchedulerConfig.max_in_flight` across every visual and channel, round robin between atlases.  `Residency.build_request` turns a key into a `ChunkRequest`.  Reads are never aborted.

**Commit** (main thread, once per drawn frame)

6. **`before_draw`** on each canvas runs **`ChunkScheduler.commit_round`** for its scene: every arrived brick takes a free slot, or evicts the least important `RECENT` brick, and **`Residency.write`** uploads it.  A 50 ms fallback timer commits for canvases that are not drawing.
7. **`Residency.rebuild_draw`**: repaints the LUT once per touched atlas.  The backstop is part of the current view, so it stays drawn under the target, which covers it wherever the target has data.  Until the current view is complete, bricks from earlier views are painted underneath it, oldest first; in 2D that background is clipped to the viewport, so stale tiles outside the view are not referenced.
8. When every wanted brick of a visual is resident, the coordinator emits `ResliceCompletedEvent` for each reslice announced since its last completion.
9. **Progress**: after each pass, commit round, given-up read or invalidation, the coordinator emits one `ResliceProgressEvent` per touched visual (coalesced to once per loop iteration), carrying a `LoadingProgress` summed over its atlases, and `BackstopCompleteEvent` once a plan's backstop is resident.  The multiscale image and labels panels show it with a loading indicator (`MultiscaleImageControlsConfig.loading_indicator`); `CellierController.loading_progress` reads it on demand.

**Store changes and paint**

A store's `contents` change, and a multiscale paint session's flush, call **`ChunkScheduler.invalidate`** with level-0 data regions: every atlas reading the store drops the bricks that overlap them, at every level, and the wanted ones are fetched again.

A store announcing a change (`notify_changed`, or reassigning a data field) is how a **live store** reaches the screen: a zarr store that another process writes to, or one that an acquisition appends to.

- **The chunk cache.** The store's tensorstore handles trust their cache (rechecks off), so the announcement also reopens them on the same context with `recheck_cached_data="open"`. Each chunk cached before the change is revalidated once, when next read: a `304` if unchanged, a refetch if changed.
- **The GPU.** The controller invalidates at once. The reslice that follows is capped at `SchedulerConfig.store_change_max_hz` per store, so a store announcing frames at any rate costs at most 30 plans a second.
- **Replanning.** A contents change does not replan multiscale readers: their invalidated bricks are already queued again.

## Triggering slicing

Re-slicing is triggered by changes to the dims model or the camera model. Only multiscale visuals reslice on a camera change: this is driven by the `requires_camera_reslice` flag on the visual model, which is `True` only on the multiscale image and label visuals. Camera-triggered re-slicing is also gated by `config.camera.reslice_enabled`.

Both triggers go through the same mechanism, the **interaction tracker**, which tells an *interaction* (the user is scrubbing a slider, or moving the camera) from a *jump* (a script moved the dims or the camera). The next section describes it; the two after it give the call path of each trigger. [Interaction tracker](interaction_tracker.md) explains the tracker itself in full, with timelines of the events and an example of listening to them.

## Interaction and progressive loading

### The two states

The controller keeps one `InteractionTracker` per **scene** for the dims (a *scrub*) and one per **canvas** for the camera (a *motion*). A tracker is `IDLE` or `ACTIVE`, and is driven by two inputs:

- **Ticks.** A change of the tracked input: a slice position moved, or the camera moved. A tick is **interactive** when it is marked so, or when a scope is open. Otherwise it is a **jump**.
- **Scopes.** "Something is holding the input": a slider held down, the camera controller driving the camera, or a `with` block in a script. Opening a scope starts nothing by itself; the first tick does.

| Input | In `IDLE` | In `ACTIVE` |
|---|---|---|
| Interactive tick | start | re-arm the stillness timer |
| Jump | nothing (it loads in full at once) | end, reason `"jump"` |
| The last open scope closes | nothing | end, reason `"release"` |
| No tick for the stillness time | nothing | end, reason `"settle"` |
| The view is replaced (displayed axes change, 2D/3D switch) | nothing | end, reason `"cancel"` |

An interaction ends **on release or on stillness, whichever comes first**. Holding the input still settles it even with the button down; the next movement starts a new interaction. The stillness times are `SchedulerConfig.dims_settle_s` (0.15 s) for the dims and `CameraConfig.settle_threshold_s` (0.3 s) for the camera.

| | Dims (per scene) | Camera (per canvas) |
|---|---|---|
| Interactive ticks | slider moves (drag, groove click, keyboard, wheel); `update_slice_indices(..., interactive=True)` | camera changes seen between frames; programmatic moves with `interactive=True` |
| Scopes | a slider held down; the ortho viewer's mirror; `controller.dims_interaction(scene_id)` | the camera controller driving the camera (a drag held, its damped tail, a wheel or key animation); `controller.camera_interaction(canvas_id)` |
| Jumps | a plain `update_slice_indices`; a thickness change | `fit_camera`, `look_at_visual`, `set_camera_depth_range`, `set_camera_state` |
| While active | visuals that opted in plan their coarse backstop only, on every reslice | nothing is resliced |
| At the end | the visuals that planned coarse plan in full | the scene's `requires_camera_reslice` visuals plan in full |
| Events | `DimsInteractionEvent`; `DimsChangedEvent.interactive` | `CameraInteractionEvent`; `CameraChangedEvent.interactive` |
| State | `controller.dims_interaction_state(scene_id)` | `controller.camera_interaction_state(canvas_id)` |

Visuals do not subscribe to these events. Each visual model says through its own explicit config what it does during a scrub (`BaseVisual.plans_coarse_on_scrub`, which a multiscale image or labels visual answers from `render_config.loading.dims_drag == "backstop"`), and the controller turns transitions into plan modes and reslices. The events are for GUIs (a "loading full resolution" indicator) and tests.

### What a script gets

A script's move is a **jump**: it loads in full at once, with no coarse pass first and no wait.

```python
controller.update_slice_indices(scene.id, {0: 12.0})   # full plan, now
controller.fit_camera(scene.id)                         # reslices for the fitted view, now
```

A jump during an interaction (a script moves the dims while the user drags) ends it with reason `"jump"`; the user's next tick starts a new one.

### Players and fly-throughs: opt in with a scope

A loop that moves the dims or the camera many times a second is an interaction, and says so by opening a scope. Inside it every move is a tick, so the loop loads coarse (dims) or does not reslice (camera) while it runs, and loads in full once, when the block exits:

```python
with controller.dims_interaction(scene.id):
    for t in frames:
        controller.update_slice_indices(scene.id, {0: t})

with controller.camera_interaction(canvas_id):
    for pose in path:
        controller.set_camera_state(canvas_id, pose)
```

`Viewer` and `OrthoViewer` mirror these: `viewer.dims_interaction()`, `viewer.set_slice_positions(...)`, `viewer.camera_interaction()`, `viewer.set_camera_state(...)`.

A loop that wants every position at full resolution does not open a scope. That is its explicit choice: each step is then a jump, and each jump issues target reads that cannot be aborted, for a position the loop is about to leave.

!!! note "Stepping the dims over a multiscale mesh"
    A multiscale mesh draws nothing for a new position until that position's level has loaded. A script that steps the dims with no scope asks for the fine level at every step; each fine read runs to completion for a position the loop has already left. A script that wants only the preview while it steps should open `dims_interaction`: the mesh then loads its coarse level at each step and its fine level once, at the end.

### Rules worth knowing

- **Every reslice during a scrub plans coarse**, not only the scrub's own ticks: a visual shown, a loading-config change, a store change, a transform change, a visual added, and a camera reslice all plan opted-in visuals backstop-only while the scene is scrubbing. The scrub's end plans them in full, once.
- **A scrub that ends while a camera of the scene is moving** leaves the camera-sensitive visuals to the camera's end. Planning them in full from a camera that is still moving would issue reads for a view the camera's end replaces.
- **A camera motion ends one frame after the camera stops**, not when the mouse button goes up: a drag has a damped tail, and the scope covers it.
- **A camera end reslices the whole scene**, one request per canvas, not only the canvas that moved. Chunk residency is per visual and the last pass wins, so a reslice of one canvas would replace what the others want.
- **A move that changes nothing is not a tick.** `update_slice_indices` with the positions already set, a position change on a *displayed* axis, or a `fit_camera` on an already fitted canvas do nothing.
- **With no event loop** (a synchronous test, a bare script) there are no timers: an interactive tick ends at once with `"settle"`.
- **`camera_reslice_enabled = False`** keeps the camera tracker and its events and skips only the reslices.

### Change to the dims model

1. **`CellierController.update_slice_indices`**: if a *sliced* axis moves, ticks the scene's dims tracker **before** writing the positions, so a scrub's start event reaches its listeners ahead of the tick's own consequences. Then writes the new positions onto the dims model (a psygnal field).
2. **`CellierController._make_dims_handler`**: the controller event bridge emits a `DimsChangedEvent` on the outgoing bus, with `interactive` set for a scrub tick.
3. **`CellierController._on_dims_changed_bus`**: calls `reslice_scene` for the whole scene. A displayed-axes change cancels a scrub first. A region change that did not come through `update_slice_indices` (a thickness) is a tick with no `interactive` flag.
4. **`CellierController._render_config_for`**: the one place a plan mode is decided. While the scene's tracker is `ACTIVE`, a visual with `plans_coarse_on_scrub` gets `PlanMode.BACKSTOP_ONLY` and joins the scene's pending set.
5. **`CellierController._on_dims_transition`**: at a scrub's end by release or stillness, plans the pending set in full with `target_visual_ids`.

`dims_drag="eager"` (the default) plans in full on every tick. `"backstop"` saves most of a scrub's reads and is recommended for remote stores. Both show the slider's slice, blurry, about one read behind it.

A slider sends more than ticks. On press it opens a scope and on release it closes it, so the target loads on release instead of 0.15 s later. The release first flushes the slider's throttle: the end plans in full, and it must plan the final position. The anywidget panel's release message carries the final position itself, because in a notebook a custom message can overtake the traitlet sync sent before it.

In the ortho viewer, `OrthoDimsController` mirrors positions to the other panels with plain `update_slice_indices` calls, which by themselves would be jumps. So when a scrub starts on one panel it opens a scope on every other panel, and closes them when that scrub ends. A panel that *displays* the scrubbed axis receives no tick (its region does not change) and its scope opens and closes silently.

### Change to the camera model

1. **`CanvasView._draw_frame`**: the view drives its pygfx camera controller itself (`auto_update=False`). Each frame it ticks the controller, applies the returned state to the camera, and then compares the camera with the last reported state. A difference emits `CameraChangedEvent(interactive=True)` and resets temporal accumulation, in the frame that draws the new camera. When the controller starts or stops having a running action, the view emits an internal `_CameraControllerEvent`.
2. **`CellierController._on_camera_changed`**: writes the model camera and ticks the canvas's camera tracker. **`_on_camera_controller_event`** opens and closes the controller's scope on that tracker.
3. **`CellierController._on_camera_transition`**: sets `CanvasView.camera_moving`, emits `CameraInteractionEvent`, and at an end by release or stillness reslices the scene's `requires_camera_reslice` visuals. The end also requests one frame on its canvas, without resetting accumulation.
4. **Programmatic moves** go through `_after_programmatic_camera_move`: `CanvasView.accept_camera_state` takes the moved camera as the canvas's baseline (so the next frame does not see it as motion), the model camera is written, `CameraChangedEvent` is emitted with `interactive=False`, and the jump reslices at once.

A transition detected inside a draw (a camera tick, the controller's scope closing) must not plan there. Its reslice is queued as a task on the event loop, one per scene. `CellierController._deferred_reslice_tasks` reports those tasks and both kinds of stillness timers, so a drain (`convenience.capture`, the test helpers) waits for a reslice that is still to come.

## Canceling requests

During interactive use the dims and camera models change faster than data loads complete, so a new reslice usually *supersedes* an in-flight one. Canceling the superseded load stops wasted reads from the data store and prevents stale data from committing to the GPU after the view has already moved on.

### What cancels in-flight requests

There are four actions/events that cancel an in-flight fetch:

- **New slice request**: if a new slice request is made before the previous one completed, the in-flight request is canceled.
- **Interaction teardown**: removing a scene or a canvas, or closing the controller, drops its interaction trackers: the stillness timers and any camera reslice queued from a draw are cancelled. This cancels the *trigger* before any request exists, rather than an in-flight fetch.
- **Safety-net resubmit**: `AsyncSlicer.submit` cancels any lingering task that shares the same `slice_request_id` before starting a new one.
- **Scene teardown**: `SliceCoordinator.cancel_scene` cancels every in-flight task for a scene.

### What gets canceled: the `cancellable` flag

Whether a visual's in-flight load is canceled is gated by its `cancellable` property:

- The image and label visuals -- both in-memory (`GFXImageMemoryVisual`, `GFXLabelMemoryVisual`) and multiscale (`GFXMultiscaleImageVisual`, `GFXMultiscaleLabelVisual`) -- default to `cancellable = True`, so a superseding reslice cancels their in-flight reads.
- The static-geometry in-memory visuals (mesh, points, lines) are `cancellable = False`. Their tasks always run to completion so every intermediate slice position reaches the GPU. This is because these tend to be very fast to slice and thus there isn’t a need to cancel.

`SliceCoordinator.submit` checks this flag for each visual it is about to re-submit and only cancels the ones marked cancellable.

### The cancellation path

1. **`SliceCoordinator.submit`** (or `cancel_scene`): decides which visuals to cancel. On submit, the visuals about to be re-loaded are canceled first, subject to the `cancellable` flag above.
2. **`SliceCoordinator.cancel_visual`**: pops the `slice_request_id` for the `(scene, canvas, visual)` key out of `_active_slice_ids` and calls `AsyncSlicer.cancel`.
3. **`AsyncSlicer.cancel`**: calls `task.cancel()`. Inside `_run`, the `CancelledError` is re-raised so asyncio marks the task canceled; the in-progress batch is discarded (its `callback` never fires), `on_complete` is skipped so no spurious `ResliceCompletedEvent` is emitted, and the `finally` block drops the task from `_tasks`.
4. **`visual.cancel_pending_2d` / `cancel_pending`**: `cancel_visual` then calls these (selected by the visual's `render_modes`); in-memory visuals reserve no GPU slots, so these are no-ops.  Multiscale visuals are skipped: the scheduler never cancels.

### GPU state after a cancel

Multiscale visuals are not cancelled at all: the chunk scheduler lets reads in flight land (tensorstore keeps the bytes anyway), keeps them as `RECENT` bricks, and drops only reads it has not yet issued.  Bricks from an earlier slice position stay on screen underneath the new one until it is complete; a key carries the slice it was read on, so bricks of different positions never collide.
