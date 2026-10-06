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
| **`CachePolicy`** | How the scheduler treats one cache's reads, declared by its `Residency` and read once at registration: a cap on target reads in flight (`max_target_fetching`; a backstop read is never held by it, and a capped cache starts its backstop and target reads of one plan together, holding the target only while a backstop result waits to be committed), the resource a read occupies (`"io"`, or `"compute"` for work done in an executor, which counts against `SchedulerConfig.compute_budget` and takes no I/O slot), and what a failure costs (`retry_max_attempts`, `retry_on_pass`). The default is what an image or labels atlas wants. |
| **`Residency`** / `ImageResidency3D` / `ImageResidency2D` | The scheduler's adapter for one atlas: writes a brick (tile) into the slot the scheduler chose, turns keys back into store requests, and repaints the LUT. |
| **GPU brick/tile cache** (multiscale) | Fixed-slot texture atlas. `write_brick` / `write_tile` upload into a slot. |
| **`TileManager2D` / `TileManager3D`** | The atlas's slot geometry and a read-only `tilemap` view of what the LUT draws.  Which brick lives in which slot is the scheduler's registry. |
| **LUT indirection texture** | Maps virtual brick-grid coordinates → physical atlas slot + level; the shader walks it to sample resident bricks. Painted coarsest→finest so finer bricks cover coarser fallbacks; bricks from earlier views are painted underneath (oldest first) until the current view is complete. The writer and the shaders share one cell → brick rule; see [Multiscale brick lookup](multiscale_brick_lookup.md). |

## Slice thickness

A sliced axis has a position and a **half-thickness**, both in world units. The thickness lives on the scene (`scene.dims.selection.thickness`, set with `controller.update_thickness(scene_id, {axis: half_thickness})` or the "+/-" box next to each slider) and it is the only thickness in the slicing path. An axis with no entry has thickness 0: a plane. No visual adds a thickness of its own.

What each kind of visual draws from that slab:

| Visual | Draws |
|---|---|
| Image, labels (in-memory and multiscale) | **One plane**: the sample nearest the slice position whose extent overlaps the slab. Nothing when no sample does (a slice past the data). |
| Points, lines | What lies **inside** the slab. At thickness 0, only what lies exactly on the plane. |
| Mesh, 3D view | Faces that lie wholly **inside** the slab. |
| Mesh, 2D view | Its **cross-section** at the slice plane: an outline where the surface crosses the plane, and a fill where the outline closes. The thickness on the cut axis is ignored unless the mesh's section mode is `"slab"`. See [A mesh in a 2D view](#a-mesh-in-a-2d-view). |
| Graph | What lies inside the slab, widened by the trail on axes that have one. A trail widens the slab and never narrows it. |

On an axis whose data axis is declared `sampling="discrete"` (a frame index), a geometry visual anchors the slab at the **sample the slider selects**, with the same round-half-up rule an image uses. So thickness 0 on a time axis draws exactly the current frame, and geometry changes frame at the same instant an image does.

!!! note "Points and lines on a continuous axis need a thickness"
    Points and lines have no extent, so a plane on a continuous axis (a spatial `z`) draws only what sits exactly on it, which is usually nothing. Give the axis a thickness to see the geometry near the slice. A mesh needs none: a 2D view cuts it.

### A mesh in a 2D view

A 2D view cuts a mesh with the slice plane and draws the cut. How is set per mesh, on its `section` config (`MeshSectionConfig`), in `add_mesh(..., section=...)`, with `controller.update_section_field(visual_id, field, value)`, or in the "2D section" control group (`MeshControlsConfig(section_controls=True)`), which is shown only while the scene displays two dimensions:

| Field | Default | Meaning |
|---|---|---|
| `outline` | `True` | Draw the curve where the surface crosses the plane. |
| `fill` | `True` | Fill the area closed loops enclose. See the fill rule below. |
| `outline_width` | `2.0` | Outline thickness in screen pixels. Applies at once; the other fields read the mesh again. |
| `mode` | `"cut"` | `"cut"`: the cut by the slice plane; the scene's thickness on that axis is ignored. `"slab"`: what lies inside the scene's slab: the surface between its two faces, flattened, with a cap at each face and both cuts as the outline. |

- **Only spatial, continuous axes are cut.** Along any other sliced axis (time, a channel, an axis declared `sampling="discrete"`) the mesh is filtered face by face, as in 3D. A `tzyx` series in a `yx` view is filtered by `t` and cut at `z`.
- **A mesh with no extent along the sliced axis is not cut.** A `yx` mesh in a `zyx` world is drawn whole, at every `z`.
- **Faces lying in the plane are drawn as themselves**, with their border as the outline. A flat mesh at one `z` shows on that plane and nowhere else.
- **The fill follows the faces' winding.** A loop whose faces point out of it is the boundary of a solid; one whose faces point into it, of a cavity. A region is filled where the solids around it are not cancelled by cavities. So an object inside another object is filled with it, and a cavity inside an object is a hole. A mesh wound inside out is filled the same. When loops nest and the faces along one of them disagree, that nest is filled even-odd instead: a loop inside another is a hole.
- **An open surface has no fill where its cut does not close.** `dataset_info` reports whether a mesh is closed, once it has been cut the first time.
- **Outline and fill share the mesh's colour and opacity.** With both on, the outline is visible only where the fill is not opaque over it.
- **Picks say what was drawn**: `MeshPickInfo.part` is `"outline"` (with the face the plane crosses there), `"fill"` (no face) or `"face"` (a face lying in the plane, or any face in 3D).
- **Where two meshes overlap in a 2D view, the last one drawn is on top.** Their cuts lie in the same plane, so depth does not separate them. The order is `appearance.render_order` (higher on top), then the order the meshes were added (later on top). An outline is drawn after the fills of the same `render_order`, so one mesh's outline shows on another's fill. A section is always drawn with the depth rule `"<="`; `appearance.depth_compare` applies to the 3D surface only. A section is drawn over an image in the same view.

`examples/mesh_cross_section.py` shows a cut with a cavity, an outline-only mesh inside a filled one, and the "2D section" controls.

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

5. The scheduler issues reads, nearest first, up to `SchedulerConfig.max_in_flight` across every visual and channel, round robin between atlases.  `Residency.build_request` turns a key into a `ChunkRequest`.  Reads are never aborted.  A cache whose policy is `resource="compute"` uses a separate budget, `SchedulerConfig.compute_budget`: a full compute lane never delays an I/O read, and the other way round.

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

Visuals do not subscribe to these events. Each visual model says through its own explicit config what it does during a scrub (`BaseVisual.plans_coarse_while_moving(n_displayed_dims)`, which a multiscale image or labels visual answers from `appearance.coarsest_while_moving_3d` in a 3D view and `appearance.coarsest_while_moving_2d` in a 2D one), and the controller turns transitions into plan modes and reslices. The events are for GUIs (a "loading full resolution" indicator) and tests.

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
4. **`CellierController._render_config_for`**: the one place a plan mode is decided. While the scene's tracker is `ACTIVE`, a visual whose `plans_coarse_while_moving` is true for the scene's view gets `PlanMode.BACKSTOP_ONLY` and joins the scene's pending set.
5. **`CellierController._on_dims_transition`**: at a scrub's end by release or stillness, plans the pending set in full with `target_visual_ids`.

A `coarsest_while_moving` setting that is `False` plans in full on every tick. `True` loads the backstop only on each tick, and the target when the scrub ends; it saves most of a scrub's reads. The 3D setting defaults to `True`, because a full 3D plan per tick never catches up with the slider; the 2D one defaults to `False`, because a slice is a few hundred tiles, and `True` is recommended there for remote stores. Both show the slider's slice, blurry, about one read behind it.

In a 3D view the same setting covers a drag of the visual's clipping planes: with it on, nothing is planned until the drag ends (see the interaction tracker explanation).

A slider sends more than ticks. On press it opens a scope and on release it closes it, so the target loads on release instead of 0.15 s later. The release first flushes the slider's throttle: the end plans in full, and it must plan the final position. The anywidget panel's release message carries the final position itself, because in a notebook a custom message can overtake the traitlet sync sent before it.

In the ortho viewer, `OrthoDimsController` mirrors positions to the other panels with plain `update_slice_indices` calls, which by themselves would be jumps. So when a scrub starts on one panel it opens a scope on every other panel, and closes them when that scrub ends. A panel that *displays* the scrubbed axis receives no tick (its region does not change) and its scope opens and closes silently.

### Change to the camera model

1. **`CanvasView._draw_frame`**: the view drives its pygfx camera controller itself (`auto_update=False`). Each frame it ticks the controller, applies the returned state to the camera, and then compares the camera with the last reported state. A difference emits `CameraChangedEvent(interactive=True)` and resets temporal accumulation, in the frame that draws the new camera. When the controller starts or stops having a running action, the view emits an internal `_CameraControllerEvent`.
2. **`CellierController._on_camera_changed`**: writes the model camera and ticks the canvas's camera tracker. **`_on_camera_controller_event`** opens and closes the controller's scope on that tracker.
3. **`CellierController._on_camera_transition`**: sets `CanvasView.camera_moving`, emits `CameraInteractionEvent`, and at an end by release or stillness reslices the scene's `requires_camera_reslice` visuals. The end also requests one frame on its canvas, without resetting accumulation.
4. **Programmatic moves** go through `_after_programmatic_camera_move`: `CanvasView.accept_camera_state` takes the moved camera as the canvas's baseline (so the next frame does not see it as motion), the model camera is written, `CameraChangedEvent` is emitted with `interactive=False`, and the jump reslices at once.

A transition detected inside a draw (a camera tick, the controller's scope closing) must not plan there. Its reslice is queued as a task on the event loop, one per scene. `CellierController._deferred_reslice_tasks` reports those tasks and both kinds of stillness timers, so a drain (`convenience.capture`, the test helpers) waits for a reslice that is still to come.

## How a mesh loads

A mesh (`GFXMeshVisual`) loads through the chunk scheduler, like a multiscale image, and not through `AsyncSlicer`. What it loads is a whole level of the mesh at one request, not bricks.

- **The read runs off the event loop.** `MeshMemoryStore.get_data` slices in an executor thread and returns arrays ready to upload. At most `SchedulerConfig.compute_budget` such reads run at once, over all meshes.
- **One read at a time per mesh, and the newest request wins.** A read cannot be cancelled, so while one runs the next waits; a request the slider has since left is dropped before it starts.
- **An unchanged request reads nothing.** A request is identified by the region it selects in the mesh's own data coordinates. A slider the mesh does not depend on, a camera move, an appearance change, and a transform change that leaves the region where it was all leave the request as it was.
- **A mesh never draws a position other than the slider's.** When the request changes, the mesh stops being drawn in the next frame and returns when the new position has loaded. A read that lands for a position the slider has left is never drawn. During a scrub faster than the reads, the mesh is absent and returns when the slider rests.
- **A canvas waits a moment for a read before it draws the gap.** After a dims change, while a mesh is hidden for a read, the canvas skips frames and keeps its last picture, for at most `RenderManagerConfig.draw_hold_ms` (default 50 ms). A read that lands in that time is drawn with no blank frame before it; a slower one leaves the mesh absent until it loads, as above. The picture kept is of the position just left. Camera input ends the wait at once, and 0 turns it off. Only visuals that hide while they load are waited for: an in-memory image or points keep their old picture until the new data arrives, and a multiscale image fills in as chunks land.
- **Nothing is kept for a revisit.** A position the slider leaves is released at once; coming back reads it again.
- **A hidden mesh loads nothing.** Shown again at the same position it reads nothing; at another position it loads.
- **A read that raises is tried once.** The error is logged and counted as failed in the visual's loading progress; the same request is not tried again until the store changes.
- **Only a store change invalidates.** Reassigning the store's arrays drops what was read; a read in flight for the old arrays is discarded when it lands.

### A mesh with levels of detail

A `MultiscaleMeshStore` holds the same surface at several levels, finest first, supplied by the caller. `controller.add_multiscale_mesh(store, scene_id, appearance, lod=GeometryLodConfig(...))` draws it with the same visual as a plain mesh, and keeps **two levels loaded**: the finest, and one coarse level (`lod.coarse_level`, 1-based, the coarsest by default). `Viewer.add_multiscale_mesh` adds it to the viewer's scene, and `OrthoViewer.add_multiscale_mesh` adds one visual per panel, all reading the one store.

- **A new position asks for both levels at once.** The coarse read is issued first and never waits for a fine read, so the mesh is back on screen after the coarse read. The finest replaces it when it has loaded, with no frame without the mesh between. A fine read that lands first is drawn directly.
- **One level is drawn at a time**, and only a level that holds the current position. In a 2D view each level is cut by itself, so the coarse section is the coarse surface's.
- **A pick names the level it hit**: `MeshPickInfo.level`, 0-based, 0 the finest, with the face in that level's numbering.
- **The bounding box and a camera fit use the finest level's extent**, whichever level is drawn.
- **Each level costs its own first read** (its index and normals) and its own memory, on the CPU and on the GPU.
- **A store of one level** loads as a plain mesh.

During a dims scrub (a slider drag, or `controller.dims_interaction`) and while the camera moves, the `lod` settings decide what loads and what is drawn:

| Setting | Default | During a scrub |
|---|---|---|
| `dims_drag` | `"coarse"` | What each tick **loads**. `"coarse"`: the coarse level only, and the finest once when the scrub ends. `"full"`: both levels on every tick. |
| `camera_motion` | `"coarse"` | Not a scrub setting: what a **3D** canvas draws while its camera moves. `"coarse"`: the coarse level from the first moved frame to the end of the motion (the drag and its damped tail, or one wheel notch), and the finest in the frame after. `"full"`: the finest throughout. Nothing is read either way. |
| `dims_drag_draw` | `"coarse"` | What is **drawn** of a mesh the scrub does not change (a static mesh beside a time series). `"coarse"`: the coarse level while the slider moves, so a very large mesh does not slow the scrub's frames; the finest stays loaded and is back in the frame after the scrub ends. `"full"`: the finest throughout. |

- **A mesh the scrub does not change reads nothing**, during the scrub or at its end. Both of its levels stay loaded.
- **Camera motion is per canvas.** A second canvas on the same scene keeps drawing the finest level while the first is orbited. A 2D view never switches level when it is panned or zoomed. A programmatic camera move (`fit_camera`, `set_camera_state`) does not switch level, unless it is made while the user is dragging, when it is part of that motion.
- **A move that is not a scrub** (a programmatic `update_slice_indices`, a click on the slider track) loads both levels at once.
- **A screenshot taken during a scrub draws the finest level.**
- **The mouse wheel switches level once per notch** when the notches are more than about 0.4 s apart. One notch is a camera motion of its own: it starts, glides, and ends 0.3 s after the camera last moved, so the mesh goes coarse at the notch and fine again before the next one. Notches closer together than that are one motion, and the mesh stays coarse until the last one ends. Set `camera_motion="full"` on a mesh that should not switch.
- **Set `dims_drag_draw="full"` on a small static mesh.** The default draws the coarse level of a mesh the scrub does not change, which shows as a level switch at the scrub's start and at its end. It is there for meshes too large to draw at the scrub's frame rate; a mesh the GPU draws easily gains nothing from it.
- `controller.set_lod_config(visual_id, dims_drag_draw="full")` changes a setting on a mesh that is in a scene. `dims_drag_draw` and `camera_motion` apply in the next frame with nothing read; `dims_drag` is read by the next scrub. `coarse_level` is fixed when the mesh is added. `Viewer.set_lod_config(visual, ...)` is the same call, and `OrthoViewer.set_lod_config(visual, ...)` changes every panel. Each change emits `LodConfigChangedEvent` (`controller.on_lod_config_changed`).
- **The "LOD" control group** (`MeshControlsConfig(lod_controls=True)`) has one row for each of the three settings and changes them the same way. A mesh with one level has no such group.

Build the levels yourself; cellier does not simplify a mesh. Use quadric decimation rather than quadric clustering: a decimated level stays closed, so its 2D section is still filled. A coarse level that is open draws its outline only where its cut does not close. `examples/multiscale_mesh_decimation.py` builds the levels with pyvista's `decimate`.

The first read of a mesh builds what later reads share: an index of the faces along each sliced axis, and the vertex normals for a 3D view. That costs once per mesh and is paid again after the store changes. A mesh whose arrays are reassigned many times a second is not what this is built for.

The mesh's bounding box and the camera fit use the store's whole extent (every timepoint of a series), not the faces of the current slice, so neither changes while the mesh loads. A camera fit on a time series therefore frames everywhere the mesh ever is, not where it is now.

### What a mesh costs

The times below were measured on one machine (Apple M-series, macOS, a 900 x 700 window). They show which sizes work; the figures on another machine will differ.

**A scrub blinks or goes blank, depending on the size of what each step reads.** A mesh is not drawn until the new position has loaded. When the read and one frame fit in the time between two steps, the mesh is back before the next step, and a read that lands within the draw hold (50 ms) shows no blank frame at all. When they do not fit, the mesh is absent until the slider rests. A smaller level shortens the gap; it does not remove it.

| Faces read per step | Read | A 20 steps-per-second scrub |
|---|---|---|
| 100k | 7-11 ms | Almost every step is drawn before the next (39 of 40). |
| 400k | 28-30 ms | About half the steps are drawn (23 of 40). |
| 1M | 71-75 ms | Almost none (0-7% of frames have the mesh). |
| 4M | about 300 ms | None. The mesh returns when the slider rests. |

- For a multiscale mesh the step reads the **coarse level**, so the row to read is the coarse level's size per position. A coarse level of about **100k faces per position** keeps up with a 20 steps-per-second scrub; at 30 steps per second about half the steps are drawn even at 100k.
- For a time series the size that counts is the faces of **one timepoint**, not of the whole series.
- When the scrub ends, the finest level is drawn one fine read and one frame later: about 95 ms at 1M faces per timepoint and 330 ms at 4M.
- A mesh the scrub does not change reads nothing and is never hidden, at any size.

**The first read of a mesh is slower than the ones after it.** It builds an index of the faces along each sliced axis and, for a 3D view, the vertex normals, over the whole level:

| Mesh | First read | Later reads |
|---|---|---|
| Static, 3D view, 4.5M faces | 290 ms | none: the request does not change |
| Static, 3D view, 16M faces | 1.0 s | none |
| Static, first slice on an axis, 4.5M faces | 115 ms | 2 ms |
| Static, first slice on an axis, 16M faces | 407 ms | 8 ms |
| Series of 10 timepoints, 100k faces each | 78 ms | 7 ms |
| Series of 10 timepoints, 1M faces each | 744 ms | 71 ms |
| Series of 10 timepoints, 4M faces each | 3.1 s | about 300 ms |

- A series pays for **every timepoint** at its first read, so a single-level series of 10 x 4M faces shows nothing for about 3 s. With a coarse level beside it the mesh is on screen after the coarse level's own first read (about 100 ms at 100k faces per timepoint, 370 ms at 400k), while the fine one runs.
- The index takes about 8 bytes per face per sliced axis (128 MB at 16M faces).
- A coarse level of 400k faces beside a fine level of 4M or 16M adds 2-5% to the process's peak memory.

**The frame that puts a new level on the GPU is a long one.** The read runs off the event loop, but the upload of the result is one frame's work: about 21 ms at 1M faces, 38 ms at 4M, and 120-180 ms at 16M. Nothing else in the window updates during that frame. It happens once per load of a level, not per frame drawn.

### The "Data fetch status" group for a mesh

A mesh that is loading is not drawn, so an empty view does not say whether the mesh is loading or has nothing at this position. `MeshControlsConfig(loading_indicator=True)` adds the "Data fetch status" group to a mesh's controls. It reads the same `LoadingProgress` a multiscale image reports (`controller.loading_progress`, `ResliceProgressEvent`), worded in levels:

| Text | Meaning |
|---|---|
| `Loading coarse level` | A new position was asked for; neither level has loaded. |
| `Loading fine level` | The coarse level is drawn; the finest is being read. |
| `Loading` | A mesh with one level is being read. |
| `Coarse level ready. Fine on stop.` | A scrub in progress: the coarse level is drawn, and the finest loads when the scrub ends. |
| `Loaded` | What the view asked for is drawn. `Loaded, 1 failed` when a read raised. |

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
- The in-memory points and lines visuals are `cancellable = False`. Their tasks always run to completion so every intermediate slice position reaches the GPU. This is because these tend to be very fast to slice and thus there isn’t a need to cancel.
- The mesh has no `cancellable` flag: it loads through the chunk scheduler, which never cancels (see [How a mesh loads](#how-a-mesh-loads)).

`SliceCoordinator.submit` checks this flag for each visual it is about to re-submit and only cancels the ones marked cancellable.

### The cancellation path

1. **`SliceCoordinator.submit`** (or `cancel_scene`): decides which visuals to cancel. On submit, the visuals about to be re-loaded are canceled first, subject to the `cancellable` flag above.
2. **`SliceCoordinator.cancel_visual`**: pops the `slice_request_id` for the `(scene, canvas, visual)` key out of `_active_slice_ids` and calls `AsyncSlicer.cancel`.
3. **`AsyncSlicer.cancel`**: calls `task.cancel()`. Inside `_run`, the `CancelledError` is re-raised so asyncio marks the task canceled; the in-progress batch is discarded (its `callback` never fires), `on_complete` is skipped so no spurious `ResliceCompletedEvent` is emitted, and the `finally` block drops the task from `_tasks`.
4. **`visual.cancel_pending_2d` / `cancel_pending`**: `cancel_visual` then calls these (selected by the visual's `render_modes`); in-memory visuals reserve no GPU slots, so these are no-ops.  Multiscale visuals are skipped: the scheduler never cancels.

### GPU state after a cancel

Multiscale visuals are not cancelled at all: the chunk scheduler lets reads in flight land (tensorstore keeps the bytes anyway), keeps them as `RECENT` bricks, and drops only reads it has not yet issued.  Bricks from an earlier slice position stay on screen underneath the new one until it is complete; a key carries the slice it was read on, so bricks of different positions never collide.
