"""SliceCoordinator — thin orchestrator that drives the async reslicing cycle."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from cellier.events._events import ResliceCompletedEvent, ResliceStartedEvent

if TYPE_CHECKING:
    from cellier.data.image import MultiscaleZarrDataStore
    from cellier.events._bus import EventBus
    from cellier.events._events import DimsChangedEvent
    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.scene_manager import SceneManager
    from cellier.slicer import AsyncSlicer


class SliceCoordinator:
    """Thin orchestrator owned by ``RenderManager``.

    Given a ``ReslicingRequest``, it looks up the target ``SceneManager``,
    runs the synchronous planning phase, cancels in-flight tasks for the
    affected visuals, and submits new async load tasks.

    One ``AsyncSlicer`` task maps to one ``(scene_id, canvas_id, visual_id)``
    triple.  A ``dict`` keyed by this triple tracks active slice IDs so that
    per-visual cancellation can cancel only the affected task while leaving
    other visuals — and other canvases — in the same scene running.

    Parameters
    ----------
    scenes : dict[UUID, SceneManager]
        Shared scene registry from ``RenderManager``.
    slicer : AsyncSlicer
        Shared async slicer instance.
    data_stores : dict[UUID, MultiscaleZarrDataStore]
        Mapping of ``visual_model_id`` to the data store for that visual.
    """

    def __init__(
        self,
        scenes: dict[UUID, SceneManager],
        slicer: AsyncSlicer,
        data_stores: dict[UUID, MultiscaleZarrDataStore],
    ) -> None:
        self.id: UUID = uuid4()
        self._scenes = scenes
        self._slicer = slicer
        self._data_stores = data_stores
        self._active_slice_ids: dict[tuple[UUID, UUID, UUID], UUID] = {}
        # Set by RenderManager.connect_event_bus so the coordinator can emit
        # ResliceStartedEvent / ResliceCompletedEvent.  None until wired (e.g.
        # in unit tests that drive submit() directly without a bus).
        self._event_bus: EventBus | None = None

    def submit(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> None:
        """Execute the full reslicing cycle for the scene in ``request.scene_id``.

        Cancels in-flight tasks for visuals that will be re-submitted, subject
        to each visual's ``cancellable`` property.  Visuals with
        ``cancellable = False`` are never cancelled; their tasks run to
        completion so every intermediate position reaches the GPU.  This is the
        case for the static-geometry in-memory visuals (mesh, lines, points).
        The image and label visuals -- both in-memory
        (``GFXImageMemoryVisual``, ``GFXLabelMemoryVisual``) and multiscale
        (``GFXMultiscaleImageVisual``, ``GFXMultiscaleLabelVisual``) -- default
        to ``cancellable = True``, so a superseding reslice cancels their
        in-flight reads.  All render-layer visual classes must expose
        ``cancellable`` as part of their public API; an ``AttributeError``
        indicates a missing implementation.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request to process.
        visual_configs : dict[UUID, VisualRenderConfig]
            Per-visual render configuration.
        """
        scene_manager = self._scenes[request.scene_id]

        if request.target_visual_ids is not None:
            to_cancel: frozenset[UUID] = request.target_visual_ids
        else:
            to_cancel = frozenset(scene_manager.visual_ids)

        for visual_id in to_cancel:
            try:
                gfx_visual = scene_manager.get_visual(visual_id)
            except KeyError:
                # Visual not yet registered in render layer; cancel defensively.
                self.cancel_visual(request.scene_id, request.canvas_id, visual_id)
                continue
            if gfx_visual.cancellable:
                self.cancel_visual(request.scene_id, request.canvas_id, visual_id)

        requests_by_visual = scene_manager.build_slice_requests(request, visual_configs)

        # Announce the start of the reslice cycle for the visuals being loaded.
        if self._event_bus is not None and requests_by_visual:
            self._event_bus.emit(
                ResliceStartedEvent(
                    source_id=self.id,
                    scene_id=request.scene_id,
                    canvas_id=request.canvas_id,
                    visual_ids=frozenset(requests_by_visual.keys()),
                )
            )

        is_2d = len(request.dims_state.selection.displayed_axes) == 2

        for visual_id, chunk_requests in requests_by_visual.items():
            visual = scene_manager.get_visual(visual_id)
            data_store = self._data_stores[visual_id]

            # Use the appropriate callback for the scene dimensionality.
            callback = visual.on_data_ready_2d if is_2d else visual.on_data_ready

            brick_count = len(chunk_requests)

            # Filled in from submit() below, so the completion closure can tell
            # whether the entry it is about to drop is still its own.  Safe to
            # assign after the fact: submit only schedules the task, so nothing
            # can run it -- and reach _on_complete -- before submit returns.
            own_slice_id: list[UUID] = []

            # Closure fired once all bricks/tiles for this visual have committed.
            # Bind the loop variables as defaults so each visual gets its own.
            def _on_complete(
                vid: UUID = visual_id,
                sid: UUID = request.scene_id,
                cid: UUID = request.canvas_id,
                n: int = brick_count,
                own: list[UUID] = own_slice_id,
            ) -> None:
                # Stop tracking the request that just finished, but only while
                # the entry still names *this* task.  A non-cancellable visual
                # is left running when a newer reslice supersedes it, so the
                # older task can finish after the newer one was recorded here;
                # popping blindly would untrack work that is still in flight.
                key = (sid, cid, vid)
                if own and self._active_slice_ids.get(key) == own[0]:
                    del self._active_slice_ids[key]
                self._emit_reslice_completed(sid, cid, vid, n)

            slice_id = self._slicer.submit(
                chunk_requests,
                fetch_fn=data_store.get_data,
                callback=callback,
                consumer_id=str(visual_id),
                on_complete=_on_complete,
            )
            if slice_id is not None:
                own_slice_id.append(slice_id)
                self._active_slice_ids[
                    (request.scene_id, request.canvas_id, visual_id)
                ] = slice_id
            else:
                # No bricks to load (e.g. an empty slab); the reslice for this
                # visual is already complete, so signal it immediately — the
                # async on_complete path never runs for an empty submission.
                self._emit_reslice_completed(
                    request.scene_id, request.canvas_id, visual_id, brick_count
                )

    def _emit_reslice_completed(
        self, scene_id: UUID, canvas_id: UUID, visual_id: UUID, brick_count: int
    ) -> None:
        """Emit a ``ResliceCompletedEvent`` for *visual_id* if a bus is wired.

        Parameters
        ----------
        scene_id : UUID
            Scene that owns the visual.
        canvas_id : UUID
            Canvas whose reslicing request produced this completion.  A scene
            with multiple canvases emits one completion per canvas, so
            quiescence trackers can distinguish them.
        visual_id : UUID
            Visual whose reslice cycle just completed.  Used by the bus as the
            routing key for ``on_reslice_completed`` subscribers.
        brick_count : int
            Number of bricks/tiles committed during the cycle.
        """
        if self._event_bus is None:
            return
        self._event_bus.emit(
            ResliceCompletedEvent(
                source_id=self.id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                visual_id=visual_id,
                brick_count=brick_count,
            )
        )

    def cancel_all(self) -> None:
        """Cancel every in-flight slice task, across every scene.

        ``cancel_scene`` can only reach what ``_active_slice_ids`` still names,
        which at teardown is not everything: ``submit`` deliberately lets a
        non-cancellable visual finish rather than cancelling it, then
        overwrites that key, so the predecessor task becomes unreachable from
        here.  Cancelling the tracked requests first keeps the per-visual GPU
        slot release running for them; the sweep afterwards catches the rest.
        """
        for scene_id, canvas_id, visual_id in list(self._active_slice_ids):
            # Pops its own key, so the loop drains _active_slice_ids.
            self.cancel_visual(scene_id, canvas_id, visual_id)
        self._slicer.cancel_all()

    def cancel_scene(self, scene_id: UUID) -> None:
        """Cancel all in-flight tasks for a scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose tasks should be cancelled.
        """
        keys = [
            (s_id, c_id, v_id)
            for (s_id, c_id, v_id) in list(self._active_slice_ids.keys())
            if s_id == scene_id
        ]
        for s_id, c_id, v_id in keys:
            self.cancel_visual(s_id, c_id, v_id)

    def cancel_visual(self, scene_id: UUID, canvas_id: UUID, visual_id: UUID) -> None:
        """Cancel the in-flight task for one visual on one canvas.

        Also calls ``visual.cancel_pending()`` or ``cancel_pending_2d()``
        to release any GPU slots reserved during the last planning phase
        that were never committed.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene containing the visual.
        canvas_id : UUID
            ID of the canvas whose request should be cancelled.
        visual_id : UUID
            ID of the visual to cancel.
        """
        key = (scene_id, canvas_id, visual_id)
        slice_id = self._active_slice_ids.pop(key, None)
        if slice_id is not None:
            self._slicer.cancel(slice_id)

        scene = self._scenes.get(scene_id)
        if scene is not None:
            try:
                visual = scene.get_visual(visual_id)
                if "2d" in visual.render_modes:
                    visual.cancel_pending_2d()
                if "3d" in visual.render_modes:
                    visual.cancel_pending()
            except KeyError:
                pass

    # ── EventBus handler methods ─────────────────────────────────────────

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        """Invalidate stale 2D tile caches when the slice position changes.

        Called synchronously by the EventBus before the controller's own
        reslice handler, ensuring stale tiles are evicted before new slice
        requests are submitted.

        When ``displayed_axes_changed`` is True the geometry rebuild path
        (``_rebuild_2d_resources``) already resets the cache; no extra
        action is needed here.

        When only ``slice_indices`` changed, every 2D-capable visual in
        the scene has its committed tile cache cleared and its LUT
        indirection table rebuilt against the empty cache.
        """
        if event.displayed_axes_changed or not event.region_changed:
            return

        scene = self._scenes.get(event.scene_id)
        if scene is None:
            return

        for visual_id in list(scene.visual_ids):
            try:
                visual = scene.get_visual(visual_id)
            except KeyError:
                continue
            if "2d" in visual.render_modes and hasattr(visual, "invalidate_2d_cache"):
                visual.invalidate_2d_cache()

    def _on_appearance_changed(self, event) -> None:
        """Dormant stub — not yet subscribed to the bus."""
        pass
