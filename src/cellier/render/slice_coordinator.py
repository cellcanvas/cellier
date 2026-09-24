"""SliceCoordinator — thin orchestrator that drives the async reslicing cycle."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from cellier.events._events import (
    BackstopCompleteEvent,
    LoadingProgress,
    ResliceCompletedEvent,
    ResliceProgressEvent,
    ResliceStartedEvent,
)
from cellier.render.scheduling import PlanMode, is_chunked_visual

if TYPE_CHECKING:
    from cellier.data.image import MultiscaleZarrDataStore
    from cellier.events._bus import EventBus
    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.scene_manager import SceneManager
    from cellier.render.scheduling import ChunkScheduler, DesiredSet
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

    **Chunked visuals** (``is_chunked_visual``: the multiscale image and
    labels) take a different path, 2D and 3D (``plans/progressive_loading_
    design_v3.md`` 4.2): they are planned into desired sets and handed to the
    shared :class:`~cellier.render.scheduling.ChunkScheduler`, which never
    cancels.  The coordinator keeps the scheduler's cache registrations in
    step with each visual's atlases, retires the atlases of a visual that
    draws nothing (and those of the mode not drawn), and turns the
    scheduler's per-atlas completion into ``ResliceCompletedEvent``, and its
    per-atlas progress into ``ResliceProgressEvent`` and
    ``BackstopCompleteEvent`` per visual (design 5.13).

    Parameters
    ----------
    scenes : dict[UUID, SceneManager]
        Shared scene registry from ``RenderManager``.
    slicer : AsyncSlicer
        Shared async slicer instance.
    data_stores : dict[UUID, MultiscaleZarrDataStore]
        Mapping of ``visual_model_id`` to the data store for that visual.
    scheduler : ChunkScheduler or None
        The chunk scheduler.  ``None`` routes every visual to the slicer.
    """

    def __init__(
        self,
        scenes: dict[UUID, SceneManager],
        slicer: AsyncSlicer,
        data_stores: dict[UUID, MultiscaleZarrDataStore],
        scheduler: ChunkScheduler | None = None,
    ) -> None:
        self.id: UUID = uuid4()
        self._scenes = scenes
        self._slicer = slicer
        self._data_stores = data_stores
        self._scheduler = scheduler
        self._active_slice_ids: dict[tuple[UUID, UUID, UUID], UUID] = {}
        # Chunked visuals: registered atlases per visual, and the reverse.
        self._visual_caches: dict[UUID, set[int]] = {}
        self._cache_owner: dict[int, UUID] = {}
        # Atlases retired and not passed since (retiring again is a no-op).
        self._retired: set[int] = set()
        # ``(scene_id, canvas_id)`` of every ResliceStartedEvent a chunked
        # visual announced since its last completion.  A completion answers
        # them all: passes coalesce, and a superseded pass never completes.
        self._announced: dict[UUID, list[tuple[UUID, UUID]]] = {}
        # Visuals whose progress changed since the last flush, which runs
        # once per loop iteration (so at most once per commit round).
        self._progress_dirty: set[UUID] = set()
        self._progress_handle: asyncio.Handle | None = None
        # Visuals whose latest pass planned the backstop only (a dims drag).
        self._deferred: set[UUID] = set()
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
        is_2d = len(request.dims_state.selection.displayed_axes) == 2

        if request.target_visual_ids is not None:
            to_cancel: frozenset[UUID] = request.target_visual_ids
        else:
            to_cancel = frozenset(scene_manager.visual_ids)

        chunked = {
            visual_id
            for visual_id in to_cancel
            if self._scheduler is not None
            and visual_id in scene_manager.visual_ids
            and is_chunked_visual(scene_manager.get_visual(visual_id))
        }
        for visual_id in to_cancel:
            if visual_id in chunked:
                continue  # the scheduler never cancels
            try:
                gfx_visual = scene_manager.get_visual(visual_id)
            except KeyError:
                # Visual not yet registered in render layer; cancel defensively.
                self.cancel_visual(request.scene_id, request.canvas_id, visual_id)
                continue
            if gfx_visual.cancellable:
                self.cancel_visual(request.scene_id, request.canvas_id, visual_id)

        requests_by_visual = scene_manager.build_slice_requests(request, visual_configs)
        planned: dict[UUID, list[DesiredSet]] = {}
        idle: set[UUID] = set()
        if chunked:
            # A pass retires the visual's atlases it did not plan: the 3D
            # ones in 2D, and the other way round.
            planned, idle = scene_manager.plan_chunked(request, visual_configs)

        # Announce the start of the reslice cycle for the visuals being loaded.
        started = frozenset(requests_by_visual) | frozenset(planned)
        if self._event_bus is not None and started:
            self._event_bus.emit(
                ResliceStartedEvent(
                    source_id=self.id,
                    scene_id=request.scene_id,
                    canvas_id=request.canvas_id,
                    visual_ids=started,
                )
            )

        for visual_id, desired_sets in planned.items():
            cfg = visual_configs.get(visual_id)
            if cfg is not None and cfg.plan_mode is PlanMode.BACKSTOP_ONLY:
                self._deferred.add(visual_id)
            else:
                self._deferred.discard(visual_id)
            self._pass_visual(request, visual_id, desired_sets)
        for visual_id in idle:
            self._deferred.discard(visual_id)
            self._retire_visual(request.scene_id, visual_id)

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
                if is_chunked_visual(visual):
                    return  # the scheduler never cancels
                if "2d" in visual.render_modes:
                    visual.cancel_pending_2d()
                if "3d" in visual.render_modes:
                    visual.cancel_pending()
            except KeyError:
                pass

    # ── Chunked visuals ──────────────────────────────────────────────────

    def _sync_caches(self, scene_id: UUID, visual_id: UUID, visual) -> set[int]:
        """Register the visual's atlases, and remove the ones it replaced.

        An atlas is replaced when its geometry or layout changes (a new
        ``Residency``, a new cache id); removing the old registration first
        keeps the scheduler from holding a dead atlas (design 5.4).
        """
        scheduler = self._scheduler
        current = dict(visual.residencies())
        known = self._visual_caches.setdefault(visual_id, set())
        for cache_id in known - set(current):
            self._forget_cache(cache_id)
        for cache_id, residency in current.items():
            if cache_id not in known:
                scheduler.register(cache_id, residency, scene_id)
                known.add(cache_id)
                self._cache_owner[cache_id] = visual_id
        return set(current)

    def _forget_cache(self, cache_id: int) -> None:
        self._scheduler.remove(cache_id)
        owner = self._cache_owner.pop(cache_id, None)
        if owner is not None:
            self._visual_caches.get(owner, set()).discard(cache_id)
        self._retired.discard(cache_id)

    def _pass_visual(
        self,
        request: ReslicingRequest,
        visual_id: UUID,
        desired_sets: list[DesiredSet],
    ) -> None:
        """Hand one visual's desired sets to the scheduler, retiring the rest."""
        visual = self._scenes[request.scene_id].get_visual(visual_id)
        current = self._sync_caches(request.scene_id, visual_id, visual)
        store = self._data_stores[visual_id]
        sets = [dataclasses.replace(ds, store=store) for ds in desired_sets]
        passed = {ds.cache_id for ds in sets}
        self._retired -= passed
        # Announce before passing: completion may fire during the pass.
        self._announced.setdefault(visual_id, []).append(
            (request.scene_id, request.canvas_id)
        )
        self._scheduler.pass_(sets)
        for cache_id in current - passed:
            self._retire_cache(cache_id)

    def _retire_visual(self, scene_id: UUID, visual_id: UUID) -> None:
        """A chunked visual draws nothing now: its atlases stop fetching."""
        scene = self._scenes.get(scene_id)
        if scene is None or visual_id not in scene.visual_ids:
            return
        for cache_id in self._sync_caches(
            scene_id, visual_id, scene.get_visual(visual_id)
        ):
            self._retire_cache(cache_id)

    def _retire_cache(self, cache_id: int) -> None:
        if cache_id in self._retired:
            return
        self._retired.add(cache_id)
        self._scheduler.retire(cache_id)

    def forget_visual(self, visual_id: UUID) -> None:
        """Drop a chunked visual's atlases from the scheduler (visual removal)."""
        if self._scheduler is None:
            return
        for cache_id in list(self._visual_caches.pop(visual_id, set())):
            self._forget_cache(cache_id)
        self._announced.pop(visual_id, None)
        self._progress_dirty.discard(visual_id)
        self._deferred.discard(visual_id)

    def on_cache_complete(self, cache_id: int, generation: int) -> None:
        """Scheduler callback: a cache's latest pass is resident or given up.

        When every atlas of the owning visual is complete, answer each
        ``ResliceStartedEvent`` announced for it since the last completion.
        """
        visual_id = self._cache_owner.get(cache_id)
        if visual_id is None or not self._announced.get(visual_id):
            return
        core = self._scheduler.core
        caches = self._visual_caches.get(visual_id, set())
        if not all(core.is_complete(cid) for cid in caches if core.is_registered(cid)):
            return
        announced = self._announced.pop(visual_id)
        n_bricks = 0
        for cid in caches:
            if core.is_registered(cid):
                progress = core.progress(cid)
                n_bricks += progress.needed_backstop + progress.needed_target
        for scene_id, canvas_id in announced:
            self._emit_reslice_completed(scene_id, canvas_id, visual_id, n_bricks)

    def visual_progress(self, visual_id: UUID) -> LoadingProgress | None:
        """A chunked visual's progress, summed over its drawn atlases.

        Retired atlases (a hidden visual, an undrawn channel, the mode not
        drawn) are left out: nothing of theirs is wanted.  ``None`` for a
        visual that has no atlas in the scheduler.
        """
        if self._scheduler is None:
            return None
        core = self._scheduler.core
        caches = [
            cid
            for cid in self._visual_caches.get(visual_id, ())
            if core.is_registered(cid) and cid not in self._retired
        ]
        if not caches:
            return None
        total = core.progress(caches[0])
        for cid in caches[1:]:
            total = total + core.progress(cid)
        return LoadingProgress(
            needed_backstop=total.needed_backstop,
            resident_backstop=total.resident_backstop,
            needed_target=total.needed_target,
            resident_target=total.resident_target,
            in_flight=total.in_flight,
            failed=total.failed,
            truncated_target=total.truncated_target,
            truncated_backstop=total.truncated_backstop,
            backstop_complete=all(core.is_backstop_complete(c) for c in caches),
            complete=all(core.is_complete(c) for c in caches),
            target_deferred=visual_id in self._deferred,
        )

    def _visual_scene(self, visual_id: UUID) -> UUID | None:
        for scene_id, scene in self._scenes.items():
            if visual_id in scene.visual_ids:
                return scene_id
        return None

    def on_cache_progress(self, cache_id: int) -> None:
        """Scheduler callback: a cache's counts may have changed.

        Marks the owning visual and flushes on the next loop iteration, so a
        commit round over several atlases (or a pass and a round in one
        iteration) emits one ``ResliceProgressEvent`` per visual.
        """
        visual_id = self._cache_owner.get(cache_id)
        if visual_id is None or self._event_bus is None:
            return
        self._progress_dirty.add(visual_id)
        if self._progress_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._flush_progress()
            return
        self._progress_handle = loop.call_soon(self._flush_progress)

    def _flush_progress(self) -> None:
        self._progress_handle = None
        dirty, self._progress_dirty = self._progress_dirty, set()
        if self._event_bus is None:
            return
        for visual_id in dirty:
            progress = self.visual_progress(visual_id)
            scene_id = self._visual_scene(visual_id)
            if progress is None or scene_id is None:
                continue
            self._event_bus.emit(
                ResliceProgressEvent(
                    source_id=self.id,
                    scene_id=scene_id,
                    visual_id=visual_id,
                    progress=progress,
                )
            )

    def on_cache_backstop_complete(self, cache_id: int, generation: int) -> None:
        """Scheduler callback: a cache's backstop is resident or given up.

        Emits ``BackstopCompleteEvent`` once the backstops of every drawn
        atlas of the visual are, and only when the plan has a backstop.
        """
        visual_id = self._cache_owner.get(cache_id)
        if visual_id is None or self._event_bus is None:
            return
        if cache_id in self._retired:
            return
        progress = self.visual_progress(visual_id)
        if progress is None or not progress.backstop_complete:
            return
        if progress.needed_backstop == 0:
            return
        scene_id = self._visual_scene(visual_id)
        if scene_id is None:
            return
        self._event_bus.emit(
            BackstopCompleteEvent(
                source_id=self.id, scene_id=scene_id, visual_id=visual_id
            )
        )

    # ── EventBus handler methods ─────────────────────────────────────────

    def _on_appearance_changed(self, event) -> None:
        """Dormant stub — not yet subscribed to the bus."""
        pass
