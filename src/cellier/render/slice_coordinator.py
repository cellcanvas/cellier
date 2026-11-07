"""SliceCoordinator: plans every visual and hands it to the chunk scheduler."""

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
from cellier.render.scheduling import PlanMode

if TYPE_CHECKING:
    from cellier.data.image import MultiscaleZarrDataStore
    from cellier.events._bus import EventBus
    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.scene_manager import SceneManager
    from cellier.render.scheduling import ChunkScheduler, DesiredSet


class SliceCoordinator:
    """Thin orchestrator owned by ``RenderManager``.

    Given a ``ReslicingRequest``, it looks up the target ``SceneManager``,
    plans every targeted visual into desired sets and hands them to the
    shared :class:`~cellier.render.scheduling.ChunkScheduler`, which never
    cancels (``plans/scheduler_only_loading_design_v2.md``; every visual is
    a chunked visual).  The coordinator keeps the scheduler's cache
    registrations in step with each visual's caches, retires the caches of
    a visual that draws nothing (and those of the mode not drawn), and turns
    the scheduler's per-cache completion into ``ResliceCompletedEvent``, and
    its per-cache progress into ``ResliceProgressEvent`` and
    ``BackstopCompleteEvent`` per visual (progressive loading design v3,
    5.13).

    With two canvases on one scene, a visual's caches follow the last plan
    made, whichever canvas made it.

    Parameters
    ----------
    scenes : dict[UUID, SceneManager]
        Shared scene registry from ``RenderManager``.
    data_stores : dict[UUID, MultiscaleZarrDataStore]
        Mapping of ``visual_model_id`` to the data store for that visual.
    scheduler : ChunkScheduler
        The chunk scheduler every visual loads through.
    """

    def __init__(
        self,
        scenes: dict[UUID, SceneManager],
        data_stores: dict[UUID, MultiscaleZarrDataStore],
        scheduler: ChunkScheduler,
    ) -> None:
        self.id: UUID = uuid4()
        self._scenes = scenes
        self._data_stores = data_stores
        self._scheduler = scheduler
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
        """Plan the targeted visuals of ``request.scene_id`` and pass them on.

        A visual that plans nothing (hidden, no data here, a slice that
        misses the data) has its caches retired; a visual whose plan mode is
        ``BACKSTOP_ONLY`` is re-planned when the motion ends.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request to process.
        visual_configs : dict[UUID, VisualRenderConfig]
            Per-visual render configuration.
        """
        scene_manager = self._scenes[request.scene_id]
        # A pass retires the visual's caches it did not plan: the 3D ones in
        # 2D, and the other way round.
        planned, idle = scene_manager.plan_chunked(request, visual_configs)

        # Announce the start of the reslice cycle for the visuals being loaded.
        started = frozenset(planned)
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
            if (cfg is not None and cfg.plan_mode is PlanMode.BACKSTOP_ONLY) or (
                # A MaskedImage's surface target waiting for its coarse mask
                # (masked image design 8.3).
                getattr(scene_manager.get_visual(visual_id), "target_deferred", False)
            ):
                self._deferred.add(visual_id)
            else:
                self._deferred.discard(visual_id)
            self._pass_visual(request, visual_id, desired_sets)
        for visual_id in idle:
            self._deferred.discard(visual_id)
            self._retire_visual(request.scene_id, visual_id)

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
        # A visual that reads through a source of its own (an in-memory
        # visual's brick pyramid, oblique 2D design v3 5.4) names it.
        sets = [
            ds if ds.store is not None else dataclasses.replace(ds, store=store)
            for ds in desired_sets
        ]
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
                n_bricks += sum(needed for _, needed, _ in progress.intermediate)
        for scene_id, canvas_id in announced:
            self._emit_reslice_completed(scene_id, canvas_id, visual_id, n_bricks)

    def visual_progress(self, visual_id: UUID) -> LoadingProgress | None:
        """A chunked visual's progress, summed over its drawn atlases.

        Retired atlases (a hidden visual, an undrawn channel, the mode not
        drawn) are left out: nothing of theirs is wanted.  ``None`` for a
        visual that has no atlas in the scheduler.
        """
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
            intermediate=total.intermediate,
            truncated_intermediate=total.truncated_intermediate,
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
