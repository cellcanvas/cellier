"""The chunk scheduler on asyncio.

Design: ``plans/progressive_loading_design_v3.md`` sections 5.4-5.7.
:class:`ChunkScheduler` wraps :class:`~cellier.render.scheduling._core.
SchedulerCore` with the parts that need an event loop:

- passes are coalesced: every :meth:`~ChunkScheduler.pass_` in one loop
  iteration is applied by a single ``call_soon`` callback;
- each issued read is a task awaiting ``store.get_data(request)``.  Reads
  are never cancelled, except by :meth:`~ChunkScheduler.close`;
- a failed read is retried from a timer;
- arrivals are committed by :meth:`~ChunkScheduler.commit_round`, which a
  canvas's ``before_draw`` hook calls with its scene (wired in by the render
  manager), or by a fallback timer once an arrival has waited
  ``commit_fallback_s`` with no round.

Everything runs on the loop's thread: there are no locks.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from cellier.logging import _SCHEDULER_LOGGER
from cellier.render._config import SchedulerConfig
from cellier.render.scheduling._core import ALL_SCENES, ReadOutcome, SchedulerCore

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from cellier.render.scheduling._types import (
        CacheProgress,
        DesiredSet,
        ReadTicket,
        Residency,
    )

_LOGGER = _SCHEDULER_LOGGER


class ChunkScheduler:
    """One global fetch budget and commit loop for every chunked cache.

    Parameters
    ----------
    config : SchedulerConfig | None
        Capacities, retry policy and the commit fallback.  Defaults to
        ``SchedulerConfig()``.
    request_draw : Callable[[Hashable | None], None] | None
        Asks every canvas of a scene to draw; called when data for that
        scene arrives, and whenever the scheduler rebuilds a cache's draw
        outside a frame (a pass, an invalidation, a given-up read, a
        fallback round).  The canvases coalesce repeated requests.
    on_complete : Callable[[int, int], None] | None
        ``(cache_id, generation)`` when a cache's latest pass is complete.
    on_backstop_complete : Callable[[int, int], None] | None
        ``(cache_id, generation)`` when its backstop is.
    on_progress : Callable[[int], None] | None
        ``cache_id`` whenever its counts may have changed (a pass, a commit
        round that touched it, a given-up read, an invalidation).
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        request_draw: Callable[[Hashable | None], None] | None = None,
        on_complete: Callable[[int, int], None] | None = None,
        on_backstop_complete: Callable[[int, int], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.config = config if config is not None else SchedulerConfig()
        self.core = SchedulerCore(self.config)
        self.core.on_complete = on_complete
        self.core.on_backstop_complete = on_backstop_complete
        self.core.on_progress = on_progress
        self._request_draw = request_draw
        self._tasks: set[asyncio.Task] = set()
        self._process_handle: asyncio.Handle | None = None
        self._fallback_handle: asyncio.TimerHandle | None = None
        self._retry_handle: asyncio.TimerHandle | None = None
        self._retry_at: float | None = None
        self._closed = False

    # -- caches --------------------------------------------------------------

    def register(
        self, cache_id: int, residency: Residency, scene: Hashable | None = None
    ) -> None:
        """Start scheduling a cache; see ``SchedulerCore.register``."""
        self.core.register(cache_id, residency, scene)

    def remove(self, cache_id: int) -> None:
        """Forget a cache and every reference to its ``Residency`` (5.4).

        Reads in flight for it land and are dropped.  Call it before
        registering a replacement under the same id.
        """
        self.core.remove(cache_id)

    def retire(self, cache_id: int) -> None:
        """Stop fetching for a cache that is no longer drawn (5.4).

        An empty pass, coalesced like any other: queued reads are deleted,
        everything else becomes ``RECENT``, residents stay.
        """
        self.core.retire(cache_id)
        self._schedule_process()

    # -- passes --------------------------------------------------------------

    def pass_(self, desired_sets: Iterable[DesiredSet]) -> None:
        """Record the latest desired set of each cache and schedule a pass.

        Every call in one loop iteration is applied together, so bursts of
        triggers coalesce.  Validation errors raise here, synchronously.
        """
        for desired in desired_sets:
            self.core.set_desired(desired)
        self._schedule_process()

    def _schedule_process(self) -> None:
        if self._closed or self._process_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop (a synchronous caller): apply the pass now so the
            # registry and the draw follow; reads start once a loop runs and
            # something pumps (the next pass, invalidation or retry).
            self.core.process()
            return
        self._process_handle = loop.call_soon(self._process)

    def _process(self) -> None:
        self._process_handle = None
        if self._closed:
            return
        passed = self.core.process()
        self._pump()
        self._draw_caches(passed)

    # -- invalidation --------------------------------------------------------

    def invalidate(self, store_id: Any, region: Any = None) -> list[int]:
        """Forget data read from a store in *region* (5.14).

        Takes effect immediately: touched caches are redrawn, and invalidated
        ``VISIBLE`` records are fetched again.

        Parameters
        ----------
        store_id : Any
            The store's id (``store_key(store)``).
        region : Any
            Handed to each cache's ``Residency.keys_in_region``; ``None`` is
            everything.

        Returns
        -------
        list[int]
            The caches touched.
        """
        touched = self.core.invalidate(store_id, region)
        if touched and not self._closed:
            self._pump()
            self._draw_caches(touched)
        return touched

    # -- fetching ------------------------------------------------------------

    def _pump(self) -> None:
        """Start every read the core will issue now, then re-arm the retry."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for ticket in self.core.next_reads():
            task = loop.create_task(self._read(ticket))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        self._arm_retry()

    async def _read(self, ticket: ReadTicket) -> None:
        data: Any = None
        error: BaseException | None = None
        try:
            data = await ticket.store.get_data(ticket.request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        if self._closed:
            return
        outcome = self.core.complete_read(ticket, data=data, error=error)
        # Refill before the commit side runs, so the window stays full.
        self._pump()
        if outcome == ReadOutcome.ARRIVED:
            self._draw(self.core.scene_of(ticket.cache_id))
            self._arm_fallback()
        elif outcome == ReadOutcome.GAVE_UP:
            # The draw was rebuilt (the background may have gone).
            self._draw(self.core.scene_of(ticket.cache_id))

    # -- retries -------------------------------------------------------------

    def _arm_retry(self) -> None:
        due = self.core.next_retry_at()
        if due is None:
            if self._retry_handle is not None:
                self._retry_handle.cancel()
                self._retry_handle = None
                self._retry_at = None
            return
        if self._retry_handle is not None and self._retry_at is not None:
            if self._retry_at <= due:
                return
            self._retry_handle.cancel()
        loop = asyncio.get_running_loop()
        delay = max(0.0, due - self.core.now())
        self._retry_at = due
        self._retry_handle = loop.call_later(delay, self._on_retry)

    def _on_retry(self) -> None:
        self._retry_handle = None
        self._retry_at = None
        if self._closed:
            return
        self.core.requeue_due()
        self._pump()

    # -- commit rounds ------------------------------------------------------

    def commit_round(self, scene: Hashable | None | object = ALL_SCENES) -> list[int]:
        """Commit what has arrived, for one scene or for every cache.

        A canvas's ``before_draw`` hook calls this with its scene, so the
        frame being drawn shows what was committed.  At most one round per
        drawn frame; nothing is time-budgeted (design 5.7, V5).

        Returns
        -------
        list[int]
            Caches that committed something.
        """
        if self._closed:
            return []
        return self.core.commit_round(scene)

    def _arm_fallback(self) -> None:
        if self._fallback_handle is not None or self._closed:
            return
        oldest = self.core.oldest_arrival()
        if oldest is None:
            return
        delay = max(0.0, oldest + self.config.commit_fallback_s - self.core.now())
        loop = asyncio.get_running_loop()
        self._fallback_handle = loop.call_later(delay, self._on_fallback)

    def _on_fallback(self) -> None:
        self._fallback_handle = None
        if self._closed:
            return
        oldest = self.core.oldest_arrival()
        if oldest is None:
            return
        # A round scoped to one scene resets only that scene's wait, so
        # re-check: fire only once an arrival has waited the full interval.
        if self.core.now() - oldest >= self.config.commit_fallback_s * 0.999:
            self._draw_caches(self.core.commit_round(ALL_SCENES))
        self._arm_fallback()

    def _draw_caches(self, cache_ids: Iterable[int]) -> None:
        """Request a draw of each scene whose caches' draw was rebuilt."""
        for scene in {self.core.scene_of(cid) for cid in cache_ids}:
            self._draw(scene)

    def _draw(self, scene: Hashable | None) -> None:
        if self._request_draw is None:
            return
        try:
            self._request_draw(scene)
        except Exception:
            _LOGGER.exception("request_draw failed for scene %s", scene)

    # -- introspection and lifecycle -------------------------------------------

    def progress(self, cache_id: int) -> CacheProgress:
        """Loading progress for one cache (design 5.13)."""
        return self.core.progress(cache_id)

    def idle(self) -> bool:
        """Nothing scheduled, in flight, waiting to commit, or to retry."""
        return self._process_handle is None and self.core.idle()

    async def drain(self, timeout_s: float = 30.0, *, commit: bool = False) -> None:
        """Wait until :meth:`idle` (reads land, commit rounds run).

        Parameters
        ----------
        timeout_s : float
            Give up after this long.
        commit : bool
            Run a commit round over every cache on each poll, as a drawing
            canvas would, instead of waiting for the fallback timer.

        Raises
        ------
        TimeoutError
            If the scheduler is still busy after *timeout_s*.
        """
        async with asyncio.timeout(timeout_s):
            while not self.idle():
                if commit:
                    self.commit_round()
                await asyncio.sleep(0.001)

    def close(self) -> None:
        """Cancel reads and timers, and drop every cache.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        for handle in (self._process_handle, self._fallback_handle, self._retry_handle):
            if handle is not None:
                handle.cancel()
        self._process_handle = self._fallback_handle = self._retry_handle = None
        self.core.clear()
