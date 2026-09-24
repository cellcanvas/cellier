"""The chunk scheduler's state machine, without an event loop.

Design: ``plans/progressive_loading_design_v3.md`` sections 5.2-5.7, 5.13
and 5.14.  :class:`SchedulerCore` owns one registry per cache and makes every
decision: what a pass keeps and drops, which read to issue next, what an
arrival becomes, which slot a commit takes and what it evicts, and when a
cache's view is complete.  It never awaits anything and never reads a clock
on its own: the caller passes ``now``, starts the reads it returns, and
reports each result back.  :class:`~cellier.render.scheduling.ChunkScheduler`
is that caller on asyncio; the property test drives the core directly with
a simulated clock.

The order of ``RECENT`` records (design 5.5), most important first::

    (cls desc, wanted_gen desc, rank asc)

Every ``VISIBLE`` record outranks every ``RECENT`` one.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.logging import _SCHEDULER_LOGGER
from cellier.render.scheduling._registry import DEAD, CacheRegistry
from cellier.render.scheduling._types import (
    CacheProgress,
    ChunkClass,
    ChunkState,
    DesiredSet,
    ReadTicket,
    Residency,
    Tier,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from cellier.render._config import SchedulerConfig

_LOGGER = _SCHEDULER_LOGGER

#: Capacity lanes: the shared window, and the backstop-only lane.
SHARED_LANE: int = 0
BACKSTOP_LANE: int = 1

_QUEUED = int(ChunkState.QUEUED)
_FETCHING = int(ChunkState.FETCHING)
_ARRIVED = int(ChunkState.ARRIVED)
_RESIDENT = int(ChunkState.RESIDENT)
_FAILED = int(ChunkState.FAILED)
_VISIBLE = int(Tier.VISIBLE)
_RECENT = int(Tier.RECENT)
_BACKSTOP = int(ChunkClass.BACKSTOP)

#: Pass as ``scene`` to commit every cache: the fallback round.
ALL_SCENES: object = object()


class ReadOutcome:
    """What :meth:`SchedulerCore.complete_read` did with a result."""

    #: The data is waiting for a commit round.
    ARRIVED = "arrived"
    #: The read failed and will be retried on a timer.
    RETRY = "retry"
    #: The read failed for the last time; the record is given up.
    GAVE_UP = "gave_up"
    #: The result was dropped: stale, unwanted and failed, or its cache is gone.
    DROPPED = "dropped"


def store_key(store: Any) -> Any:
    """The id :meth:`SchedulerCore.invalidate` knows *store* by.

    The store's ``id`` (every Cellier data store has one), or ``id(store)``
    for an object without it.
    """
    store_id = getattr(store, "id", None)
    return store_id if store_id is not None else id(store)


@dataclass(eq=False)
class _Cache:
    """The scheduler's state for one cache."""

    cache_id: int
    residency: Residency
    scene: Hashable | None
    n_slots: int
    token: object = field(default_factory=object)
    registry: CacheRegistry = field(default_factory=CacheRegistry)
    free: list[int] = field(default_factory=list)
    generation: int = 0
    completed_gen: int = 0
    backstop_gen: int = 0
    store: Any = None
    store_id: Any = None
    build_request: Callable[[np.ndarray], Any] | None = None
    truncated_target: int = 0
    truncated_backstop: int = 0
    queue: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    cursor: int = 0
    queue_dirty: bool = False
    served: int = -1
    n_fetching: int = 0
    arrived: dict[int, Any] = field(default_factory=dict)
    arrived_since: float | None = None
    last_pass: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        # A stack: the lowest slot is handed out first.
        self.free = list(range(self.n_slots - 1, -1, -1))


def _snapshot(reg: CacheRegistry, row: int) -> tuple[int, int, int, int, int]:
    """``(key, tier, cls, wanted_gen, rank)`` of *row*, for traces."""
    return (
        int(reg.key[row]),
        int(reg.tier[row]),
        int(reg.cls[row]),
        int(reg.wanted_gen[row]),
        int(reg.rank[row]),
    )


def recent_importance(cls: int, wanted_gen: int, rank: int) -> tuple[int, int, int]:
    """The ``RECENT`` order as a tuple; larger is more important."""
    return (cls, wanted_gen, -rank)


class SchedulerCore:
    """Registries, passes, fetch selection, commit rounds and progress.

    Parameters
    ----------
    config : SchedulerConfig
        Capacities and the retry policy.
    now : Callable[[], float]
        The clock, in seconds.  Monotonic by default.

    Attributes
    ----------
    on_complete : Callable[[int, int], None] | None
        ``(cache_id, generation)``, once per generation, when every
        ``VISIBLE`` record of the cache is resident or given up.
    on_backstop_complete : Callable[[int, int], None] | None
        The same, for the ``VISIBLE`` backstop records.
    on_progress : Callable[[int], None] | None
        ``cache_id``, whenever its counts may have changed: after a pass, a
        commit round that touched it, a read given up, or an invalidation.
        Consumers coalesce (design 5.13).
    trace : Callable[[str, tuple], None] | None
        Test hook: every decision, as ``(kind, payload)``.  ``None`` costs
        nothing.
    in_flight : list[int]
        Reads outstanding per lane: ``[shared, backstop]``.
    """

    def __init__(
        self, config: SchedulerConfig, now: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self.now = now
        self.in_flight: list[int] = [0, 0]
        self.on_complete: Callable[[int, int], None] | None = None
        self.on_backstop_complete: Callable[[int, int], None] | None = None
        self.on_progress: Callable[[int], None] | None = None
        self.trace: Callable[[str, tuple], None] | None = None
        self._caches: dict[int, _Cache] = {}
        self._pending: dict[int, DesiredSet | None] = {}
        self._serve_clock = 0

    # -- caches --------------------------------------------------------------

    def register(
        self, cache_id: int, residency: Residency, scene: Hashable | None = None
    ) -> None:
        """Start scheduling for a cache.

        Parameters
        ----------
        cache_id : int
            The id the cache's desired sets carry.
        residency : Residency
            The cache's adapter.  ``n_slots`` is read once, here.
        scene : Hashable | None
            The scene whose canvases draw the cache; commit rounds are scoped
            to it.

        Raises
        ------
        ValueError
            If *cache_id* is registered already, or the cache has fewer than
            two slots.
        """
        if cache_id in self._caches:
            raise ValueError(f"cache {cache_id} is already registered.")
        n_slots = int(residency.n_slots)
        if n_slots < 2:
            raise ValueError(f"a cache needs at least 2 slots, got {n_slots}.")
        self._caches[cache_id] = _Cache(cache_id, residency, scene, n_slots)

    def remove(self, cache_id: int) -> None:
        """Forget a cache: its registry, queue, arrivals and adapter.

        A read in flight for it is dropped when it lands.  A no-op for an
        unknown id.
        """
        self._caches.pop(cache_id, None)
        self._pending.pop(cache_id, None)

    def is_registered(self, cache_id: int) -> bool:
        """Whether *cache_id* is registered."""
        return cache_id in self._caches

    def scene_of(self, cache_id: int) -> Hashable | None:
        """The scene *cache_id* was registered with."""
        cache = self._caches.get(cache_id)
        return None if cache is None else cache.scene

    @property
    def cache_ids(self) -> list[int]:
        """Registered caches."""
        return list(self._caches)

    # -- passes (5.4) ----------------------------------------------------------

    def set_desired(self, desired: DesiredSet) -> None:
        """Record the latest desired set for its cache; applied by :meth:`process`.

        Raises
        ------
        KeyError
            If the cache is not registered.
        ValueError
            If the keys repeat, or there are more than ``n_slots - 1`` of them
            (the planner must truncate, design 5.3).
        """
        cache = self._caches.get(desired.cache_id)
        if cache is None:
            raise KeyError(f"cache {desired.cache_id} is not registered.")
        n = len(desired.keys)
        if n > cache.n_slots - 1:
            raise ValueError(
                f"cache {desired.cache_id}: {n} desired keys for {cache.n_slots} "
                f"slots; a planner must truncate to n_slots - 1."
            )
        if n > 1 and len(np.unique(desired.keys)) != n:
            raise ValueError(f"cache {desired.cache_id}: desired keys repeat.")
        self._pending[desired.cache_id] = desired

    def retire(self, cache_id: int) -> None:
        """Queue an empty pass for *cache_id* (design 5.4).

        Queued records are deleted and everything else becomes ``RECENT``;
        residents stay until the ``RECENT`` order evicts them.
        """
        if cache_id in self._caches:
            self._pending[cache_id] = None

    @property
    def has_pending(self) -> bool:
        """Whether passes are waiting for :meth:`process`."""
        return bool(self._pending)

    def process(self) -> list[int]:
        """Apply every waiting pass, then redraw each cache it touched.

        Returns
        -------
        list[int]
            The caches passed.
        """
        pending, self._pending = self._pending, {}
        passed: list[int] = []
        for cache_id, desired in pending.items():
            cache = self._caches.get(cache_id)
            if cache is None:
                continue
            self._apply(cache, desired)
            self._settle(cache)
            passed.append(cache_id)
            if _LOGGER.isEnabledFor(logging.INFO):
                n_desired, n_new = cache.last_pass
                _LOGGER.info(
                    "pass  cache=%s gen=%d desired=%d new=%d%s",
                    cache_id,
                    cache.generation,
                    n_desired,
                    n_new,
                    "  (retired)" if desired is None else "",
                )
        return passed

    def _apply(self, cache: _Cache, desired: DesiredSet | None) -> None:
        """Diff one desired set against the cache's registry (design 5.4)."""
        reg = cache.registry
        reg.compact()
        cache.generation += 1
        gen = cache.generation
        if desired is None:
            keys = np.empty(0, np.int64)
            cls = np.empty(0, np.uint8)
            slice_ids = np.empty(0, np.int32)
            cache.truncated_target = cache.truncated_backstop = 0
        else:
            keys, cls, slice_ids = desired.keys, desired.cls, desired.slice_ids
            cache.store = desired.store
            cache.store_id = store_key(desired.store)
            cache.build_request = desired.build_request
            cache.truncated_target = int(desired.n_truncated_target)
            cache.truncated_backstop = int(desired.n_truncated_backstop)
        rank = np.arange(len(keys), dtype=np.int32)

        rows, found = reg.find_many(keys)
        hit = rows[found]

        # Kept: re-prioritised.  A given-up record gets one more attempt.
        gave_up = (reg.state[hit] == _FAILED) & (
            reg.attempts[hit] >= self.config.retry_max_attempts
        )
        again = hit[gave_up]
        reg.state[again] = _QUEUED
        reg.attempts[again] = self.config.retry_max_attempts - 1
        reg.tier[hit] = _VISIBLE
        reg.cls[hit] = cls[found]
        reg.rank[hit] = rank[found]
        reg.slice_id[hit] = slice_ids[found]
        reg.wanted_gen[hit] = gen

        # Dropped: VISIBLE -> RECENT; unfetched and failed records go.
        wanted = np.zeros(len(reg.key), dtype=bool)
        wanted[hit] = True
        dropped = (reg.tier == _VISIBLE) & ~wanted & (reg.state != DEAD)
        reg.tier[dropped] = _RECENT
        doomed = dropped & ((reg.state == _QUEUED) | (reg.state == _FAILED))
        reg.kill(np.flatnonzero(doomed))

        # New: queued.
        new = ~found
        cache.last_pass = (len(keys), int(new.sum()))
        reg.insert(
            keys[new],
            {
                "cls": cls[new],
                "rank": rank[new],
                "slice_id": slice_ids[new],
                "wanted_gen": np.full(int(new.sum()), gen, np.int32),
            },
        )
        cache.queue_dirty = True
        if self.trace is not None:
            self.trace("pass", (cache.cache_id, gen, keys.copy(), cls.copy()))

    # -- fetching (5.5, 5.6) -----------------------------------------------------

    def _head(self, cache: _Cache) -> tuple[int, int] | None:
        """``(cls, key)`` of the cache's next read, or ``None``."""
        reg = cache.registry
        if cache.queue_dirty:
            rows = np.flatnonzero((reg.state == _QUEUED) & (reg.tier == _VISIBLE))
            order = np.lexsort((reg.rank[rows], -reg.cls[rows].astype(np.int16)))
            cache.queue = reg.key[rows[order]]
            cache.cursor = 0
            cache.queue_dirty = False
        while cache.cursor < len(cache.queue):
            key = int(cache.queue[cache.cursor])
            row = reg.find(key)
            if row >= 0 and reg.state[row] == _QUEUED and reg.tier[row] == _VISIBLE:
                return int(reg.cls[row]), key
            cache.cursor += 1
        return None

    def _lane_for(self, cls: int) -> int | None:
        if cls == _BACKSTOP and self.in_flight[BACKSTOP_LANE] < (
            self.config.backstop_reserved
        ):
            return BACKSTOP_LANE
        if self.in_flight[SHARED_LANE] < self.config.max_in_flight:
            return SHARED_LANE
        return None

    def next_reads(self) -> list[ReadTicket]:
        """Issue reads while capacity remains; the caller starts each one.

        Between caches, the head with the highest class goes first, ties to
        the cache served longest ago (round robin).  Within a cache, the
        planner's order.  A backstop read takes the backstop lane first and a
        shared slot after that; a target read only a shared slot.

        Returns
        -------
        list[ReadTicket]
            Reads now ``FETCHING``.  Report each through
            :meth:`complete_read`.
        """
        picks: dict[int, list[tuple[int, int]]] = defaultdict(list)
        while True:
            best: tuple[tuple[int, int], _Cache, int, int] | None = None
            for cache in self._caches.values():
                head = self._head(cache)
                if head is None:
                    continue
                score = (head[0], -cache.served)
                if best is None or score > best[0]:
                    best = (score, cache, head[0], head[1])
            if best is None:
                break
            _, cache, cls, key = best
            lane = self._lane_for(cls)
            if lane is None:
                # The best head cannot start, so none can: heads rank
                # class first, and a target needs what a backstop falls
                # back to.
                break
            reg = cache.registry
            row = reg.find(key)
            reg.state[row] = _FETCHING
            reg.attempts[row] += 1
            cache.cursor += 1
            cache.n_fetching += 1
            self.in_flight[lane] += 1
            self._serve_clock += 1
            cache.served = self._serve_clock
            picks[cache.cache_id].append((key, lane))
            if self.trace is not None:
                self.trace("issue", (cache.cache_id, key, cls, lane))
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "issue  cache=%s key=%d class=%d lane=%d",
                    cache.cache_id,
                    key,
                    cls,
                    lane,
                )
        tickets: list[ReadTicket] = []
        for cache_id, batch in picks.items():
            tickets.extend(self._tickets(self._caches[cache_id], batch))
        return tickets

    def _tickets(self, cache: _Cache, batch: list[tuple[int, int]]) -> list[ReadTicket]:
        """Build one batch of requests; a failing builder fails the reads."""
        keys = np.fromiter((k for k, _ in batch), dtype=np.int64, count=len(batch))
        try:
            requests = list(cache.build_request(keys))
            if len(requests) != len(batch):
                raise ValueError(
                    f"build_request returned {len(requests)} requests for "
                    f"{len(batch)} keys."
                )
        except Exception as error:
            _LOGGER.exception("cache %s: build_request failed", cache.cache_id)
            for key, lane in batch:
                ticket = ReadTicket(
                    cache.cache_id, key, None, cache.store, lane, cache.token
                )
                self.complete_read(ticket, error=error)
            return []
        return [
            ReadTicket(cache.cache_id, key, request, cache.store, lane, cache.token)
            for (key, lane), request in zip(batch, requests, strict=True)
        ]

    def complete_read(
        self, ticket: ReadTicket, data: Any = None, error: BaseException | None = None
    ) -> str:
        """Report a finished read (design 5.6).

        Parameters
        ----------
        ticket : ReadTicket
            From :meth:`next_reads`.
        data : Any
            The result, when the read succeeded.
        error : BaseException | None
            The exception, when it failed.

        Returns
        -------
        str
            A :class:`ReadOutcome` value.
        """
        self.in_flight[ticket.lane] -= 1
        cache = self._caches.get(ticket.cache_id)
        if cache is None or cache.token is not ticket.token:
            return ReadOutcome.DROPPED
        cache.n_fetching -= 1
        reg = cache.registry
        row = reg.find(ticket.key)
        if row < 0 or reg.state[row] != _FETCHING:
            raise AssertionError(
                f"cache {ticket.cache_id}: read landed for key {ticket.key} "
                f"that is not FETCHING"
            )
        if reg.stale[row]:
            # Invalidated in flight (5.14): never commit pre-invalidation data.
            reg.stale[row] = False
            reg.attempts[row] = 0
            if reg.tier[row] == _VISIBLE:
                reg.state[row] = _QUEUED
                cache.queue_dirty = True
            else:
                reg.kill(row)
            return ReadOutcome.DROPPED
        if error is not None:
            if reg.tier[row] != _VISIBLE:
                reg.kill(row)
                return ReadOutcome.DROPPED
            reg.state[row] = _FAILED
            attempts = int(reg.attempts[row])
            reg.retry_at[row] = self.now() + self.config.retry_backoff_s * 2 ** (
                attempts - 1
            )
            _LOGGER.warning(
                "cache %s: read failed (attempt %d of %d): %r",
                cache.cache_id,
                attempts,
                self.config.retry_max_attempts,
                error,
            )
            if attempts >= self.config.retry_max_attempts:
                # Trigger 4 (5.8): completion, and so the background, may change.
                self._settle(cache)
                return ReadOutcome.GAVE_UP
            return ReadOutcome.RETRY
        # 5.2: a good read is kept, wanted or not; the commit round decides.
        reg.state[row] = _ARRIVED
        cache.arrived[ticket.key] = data
        if cache.arrived_since is None:
            cache.arrived_since = self.now()
        if self.trace is not None:
            self.trace("arrive", (cache.cache_id, ticket.key, int(reg.tier[row])))
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "arrive  cache=%s key=%d wanted=%s",
                cache.cache_id,
                ticket.key,
                reg.tier[row] == _VISIBLE,
            )
        return ReadOutcome.ARRIVED

    # -- retry ---------------------------------------------------------------

    def next_retry_at(self) -> float | None:
        """When the earliest waiting retry is due, or ``None``."""
        soonest: float | None = None
        for cache in self._caches.values():
            reg = cache.registry
            waiting = (reg.state == _FAILED) & (
                reg.attempts < self.config.retry_max_attempts
            )
            if waiting.any():
                t = float(reg.retry_at[waiting].min())
                soonest = t if soonest is None else min(soonest, t)
        return soonest

    def requeue_due(self) -> int:
        """Queue every failed record whose retry is due.

        Returns
        -------
        int
            Records requeued.
        """
        now = self.now()
        n = 0
        for cache in self._caches.values():
            reg = cache.registry
            due = (
                (reg.state == _FAILED)
                & (reg.attempts < self.config.retry_max_attempts)
                & (reg.retry_at <= now)
            )
            k = int(due.sum())
            if k:
                reg.state[due] = _QUEUED
                cache.queue_dirty = True
                n += k
        return n

    # -- commit rounds (5.7) -----------------------------------------------------

    def oldest_arrival(self) -> float | None:
        """When the longest-waiting arrival landed, or ``None``."""
        times = [
            c.arrived_since
            for c in self._caches.values()
            if c.arrived_since is not None
        ]
        return min(times) if times else None

    def commit_round(self, scene: Hashable | None | object = ALL_SCENES) -> list[int]:
        """Commit every arrival, in one scene or in all of them.

        Parameters
        ----------
        scene : Hashable | None | object
            Commit only the caches registered with this scene (a canvas's
            frame round), or every cache for :data:`ALL_SCENES` (the
            fallback round, and the default).

        Returns
        -------
        list[int]
            The caches that committed something; each was redrawn once.
        """
        touched: list[int] = []
        for cache in list(self._caches.values()):
            if not cache.arrived or not (scene is ALL_SCENES or cache.scene == scene):
                continue
            if self._commit_cache(cache):
                touched.append(cache.cache_id)
            self._settle(cache)
        return touched

    def _commit_cache(self, cache: _Cache) -> int:
        """One cache's share of a round.  Returns records committed."""
        reg = cache.registry
        reg.compact()
        keys = np.fromiter(cache.arrived, dtype=np.int64, count=len(cache.arrived))
        rows, found = reg.find_many(keys)
        if not found.all() or (reg.state[rows] != _ARRIVED).any():
            raise AssertionError(f"cache {cache.cache_id}: arrival table out of sync")
        tier, cls = reg.tier[rows], reg.cls[rows].astype(np.int32)
        rank, gen = reg.rank[rows], reg.wanted_gen[rows]
        vis = np.flatnonzero(tier == _VISIBLE)
        rec = np.flatnonzero(tier != _VISIBLE)
        vis = vis[np.lexsort((rank[vis], -cls[vis]))]
        rec = rec[np.lexsort((rank[rec], -gen[rec], -cls[rec]))]
        order = rows[np.concatenate([vis, rec])]

        victims: np.ndarray | None = None
        next_victim = 0
        committed = 0
        discarded = 0
        for row in order.tolist():
            key = int(reg.key[row])
            data = cache.arrived.pop(key)
            visible = reg.tier[row] == _VISIBLE
            if cache.free:
                slot = cache.free.pop()
            else:
                if victims is None:
                    victims = self._recent_residents_least_first(reg)
                victim = int(victims[next_victim]) if next_victim < len(victims) else -1
                if visible:
                    if victim < 0:
                        raise AssertionError(
                            f"cache {cache.cache_id}: no RECENT resident to evict "
                            f"for a VISIBLE arrival (invariant 3b; the planner "
                            f"must truncate to n_slots - 1)"
                        )
                elif victim < 0 or self._importance(reg, victim) >= self._importance(
                    reg, row
                ):
                    # A kept arrival evicts only a strictly lower-ranked one.
                    if self.trace is not None:
                        self.trace(
                            "discard",
                            (
                                cache.cache_id,
                                _snapshot(reg, row),
                                len(cache.free),
                                self._recent_resident_snapshots(reg),
                            ),
                        )
                    reg.kill(row)
                    discarded += 1
                    continue
                next_victim += 1
                slot = int(reg.slot[victim])
                if self.trace is not None:
                    self.trace(
                        "evict",
                        (
                            cache.cache_id,
                            _snapshot(reg, victim),
                            _snapshot(reg, row),
                            self._recent_resident_snapshots(reg),
                        ),
                    )
                reg.kill(victim)
            reg.state[row] = _RESIDENT
            reg.slot[row] = slot
            try:
                cache.residency.write(slot, key, data)
            except Exception:
                # A write that raises will raise again: give the record up
                # (a later pass that wants it grants one more attempt) rather
                # than retry into a storm.
                _LOGGER.exception(
                    "cache %s: write of key %d failed", cache.cache_id, key
                )
                cache.free.append(slot)
                reg.slot[row] = -1
                if visible:
                    reg.state[row] = _FAILED
                    reg.attempts[row] = self.config.retry_max_attempts
                else:
                    reg.kill(row)
                continue
            committed += 1
            if self.trace is not None:
                self.trace("commit", (cache.cache_id, key, int(reg.tier[row])))
        cache.arrived_since = None
        if _LOGGER.isEnabledFor(logging.INFO):
            _LOGGER.info(
                "commit  cache=%s committed=%d evicted=%d discarded=%d free=%d",
                cache.cache_id,
                committed,
                next_victim,
                discarded,
                len(cache.free),
            )
        return committed

    def _recent_residents_least_first(self, reg: CacheRegistry) -> np.ndarray:
        rows = np.flatnonzero((reg.state == _RESIDENT) & (reg.tier == _RECENT))
        cls = reg.cls[rows].astype(np.int32)
        order = np.lexsort((-reg.rank[rows], reg.wanted_gen[rows], cls))
        return rows[order]

    def _recent_resident_snapshots(self, reg: CacheRegistry) -> list[tuple]:
        rows = np.flatnonzero((reg.state == _RESIDENT) & (reg.tier == _RECENT))
        return [_snapshot(reg, int(r)) for r in rows]

    @staticmethod
    def _importance(reg: CacheRegistry, row: int) -> tuple[int, int, int]:
        return recent_importance(
            int(reg.cls[row]), int(reg.wanted_gen[row]), int(reg.rank[row])
        )

    # -- invalidation (5.14) -----------------------------------------------------

    def invalidate(self, store_id: Any, region: Any = None) -> list[int]:
        """Forget data read from a store in *region* (all of it for ``None``).

        A cache is on the store if its latest desired set named it.

        ``FETCHING`` records are marked stale: the read lands and is dropped.
        ``RESIDENT``, ``ARRIVED`` and ``FAILED`` records give up their data
        and slot.  Each is requeued if ``VISIBLE`` and deleted if ``RECENT``.
        ``QUEUED`` records have read nothing and are left alone.

        Parameters
        ----------
        store_id : Any
            The store's id, as :func:`store_key` gives it.
        region : Any
            Handed to each cache's ``Residency.keys_in_region``.

        Returns
        -------
        list[int]
            The caches touched; each was redrawn.
        """
        touched: list[int] = []
        for cache in self._caches.values():
            if cache.store_id is None or cache.store_id != store_id:
                continue
            reg = cache.registry
            reg.compact()
            if not len(reg):
                continue
            if region is None:
                hit = np.ones(len(reg.key), dtype=bool)
            else:
                hit = np.asarray(
                    cache.residency.keys_in_region(reg.key, region), dtype=bool
                )
            if not hit.any():
                continue
            if self.trace is not None:
                self.trace("invalidate", (cache.cache_id, reg.key[hit].copy()))
            if _LOGGER.isEnabledFor(logging.INFO):
                _LOGGER.info(
                    "invalidate  cache=%s hit=%d in_flight=%d resident=%d",
                    cache.cache_id,
                    int(hit.sum()),
                    int((hit & (reg.state == _FETCHING)).sum()),
                    int((hit & (reg.state == _RESIDENT)).sum()),
                )
            reg.stale[hit & (reg.state == _FETCHING)] = True
            lose = hit & (
                (reg.state == _RESIDENT)
                | (reg.state == _ARRIVED)
                | (reg.state == _FAILED)
            )
            for row in np.flatnonzero(lose & (reg.state == _RESIDENT)).tolist():
                cache.free.append(int(reg.slot[row]))
            for key in reg.key[lose & (reg.state == _ARRIVED)].tolist():
                cache.arrived.pop(int(key), None)
            if not cache.arrived:
                cache.arrived_since = None
            reg.slot[lose] = -1
            requeue = lose & (reg.tier == _VISIBLE)
            reg.state[requeue] = _QUEUED
            reg.attempts[requeue] = 0
            reg.kill(np.flatnonzero(lose & (reg.tier != _VISIBLE)))
            cache.queue_dirty = True
            # Compacts the registry, so the masks above are stale after it.
            self._settle(cache)
            touched.append(cache.cache_id)
        return touched

    # -- completion, drawing, progress (5.8, 5.13) -------------------------------

    def _done_mask(self, reg: CacheRegistry) -> np.ndarray:
        return (reg.state == _RESIDENT) | (
            (reg.state == _FAILED) & (reg.attempts >= self.config.retry_max_attempts)
        )

    def is_complete(self, cache_id: int) -> bool:
        """Every ``VISIBLE`` record is resident or given up."""
        reg = self._caches[cache_id].registry
        visible = (reg.tier == _VISIBLE) & (reg.state != DEAD)
        return bool(self._done_mask(reg)[visible].all())

    def is_backstop_complete(self, cache_id: int) -> bool:
        """Every ``VISIBLE`` backstop record of the latest pass is done."""
        cache = self._caches[cache_id]
        return cache.backstop_gen == cache.generation

    def _settle(self, cache: _Cache) -> None:
        """Fire completion events, rebuild the cache's draw, report progress."""
        reg = cache.registry
        visible = (reg.tier == _VISIBLE) & (reg.state != DEAD)
        done = self._done_mask(reg)
        complete = bool(done[visible].all())
        gen = cache.generation
        if cache.backstop_gen < gen and done[visible & (reg.cls == _BACKSTOP)].all():
            cache.backstop_gen = gen
            if self.on_backstop_complete is not None:
                self.on_backstop_complete(cache.cache_id, gen)
        if cache.completed_gen < gen and complete:
            cache.completed_gen = gen
            if self.on_complete is not None:
                self.on_complete(cache.cache_id, gen)
        cache.residency.rebuild_draw(reg.view(gen, complete))
        if self.on_progress is not None:
            self.on_progress(cache.cache_id)

    def pass_stats(self, cache_id: int) -> tuple[int, int]:
        """``(desired, new)`` for the cache's latest pass.

        *new* counts keys that had no record: everything else was already
        resident, arriving, in flight or queued, and costs no new read.
        """
        return self._caches[cache_id].last_pass

    def generation(self, cache_id: int) -> int:
        """The cache's current pass generation."""
        return self._caches[cache_id].generation

    def progress(self, cache_id: int) -> CacheProgress:
        """Loading progress for one cache (design 5.13)."""
        cache = self._caches[cache_id]
        reg = cache.registry
        visible = (reg.tier == _VISIBLE) & (reg.state != DEAD)
        backstop = visible & (reg.cls == _BACKSTOP)
        target = visible & (reg.cls != _BACKSTOP)
        resident = reg.state == _RESIDENT
        gave_up = (reg.state == _FAILED) & (
            reg.attempts >= self.config.retry_max_attempts
        )
        return CacheProgress(
            needed_backstop=int(backstop.sum()),
            resident_backstop=int((backstop & resident).sum()),
            needed_target=int(target.sum()),
            resident_target=int((target & resident).sum()),
            in_flight=cache.n_fetching,
            failed=int((visible & gave_up).sum()),
            truncated_target=cache.truncated_target,
            truncated_backstop=cache.truncated_backstop,
        )

    # -- introspection -----------------------------------------------------------

    def registry(self, cache_id: int) -> CacheRegistry:
        """The cache's registry (read it; do not mutate it)."""
        return self._caches[cache_id].registry

    def free_slots(self, cache_id: int) -> list[int]:
        """The cache's free slots (a copy)."""
        return list(self._caches[cache_id].free)

    def arrived_keys(self, cache_id: int) -> list[int]:
        """Keys waiting for a commit round."""
        return list(self._caches[cache_id].arrived)

    def idle(self) -> bool:
        """Nothing waiting: no pass, read, arrival or retry."""
        if self._pending or any(self.in_flight):
            return False
        for cache in self._caches.values():
            if cache.arrived:
                return False
            reg = cache.registry
            if (reg.state == _QUEUED).any():
                return False
            waiting = (reg.state == _FAILED) & (
                reg.attempts < self.config.retry_max_attempts
            )
            if waiting.any():
                return False
        return True

    def clear(self) -> None:
        """Forget every cache (``close``)."""
        self._caches.clear()
        self._pending.clear()
