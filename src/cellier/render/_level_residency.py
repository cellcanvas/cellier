"""Whole-level geometry on the chunk scheduler, and what of it is drawn.

``plans/mesh_refactor_v3.md`` L2 and L3.  :class:`LevelResidency` is the
scheduler ``Residency`` of a visual whose data is a few whole levels (a mesh:
the fine level, and for a multiscale mesh a coarse one), not many bricks.
One per visual, so one scheduler cache per visual.

- A key is one level at one request: ``(level, request key)``, interned to
  the ``int64`` the scheduler works with.  The same level at another slider
  position is another key, so the scheduler's pass diff reads nothing for an
  unchanged request and never reads a position the slider has left twice.
- The helper knows nothing about meshes.  The visual gives it two
  callbacks: ``upload(level, data)`` puts a result on the GPU, and
  ``release(level)`` takes it off.

The display rule (L3, decision D8): **a level is drawn only if it holds the
key of the newest plan.**  ``planned`` is set when the plan is made, so a
slider tick stops the old position being drawn in the next frame, and an
arrival for a position the slider has left is never drawn.  While the new
position loads, nothing is drawn.

No revisit guarantee (D9): a level whose key leaves the plan is released at
once, and coming back to that position reads again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.logging import _SCHEDULER_LOGGER
from cellier.render.block_cache._image_residency import new_cache_id
from cellier.render.scheduling import (
    CachePolicy,
    ChunkClass,
    ChunkState,
    DesiredSet,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Mapping

    from cellier.render.scheduling import RegistryView

_RESIDENT = int(ChunkState.RESIDENT)

#: How the scheduler treats a level cache's reads (L2):
#:
#: - one target (fine) read in flight per visual; a coarse read is never
#:   held by the cap (K1);
#: - reads are computation in an executor, so they count against
#:   ``SchedulerConfig.compute_budget`` (K2);
#: - a read that raises fails the same way every time: one attempt, and no
#:   new attempt on later passes (K6).
LEVEL_CACHE_POLICY = CachePolicy(
    max_target_fetching=1,
    resource="compute",
    retry_max_attempts=1,
    retry_on_pass=False,
)


class LevelResidency:
    """Resident levels of one visual, and which of them may be drawn.

    Parameters
    ----------
    n_levels : int
        Levels the visual keeps resident: 1 (a single-level visual) or 2
        (coarse and fine).
    upload : Callable[[int, Hashable, Any], None]
        ``(level, request_key, data)``: put a read's result on the GPU as
        that level.  Called from ``rebuild_draw``, on the UI thread.  If it
        raises, the error is logged and the level holds nothing.
    release : Callable[[int], None]
        ``(level)``: the level holds nothing any more.
    on_change : Callable[[], None] or None
        Called after a plan and after ``rebuild_draw``: what may be drawn
        may have changed.
    fine_level : int
        The level that is the ``TARGET`` class; every other level is
        ``BACKSTOP``.

    Attributes
    ----------
    cache_id : int
        The scheduler cache id the visual's desired sets carry.
    n_slots : int
        ``n_levels + 1``: one spare slot lets a stale arrival land.
    policy : CachePolicy
        :data:`LEVEL_CACHE_POLICY`.
    planned : dict[int, int]
        ``level -> key`` of the newest plan.
    held : dict[int, tuple[int, int]]
        ``level -> (key, write serial)`` of what the level has uploaded.
    """

    policy: CachePolicy = LEVEL_CACHE_POLICY

    def __init__(
        self,
        n_levels: int,
        upload: Callable[[int, Hashable, Any], None],
        release: Callable[[int], None],
        *,
        on_change: Callable[[], Any] | None = None,
        fine_level: int = 0,
    ) -> None:
        if n_levels < 1:
            raise ValueError(f"n_levels must be at least 1, got {n_levels}.")
        self.cache_id: int = new_cache_id()
        self.n_slots: int = n_levels + 1
        self.fine_level = fine_level
        self._upload = upload
        self._release = release
        self._on_change = on_change
        self.planned: dict[int, int] = {}
        self.held: dict[int, tuple[int, int]] = {}
        # slot -> (key, data, write serial); data is None once uploaded or
        # dropped.
        self._payloads: dict[int, tuple[int, Any, int]] = {}
        self._writes = 0
        # The interner: (level, request key) <-> int64, and the store
        # request each live key reads with.
        self._ids: dict[tuple[int, Hashable], int] = {}
        self._items: dict[int, tuple[int, Hashable]] = {}
        self._requests: dict[int, Any] = {}
        self._next_key = 1
        self._slice_ids: dict[Hashable, int] = {}

    # -- the interner --------------------------------------------------------

    def _intern(self, level: int, request_key: Hashable) -> int:
        item = (level, request_key)
        key = self._ids.get(item)
        if key is None:
            key = self._next_key
            self._next_key += 1
            self._ids[item] = key
            self._items[key] = item
        return key

    def _forget(self, key: int) -> None:
        """Stop handing *key* out: its item gets a new key next time."""
        item = self._items.get(key)
        if item is not None and self._ids.get(item) == key:
            del self._ids[item]

    def key_of(self, level: int, request_key: Hashable) -> int | None:
        """The live key of ``(level, request_key)``, if it has one."""
        return self._ids.get((level, request_key))

    def item_of(self, key: int) -> tuple[int, Hashable] | None:
        """``(level, request key)`` of *key*, while the helper knows it."""
        return self._items.get(int(key))

    # -- planning ------------------------------------------------------------

    def desired(
        self,
        request_keys: Mapping[int, Hashable],
        asked: Collection[int],
        requests: Mapping[int, Any],
    ) -> DesiredSet:
        """Adopt a plan, and describe it to the scheduler.

        Parameters
        ----------
        request_keys : Mapping[int, Hashable]
            ``level -> request key`` for every level of the visual.  A
            request key holds everything the read depends on except the
            store's contents.
        asked : Collection[int]
            The levels the plan mode asks for.  A level left out is kept in
            the plan while its key is unchanged (decision D23): it costs no
            read, and a level that is still valid is not thrown away because
            a scrub of some other axis plans coarse only.
        requests : Mapping[int, Any]
            ``level -> store request`` for every level.

        Returns
        -------
        DesiredSet
            Coarsest level first.  ``store`` is ``None``: the coordinator
            attaches the visual's store.
        """
        planned: dict[int, int] = {}
        # Coarsest first: the backstop block leads a desired set.
        for level in sorted(request_keys, reverse=True):
            request_key = request_keys[level]
            if level in asked:
                key = self._intern(level, request_key)
            else:
                key = self._ids.get((level, request_key))
                if key is None or self.planned.get(level) != key:
                    continue
            planned[level] = key
            self._requests[key] = requests[level]
        # D9: a level that no longer holds the plan's key is released now,
        # when the plan is made, so no frame draws the position just left.
        for level in [lv for lv, (key, _) in self.held.items()]:
            if planned.get(level) != self.held[level][0]:
                self._let_go(level)
        self.planned = planned
        if self._on_change is not None:
            self._on_change()

        levels = list(planned)
        keys = [planned[level] for level in levels]
        cls = [
            ChunkClass.TARGET if level == self.fine_level else ChunkClass.BACKSTOP
            for level in levels
        ]
        slice_key = tuple(request_keys[level] for level in levels)
        if len(self._slice_ids) > 4096:
            self._slice_ids.clear()
        slice_id = self._slice_ids.setdefault(slice_key, len(self._slice_ids))
        return DesiredSet(
            cache_id=self.cache_id,
            keys=np.array(keys, dtype=np.int64),
            cls=np.array(cls, dtype=np.uint8),
            slice_ids=np.full(len(keys), slice_id, dtype=np.int32),
            build_request=self._build_requests,
            store=None,
        )

    def _build_requests(self, keys: np.ndarray) -> list[Any]:
        return [self._requests[int(key)] for key in keys]

    def _let_go(self, level: int) -> None:
        """Release what *level* holds and forget its key, together."""
        key, _serial = self.held.pop(level)
        self._forget(key)
        for slot, (slot_key, _data, serial) in list(self._payloads.items()):
            if slot_key == key:
                self._payloads[slot] = (slot_key, None, serial)
        self._release(level)

    # -- the Residency protocol ----------------------------------------------

    def write(self, slot: int, key: int, data: Any) -> None:
        """Keep a committed result by reference; upload nothing yet.

        The scheduler commits arrivals that are no longer wanted too, so the
        upload waits for ``rebuild_draw`` to find the key in the plan.
        """
        self._writes += 1
        self._payloads[int(slot)] = (int(key), data, self._writes)

    def rebuild_draw(self, view: RegistryView) -> None:
        """Bring the uploads in line with the registry and the newest plan."""
        resident = np.flatnonzero(view.state == _RESIDENT)
        slot_of = {int(view.key[row]): int(view.slot[row]) for row in resident.tolist()}

        # 1. A held level whose record is no longer resident: an invalidation
        #    (or an eviction while retired) took it.  The key stays; if it is
        #    still wanted the scheduler reads it again.
        for level in [lv for lv, (key, _) in self.held.items() if key not in slot_of]:
            self.held.pop(level)
            self._release(level)

        # Payloads of slots the registry gave to something else.
        live = set(slot_of.values())
        for slot in [s for s in self._payloads if s not in live]:
            del self._payloads[slot]

        # 2. A resident key of the newest plan that its level does not hold.
        for level, key in self.planned.items():
            slot = slot_of.get(key)
            if slot is None:
                continue
            payload = self._payloads.get(slot)
            if payload is None or payload[0] != key:
                continue
            _key, data, serial = payload
            if self.held.get(level) == (key, serial) or data is None:
                continue
            self._payloads[slot] = (key, None, serial)
            try:
                self._upload(level, self._items[key][1], data)
            except Exception:
                # The same data would fail the same way: the level holds
                # nothing, and the payload is not tried again.
                _SCHEDULER_LOGGER.exception(
                    "cache %s: upload of level %d failed", self.cache_id, level
                )
                self._release(level)
                self.held.pop(level, None)
                continue
            self.held[level] = (key, serial)

        # 3. Every other resident result is dropped, and its key forgotten
        #    with it, so the same position later gets a new key and a read.
        wanted = set(self.planned.values())
        for slot, (key, data, serial) in list(self._payloads.items()):
            if key in wanted:
                continue
            self._forget(key)
            if data is not None:
                self._payloads[slot] = (key, None, serial)

        # Forget what neither the registry nor the plan names any more.  A
        # planned key is kept although the registry may not know it yet:
        # passes are applied one loop iteration after the plan.
        known = {int(key) for key in view.key.tolist()} | wanted
        for key in [k for k in self._items if k not in known]:
            item = self._items.pop(key)
            if self._ids.get(item) == key:
                del self._ids[item]
            self._requests.pop(key, None)
        if self._on_change is not None:
            self._on_change()

    def keys_in_region(self, keys: np.ndarray, region: Any) -> np.ndarray:
        """Every key: a result is a whole level, so any change touches it."""
        return np.ones(len(keys), dtype=bool)

    # -- the display rule (L3) -----------------------------------------------

    def is_drawable(self, level: int) -> bool:
        """Whether *level* holds the newest plan's key for it."""
        held = self.held.get(level)
        return held is not None and held[0] == self.planned.get(level)

    @property
    def awaiting(self) -> bool:
        """Whether nothing is drawable while a plan's read is still to land."""
        return bool(self.planned) and not any(
            self.is_drawable(level) for level in self.held
        )

    def level_to_draw(self, *, prefer_coarse: bool = False) -> int | None:
        """The one level to draw now, or ``None`` to draw nothing.

        The fine level, or the coarsest drawable level when *prefer_coarse*
        (a moving camera, a dims scrub); when the preferred level is not
        drawable, another drawable one.  Never two.
        """
        drawable = [level for level in self.held if self.is_drawable(level)]
        if not drawable:
            return None
        return max(drawable) if prefer_coarse else min(drawable)

    def drawn_serial(self, level: int | None) -> tuple[int, int] | None:
        """``(level, write serial)`` of what drawing *level* shows."""
        if level is None or level not in self.held:
            return None
        return level, self.held[level][1]
